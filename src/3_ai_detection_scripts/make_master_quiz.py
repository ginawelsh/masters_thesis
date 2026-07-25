"""Build the canonical human-vs-AI detection quiz (`quiz_master`).

Single source of truth for the detection experiment's HUMAN arm: one balanced,
GPT-only quiz spanning both registers, that humans, the LLM detectors, and the
own detector all classify. Machine-only, both-generator power comes from a
separate, larger corpus (make_detection_corpus.py) — this file is the ~55-item
subset a person can realistically sit.

Generator scope: GPT-5.2 only (matches the main §1–§3 analysis and keeps the
per-cell n usable at this size). Mistral is NOT in the human quiz by design.

Condition scope: human + baseline + human_like + detector_evasive — the three
REPORTED prompt conditions plus human, aligned to the thesis (detector_aware is
excluded). Texts are copied VERBATIM from the source CSV cells; human comments
and abstracts are never altered.

Composition (edit the constants to resize):
  Formal:   N_FORMAL_PER_COND items per condition  -> 4 * that many abstracts
  Informal: N_INFORMAL_PER_COND items per condition -> 4 * that many comments
Each quiz item is a DISTINCT source row for its register, so a human text and
its own LLM regeneration never co-occur.

Outputs (into this folder):
  quiz_master.csv        full data + answer key
                         (item_id, register, generator, condition, is_human,
                          context, text, source_row)
  quiz_master_human.md   human-facing quiz: two sections, no answers up top,
                         answer key hidden at the bottom
"""
import os
import random
import re

import pandas as pd

# ---- composition (4 conditions: human / baseline / human_like / detector_evasive)
N_FORMAL_PER_COND = 6       # 6 * 4 = 24 abstracts
N_INFORMAL_PER_COND = 8     # 8 * 4 = 32 comments   -> 56 items total
SEED = 42

# informal readability band (text actually shown); formal uses a min length only
INF_MIN_LEN, INF_MAX_LEN = 40, 1000
INF_MAX_QLEN = 300          # keep thread questions short enough to display
FORMAL_MIN_LEN = 150        # abstracts shorter than this are dropped as candidates

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
FORMAL_CSV = os.path.join(_src, "1_data_collection", "llm_abstracts", "abstracts",
                          "sv_abstracts_adversarial.csv")
INFORMAL_CSV = os.path.join(_src, "1_data_collection", "llm_comments",
                            "consolidated_informal_comments_adversarial.csv")
OUT_CSV = os.path.join(_here, "quiz_master.csv")
OUT_MD = os.path.join(_here, "quiz_master_human.md")

CONDITIONS = ["human", "baseline", "human_like", "detector_evasive"]

# condition -> source column, per register (GPT columns)
FORMAL_COL = {
    "human": "Abstract",
    "baseline": "Abstract_baseline",
    "human_like": "Abstract_human_like",
    "detector_evasive": "Abstract_detector_evasive",
}
INFORMAL_COL = {
    "human": "human_comment",
    "baseline": "generated_comment",
    "human_like": "comment_human_like",
    "detector_evasive": "comment_detector_evasive",
}

# Explicit / adult-content blocklist (informal only). A whole question is dropped
# if the question text or ANY condition cell matches. Word-boundary / stem anchored
# to avoid false positives ("analys", "kanal", "sexton", "Fittja"). Never edits text.
_BLOCK_PATTERNS = [
    r"porr", r"porno", r"pornogr", r"\bporn\b", r"\bxxx\b",
    r"knull", r"\bfitt(a|an|or|orna)\b", r"\bkuk(ar|en)?\b", r"snopp",
    r"runk", r"onan", r"samlag", r"orgasm", r"avsug", r"\bdildo",
    r"vibrator", r"\bsex\b", r"sexuell", r"sexst", r"sexchatt", r"sexfilm",
    r"sexvideo", r"analsex", r"\banala\b", r"trekant", r"gangbang",
    r"blowjob", r"\bhora\b", r"\bhoror\b", r"bordell", r"prostitu",
    r"eskort", r"\bescort", r"nakenbild", r"\bnaken\b", r"brostvart",
    r"stånd", r"erektion", r"\bslampa\b", r"\bhorunge\b",
]
_BLOCK_RE = re.compile("|".join(_BLOCK_PATTERNS), re.IGNORECASE)


def _nonempty(v):
    s = str(v).strip()
    return s and s.lower() != "nan"


def _blocked(v):
    return bool(_BLOCK_RE.search(str(v)))


def _select(df, cols, per_cond, *, min_len, max_len=None, blocklist=False,
            context_col=None, max_ctx_len=None, rng=None):
    """Pick distinct fully-valid rows and assign one condition-role to each.

    A row is a candidate only if every condition cell (and the context, for
    informal) is present, within the length band, and — for informal — clears
    the content blocklist. This guarantees a chosen row can serve whichever
    condition it's assigned, and that a human item is never the twin of an LLM
    item (each item is a distinct row).
    """
    needed = per_cond * len(CONDITIONS)
    cand = []
    for idx, r in df.iterrows():
        if context_col is not None:
            ctx = r[context_col]
            if not _nonempty(ctx):
                continue
            if max_ctx_len and len(str(ctx)) > max_ctx_len:
                continue
        ok = True
        for c in CONDITIONS:
            v = r[cols[c]]
            if not _nonempty(v):
                ok = False
                break
            L = len(str(v))
            if L < min_len or (max_len and L > max_len):
                ok = False
                break
            if blocklist and _blocked(v):
                ok = False
                break
        if blocklist and context_col is not None and _blocked(r[context_col]):
            ok = False
        if ok:
            cand.append(idx)

    if len(cand) < needed:
        raise SystemExit(f"Only {len(cand)} valid rows; need {needed} "
                         f"({per_cond}/condition x {len(CONDITIONS)}).")

    rows = rng.sample(cand, needed)
    roles = []
    for c in CONDITIONS:
        roles += [c] * per_cond
    rng.shuffle(roles)
    return list(zip(rows, roles))


def _emit(df, assigned, cols, register, context_col):
    recs = []
    for row_idx, cond in assigned:
        recs.append({
            "register": register,
            "generator": "human" if cond == "human" else "gpt",
            "condition": cond,
            "is_human": cond == "human",
            "context": str(df.at[row_idx, context_col]),        # title / question
            "text": str(df.at[row_idx, cols[cond]]),             # verbatim
            "source_row": int(row_idx),
        })
    return recs


def main():
    rng = random.Random(SEED)

    formal = pd.read_csv(FORMAL_CSV, encoding="utf-8").reset_index(drop=True)
    informal = pd.read_csv(INFORMAL_CSV, encoding="utf-8").reset_index(drop=True)
    if len(formal) != 170 or len(informal) != 1149:
        print(f"[warn] unexpected corpus sizes: formal={len(formal)} (exp 170), "
              f"informal={len(informal)} (exp 1149) — post-exclusion state may differ.")

    f_assigned = _select(formal, FORMAL_COL, N_FORMAL_PER_COND,
                         min_len=FORMAL_MIN_LEN, context_col="Title", rng=rng)
    i_assigned = _select(informal, INFORMAL_COL, N_INFORMAL_PER_COND,
                         min_len=INF_MIN_LEN, max_len=INF_MAX_LEN, blocklist=True,
                         context_col="question", max_ctx_len=INF_MAX_QLEN, rng=rng)

    recs = _emit(formal, f_assigned, FORMAL_COL, "formal", "Title") \
        + _emit(informal, i_assigned, INFORMAL_COL, "informal", "question")

    # stable, register-blocked display order (formal section then informal section);
    # within each section shuffle so conditions aren't positional
    formal_recs = [r for r in recs if r["register"] == "formal"]
    informal_recs = [r for r in recs if r["register"] == "informal"]
    rng.shuffle(formal_recs)
    rng.shuffle(informal_recs)
    ordered = formal_recs + informal_recs
    for i, r in enumerate(ordered, 1):
        r["item_id"] = i

    out = pd.DataFrame(ordered, columns=[
        "item_id", "register", "generator", "condition", "is_human",
        "context", "text", "source_row",
    ])
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # ---- human-facing quiz ----
    lines = [
        "# Human vs AI text quiz",
        "",
        "Two sections of Swedish texts. For each text, decide whether it was "
        "**written by a human** or **generated by an AI**. Judge the text itself — "
        "the title/thread question is only context.",
        "",
    ]

    def section(title, items, kind):
        block = [f"## Section {title}", ""]
        for it in items:
            ctx_label = "Thesis title" if kind == "formal" else "Thread"
            block += [
                f"### {it['item_id']}. _{ctx_label}: {it['context']}_",
                "",
                it["text"],
                "",
                "**Human or AI?** ______",
                "",
                "---",
                "",
            ]
        return block

    lines += section("A — thesis abstracts", formal_recs, "formal")
    lines += section("B — forum comments", informal_recs, "informal")

    lines += [
        "",
        "<!-- ================= ANSWER KEY (do not read until done) ================= -->",
        "",
        "## Answer key",
        "",
        "| # | Answer | Generator | Condition | Register |",
        "|---|--------|-----------|-----------|----------|",
    ]
    for it in ordered:
        ans = "Human" if it["is_human"] else "AI"
        lines.append(f"| {it['item_id']} | {ans} | {it['generator']} | "
                     f"{it['condition']} | {it['register']} |")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    counts = out.groupby(["register", "condition"]).size().to_dict()
    print(f"Wrote {len(out)} items "
          f"({len(formal_recs)} formal + {len(informal_recs)} informal), GPT-only.")
    print(f"  per (register, condition): {counts}")
    print(f"  distinct source rows: {out['source_row'].nunique()} of {len(out)}")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_MD}")


if __name__ == "__main__":
    main()
