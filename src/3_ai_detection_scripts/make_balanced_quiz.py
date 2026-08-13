"""Build a LARGER, balanced, MATCHED human-vs-AI detection corpus for the LLM judges.

Why this exists
---------------
`make_master_quiz.py` builds the ~56-item quiz a *person* can sit: it uses one
DISTINCT source row per item so a human and its own regeneration never co-occur.
That distinct-row rule caps the FORMAL register at 170 items total (~28 per cell),
which leaves the per-condition detection rates statistically underpowered
(n≈7-14 per cell in the shipped quiz).

This script targets the *automated* judges instead (Claude / DeepSeek / Gemini in
ai_detection_llm.py), which score every item in an ISOLATED API call. Because a
text and its twin never share a prompt, nothing leaks — so we can use the far more
powerful MATCHED-PAIRS design:

  * sample N_DOCS distinct source documents per register;
  * from EACH document emit all four aligned texts
      human / baseline / human_like / detector_evasive
    (same source row -> same `pair_id`).

Per register this yields  N_DOCS human + N_DOCS per AI condition, i.e. every
per-condition comparison is a balanced  N_DOCS-vs-N_DOCS  paired contrast.
At the default N_DOCS=100 each (register x condition) cell has n=100
(95% CI half-width ~10pp) versus ~7-14 before.

Generator scope: GPT-5.2 only (matches the reported §1-§3 analysis; Mistral is a
separate factor, not a way to grow n). Conditions: the three REPORTED prompts plus
human (detector_aware excluded). Texts are copied VERBATIM from the source cells;
human texts are never altered.

Class balance
-------------
Within each per-condition contrast the classes are 50/50 (N_DOCS human vs N_DOCS
AI, the human set reused across conditions). The POOLED file is therefore
1 human : 3 AI by construction (three AI conditions share one human set) --
analyse it per condition (+ balanced accuracy), not as one pooled accuracy.

Analysis hook
-------------
ai_detection_llm.py only carries item_id/register/generator/condition/is_human/text
into its results CSV, so `pair_id` is not written to the results. Recover it by
joining the results back to this quiz on `item_id` (1:1), then run McNemar / a
mixed-effects logistic `correct ~ register*condition*judge + (1|pair_id)`.

Output (into this folder)
  quiz_master_balanced.csv
    item_id, register, generator, condition, is_human, context, text,
    source_row, pair_id

Usage
  python make_balanced_quiz.py                     # 100 docs/register (default)
  python make_balanced_quiz.py --formal 100 --informal 100 --seed 42
  # then, from this folder:
  python ai_detection_llm.py --provider all --corpus quiz_master_balanced.csv
"""
import argparse
import os
import re

import pandas as pd

# ---- defaults (per register; a "doc" contributes 1 human + 3 AI items) ---------
DEFAULT_FORMAL_DOCS = 100
DEFAULT_INFORMAL_DOCS = 100
DEFAULT_SEED = 42

# text length bands (of the text actually shown), mirroring make_master_quiz.py
INF_MIN_LEN, INF_MAX_LEN = 40, 1000
INF_MAX_QLEN = 300          # keep thread questions short enough to display
FORMAL_MIN_LEN = 150        # abstracts shorter than this are dropped as candidates

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)
FORMAL_CSV = os.path.join(_src, "1_data_collection", "llm_abstracts", "abstracts",
                          "sv_abstracts_adversarial.csv")
INFORMAL_CSV = os.path.join(_src, "1_data_collection", "llm_comments",
                            "consolidated_informal_comments_adversarial.csv")
OUT_CSV = os.path.join(_here, "quiz_master_balanced.csv")

# human + the three REPORTED conditions (detector_aware excluded), aligned to thesis
CONDITIONS = ["human", "baseline", "human_like", "detector_evasive"]

FORMAL_COL = {
    "human": "Abstract",
    "baseline": "Abstract_baseline",
    "human_like": "Abstract_human_like",
    "detector_evasive": "Abstract_detector_evasive",
}
INFORMAL_COL = {
    "human": "human_comment",
    # DELIBERATELY STILL `generated_comment`, even though that column is gpt-4o-mini
    # text rather than GPT-5.2 (see ../../BASELINE_PROVENANCE.md). data_utils.py has been
    # repointed at the corrected `comment_baseline`; this script has NOT, on purpose:
    #
    #   _candidates() filters on every condition cell's length and blocklist match, so a
    #   different baseline column can change which documents qualify. Re-running with
    #   comment_baseline would therefore produce a DIFFERENT 100-document sample, and the
    #   800 judgements in gpt52_run_results/ are tied to the current one.
    #
    # Reproducibility of the shipped quiz wins. Nothing needs this changed:
    # make_followup_quiz.py reads the corrected baseline directly from the generation
    # output, and pins to the documents this script already chose.
    "baseline": "generated_comment",
    "human_like": "comment_human_like",
    "detector_evasive": "comment_detector_evasive",
}

# Explicit / adult-content blocklist (informal only), copied verbatim from
# make_master_quiz.py so the two quizzes exclude the same material. Never edits text.
#
# KNOWN FALSE POSITIVES -- DOCUMENTED, DELIBERATELY NOT FIXED HERE.
# Two patterns below are near-pure false positives in Swedish:
#   r"\bsex\b"  matches the numeral SIX ("efter sex månader")  -- 72/1149 documents
#   r"stånd"    is unanchored, so it fires inside avstånd, uppehållstillstånd,
#               motstånd(are), missförstånd, förstånd ...      -- 106/1149 documents
# Together they excluded 141 of 1,149 informal documents (12%) from the candidate pool:
# the eligible pool was 909 rather than 1,050. The 100 sampled documents are still a
# random draw from that 909, and the paired human-vs-AI design is unaffected, but topic
# coverage is skewed -- asylum/residence-permit, opposition and employment threads are
# under-represented. Report as a sampling limitation.
#
# NOT corrected here on purpose: changing the blocklist changes which documents are
# eligible, which changes the sample, which invalidates the 800 existing judgements in
# detection_results_*.csv. Reproducibility of the shipped quiz wins. The corrected
# patterns are used in make_followup_quiz.py, where the blocklist only warns.
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
    return bool(s) and s.lower() != "nan"


def _blocked(v):
    return bool(_BLOCK_RE.search(str(v)))


def _candidates(df, cols, *, min_len, max_len=None, blocklist=False,
                context_col=None, max_ctx_len=None):
    """Row indices where EVERY condition cell (and the context) is valid.

    A candidate document must be usable in all four roles at once, because the
    matched design emits all four from the same row.
    """
    out = []
    for idx, r in df.iterrows():
        if context_col is not None:
            ctx = r[context_col]
            if not _nonempty(ctx):
                continue
            if max_ctx_len and len(str(ctx)) > max_ctx_len:
                continue
            if blocklist and _blocked(ctx):
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
        if ok:
            out.append(idx)
    return out


def _emit(df, row_indices, cols, register, context_col):
    """One document -> four aligned records sharing a pair_id (the source row)."""
    recs = []
    for row_idx in row_indices:
        pair_id = f"{register}-{int(row_idx)}"
        for cond in CONDITIONS:
            recs.append({
                "register": register,
                "generator": "human" if cond == "human" else "gpt",
                "condition": cond,
                "is_human": cond == "human",
                "context": str(df.at[row_idx, context_col]),   # title / question
                "text": str(df.at[row_idx, cols[cond]]),        # verbatim
                "source_row": int(row_idx),
                "pair_id": pair_id,
            })
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--formal", type=int, default=DEFAULT_FORMAL_DOCS,
                    help="distinct formal documents to sample (default 100)")
    ap.add_argument("--informal", type=int, default=DEFAULT_INFORMAL_DOCS,
                    help="distinct informal documents to sample (default 100)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    import random
    rng = random.Random(args.seed)

    formal = pd.read_csv(FORMAL_CSV, encoding="utf-8").reset_index(drop=True)
    informal = pd.read_csv(INFORMAL_CSV, encoding="utf-8").reset_index(drop=True)
    if len(formal) != 170 or len(informal) != 1149:
        print(f"[warn] unexpected corpus sizes: formal={len(formal)} (exp 170), "
              f"informal={len(informal)} (exp 1149) — post-exclusion state may differ.")

    f_cand = _candidates(formal, FORMAL_COL, min_len=FORMAL_MIN_LEN,
                         context_col="Title")
    i_cand = _candidates(informal, INFORMAL_COL, min_len=INF_MIN_LEN,
                         max_len=INF_MAX_LEN, blocklist=True,
                         context_col="question", max_ctx_len=INF_MAX_QLEN)
    print(f"[candidates] formal={len(f_cand)} informal={len(i_cand)}")

    if len(f_cand) < args.formal:
        raise SystemExit(f"Only {len(f_cand)} valid formal docs; need {args.formal}. "
                         f"Lower --formal (formal is hard-capped at ~170).")
    if len(i_cand) < args.informal:
        raise SystemExit(f"Only {len(i_cand)} valid informal docs; need {args.informal}. "
                         f"Lower --informal.")

    f_pick = rng.sample(f_cand, args.formal)
    i_pick = rng.sample(i_cand, args.informal)

    recs = _emit(formal, f_pick, FORMAL_COL, "formal", "Title") \
        + _emit(informal, i_pick, INFORMAL_COL, "informal", "question")

    # register-blocked, then shuffle within each register so condition is not
    # positional (irrelevant to isolated LLM calls, but keeps the file tidy).
    formal_recs = [r for r in recs if r["register"] == "formal"]
    informal_recs = [r for r in recs if r["register"] == "informal"]
    rng.shuffle(formal_recs)
    rng.shuffle(informal_recs)
    ordered = formal_recs + informal_recs
    for i, r in enumerate(ordered, 1):
        r["item_id"] = i

    out = pd.DataFrame(ordered, columns=[
        "item_id", "register", "generator", "condition", "is_human",
        "context", "text", "source_row", "pair_id",
    ])
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")

    counts = out.groupby(["register", "condition"]).size().to_dict()
    n_human = int(out["is_human"].sum())
    print(f"Wrote {len(out)} items "
          f"({len(formal_recs)} formal + {len(informal_recs)} informal), GPT-only.")
    print(f"  class balance: human={n_human}  ai={len(out) - n_human}  "
          f"(per-condition contrasts are {args.formal}/{args.informal}-vs-same balanced)")
    print(f"  per (register, condition): {counts}")
    print(f"  distinct documents: formal={out[out.register=='formal'].pair_id.nunique()}, "
          f"informal={out[out.register=='informal'].pair_id.nunique()}")
    print(f"  -> {OUT_CSV}")


if __name__ == "__main__":
    main()
