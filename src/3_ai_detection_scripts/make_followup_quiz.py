"""Build the FOLLOW-UP detection quiz: GPT-5.6 + a corrected GPT-5.2 informal baseline.

Why a separate script (not a flag on make_balanced_quiz.py)
-----------------------------------------------------------
make_balanced_quiz.py *samples* documents with an RNG seed. It must stay byte-
reproducible, because quiz_master_balanced.csv and the three completed
detection_results_*.csv runs depend on it. This script does the opposite: it
*pins* to the documents that quiz already chose, so every new item is a matched
partner of an item that has already been judged.

That pinning is what makes the new contrasts PAIRED (McNemar) rather than
unpaired, which is worth roughly a 50% increase in n at zero API cost.

What goes in
------------
  GPT-5.6 INFORMAL, 3 conditions x 100 docs                  = 300 items
  GPT-5.2 informal baseline (corrected), 100 docs            = 100 items
  GPT-5.2 informal human_like, markdown-stripped re-judge    =   6 items
                                                               ---------
                                                               406 items

FORMAL IS DELIBERATELY EXCLUDED for GPT-5.6. Measured over 20 generations per model,
GPT-5.6's formal output carried four confounds that GPT-5.2's did not: it refused the
detector_evasive instruction 25% of the time (95% CI 11-47%), emitted markdown in 90%
of outputs, added a section heading and a "Nyckelord:" footer in ~75% (which SURVIVES
markdown stripping), and ran 45% shorter. A formal 5.2-vs-5.6 contrast would measure
formatting conventions, not Swedish. It would also have been uninformative regardless:
formal detection is already at ceiling (Claude 100%, Gemini 96-100% across conditions),
so every formal generator contrast comes back as no-discordance. The 25% refusal rate is
reported as a finding in its own right instead.

The 6 re-judge items exist because 6 of the 100 informal human_like items in the original
quiz contain markdown emphasis (e.g. *Sybil*) and were judged with it in, while newly
generated text is now stripped at write time -- an asymmetry in GPT-5.6's favour. They
carry a `supersedes` column naming the original item_id; the scorer drops any item that
another item supersedes, so the stripped judgement replaces the original rather than
double-counting the document.

The 200 human items and the already-judged GPT-5.2 cells are NOT re-emitted --
they are already scored in detection_results_*.csv. Join on `pair_id` to pool.

Why the GPT-5.2 informal baseline is re-generated: the original informal baseline
column (`generated_comment`) is gpt-4o-mini text, not GPT-5.2. See
../../BASELINE_PROVENANCE.md. The old judgments stay valid *as gpt-4o-mini* -- relabel
them, do not discard them.

Alignment
---------
Every input row carries `quiz_pair_id`, written by generation_pipeline.quiz_subset(),
which is the authoritative link back to the quiz document. Rows are matched on that
key, never on row order (generation output follows the generation input's order, not
the quiz's). Any missing or unexpected pair_id is a hard error: a partially-matched
quiz would look fine and not be comparable to the GPT-5.2 run.

item_id offsets
---------------
ai_detection_llm.py resumes by matching `item_id` against its existing results CSV,
so ids MUST NOT collide with the 1-800 already used. Default offset is 1001.

Output (into this folder)
  quiz_followup.csv
    item_id, register, generator, condition, is_human, context, text,
    source_row, pair_id

Usage
  python make_followup_quiz.py
  python make_followup_quiz.py --offset 2001 --out quiz_followup_v2.csv
  # then back up the old results, and:
  python ai_detection_llm.py --provider all --corpus quiz_followup.csv
"""
import argparse
import os
import re
import sys

import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.dirname(_here)

BASE_QUIZ = os.path.join(_here, "quiz_master_balanced.csv")
DEFAULT_OUT = os.path.join(_here, "quiz_followup.csv")
DEFAULT_OFFSET = 1001

_FORMAL_DIR = os.path.join(_src, "1_data_collection", "llm_abstracts", "abstracts")
_INFORMAL_DIR = os.path.join(_src, "1_data_collection", "llm_comments")

CONDITIONS = ["baseline", "human_like", "detector_evasive"]

# Each arm: (generator label, register, csv path, context col, {condition: text col})
def _arms(subset_tag="_quiz100"):
    """Each arm lists CANDIDATE filenames; the first that exists is used.

    The generation scripts fold every scoping choice into the filename, so a
    baseline-only run writes '..._quiz100_baseline.csv' while an all-conditions run
    writes '..._quiz100.csv'. Both are valid ways to produce the cells this quiz needs,
    so accept either rather than forcing a 3x more expensive run just to get the name
    the builder happens to expect.
    """
    stem = "generated_three_prompt_14_07_26"
    return [
        # NO GPT-5.6 formal arm -- see module docstring (refusals, markdown, structural
        # artefacts, length; and formal detection is already at ceiling).
        ("gpt-5.6", "informal",
         [os.path.join(_INFORMAL_DIR, f"{stem}_gpt-5.6{subset_tag}.csv")],
         "question",
         {c: f"comment_{c}" for c in CONDITIONS}),
        # corrected GPT-5.2 informal baseline only -- the other two 5.2 informal
        # conditions are already judged and are not re-emitted.
        ("gpt-5.2", "informal",
         [os.path.join(_INFORMAL_DIR, f"{stem}_gpt-5.2{subset_tag}_baseline.csv"),
          os.path.join(_INFORMAL_DIR, f"{stem}_gpt-5.2{subset_tag}.csv")],
         "question",
         {"baseline": "comment_baseline"}),
    ]

# Blocklist from make_balanced_quiz.py, applied here as a WARNING only -- dropping a
# row would unpair it from its already-judged GPT-5.2 partner and break the matched
# design, so the decision is left to the operator.
#
# TWO PATTERNS FROM THE ORIGINAL ARE DELIBERATELY REMOVED, because in Swedish they are
# almost pure false positives:
#   r"\bsex\b"  -- "sex" is the Swedish numeral SIX ("efter sex månader"). 72 of 1,149
#                  informal documents match it; the sexual senses are already covered by
#                  sexuell / sexst / sexchatt / sexfilm / sexvideo / analsex.
#   r"stånd"    -- unanchored, so it fires inside avstånd, uppehållstillstånd,
#                  motstånd(are), missförstånd, förstånd ... 106 of 1,149 documents match.
#                  Narrowed to \bstånd\b.
# Together those two excluded 141 of 1,149 documents (12%) from the eligible pool in the
# ORIGINAL quiz build. That is not fixed retroactively -- see the note in
# make_balanced_quiz.py -- because re-sampling would invalidate 800 existing judgements.
_BLOCK_RE = re.compile("|".join([
    r"porr", r"porno", r"pornogr", r"\bporn\b", r"\bxxx\b",
    r"knull", r"\bfitt(a|an|or|orna)\b", r"\bkuk(ar|en)?\b", r"snopp",
    r"runk", r"onan", r"samlag", r"orgasm", r"avsug", r"\bdildo",
    r"vibrator", r"sexuell", r"sexst", r"sexchatt", r"sexfilm",
    r"sexvideo", r"analsex", r"\banala\b", r"trekant", r"gangbang",
    r"blowjob", r"\bhora\b", r"\bhoror\b", r"bordell", r"prostitu",
    r"eskort", r"\bescort", r"nakenbild", r"\bnaken\b", r"brostvart",
    r"\bstånd\b", r"erektion", r"\bslampa\b", r"\bhorunge\b",
]), re.IGNORECASE)


def _nonempty(v):
    s = str(v).strip()
    return bool(s) and s.lower() != "nan"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--offset", type=int, default=DEFAULT_OFFSET,
                    help=f"first item_id (default {DEFAULT_OFFSET}; must not collide with 1-800)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--subset-tag", default="_quiz100",
                    help="filename tag of the generation run to read (default _quiz100)")
    ap.add_argument("--no-restrip", action="store_true",
                    help="skip the markdown-stripped re-judge of already-judged GPT-5.2 "
                         "informal items (see module docstring; leaves the 6-item "
                         "formatting asymmetry in place)")
    args = ap.parse_args()

    base = pd.read_csv(BASE_QUIZ, encoding="utf-8")
    if args.offset <= int(base["item_id"].max()):
        raise SystemExit(
            f"--offset {args.offset} collides with existing item_ids "
            f"(max {int(base['item_id'].max())} in {os.path.basename(BASE_QUIZ)}). "
            f"ai_detection_llm.py resumes on item_id and would silently skip the new items."
        )
    known = {reg: set(g["pair_id"]) for reg, g in base.groupby("register")}

    recs, warnings = [], []
    for generator, register, candidates, ctx_col, colmap in _arms(args.subset_tag):
        path = next((c for c in candidates if os.path.exists(c)), None)
        if path is None:
            listing = "\n".join(f"    {os.path.basename(c)}" for c in candidates)
            raise SystemExit(
                f"missing input for {generator} / {register}. Looked for:\n{listing}\n"
                f"Run the generation step first, e.g.\n"
                f"    GEN_MODEL={generator} QUIZ_SUBSET=1 python generate_adversarial_openai.py"
            )
        print(f"  {generator:8} {register:9} <- {os.path.basename(path)}")
        df = pd.read_csv(path, encoding="utf-8")
        if "quiz_pair_id" not in df.columns:
            raise SystemExit(
                f"{os.path.basename(path)} has no quiz_pair_id column -- it was produced "
                f"without QUIZ_SUBSET=1. Re-run generation with QUIZ_SUBSET=1 so the quiz "
                f"link is recorded; row order alone is NOT a safe key."
            )

        pids = list(df["quiz_pair_id"])
        if len(set(pids)) != len(pids):
            raise SystemExit(f"{os.path.basename(path)}: duplicate quiz_pair_id values")
        unknown = set(pids) - known.get(register, set())
        if unknown:
            raise SystemExit(
                f"{os.path.basename(path)}: {len(unknown)} quiz_pair_id values are not in "
                f"{os.path.basename(BASE_QUIZ)} for register {register!r}, e.g. "
                f"{sorted(unknown)[:3]}. The corpora are not aligned; refusing to build."
            )

        for _, r in df.iterrows():
            for cond, col in colmap.items():
                if col not in df.columns:
                    raise SystemExit(f"{os.path.basename(path)}: missing column {col!r}")
                text = r[col]
                if not _nonempty(text):
                    raise SystemExit(
                        f"{os.path.basename(path)}: empty {col!r} for "
                        f"{r['quiz_pair_id']}. Re-run generation to fill that cell "
                        f"(the cache will skip everything already done)."
                    )
                if register == "informal" and _BLOCK_RE.search(str(text)):
                    warnings.append(f"{generator}/{register}/{cond} {r['quiz_pair_id']}")
                recs.append({
                    "register": register,
                    "generator": generator,
                    "condition": cond,
                    "is_human": False,
                    "context": str(r[ctx_col]),
                    "text": str(text),          # verbatim
                    "source_row": int(r["quiz_source_row"]),
                    "pair_id": str(r["quiz_pair_id"]),   # SAME id as the judged 5.2 item
                })

    for r in recs:
        r["supersedes"] = pd.NA

    # ---- markdown-stripped re-judge of already-judged GPT-5.2 informal items ----
    # Only informal, only non-baseline (baseline is replaced wholesale by the 5.2 run
    # above, so stripping it would be redundant). Formal is skipped: there is no 5.6
    # formal arm, so no asymmetry to correct there.
    n_restrip = 0
    if not args.no_restrip:
        sys.path.insert(0, os.path.join(_src, "1_data_collection"))
        import generation_pipeline as pipe  # noqa: E402
        cand = base[(base["register"] == "informal") & (~base["is_human"].astype(bool))
                    & (base["condition"] != "baseline")]
        for _, r in cand.iterrows():
            stripped = pipe.strip_markdown(r["text"])
            if stripped == str(r["text"]):
                continue
            recs.append({
                "register": r["register"],
                "generator": "gpt-5.2",
                "condition": r["condition"],
                "is_human": False,
                "context": str(r["context"]),
                "text": stripped,
                "source_row": int(r["source_row"]),
                "pair_id": str(r["pair_id"]),
                "supersedes": int(r["item_id"]),   # scorer drops the original
            })
            n_restrip += 1

    for i, r in enumerate(recs, args.offset):
        r["item_id"] = i

    out = pd.DataFrame(recs, columns=[
        "item_id", "register", "generator", "condition", "is_human",
        "context", "text", "source_row", "pair_id", "supersedes",
    ])
    out.to_csv(args.out, index=False, encoding="utf-8")

    print(f"Wrote {len(out)} items -> {args.out}")
    print(f"  item_id range: {out.item_id.min()}-{out.item_id.max()} "
          f"(existing quiz uses 1-{int(base['item_id'].max())})")
    for (g, reg, cond), n in out.groupby(["generator", "register", "condition"]).size().items():
        print(f"    {g:8} {reg:9} {cond:17} n={n}")
    print(f"  distinct documents per register: "
          f"{ {reg: g.pair_id.nunique() for reg, g in out.groupby('register')} }")
    if n_restrip:
        sup = out[out.supersedes.notna()]
        print(f"  markdown-stripped re-judge: {n_restrip} item(s), superseding "
              f"original item_id(s) {sorted(int(x) for x in sup.supersedes)}")
        print(f"    the scorer drops those originals, so each document is counted once")
    if warnings:
        print(f"\n[warn] {len(warnings)} generated informal texts match the explicit-content "
              f"blocklist. They are INCLUDED (dropping them would unpair them from their "
              f"already-judged GPT-5.2 partners). Review before showing to human raters:")
        for w in warnings[:10]:
            print(f"    {w}")
    print("\nNext: back up detection_results_*.csv, then\n"
          f"  python ai_detection_llm.py --provider all --corpus {os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
