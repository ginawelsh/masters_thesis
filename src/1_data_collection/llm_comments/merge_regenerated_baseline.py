"""Merge a regenerated GPT-5.2 informal baseline into the canonical informal corpus.

Context
-------
The informal `generated_comment` column is NOT GPT-5.2 -- 863 of 1,149 texts are traced
to gpt-4o-mini output and the remaining 286 have no locatable source. See
../../../BASELINE_PROVENANCE.md. Run 4 regenerates a genuine GPT-5.2 baseline:

  GEN_MODEL=gpt-5.2 GEN_CONDITIONS=baseline \
  GEN_INPUT=<...>/consolidated_informal_comments_adversarial.csv \
  python generate_adversarial_openai.py

This script folds that run's `comment_baseline` column into the canonical corpus,
ADDITIVELY: one new column, every existing column byte-identical. The old gpt-4o-mini
`generated_comment` is deliberately preserved so it stays usable as a third generator.

Why a script rather than a manual column copy
---------------------------------------------
Row order is the failure mode that caused the original problem: the two informal corpus
files hold the same 1,149 rows in DIFFERENT order (positional agreement on `question` is
~37%). Because run 4 is given the canonical corpus as GEN_INPUT, its output IS that file
plus a column and the orders match by construction -- but "by construction" is an
assumption, and a silent misalignment here would attach every baseline comment to the
wrong document and quietly corrupt the feature chapter. So every assumption is asserted.

Checks (all must pass before anything is written)
  1. row counts equal
  2. row order identical, verified on human_comment + question + thread_id
  3. comment_baseline present, non-empty in every row
  4. generated_comment byte-identical (the gpt-4o-mini control must survive)
  5. the other generated columns byte-identical (nothing else regenerated)
  6. the new baseline is genuinely NEW (low overlap with the old column)

Usage
  python merge_regenerated_baseline.py                 # dry run: check + report only
  python merge_regenerated_baseline.py --write         # back up, then merge in place
  python merge_regenerated_baseline.py --write --out other.csv   # write elsewhere

After merging, point these at the new column:
  2_text_analysis_scripts/scripts/data_utils.py  INFORMAL_CONDITIONS["baseline"]
  3_ai_detection_scripts/make_balanced_quiz.py   INFORMAL_COL["baseline"]
"""
import argparse
import os
import re
import shutil
import sys

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_dir = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(_dir, "consolidated_informal_comments_adversarial.csv")
# default name written by run 4 (GEN_INPUT=<target>, GEN_CONDITIONS=baseline)
REGEN = os.path.join(
    _dir, "generated_three_prompt_14_07_26_gpt-5.2_baseline_in-consolidated-f827.csv")

NEW_COL = "comment_baseline"
ORDER_KEYS = ["human_comment", "question", "thread_id"]
MUST_MATCH = ["generated_comment", "comment_human_like",
              "comment_detector_aware", "comment_detector_evasive"]


def N(v):
    return re.sub(r"\s+", " ", str(v)).strip()


def _fail(msg):
    print(f"\nFAILED: {msg}")
    print("Nothing was written.")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regen", default=REGEN, help="run 4's output CSV")
    ap.add_argument("--target", default=TARGET, help="canonical corpus to merge into")
    ap.add_argument("--out", default=None,
                    help="write here instead of updating --target in place")
    ap.add_argument("--write", action="store_true",
                    help="actually write (default is a dry run)")
    args = ap.parse_args()

    for p in (args.regen, args.target):
        if not os.path.exists(p):
            _fail(f"missing file: {p}\n"
                  f"        Has run 4 finished? Expected output name:\n"
                  f"        {os.path.basename(REGEN)}")

    regen = pd.read_csv(args.regen, encoding="utf-8")
    target = pd.read_csv(args.target, encoding="utf-8")
    print(f"regen : {os.path.basename(args.regen)}  ({len(regen)} rows)")
    print(f"target: {os.path.basename(args.target)}  ({len(target)} rows)")

    # 1. row counts
    if len(regen) != len(target):
        _fail(f"row count mismatch: regen={len(regen)} target={len(target)}. "
              f"Run 4 must be given the canonical corpus as GEN_INPUT and must NOT "
              f"have QUIZ_SUBSET set.")

    # 2. row order, on three independent keys
    for k in ORDER_KEYS:
        if k not in regen.columns or k not in target.columns:
            _fail(f"order key {k!r} missing from one of the files")
        a = regen[k].map(N).values
        b = target[k].map(N).values
        agree = (a == b).mean()
        print(f"  order check {k:15} positional agreement: {agree:.4%}")
        if agree < 1.0:
            bad = int((a != b).sum())
            _fail(f"row order differs on {k!r} ({bad} rows). Merging positionally would "
                  f"attach baselines to the WRONG documents. Re-run run 4 with "
                  f"GEN_INPUT pointed at {os.path.basename(args.target)}.")

    # 3. new column present and complete
    if NEW_COL not in regen.columns:
        _fail(f"{NEW_COL!r} not in the regen file. Was GEN_CONDITIONS=baseline set?")
    empty = regen[NEW_COL].map(lambda v: not N(v) or N(v).lower() == "nan")
    print(f"  {NEW_COL}: {len(regen) - int(empty.sum())}/{len(regen)} populated")
    if empty.any():
        idx = list(regen.index[empty])[:5]
        _fail(f"{int(empty.sum())} empty {NEW_COL} cells (e.g. rows {idx}). Re-run run 4 "
              f"— the cache skips everything already done, so only the gaps are charged.")

    # 4/5. nothing else changed
    for c in MUST_MATCH:
        if c in regen.columns and c in target.columns:
            same = (regen[c].map(N).values == target[c].map(N).values).mean()
            flag = "OK" if same == 1.0 else "CHANGED"
            print(f"  unchanged check {c:26} {same:.4%}  {flag}")
            if same < 1.0:
                _fail(f"{c!r} differs between the files. Run 4 should only have added "
                      f"{NEW_COL}. Check GEN_CONDITIONS was 'baseline'.")

    # 6. the new baseline is actually new
    old = set(target["generated_comment"].map(N))
    new = regen[NEW_COL].map(N)
    dup = int(new.isin(old).sum())
    print(f"  novelty: {dup}/{len(regen)} new baselines identical to the old column")
    if dup > len(regen) * 0.05:
        _fail(f"{dup} regenerated baselines are identical to the old gpt-4o-mini text. "
              f"That suggests the column was copied, not generated.")

    # descriptive sanity (report only)
    ol = target["generated_comment"].map(lambda v: len(str(v)))
    nl = new.map(len)
    print(f"\n  length (chars)  old baseline: mean {ol.mean():6.0f}  median {ol.median():6.0f}")
    print(f"                  new baseline: mean {nl.mean():6.0f}  median {nl.median():6.0f}")
    if not (0.5 <= nl.mean() / max(ol.mean(), 1) <= 2.0):
        print("  [warn] mean length differs from the old column by more than 2x. Not a "
              "blocker (different model), but eyeball a few before relying on it.")

    print("\nAll checks passed.")

    out = args.out or args.target
    if not args.write:
        print(f"\nDry run. Re-run with --write to add {NEW_COL!r} to "
              f"{os.path.basename(out)}.")
        return

    merged = target.copy()
    merged[NEW_COL] = regen[NEW_COL].values
    if args.out is None:
        bak = args.target + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(args.target, bak)
            print(f"Backed up -> {os.path.basename(bak)}")
        else:
            print(f"[note] {os.path.basename(bak)} already exists; left as-is")
    merged.to_csv(out, index=False, encoding="utf-8")
    print(f"Wrote {len(merged)} rows, {len(merged.columns)} columns -> "
          f"{os.path.basename(out)}")
    print("\nNext: point these at the new column, then re-run the informal feature "
          "pipeline:\n"
          "  2_text_analysis_scripts/scripts/data_utils.py  INFORMAL_CONDITIONS['baseline']\n"
          "  3_ai_detection_scripts/make_balanced_quiz.py   INFORMAL_COL['baseline']")


if __name__ == "__main__":
    main()
