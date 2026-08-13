"""What does informal scrape-artefact normalization actually change?

Two reports, written to csv_files/ and printed:

  1. CORPUS AUDIT -- which comments text_normalizer.normalize_informal touches, and
     under which artefact class (HTML entity / blockquote marker / Mvh sign-off).
     Pure leading/trailing whitespace changes are counted separately and excluded
     from the "substantive" tally, because compute_metrics() already .strip()s.

  2. FEATURE DELTA -- per stylometric feature, the paired human-vs-LLM statistic
     before and after normalization, using the SAME test as significance_tests.py
     (paired Wilcoxon, Cohen's dz, rank-biserial) so the numbers are comparable to
     the ones in the thesis tables.

Reads the un-normalized CSVs written by
    stylometric_surface.py --dataset informal
and the normalized ones written by
    stylometric_surface.py --dataset informal --normalize
so run both before this.

Usage:  python normalization_delta.py [--drop-quoted]
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

from data_utils import resolve_conditions, condition_tag
from significance_tests import test_feature

HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(HERE)
CSV_DIR = os.path.join(_2TAS, "csv_files")
_SRC = os.path.dirname(_2TAS)
CORPUS = os.path.join(_SRC, "1_data_collection", "llm_comments",
                      "consolidated_informal_comments_adversarial.csv")

TEXT_COLS = ["human_comment", "comment_baseline", "comment_human_like",
             "comment_detector_aware", "comment_detector_evasive"]

sys.path.insert(0, os.path.join(_SRC, "3_ai_detection_scripts"))
from text_normalizer import (normalize_informal, _ENTITIES,   # noqa: E402
                             _QUOTE_MARKERS, _SIGNOFF)


def classify(before: str) -> str:
    """Which artefact class(es) fire on this text, as a '+'-joined label."""
    hits = []
    if any(rx.search(before) for rx, _ in _ENTITIES):
        hits.append("html_entity")
    # markers are matched on the post-unescape text, as the normalizer sees them
    unescaped = before
    for rx, repl in _ENTITIES:
        unescaped = rx.sub(repl, unescaped)
    if any(rx.search(unescaped) for rx in _QUOTE_MARKERS):
        hits.append("quote_marker")
    if _SIGNOFF.search(unescaped):
        hits.append("signoff")
    if not hits and re.search(r"\n{3,}", before):
        # inherited from normalize_formal: collapses runs of blank lines. No feature
        # depends on blank lines, so this is cosmetic — tracked so it is not silently
        # lumped in with the artefact classes.
        hits.append("blank_lines")
    return "+".join(hits) if hits else "whitespace_only"


def corpus_audit(drop_quoted: bool):
    df = pd.read_csv(CORPUS)
    rows, summary = [], []
    for col in TEXT_COLS:
        s = df[col].fillna("").astype(str)
        n = s.map(lambda t: normalize_informal(t, drop_quoted=drop_quoted))
        changed = n != s
        substantive = n != s.str.strip()
        summary.append(dict(column=col, n=len(s),
                            changed_any=int(changed.sum()),
                            changed_substantive=int(substantive.sum()),
                            chars_before=int(s.str.len().sum()),
                            chars_after=int(n.str.len().sum())))
        for i in s[substantive].index:
            rows.append(dict(row=i, column=col, source=df.at[i, "source"],
                             artefact=classify(s[i]),
                             chars_before=len(s[i]), chars_after=len(n[i]),
                             before=s[i][:300], after=n[i][:300]))
    return pd.DataFrame(summary), pd.DataFrame(rows)


def feature_delta(drop_quoted: bool):
    suffix = "_norm_dropquoted" if drop_quoted else "_norm"
    out = []
    for cond, _ in resolve_conditions("informal", "all"):
        tag = condition_tag("informal", cond)
        pre_p = os.path.join(CSV_DIR, f"stylometric_surface_{tag}.csv")
        post_p = os.path.join(CSV_DIR, f"stylometric_surface_{tag}{suffix}.csv")
        if not (os.path.exists(pre_p) and os.path.exists(post_p)):
            print(f"  skipping {cond}: missing {os.path.basename(pre_p)} "
                  f"or {os.path.basename(post_p)}")
            continue
        pre, post = pd.read_csv(pre_p), pd.read_csv(post_p)
        feats = [c[len("human_"):] for c in pre.columns if c.startswith("human_")]
        for f in feats:
            h, l = f"human_{f}", f"llm_{f}"
            if h not in post.columns:
                continue
            a = test_feature(pre[h], pre[l])
            b = test_feature(post[h], post[l])
            if a is None or b is None:
                continue
            out.append(dict(
                condition=cond, feature=f,
                human_mean_pre=pre[h].mean(), human_mean_post=post[h].mean(),
                human_mean_pct_change=(
                    100.0 * (post[h].mean() - pre[h].mean()) / pre[h].mean()
                    if pre[h].mean() else np.nan),
                llm_mean_pre=pre[l].mean(), llm_mean_post=post[l].mean(),
                dz_pre=a["dz"], dz_post=b["dz"], dz_change=b["dz"] - a["dz"],
                # RAW Wilcoxon p, not BH-corrected: this script tests one feature at a
                # time, so there is no family to correct over. Re-run
                # significance_tests.py on the _norm CSVs for the q-values.
                p_pre=a["p"], p_post=b["p"],
                n_pre=a["n"], n_post=b["n"],
            ))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drop-quoted", action="store_true",
                    help="report the destructive blockquote-line-deletion variant")
    args = ap.parse_args()
    sfx = "_dropquoted" if args.drop_quoted else ""

    summary, detail = corpus_audit(args.drop_quoted)
    print("=== 1. CORPUS AUDIT ===")
    print(summary.to_string(index=False))
    print("\nsubstantive changes by artefact class:")
    if not detail.empty:
        print(detail.groupby(["column", "artefact"]).size().to_string())
    s_path = os.path.join(CSV_DIR, f"informal_normalization_audit{sfx}.csv")
    detail.to_csv(s_path, index=False, encoding="utf-8")
    print(f"\n-> {s_path}")

    delta = feature_delta(args.drop_quoted)
    if delta.empty:
        print("\nNo feature CSV pairs found -- run stylometric_surface.py first.")
        return
    d_path = os.path.join(CSV_DIR, f"informal_normalization_delta{sfx}.csv")
    delta.to_csv(d_path, index=False, encoding="utf-8")

    print("\n=== 2. FEATURE DELTA (paired human-vs-LLM, per condition) ===")
    print("features whose |dz| moves by >=0.01, or whose human mean moves by >=0.5%:")
    moved = delta[(delta["dz_change"].abs() >= 0.01)
                  | (delta["human_mean_pct_change"].abs() >= 0.5)]
    if moved.empty:
        print("  none -- every feature is unchanged to that tolerance.")
    else:
        for _, r in moved.iterrows():
            print(f"  {r['condition']:16} {r['feature']:24} "
                  f"human {r['human_mean_pre']:.4f}->{r['human_mean_post']:.4f} "
                  f"({r['human_mean_pct_change']:+.1f}%)  "
                  f"dz {r['dz_pre']:+.3f}->{r['dz_post']:+.3f} "
                  f"({r['dz_change']:+.3f})")
    print(f"\nlargest |dz| change overall: {delta['dz_change'].abs().max():.4f}")
    print(f"-> {d_path}")


if __name__ == "__main__":
    main()
