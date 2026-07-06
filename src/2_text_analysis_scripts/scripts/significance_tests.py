"""
significance_tests.py

Paired significance testing of human vs LLM-generated Swedish text features.

Design: each row of a feature CSV is a matched pair (a human document and its
LLM regeneration from the same title/keywords or question). For every numeric
`human_<feat>` / `llm_<feat>` column pair we run a paired Wilcoxon signed-rank
test, compute effect sizes (matched-pairs rank-biserial correlation and
Cohen's dz), then correct for multiple comparisons with Benjamini-Hochberg FDR
*within each dataset* (formal / informal).

Why Wilcoxon (not paired t-test): most features are rates/proportions/counts
that are not normally distributed. Wilcoxon is the robust paired choice.

Output: csv_files/significance_tests_results.csv  (one row per feature/dataset)
        + a printed summary of the significant features ranked by effect size.

Run:  python src/2_text_analysis_scripts/scripts/significance_tests.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(os.path.dirname(HERE), "csv_files")   # ...2_text_analysis_scripts/csv_files
OUT = os.path.join(CSV_DIR, "significance_tests_results.csv")

# per-document paired feature files (group -> file stem); {dataset} filled in below
FEATURE_GROUPS = ["syntactic_complexity", "stylometric_surface",
                  "pragmatic_markers", "affective_analysis"]
DATASETS = ["formal", "informal"]

# columns to skip (non-numeric labels, or binary that wants McNemar not Wilcoxon)
SKIP_EXACT = {"sentiment_label"}
BINARY_FLAG = {"is_subjective"}   # tested, but flagged: prefer McNemar for binary


def rank_biserial(diffs):
    """Matched-pairs rank-biserial correlation for the signed-rank test.
    r = (W+ - W-) / (W+ + W-), computed over non-zero differences.
    +ve => llm > human ; -ve => human > llm. Range [-1, 1]."""
    d = diffs[diffs != 0]
    if len(d) == 0:
        return np.nan
    ranks = rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    tot = w_pos + w_neg
    return float((w_pos - w_neg) / tot) if tot else np.nan


def cohens_dz(diffs):
    """Cohen's dz for a paired design = mean(diff) / sd(diff)."""
    if len(diffs) < 2:
        return np.nan
    sd = diffs.std(ddof=1)
    return float(diffs.mean() / sd) if sd > 0 else np.nan


def bh_fdr(pvals):
    """Benjamini-Hochberg FDR-adjusted p-values (returns q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.clip(ranked, 0, 1)
    return q


def test_feature(h, l):
    """Return dict of stats for one paired human/llm numeric column pair."""
    df = pd.DataFrame({"h": pd.to_numeric(h, errors="coerce"),
                       "l": pd.to_numeric(l, errors="coerce")}).dropna()
    n = len(df)
    if n < 3:
        return None
    diffs = (df["l"] - df["h"]).to_numpy()
    if np.all(diffs == 0):
        return dict(n=n, stat=np.nan, p=1.0, rbc=0.0, dz=0.0,
                    median_human=float(df["h"].median()),
                    median_llm=float(df["l"].median()))
    try:
        stat, p = wilcoxon(df["h"], df["l"], zero_method="wilcox",
                           alternative="two-sided", mode="auto")
    except ValueError:
        return None
    return dict(n=n, stat=float(stat), p=float(p),
                rbc=rank_biserial(diffs), dz=cohens_dz(diffs),
                median_human=float(df["h"].median()),
                median_llm=float(df["l"].median()))


def main():
    rows = []
    for ds in DATASETS:
        ds_rows = []
        for grp in FEATURE_GROUPS:
            path = os.path.join(CSV_DIR, f"{grp}_{ds}.csv")
            if not os.path.exists(path):
                print(f"  [skip] {os.path.basename(path)} not found")
                continue
            df = pd.read_csv(path, encoding="utf-8")
            for hcol in [c for c in df.columns if c.startswith("human_")]:
                feat = hcol[len("human_"):]
                if feat in SKIP_EXACT:
                    continue
                lcol = "llm_" + feat
                if lcol not in df.columns:
                    continue
                res = test_feature(df[hcol], df[lcol])
                if res is None:
                    continue
                direction = ("llm > human" if res["median_llm"] > res["median_human"]
                             else "human > llm" if res["median_llm"] < res["median_human"]
                             else "equal")
                ds_rows.append(dict(dataset=ds, feature_group=grp, feature=feat,
                                    n_pairs=res["n"], median_human=round(res["median_human"], 4),
                                    median_llm=round(res["median_llm"], 4), direction=direction,
                                    wilcoxon_stat=res["stat"], p_value=res["p"],
                                    rank_biserial=res["rbc"], cohens_dz=res["dz"],
                                    binary_flag=(feat in BINARY_FLAG)))
        # FDR within this dataset
        if ds_rows:
            q = bh_fdr([r["p_value"] for r in ds_rows])
            for r, qi in zip(ds_rows, q):
                r["p_fdr_bh"] = float(qi)
                r["significant_fdr_0.05"] = bool(qi < 0.05)
            rows.extend(ds_rows)

    out = pd.DataFrame(rows)
    # order columns
    cols = ["dataset", "feature_group", "feature", "n_pairs", "median_human",
            "median_llm", "direction", "cohens_dz", "rank_biserial",
            "p_value", "p_fdr_bh", "significant_fdr_0.05", "wilcoxon_stat", "binary_flag"]
    out = out[cols].sort_values(["dataset", "p_fdr_bh"]).reset_index(drop=True)
    out.to_csv(OUT, index=False, encoding="utf-8")

    print(f"\nWrote {OUT}\n")
    for ds in DATASETS:
        sub = out[out.dataset == ds]
        if sub.empty:
            continue
        sig = sub[sub["significant_fdr_0.05"]]
        n = int(sub["n_pairs"].max()) if not sub.empty else 0
        print(f"=== {ds.upper()} (n~{n} pairs) - {len(sig)}/{len(sub)} features significant after BH-FDR ===")
        show = sig.reindex(sig["cohens_dz"].abs().sort_values(ascending=False).index)
        for _, r in show.head(15).iterrows():
            flag = "  [binary: use McNemar]" if r["binary_flag"] else ""
            print(f"   {r['feature']:<26} dz={r['cohens_dz']:+.2f}  q={r['p_fdr_bh']:.1e}  ({r['direction']}){flag}")
        print()


if __name__ == "__main__":
    main()
