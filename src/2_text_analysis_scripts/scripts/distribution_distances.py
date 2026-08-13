"""
Distribution distance measures between human and LLM text distributions.

  JS divergence (Jensen-Shannon)
    Primary metric.  Symmetric, bounded [0, 1] (using base-2 log so max = 1).
    Used for CATEGORICAL distributions: POS tags, dependency relations, NER types,
    sentiment labels.

  Wasserstein distance (earth mover's distance, L1)
    For ORDERED / CONTINUOUS features where absolute differences matter:
    token length, sentence length, dependency distance, tree depth,
    punctuation rates, vocabulary richness scores, pragmatic marker rates, etc.

Input sources (loaded automatically when they exist):
  1. Existing spaCy counts CSVs in results/formal/ or results/informal/
       linguistic_analysis_dep_counts.csv
       linguistic_analysis_entity_counts.csv
     (POS counts are stored per-row; the script aggregates them to corpus level)
  2. Feature CSVs written by the other scripts in features/:
       syntactic_complexity_{dataset}.csv
       stylometric_surface_{dataset}.csv
       affective_analysis_{dataset}.csv
       pragmatic_markers_{dataset}.csv

Usage:
  python src/2_text_analysis_scripts/distribution_distances.py --dataset formal
  python src/2_text_analysis_scripts/distribution_distances.py --dataset informal
  python src/2_text_analysis_scripts/distribution_distances.py --dataset both
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

from data_utils import add_condition_arg, resolve_conditions, condition_tag

_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
FEATURES_DIR = os.path.join(_2tas_dir, "csv_files")

POS_SEP = " | "


def parse_args():
    p = argparse.ArgumentParser(description="JS divergence and Wasserstein distances")
    p.add_argument("--dataset", choices=["formal", "informal", "both"], default="both")
    add_condition_arg(p)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_counts_string(s):
    """Parse 'TAG:n | TAG:n | …' into a Counter."""
    counts = Counter()
    if not s or (isinstance(s, float) and np.isnan(s)):
        return counts
    for part in str(s).split(POS_SEP):
        part = part.strip()
        if ":" not in part:
            continue
        tag, n_str = part.rsplit(":", 1)
        try:
            counts[tag.strip()] += int(n_str.strip())
        except ValueError:
            continue
    return counts


def _aggregate_counts(series):
    total = Counter()
    for s in series.dropna():
        total += _parse_counts_string(s)
    return total


# ---------------------------------------------------------------------------
# Distance calculators
# ---------------------------------------------------------------------------

def js_div(counter_a, counter_b):
    """JS divergence (base-2 log, range [0, 1]) between two Counters."""
    vocab = sorted(set(counter_a) | set(counter_b))
    if not vocab:
        return float("nan")
    total_a = sum(counter_a.values()) or 1
    total_b = sum(counter_b.values()) or 1
    p = np.array([counter_a.get(k, 0) / total_a for k in vocab], dtype=float)
    q = np.array([counter_b.get(k, 0) / total_b for k in vocab], dtype=float)
    # jensenshannon uses natural log by default; pass base=2 for bounded [0,1]
    return float(jensenshannon(p, q, base=2))


def wass(series_a, series_b):
    """Wasserstein-1 distance between two numeric Series."""
    a = series_a.dropna().values.astype(float)
    b = series_b.dropna().values.astype(float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    return float(wasserstein_distance(a, b))


# ---------------------------------------------------------------------------
# Dataset-specific loaders
# ---------------------------------------------------------------------------

def _counts_paths(dataset, condition=None):
    """Return (dir, human_prefix) for the spaCy counts files.

    Mirrors spacy_linguistic_analysis output layout: each formal condition lives in
    results/<dataset>/<condition>/; informal (condition=None) in results/<dataset>/.
    """
    base = os.path.join(_2tas_dir, "results", dataset)
    if condition:
        base = os.path.join(base, condition)
    return base, "Abstract"


def categorical_js(dataset, condition=None):
    """JS divergence for POS / dep / entity distributions from spaCy counts CSVs."""
    counts_dir, prefix = _counts_paths(dataset, condition)
    results = []

    for feat, fname in [
        ("dependency_distribution", "linguistic_analysis_dep_counts.csv"),
        ("NER_distribution",        "linguistic_analysis_entity_counts.csv"),
    ]:
        path = os.path.join(counts_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {path} not found")
            continue
        df = pd.read_csv(path, encoding="utf-8")
        hcol = f"{prefix}_Dep_Counts" if "dep" in fname else f"{prefix}_Entity_Counts"
        gcol = f"Generated_{prefix}_Dep_Counts" if "dep" in fname else f"Generated_{prefix}_Entity_Counts"
        if hcol not in df.columns or gcol not in df.columns:
            # Try title-case variants produced by all_ling_analysis_plot
            print(f"  [skip] expected columns '{hcol}'/'{gcol}' not in {fname}")
            continue
        h_agg = _aggregate_counts(df[hcol])
        l_agg = _aggregate_counts(df[gcol])
        results.append({
            "dataset":     dataset,
            "feature":     feat,
            "metric":      "JS_divergence",
            "value":       js_div(h_agg, l_agg),
            "human_tokens": sum(h_agg.values()),
            "llm_tokens":   sum(l_agg.values()),
        })
    return results


def continuous_wass(dataset, tag, feat_name):
    """Wasserstein distance for every numeric human_* / llm_* pair in a feature CSV."""
    path = os.path.join(FEATURES_DIR, f"{feat_name}_{tag}.csv")
    results = []
    if not os.path.exists(path):
        print(f"  [skip] {path} not found — run {feat_name}.py first")
        return results

    df = pd.read_csv(path, encoding="utf-8")
    for hcol in [c for c in df.columns if c.startswith("human_")]:
        lcol = "llm_" + hcol[len("human_"):]
        if lcol not in df.columns:
            continue
        # Only numeric columns make sense for Wasserstein
        try:
            h_vals = pd.to_numeric(df[hcol], errors="raise")
            l_vals = pd.to_numeric(df[lcol], errors="raise")
        except (ValueError, TypeError):
            continue
        results.append({
            "dataset": dataset,
            "feature": hcol[len("human_"):],
            "metric":  "Wasserstein",
            "value":   wass(h_vals, l_vals),
        })
    return results


def sentiment_js(dataset, tag):
    """JS divergence over sentiment label distributions from affective_analysis CSV."""
    path = os.path.join(FEATURES_DIR, f"affective_analysis_{tag}.csv")
    if not os.path.exists(path):
        print(f"  [skip] {path} not found — run affective_analysis.py first")
        return []
    df = pd.read_csv(path, encoding="utf-8")
    if "human_sentiment_label" not in df.columns:
        return []
    h_cnt = Counter(df["human_sentiment_label"].dropna())
    l_cnt = Counter(df["llm_sentiment_label"].dropna())
    return [{
        "dataset": dataset,
        "feature": "sentiment_distribution",
        "metric":  "JS_divergence",
        "value":   js_div(h_cnt, l_cnt),
    }]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_dataset(dataset, requested="all"):
    results = []
    for cond, _ in resolve_conditions(dataset, requested):
        tag = condition_tag(dataset, cond)
        cond_label = cond or "baseline"
        print(f"  [{cond_label}]")
        cond_results = []
        cond_results += categorical_js(dataset, cond)
        cond_results += sentiment_js(dataset, tag)
        for feat in ["syntactic_complexity", "stylometric_surface",
                     "pragmatic_markers", "affective_analysis", "perplexity"]:
            cond_results += continuous_wass(dataset, tag, feat)
        for r in cond_results:
            r["condition"] = cond_label
        results += cond_results
    return results


def main():
    args = parse_args()
    datasets = ["formal", "informal"] if args.dataset == "both" else [args.dataset]

    all_results = []
    for ds in datasets:
        print(f"\n--- {ds} ---")
        all_results += run_dataset(ds, args.condition)

    os.makedirs(FEATURES_DIR, exist_ok=True)
    out_df = pd.DataFrame(all_results)
    if not out_df.empty:
        lead = ["dataset", "condition", "feature", "metric", "value"]
        out_df = out_df[lead + [c for c in out_df.columns if c not in lead]]
    out_path = os.path.join(FEATURES_DIR, f"distribution_distances_{args.dataset}.csv")
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(out_df)} rows → {out_path}")

    if not out_df.empty:
        print("\n" + out_df[["dataset", "condition", "feature", "metric", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
