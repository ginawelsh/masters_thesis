"""Document-matched control for tab:embedding-auc.

Recomputes the GPT-5.2 and Mistral embedding two-sample AUCs restricted to exactly
the documents the GPT-5.6 quiz100 subset used, so the three generators are compared
on the same texts at the same n. Methodology mirrors embedding_comparison.py:
same model, same L2-normalized embeddings, same LogisticRegression(max_iter=2000),
same StratifiedKFold(5, shuffle=True, random_state=0), scoring=roc_auc.

With --permutations N (default 0) each cell additionally gets a label-permutation
test, mirroring embedding_comparison.py's permutation_test_score, plus a
Hanley-McNeil 95% CI on the matched AUC. Default 0 reproduces the original
point-estimate-only output exactly.

Writes csv_files/embedding_auc_matched_gpt56subset.csv; does NOT touch any existing
embedding_* output.

Why it exists: tab:embedding-auc reports GPT-5.6 at n=100 but GPT-5.2/Mistral at
n=170 (formal) / 1,149 (informal). Because the statistic is a CROSS-VALIDATED
classifier AUC, fewer documents means less training data per fold and a
systematically LOWER AUC -- so the n gap biases the cross-generator comparison,
it does not merely blur it. Running this restores like-for-like.
Findings are written up in 3_ai_detection_scripts/EMBEDDING_AUC_CAVEATS.md.

Self-check: re-deriving GPT-5.6 through this path must reproduce its published
AUCs exactly (delta 0.0000); if it does not, this script has drifted from
embedding_comparison.py and its GPT-5.2/Mistral deltas cannot be trusted.

Run: python src/2_text_analysis_scripts/scripts/matched_auc_control.py
     python src/2_text_analysis_scripts/scripts/matched_auc_control.py --permutations 1000
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS)
os.chdir(SCRIPTS)
# line_buffering so per-cell progress is visible when stdout is redirected to a
# file; --permutations runs take long enough that block buffering looks like a hang.
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import embedding_comparison as ec  # noqa: E402
from data_utils import resolve_conditions  # noqa: E402

OUT = os.path.join(ec.CSV_DIR, "embedding_auc_matched_gpt56subset.csv")
CONDS = ["baseline", "human_like", "detector_evasive"]


def norm(series):
    return series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def hanley_mcneil_ci(auc, n_pos, n_neg, z=1.96):
    """Normal-approximation 95% CI on an AUC (Hanley & McNeil 1982).

    Indicative of relative precision only -- the point estimate here is a
    cross-validated classifier AUC, not the single-sample statistic the formula
    assumes. Same method already used in EMBEDDING_AUC_CAVEATS.md, kept for
    consistency with the numbers reported there.
    """
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return max(0.0, auc - z * se), min(1.0, auc + z * se)


def doc_key(df, reg):
    """Document identity: (label, human text). Informal repeats questions across rows,
    so the label alone is not unique -- the human comment pins the exact document."""
    return norm(df[reg["label_col"]]) + " ||| " + norm(df[reg["human_col"]])


def main(n_perm=0):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                         permutation_test_score)

    model = SentenceTransformer(ec.MODEL)
    rows = []

    for dataset in ("formal", "informal"):
        reg = ec.REGISTER[dataset]
        g56 = ec.load_dataset("gpt56", dataset, 1.0)
        keys56 = set(doc_key(g56, reg))
        print(f"\n=== {dataset}: GPT-5.6 subset defines {len(keys56)} unique documents "
              f"(from {len(g56)} rows) ===")

        sources = {
            "GPT-5.2": ec.load_dataset("gpt52", dataset, 1.0),
            "Mistral 3.2": ec.load_mistral(dataset, 1.0),
            "GPT-5.6": g56,
        }

        for gen, df_all in sources.items():
            df_all = df_all.copy()
            df_all["_k"] = doc_key(df_all, reg)
            matched = df_all[df_all["_k"].isin(keys56)]
            # one row per document (guards against any duplicate keys)
            matched = matched.drop_duplicates("_k")
            print(f"  {gen:<12} matched {len(matched)} / {len(keys56)} documents")

            for cond, llm_col in resolve_conditions(dataset, "all"):
                if cond not in CONDS or llm_col not in matched.columns:
                    continue
                d = matched[matched[reg["human_col"]].notna() & matched[llm_col].notna()]
                n = len(d)
                if n < 20:
                    print(f"    [skip] {cond}: only {n} usable rows")
                    continue
                h = model.encode(d[reg["human_col"]].astype(str).tolist(),
                                 normalize_embeddings=True, show_progress_bar=False)
                l = model.encode(d[llm_col].astype(str).tolist(),
                                 normalize_embeddings=True, show_progress_bar=False)
                X = np.vstack([h, l])
                y = np.array([0] * n + [1] * n)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
                clf = LogisticRegression(max_iter=2000)
                auc = float(cross_val_score(clf, X, y,
                                            scoring="roc_auc", cv=cv, n_jobs=-1).mean())
                pair_cos = float(np.sum(h * l, axis=1).mean())
                row = dict(generator=gen, register=dataset, condition=cond,
                           n_per_class=n, matched_cv_roc_auc=round(auc, 4),
                           mean_pair_cosine=round(pair_cos, 4))

                if n_perm:
                    # permutation_test_score refits from scratch, so its own point
                    # estimate is what the p-value is anchored to; keep both and
                    # assert they agree with the cross_val_score AUC above.
                    perm_auc, perm_scores, p = permutation_test_score(
                        clf, X, y, scoring="roc_auc", cv=cv,
                        n_permutations=n_perm, random_state=0, n_jobs=-1)
                    assert abs(perm_auc - auc) < 1e-9, (
                        f"{gen}/{dataset}/{cond}: permutation_test_score AUC "
                        f"{perm_auc:.6f} != cross_val_score AUC {auc:.6f}")
                    lo, hi = hanley_mcneil_ci(auc, n, n)
                    row.update(permutation_p_value=float(p),
                               null_mean_auc=round(float(perm_scores.mean()), 4),
                               n_permutations=n_perm,
                               ci95_lo=round(lo, 4), ci95_hi=round(hi, 4))
                    print(f"    {cond:<17} n={n:<5} matched AUC = {auc:.4f}  "
                          f"[{lo:.3f}, {hi:.3f}]  perm p = {p:.4g}  "
                          f"(null mean {perm_scores.mean():.4f})")
                else:
                    print(f"    {cond:<17} n={n:<5} matched AUC = {auc:.4f}")

                rows.append(row)

    out = pd.DataFrame(rows)

    # attach the published full-corpus AUC for side-by-side comparison
    suffix = {"GPT-5.2": "", "GPT-5.6": "_gpt56", "Mistral 3.2": "_mistral_t1.0"}

    def published(r):
        p = os.path.join(ec.CSV_DIR,
                         f"embedding_twosample_{r.register}_{r.condition}{suffix[r.generator]}.csv")
        if not os.path.exists(p):
            return pd.Series({"published_auc": np.nan, "published_n": np.nan})
        d = dict(zip(*[pd.read_csv(p)[c] for c in ("metric", "value")]))
        return pd.Series({"published_auc": float(d["classifier_cv_roc_auc"]),
                          "published_n": int(d["n_per_class"])})

    out = pd.concat([out, out.apply(published, axis=1)], axis=1)
    out["delta"] = (out["matched_cv_roc_auc"] - out["published_auc"]).round(4)
    out.to_csv(OUT, index=False, encoding="utf-8")

    print("\n" + "=" * 92)
    print("DOCUMENT-MATCHED vs PUBLISHED (all matched rows use the GPT-5.6 quiz documents)")
    print("=" * 92)
    print(out.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permutations", type=int, default=0,
                    help="label permutations per cell (0 = point estimates only, "
                         "reproducing the original output)")
    main(n_perm=ap.parse_args().permutations)
