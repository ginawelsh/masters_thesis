"""
embedding_comparison.py

Compares human vs LLM abstract embeddings two ways:

  (1) Per-pair cosine similarity: each human abstract is paired with its LLM
      regeneration (same title/keywords). Cosine similarity of the two
      embeddings per pair -> distribution of how close each LLM output is to
      its human original.

  (2) Classifier two-sample test: label human embeddings 0 and LLM embeddings 1,
      train logistic regression with 5-fold CV, report cross-validated ROC AUC
      (0.5 = indistinguishable, 1.0 = perfectly separable) and a
      label-permutation p-value.

Same sentence-transformer model as multi_layer_analysis.
Outputs to csv_files/: embedding_pair_cosine_<dataset>.csv, embedding_twosample_<dataset>.csv

Run: python src/2_text_analysis_scripts/scripts/embedding_comparison.py --dataset formal
"""
import os, argparse
import numpy as np
import pandas as pd
from data_utils import read_csv_robust

HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(HERE)
_DATA = os.path.join(os.path.dirname(_2TAS), "1_data_collection")
CSV_DIR = os.path.join(_2TAS, "csv_files")
MODEL = "paraphrase-multilingual-mpnet-base-v2"

SOURCES = {
    "formal": dict(csv=os.path.join(_DATA, "llm_abstracts", "abstracts", "sv_abstracts_openai_2.csv"),
                   human_col="Abstract", llm_col="Generated_OpenAI_Abstract", label_col="Title"),
    "informal": dict(csv=os.path.join(_DATA, "llm_comments", "consolidated_informal_comments_JUN26.csv"),
                     human_col="human_comment", llm_col="generated_comment", label_col="question"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    ap.add_argument("--permutations", type=int, default=1000)
    args = ap.parse_args()
    cfg = SOURCES[args.dataset]

    df = read_csv_robust(cfg["csv"])
    df = df[df[cfg["human_col"]].notna() & df[cfg["llm_col"]].notna()].reset_index(drop=True)
    human = df[cfg["human_col"]].astype(str).tolist()
    llm = df[cfg["llm_col"]].astype(str).tolist()
    n = len(df)
    print(f"{args.dataset}: {n} paired documents")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    print("encoding human + llm texts...")
    h = model.encode(human, normalize_embeddings=True, show_progress_bar=False)
    l = model.encode(llm, normalize_embeddings=True, show_progress_bar=False)

    # (1) per-pair cosine similarity (embeddings are L2-normalized -> dot = cosine)
    pair_cos = np.sum(h * l, axis=1)
    pd.DataFrame({cfg["label_col"]: df[cfg["label_col"]], "cosine_similarity": pair_cos}) \
        .to_csv(os.path.join(CSV_DIR, f"embedding_pair_cosine_{args.dataset}.csv"), index=False, encoding="utf-8")
    print(f"\nper-pair cosine similarity (n={n}):")
    print(f"  mean={pair_cos.mean():.4f}  median={np.median(pair_cos):.4f}  "
          f"sd={pair_cos.std(ddof=1):.4f}  min={pair_cos.min():.4f}  max={pair_cos.max():.4f}")

    # (2) classifier two-sample test
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    X = np.vstack([h, l])
    y = np.array([0] * n + [1] * n)
    clf = LogisticRegression(max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    print(f"\nrunning classifier two-sample test ({args.permutations} permutations)...")
    auc, perm_scores, p = permutation_test_score(
        clf, X, y, scoring="roc_auc", cv=cv, n_permutations=args.permutations,
        random_state=0, n_jobs=-1)
    print(f"  CV ROC AUC = {auc:.4f}   permutation p = {p:.4g}   "
          f"(null mean AUC = {perm_scores.mean():.4f})")

    pd.DataFrame([
        dict(metric="n_per_class", value=n),
        dict(metric="mean_pair_cosine", value=round(float(pair_cos.mean()), 4)),
        dict(metric="median_pair_cosine", value=round(float(np.median(pair_cos)), 4)),
        dict(metric="classifier_cv_roc_auc", value=round(float(auc), 4)),
        dict(metric="permutation_p_value", value=float(p)),
        dict(metric="null_mean_auc", value=round(float(perm_scores.mean()), 4)),
        dict(metric="n_permutations", value=args.permutations),
        dict(metric="embedding_model", value=MODEL),
    ]).to_csv(os.path.join(CSV_DIR, f"embedding_twosample_{args.dataset}.csv"), index=False, encoding="utf-8")
    print(f"\nwrote embedding_pair_cosine_{args.dataset}.csv and embedding_twosample_{args.dataset}.csv")


if __name__ == "__main__":
    main()
