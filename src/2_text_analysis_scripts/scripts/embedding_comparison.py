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
Outputs to csv_files/: embedding_pair_cosine_<tag>.csv, embedding_twosample_<tag>.csv

Three generators are supported via --generator:
  gpt52   (default) the main adversarial corpora; 170 formal / 1,149 informal
  gpt56   the 100-document quiz subsets; no detector_aware condition
  mistral mistral-small-2506 from the temperature A/B corpus, --temp selects the arm

Tags keep the gpt52 filenames bare (`formal_baseline`) so the existing outputs and
the tables built off them are unaffected; the other generators add a suffix
(`formal_baseline_gpt56`, `formal_baseline_mistral_t1.0`).

Run: python src/2_text_analysis_scripts/scripts/embedding_comparison.py --dataset formal
     python src/2_text_analysis_scripts/scripts/embedding_comparison.py --generator gpt56 --dataset informal
     python src/2_text_analysis_scripts/scripts/embedding_comparison.py --generator mistral --temp 1.0
"""
import os, argparse
import numpy as np
import pandas as pd
from data_utils import read_csv_robust, add_condition_arg, resolve_conditions, condition_tag

HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(HERE)
_ROOT = os.path.dirname(os.path.dirname(_2TAS))
_DATA = os.path.join(os.path.dirname(_2TAS), "1_data_collection")
CSV_DIR = os.path.join(_2TAS, "csv_files")
MODEL = "paraphrase-multilingual-mpnet-base-v2"

MISTRAL_CSV = os.path.join(_2TAS, "mistral_temperature_ab", "generated_corpus_mistral_temps.csv")

# Mistral doc_id is the ROW INDEX of the corpus generate_mistral_temps.py iterated over,
# NOT of the adversarial CSVs. For formal the two orders coincide (Title matches 170/170),
# but the informal adversarial file is a re-ordering of the JUN26 archive: only 37% of rows
# line up. Pairing Mistral informal text against the adversarial file therefore mispairs
# ~63% of documents, which corrupts the per-pair cosine (metric 1). We join on the archive
# the generator actually indexed. Same 1,149 documents either way -- verified as equal
# multisets of (question, human_comment) after whitespace normalisation -- so metric 2
# (set-vs-set) is unaffected by the choice.
MISTRAL_HUMAN = {
    "formal": dict(csv=os.path.join(_DATA, "llm_abstracts", "abstracts", "sv_abstracts_openai_2.csv"),
                   require="Title"),
    "informal": dict(csv=os.path.join(_ROOT, "src", "4_archive", "consolidated_informal_comments_JUN26.csv"),
                     require="question"),
}

# per-register column naming, shared by every generator so resolve_conditions() applies unchanged
REGISTER = {
    "formal": dict(human_col="Abstract", label_col="Title", llm_prefix="Abstract"),
    "informal": dict(human_col="human_comment", label_col="question", llm_prefix="comment"),
}

SOURCES = {
    "gpt52": {
        "formal": os.path.join(_DATA, "llm_abstracts", "abstracts", "sv_abstracts_adversarial.csv"),
        "informal": os.path.join(_DATA, "llm_comments", "consolidated_informal_comments_adversarial.csv"),
    },
    "gpt56": {
        "formal": os.path.join(_DATA, "llm_abstracts", "abstracts",
                               "generated_three_prompt_formal_14_07_26_gpt-5.6_quiz100.csv"),
        "informal": os.path.join(_DATA, "llm_comments",
                                 "generated_three_prompt_14_07_26_gpt-5.6_quiz100.csv"),
    },
}


def load_mistral(dataset, temp):
    """Pivot the long Mistral corpus to one wide frame matching the adversarial layout.

    Returns a frame with the register's human_col + label_col and one column per
    condition, named with the register's prefix (Abstract_baseline / comment_baseline)
    so the shared condition maps in data_utils resolve against it.
    """
    reg = REGISTER[dataset]
    hcfg = MISTRAL_HUMAN[dataset]

    gen = pd.read_csv(MISTRAL_CSV, encoding="utf-8")
    gen = gen[gen["register"] == dataset]
    available = sorted(gen["temp"].unique())
    gen = gen[gen["temp"] == temp]
    if gen.empty:
        raise SystemExit(f"no Mistral rows for {dataset} at temp={temp}; available temps: {available}")

    wide = gen.pivot(index="doc_id", columns="condition", values="text")
    wide.columns = [f"{reg['llm_prefix']}_{c}" for c in wide.columns]

    # rebuild the doc_id -> human-text mapping exactly as the generator built it:
    # positional index over rows whose prompt field is non-empty
    human = pd.read_csv(hcfg["csv"], encoding="utf-8")
    keep = human[hcfg["require"]].notna() & human[hcfg["require"]].astype(str).str.strip().ne("")
    human = human.loc[keep].reset_index(drop=True)
    human.index.name = "doc_id"

    missing = set(wide.index) - set(human.index)
    if missing:
        raise SystemExit(f"{len(missing)} Mistral doc_ids have no human row in {os.path.basename(hcfg['csv'])}")

    return human[[reg["human_col"], reg["label_col"]]].join(wide, how="inner").reset_index(drop=True)


def load_dataset(generator, dataset, temp):
    if generator == "mistral":
        return load_mistral(dataset, temp)
    return read_csv_robust(SOURCES[generator][dataset])


def output_tag(dataset, cond, generator, temp):
    tag = condition_tag(dataset, cond)
    if generator == "gpt52":
        return tag                      # keep the established filenames untouched
    if generator == "mistral":
        return f"{tag}_mistral_t{temp}"
    return f"{tag}_{generator}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["formal", "informal", "both"], default="formal")
    ap.add_argument("--generator", choices=["gpt52", "gpt56", "mistral"], default="gpt52")
    ap.add_argument("--temp", type=float, default=1.0,
                    help="Mistral temperature arm to analyse (ignored for the GPT generators).")
    ap.add_argument("--permutations", type=int, default=1000)
    add_condition_arg(ap)
    args = ap.parse_args()

    datasets = ["formal", "informal"] if args.dataset == "both" else [args.dataset]
    os.makedirs(CSV_DIR, exist_ok=True)

    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, permutation_test_score
    model = SentenceTransformer(MODEL)

    for dataset in datasets:
        reg = REGISTER[dataset]
        df_all = load_dataset(args.generator, dataset, args.temp)

        for cond, llm_col in resolve_conditions(dataset, args.condition):
            if llm_col not in df_all.columns:
                print(f"  skipping condition '{cond}': column '{llm_col}' not found")
                continue
            tag = output_tag(dataset, cond, args.generator, args.temp)
            df = df_all[df_all[reg["human_col"]].notna() & df_all[llm_col].notna()].reset_index(drop=True)
            human = df[reg["human_col"]].astype(str).tolist()
            llm = df[llm_col].astype(str).tolist()
            n = len(df)
            print(f"\n[{args.generator}/{dataset}/{cond or 'generated'}] {n} paired documents")

            print("encoding human + llm texts...")
            h = model.encode(human, normalize_embeddings=True, show_progress_bar=False)
            l = model.encode(llm, normalize_embeddings=True, show_progress_bar=False)

            # (1) per-pair cosine similarity (embeddings are L2-normalized -> dot = cosine)
            pair_cos = np.sum(h * l, axis=1)
            pd.DataFrame({reg["label_col"]: df[reg["label_col"]], "cosine_similarity": pair_cos}) \
                .to_csv(os.path.join(CSV_DIR, f"embedding_pair_cosine_{tag}.csv"), index=False, encoding="utf-8")
            print(f"per-pair cosine similarity (n={n}):")
            print(f"  mean={pair_cos.mean():.4f}  median={np.median(pair_cos):.4f}  "
                  f"sd={pair_cos.std(ddof=1):.4f}  min={pair_cos.min():.4f}  max={pair_cos.max():.4f}")

            # (2) classifier two-sample test
            X = np.vstack([h, l])
            y = np.array([0] * n + [1] * n)
            clf = LogisticRegression(max_iter=2000)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            print(f"running classifier two-sample test ({args.permutations} permutations)...")
            auc, perm_scores, p = permutation_test_score(
                clf, X, y, scoring="roc_auc", cv=cv, n_permutations=args.permutations,
                random_state=0, n_jobs=-1)
            print(f"  CV ROC AUC = {auc:.4f}   permutation p = {p:.4g}   "
                  f"(null mean AUC = {perm_scores.mean():.4f})")

            rows = [
                dict(metric="generator", value=args.generator),
                dict(metric="condition", value=cond or "generated"),
                dict(metric="n_per_class", value=n),
                dict(metric="mean_pair_cosine", value=round(float(pair_cos.mean()), 4)),
                dict(metric="median_pair_cosine", value=round(float(np.median(pair_cos)), 4)),
                dict(metric="classifier_cv_roc_auc", value=round(float(auc), 4)),
                dict(metric="permutation_p_value", value=float(p)),
                dict(metric="null_mean_auc", value=round(float(perm_scores.mean()), 4)),
                dict(metric="n_permutations", value=args.permutations),
                dict(metric="embedding_model", value=MODEL),
            ]
            if args.generator == "mistral":
                rows.insert(1, dict(metric="temperature", value=args.temp))
            pd.DataFrame(rows).to_csv(
                os.path.join(CSV_DIR, f"embedding_twosample_{tag}.csv"), index=False, encoding="utf-8")
            print(f"wrote embedding_pair_cosine_{tag}.csv and embedding_twosample_{tag}.csv")


if __name__ == "__main__":
    main()
