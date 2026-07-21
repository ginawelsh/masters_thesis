"""
feature_detector.py

A NON-LLM AI-text detector: a supervised classifier trained on the handcrafted
linguistic features this pipeline already extracts (no prompting, no LLM judge).

It reuses the per-document paired feature CSVs written by the other scripts
(`<group>_<dataset>_<condition>.csv` in csv_files/), where every row is a matched
pair: a human document (`human_<feat>` columns) and its LLM regeneration
(`llm_<feat>` columns) from the same title/keywords or question. The two halves
are un-paired into a two-sample detection task:

    human documents -> label 0     LLM documents -> label 1

and a classifier is trained to tell them apart. We report cross-validated
**ROC AUC** (0.5 = indistinguishable, 1.0 = perfectly separable) so the number
is directly comparable to the sentence-transformer two-sample test in
embedding_comparison.py -- the difference being that THIS detector runs on
interpretable linguistic features, so it also emits feature importances.

Why this detector matters for the thesis: the `detector_evasive` prompt was
written to *invert this study's measured feature signals*. Reporting detection
AUC per prompt condition (baseline / human_like / detector_evasive) therefore
measures directly how far prompt-based evasion degrades a feature detector.
(`detector_aware` is generated but not part of the reported analysis.)

Feature families combined (whatever files exist for the tag):
    syntactic_complexity, stylometric_surface, pragmatic_markers,
    affective_analysis (informal only), ner

Models (--model): logreg (default, standardized + balanced), rf, gboost.
CV (--cv): stratified (default; matches embedding_comparison) or grouped
(StratifiedGroupKFold, keeps both halves of a pair in the same fold to avoid
topic leakage inflating AUC).

Outputs to csv_files/:
    feature_detector_results.csv                 one row per (dataset, condition)
    feature_detector_importance_<tag>.csv        signed/ranked feature importances

Run:
    python src/2_text_analysis_scripts/scripts/feature_detector.py
    python .../feature_detector.py --dataset informal --condition detector_evasive
    python .../feature_detector.py --model rf --cv grouped --permutations 1000
"""
import argparse
import os
import numpy as np
import pandas as pd

from data_utils import add_condition_arg, resolve_conditions, condition_tag

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(os.path.dirname(HERE), "csv_files")
RESULTS = os.path.join(CSV_DIR, "feature_detector_results.csv")

# Same paired per-document feature files significance_tests.py reads.
FEATURE_GROUPS = ["syntactic_complexity", "stylometric_surface",
                  "pragmatic_markers", "affective_analysis", "ner"]

# Non-numeric / non-feature columns to never use as predictors.
SKIP_FEATS = {"sentiment_label"}

# The conditions actually reported in the thesis/paper (detector_aware is
# generated but excluded); used only for the closing note, not to filter.
REPORTED = ("baseline", "human_like", "detector_evasive")


def load_feature_matrix(tag):
    """Assemble one (X, y, groups, feature_names) two-sample task for a tag.

    Reads every feature-group file present for `tag`, aligns them by row
    position (row i = the same pair across all groups), keeps every feature that
    appears as BOTH human_<feat> and llm_<feat> and is numeric, then stacks the
    human rows (label 0) above the LLM rows (label 1). `groups` gives each pair a
    shared id so grouped CV can keep both halves of a pair together.
    """
    human_blocks, llm_blocks = [], []
    feat_names = []
    n_rows = None
    found = []

    for grp in FEATURE_GROUPS:
        path = os.path.join(CSV_DIR, f"{grp}_{tag}.csv")
        if not os.path.exists(path):
            continue
        found.append(grp)
        df = pd.read_csv(path, encoding="utf-8")

        # feats present on both sides of this group
        h_feats = {c[len("human_"):] for c in df.columns if c.startswith("human_")}
        l_feats = {c[len("llm_"):] for c in df.columns if c.startswith("llm_")}
        feats = sorted((h_feats & l_feats) - SKIP_FEATS)

        h = df[[f"human_{f}" for f in feats]].apply(pd.to_numeric, errors="coerce")
        l = df[[f"llm_{f}" for f in feats]].apply(pd.to_numeric, errors="coerce")
        # drop feats that are entirely non-numeric on either side
        keep = [f for f in feats
                if not h[f"human_{f}"].isna().all() and not l[f"llm_{f}"].isna().all()]
        if not keep:
            continue
        h = h[[f"human_{f}" for f in keep]]
        l = l[[f"llm_{f}" for f in keep]]

        # namespace the feature with its group so identically-named feats don't collide
        cols = [f"{grp}::{f}" for f in keep]
        h.columns = cols
        l.columns = cols

        if n_rows is None:
            n_rows = len(df)
        elif len(df) != n_rows:
            m = min(n_rows, len(df))
            print(f"  [warn] {grp}_{tag}.csv has {len(df)} rows vs {n_rows} in earlier "
                  f"groups; truncating all to {m} (row-position alignment).")
            n_rows = m

        human_blocks.append(h)
        llm_blocks.append(l)
        feat_names.extend(cols)

    if not found:
        return None

    # truncate every block to the common row count, then concat feature groups side-by-side
    H = pd.concat([b.iloc[:n_rows].reset_index(drop=True) for b in human_blocks], axis=1)
    L = pd.concat([b.iloc[:n_rows].reset_index(drop=True) for b in llm_blocks], axis=1)

    pair_id = np.arange(n_rows)
    X = pd.concat([H, L], axis=0, ignore_index=True)
    y = np.array([0] * n_rows + [1] * n_rows)
    groups = np.concatenate([pair_id, pair_id])

    # drop rows with any missing feature (keeps human/LLM halves independent)
    ok = X.notna().all(axis=1).to_numpy()
    dropped = int((~ok).sum())
    if dropped:
        print(f"  [info] dropped {dropped} rows with missing features "
              f"({ok.sum()} of {len(X)} kept)")
    X, y, groups = X.loc[ok].reset_index(drop=True), y[ok], groups[ok]
    return X, y, groups, feat_names, found


def build_model(name):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return Pipeline([("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000,
                                                    class_weight="balanced"))])
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                      random_state=0, n_jobs=-1)
    if name == "gboost":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(random_state=0)
    raise ValueError(f"unknown model {name!r}")


def make_cv(scheme, y, groups):
    """Return (cv_splitter, cv_kwargs) for cross_validate/permutation_test_score."""
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    n_min = int(np.bincount(y).min())
    n_splits = min(5, n_min) if n_min >= 2 else 2
    if scheme == "grouped":
        return StratifiedGroupKFold(n_splits=n_splits), dict(groups=groups)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0), {}


def feature_importance(model, model_name, X, y, feat_names):
    """Signed importance for logreg (standardized coef), else non-negative
    impurity/gain importance. Fit on the full sample."""
    model.fit(X, y)
    if model_name == "logreg":
        coef = model.named_steps["clf"].coef_.ravel()
        imp = pd.DataFrame({"feature": feat_names, "importance": coef,
                            "abs_importance": np.abs(coef)})
    else:
        vals = getattr(model, "feature_importances_")
        imp = pd.DataFrame({"feature": feat_names, "importance": vals,
                            "abs_importance": np.abs(vals)})
    return imp.sort_values("abs_importance", ascending=False).reset_index(drop=True)


def parse_args():
    p = argparse.ArgumentParser(description="Feature-based (non-LLM) AI-text detector")
    p.add_argument("--dataset", choices=["formal", "informal", "both"], default="both")
    p.add_argument("--model", choices=["logreg", "rf", "gboost"], default="logreg")
    p.add_argument("--cv", choices=["stratified", "grouped"], default="stratified",
                   help="grouped keeps both halves of a pair in the same fold "
                        "(avoids topic leakage); stratified matches embedding_comparison.py")
    p.add_argument("--permutations", type=int, default=0,
                   help="label-permutation test on ROC AUC (0 = skip; 1000 matches embedding test)")
    add_condition_arg(p)
    return p.parse_args()


def main():
    from sklearn.model_selection import cross_validate, permutation_test_score
    args = parse_args()
    datasets = ["formal", "informal"] if args.dataset == "both" else [args.dataset]

    rows = []
    for ds in datasets:
        for cond, _ in resolve_conditions(ds, args.condition):
            tag = condition_tag(ds, cond)
            loaded = load_feature_matrix(tag)
            if loaded is None:
                print(f"[skip] no feature files found for {tag}")
                continue
            X, y, groups, feat_names, found = loaded
            n_per_class = int(np.bincount(y).min())
            print(f"\n[{ds}/{cond}] {n_per_class} per class, {len(feat_names)} features "
                  f"from {found}")

            model = build_model(args.model)
            cv, cv_kw = make_cv(args.cv, y, groups)
            scoring = ["roc_auc", "accuracy", "f1"]
            cvres = cross_validate(model, X, y, scoring=scoring, cv=cv,
                                   n_jobs=-1, **cv_kw)
            auc = float(cvres["test_roc_auc"].mean())
            auc_sd = float(cvres["test_roc_auc"].std())
            acc = float(cvres["test_accuracy"].mean())
            f1 = float(cvres["test_f1"].mean())
            print(f"  CV ROC AUC = {auc:.4f} (+/-{auc_sd:.4f})   "
                  f"accuracy = {acc:.4f}   F1 = {f1:.4f}")

            perm_p = None
            if args.permutations > 0:
                print(f"  permutation test ({args.permutations} perms)...")
                _, _, perm_p = permutation_test_score(
                    build_model(args.model), X, y, scoring="roc_auc", cv=cv,
                    n_permutations=args.permutations, random_state=0, n_jobs=-1,
                    **cv_kw)
                perm_p = float(perm_p)
                print(f"  permutation p = {perm_p:.4g}")

            imp = feature_importance(build_model(args.model), args.model, X, y, feat_names)
            imp_path = os.path.join(CSV_DIR, f"feature_detector_importance_{tag}.csv")
            imp.to_csv(imp_path, index=False, encoding="utf-8")
            top = imp.head(8)["feature"].tolist()
            print(f"  top features: {', '.join(top)}")
            print(f"  wrote {os.path.basename(imp_path)}")

            rows.append(dict(dataset=ds, condition=cond or "baseline",
                             n_per_class=n_per_class, n_features=len(feat_names),
                             model=args.model, cv=args.cv,
                             roc_auc=round(auc, 4), roc_auc_sd=round(auc_sd, 4),
                             accuracy=round(acc, 4), f1=round(f1, 4),
                             permutation_p=perm_p))

    if not rows:
        print("\nNothing produced -- run the feature scripts first "
              "(stylometric_surface.py, syntactic_complexity.py, ...).")
        return

    new = pd.DataFrame(rows)
    # merge with any existing results, replacing recomputed (dataset, condition) rows
    if os.path.exists(RESULTS):
        old = pd.read_csv(RESULTS)
        key = ["dataset", "condition"]
        old = old.merge(new[key], on=key, how="left", indicator=True)
        old = old[old["_merge"] == "left_only"].drop(columns="_merge")
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out = out.sort_values(["dataset", "condition"]).reset_index(drop=True)
    out.to_csv(RESULTS, index=False, encoding="utf-8")

    print(f"\nWrote {RESULTS}")
    print("\nDetectability (CV ROC AUC) by condition — higher = easier to detect:")
    for ds, sub in out.groupby("dataset"):
        print(f"  {ds}:")
        for _, r in sub.iterrows():
            star = "  *reported" if r["condition"] in REPORTED else ""
            print(f"    {r['condition']:<16} AUC={r['roc_auc']:.4f}  "
                  f"acc={r['accuracy']:.3f}  F1={r['f1']:.3f}{star}")


if __name__ == "__main__":
    main()
