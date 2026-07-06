"""
Syntactic complexity metrics complementing the dependency distribution already in
spacy_linguistic_analysis.py:

  - Dependency distance: mean linear distance (|token.i - head.i|) across all
    non-ROOT tokens in a document, averaged per sentence.
  - Parse tree depth: max depth of the dependency tree per sentence, then averaged
    across sentences in a document.
  - Subordinate-clause rate: number of subordinating dependency arcs
    (advcl, relcl, acl, acl:relcl, csubj, csubj:pass, ccomp) per sentence.

Reads paired human/LLM text from the same CSV used by all_ling_analysis_plot.py.
Outputs a per-document feature CSV plus a summary to features/.

Usage:
  python src/2_text_analysis_scripts/syntactic_complexity.py --dataset formal
  python src/2_text_analysis_scripts/syntactic_complexity.py --dataset informal
"""
import argparse
import os

import numpy as np
import pandas as pd
import spacy

from data_utils import read_csv_robust, add_condition_arg, resolve_conditions, condition_tag

MODEL = "sv_core_news_lg"

_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
_data = os.path.join(_root, "1_data_collection")
OUT_DIR = os.path.join(_2tas_dir, "csv_files")

SUBORDINATE_DEPS = {
    "advcl", "relcl", "acl", "acl:relcl",
    "csubj", "csubj:pass", "ccomp",
}


def parse_args():
    p = argparse.ArgumentParser(description="Syntactic complexity metrics")
    p.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    add_condition_arg(p)
    return p.parse_args()


def resolve_config(dataset):
    if dataset == "informal":
        return {
            "csv": os.path.join(_data, "llm_comments", "consolidated_informal_comments_JUN26.csv"),
            "human_col": "human_comment",
            "llm_col": "generated_comment",
            "min_tokens": 5,
        }
    return {
        "csv": os.path.join(_data, "llm_abstracts", "abstracts", "sv_abstracts_adversarial.csv"),
        "human_col": "Abstract",
        "llm_col": "Abstract_baseline",
        "min_tokens": 30,
    }


def load_nlp():
    try:
        return spacy.load(MODEL)
    except OSError:
        raise OSError(f"Model not found. Run: python -m spacy download {MODEL}")


def mean_dep_distance(doc):
    """Mean |token.i - head.i| for all non-ROOT tokens."""
    distances = [abs(t.i - t.head.i) for t in doc if t.dep_ != "ROOT"]
    return float(np.mean(distances)) if distances else 0.0


def max_tree_depth(sent):
    """Maximum depth from ROOT to any leaf in a sentence's parse tree."""
    root_nodes = [t for t in sent if t.dep_ == "ROOT"]
    if not root_nodes:
        return 0

    def _depth(tok, d=0):
        children = list(tok.children)
        return d if not children else max(_depth(c, d + 1) for c in children)

    return _depth(root_nodes[0])


def subordinate_clause_rate(doc):
    """Count of subordinating arcs divided by number of sentences."""
    sents = list(doc.sents)
    if not sents:
        return 0.0
    n_sub = sum(1 for t in doc if t.dep_ in SUBORDINATE_DEPS)
    return n_sub / len(sents)


def compute_metrics(nlp, text, min_tokens=0):
    if pd.isna(text) or not str(text).strip():
        return None
    doc = nlp(str(text).strip())
    if len(doc) < min_tokens:
        return None
    sents = list(doc.sents)
    depths = [max_tree_depth(s) for s in sents]
    return {
        "dep_distance_mean": mean_dep_distance(doc),
        "tree_depth_mean": float(np.mean(depths)) if depths else 0.0,
        "tree_depth_max": float(max(depths)) if depths else 0.0,
        "subordinate_clause_rate": subordinate_clause_rate(doc),
        "n_sentences": len(sents),
    }


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    nlp = load_nlp()
    df = read_csv_robust(cfg["csv"])
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(df)

    # human side is identical across conditions -> parse it once and reuse
    print("parsing human texts…", flush=True)
    human_feats = [compute_metrics(nlp, row.get(cfg["human_col"]), cfg["min_tokens"]) for _, row in df.iterrows()]

    for cond, llm_col in resolve_conditions(args.dataset, args.condition):
        if llm_col not in df.columns:
            print(f"  skipping condition '{cond}': column '{llm_col}' not found")
            continue
        tag = condition_tag(args.dataset, cond)
        print(f"parsing LLM texts [{cond or 'generated'}]…", flush=True)
        rows = []
        for i, (_, row) in enumerate(df.iterrows()):
            print(f"  {i + 1}/{n}", end="\r", flush=True)
            h = human_feats[i]
            l = compute_metrics(nlp, row.get(llm_col), cfg["min_tokens"])
            if h is None or l is None:
                continue
            out_row = {f"human_{k}": v for k, v in h.items()}
            out_row.update({f"llm_{k}": v for k, v in l.items()})
            rows.append(out_row)
        print()

        out_df = pd.DataFrame(rows)
        out_path = os.path.join(OUT_DIR, f"syntactic_complexity_{tag}.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"[{cond or 'generated'}] Wrote {len(out_df)} rows → {out_path}")

        print("Means (human | llm):")
        metrics = ["dep_distance_mean", "tree_depth_mean", "tree_depth_max", "subordinate_clause_rate"]
        for m in metrics:
            hcol, lcol = f"human_{m}", f"llm_{m}"
            print(f"  {m:30s}  {out_df[hcol].mean():.3f}  |  {out_df[lcol].mean():.3f}")


if __name__ == "__main__":
    main()
