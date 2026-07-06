"""
Linguistic analysis (spaCy) on Abstract and Generated Abstract from formal CSV.
Install: pip install -r requirements.txt
Swedish: python -m spacy download sv_core_news_lg

The sv_core_news_lg model supports:
  - POS tagging  (already in pipeline)
  - Dependency parsing  (arc labels: ROOT, nsubj, obj, nmod, …)
  - Named entity recognition  (PER, ORG, LOC, MISC)
"""
import argparse
import os
from collections import Counter

import pandas as pd
import spacy

from data_utils import add_condition_arg, resolve_conditions, condition_tag

# Use Swedish small model for thesis data
MODEL = "sv_core_news_lg"

# NOTE: This repo currently stores outputs under `src/1_data_collection/...`.
# Some older paths in this script referenced `src/data_collection/...`.
_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)
HUMAN_FORMAL_DIR = os.path.join(_root, "1_data_collection", "human_abstracts")
LLM_INFORMAL_DIR = os.path.join(_root, "1_data_collection", "llm_comments")
RESULTS_FORMAL_DIR = os.path.join(_2tas_dir, "results", "formal")
RESULTS_INFORMAL_DIR = os.path.join(_2tas_dir, "results", "informal")

TOKEN_SEP = ","  # separator for tokens/POS in CSV cells
POS_SEP = " | "  # separator for TAG:n lists in CSV cells (matches existing checked-in counts file)
VERBOSE = False  # set True to print per-row token/POS debug output
# per-register token floor: abstracts are long (drop fragments < 30 tokens); comments are
# short, so keep everything (0 = no floor) rather than lose ~40% of the corpus.
MIN_TOKENS_BY_DATASET = {"formal": 30, "informal": 0}


def parse_args():
    parser = argparse.ArgumentParser(description="POS analysis for formal/informal paired datasets")
    parser.add_argument(
        "--dataset",
        choices=["formal", "informal"],
        default="formal",
        help="Preset input/output config.",
    )
    add_condition_arg(parser)
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Reuse an existing POS-counts file (derive proportions without re-running spaCy). "
             "Off by default so a data change triggers a fresh parse.",
    )
    return parser.parse_args()


def resolve_config(dataset: str) -> dict:
    if dataset == "informal":
        return {
            "csv_path": os.path.join(LLM_INFORMAL_DIR, "consolidated_informal_comments_adversarial.csv"),
            "results_dir": RESULTS_INFORMAL_DIR,
            "human_col": "human_comment",
            "human_label": "Comment",
            "generated_label": "Generated Comment",
        }
    return {
        "csv_path": os.path.join(os.path.dirname(HUMAN_FORMAL_DIR), "llm_abstracts", "abstracts", "sv_abstracts_adversarial.csv"),
        "results_dir": RESULTS_FORMAL_DIR,
        "human_col": "Abstract",
        "human_label": "Abstract",
        "generated_label": "Generated Abstract",
    }


_OUT_FILE_NAMES = {
    "out_tokens": "linguistic_analysis_tokens.csv",
    "out_pos": "linguistic_analysis_pos.csv",
    "out_pos_counts": "linguistic_analysis_pos_counts.csv",
    "out_pos_proportions": "linguistic_analysis_pos_proportions.csv",
    "out_pos_differences": "linguistic_analysis_pos_differences.csv",
    "out_dep": "linguistic_analysis_dep.csv",
    "out_dep_counts": "linguistic_analysis_dep_counts.csv",
    "out_entities": "linguistic_analysis_entities.csv",
    "out_entity_counts": "linguistic_analysis_entity_counts.csv",
}


def build_out_paths(output_dir: str) -> dict:
    """Full paths for the nine linguistic-analysis CSVs under output_dir (created if missing)."""
    os.makedirs(output_dir, exist_ok=True)
    return {key: os.path.join(output_dir, name) for key, name in _OUT_FILE_NAMES.items()}


def load_nlp(model_name: str = MODEL):
    """Load spaCy pipeline. Download with: python -m spacy download <model_name>"""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found. Run: python -m spacy download {model_name}")
        raise


def process_text(nlp, text: str):
    """Return a Doc for the given text."""
    if pd.isna(text) or not str(text).strip():
        return None
    return nlp(str(text).strip())


def pos_counts_string(doc) -> str:
    """Return POS counts as 'TAG:n | TAG:n | ...' sorted by tag."""
    if doc is None:
        return ""
    counts = Counter(t.pos_ for t in doc)
    return POS_SEP.join(f"{tag}:{n}" for tag, n in sorted(counts.items()))


# ---------------------------------------------------------------------------
# Dependency parsing helpers
# ---------------------------------------------------------------------------

def dep_counts_string(doc) -> str:
    """Return dependency-relation counts as 'DEP:n | DEP:n | ...' sorted by label."""
    if doc is None:
        return ""
    counts = Counter(t.dep_ for t in doc)
    return POS_SEP.join(f"{dep}:{n}" for dep, n in sorted(counts.items()))


def dep_tree_string(doc) -> str:
    """Return 'token/dep/head' triples joined by TOKEN_SEP for inspecting the arc structure."""
    if doc is None:
        return ""
    return TOKEN_SEP.join(f"{t.text}/{t.dep_}/{t.head.text}" for t in doc)


def parse_dep_counts_string(s: str) -> Counter:
    """Parse a 'DEP:n | DEP:n | ...' string into a Counter (same format as POS counts)."""
    return parse_pos_counts_string(s)  # identical format


# ---------------------------------------------------------------------------
# Named entity recognition helpers
# ---------------------------------------------------------------------------

def entity_counts_string(doc) -> str:
    """Return entity-type counts as 'TYPE:n | TYPE:n | ...' sorted by label.

    Swedish sv_core_news_lg labels: PER, ORG, LOC, MISC.
    """
    if doc is None:
        return ""
    counts = Counter(ent.label_ for ent in doc.ents)
    return POS_SEP.join(f"{label}:{n}" for label, n in sorted(counts.items()))


def entities_string(doc) -> str:
    """Return 'text:LABEL' pairs joined by TOKEN_SEP for all recognised entities."""
    if doc is None:
        return ""
    return TOKEN_SEP.join(f"{ent.text}:{ent.label_}" for ent in doc.ents)


def parse_pos_counts_string(s: str) -> Counter:
    """Parse a 'TAG:n | TAG:n | ...' string into a Counter."""
    counts: Counter = Counter()
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return counts
    text = str(s).strip()
    if not text:
        return counts
    for part in text.split(POS_SEP):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        tag, n_str = part.split(":", 1)
        tag = tag.strip()
        n_str = n_str.strip()
        if not tag or not n_str:
            continue
        try:
            counts[tag] += int(n_str)
        except ValueError:
            continue
    return counts


def pos_proportions_string(counts: Counter, ndigits: int = 4) -> str:
    """Return POS proportions as 'TAG:0.xxxx | TAG:0.xxxx | ...' sorted by tag."""
    total = sum(counts.values())
    if total <= 0:
        return ""
    return POS_SEP.join(
        f"{tag}:{round(n / total, ndigits):.{ndigits}f}" for tag, n in sorted(counts.items())
    )


def token_length_from_counts(counts: Counter) -> int:
    """Token length using option B: spaCy tokens including everything (sum of POS counts)."""
    return int(sum(counts.values()))


def parse_pos_proportions_string(s: str) -> dict:
    """Parse a 'TAG:0.xxxx | TAG:0.xxxx | ...' string into a dict."""
    props = {}
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return props
    text = str(s).strip()
    if not text:
        return props
    for part in text.split(POS_SEP):
        part = part.strip()
        if not part or ":" not in part:
            continue
        tag, p_str = part.split(":", 1)
        tag = tag.strip()
        p_str = p_str.strip()
        if not tag or not p_str:
            continue
        try:
            props[tag] = float(p_str)
        except ValueError:
            continue
    return props


def write_pos_differences_csv(df_pos_props: pd.DataFrame, out_path: str, ndigits: int = 4) -> None:
    """
    Write POS proportion differences (Human - Generated) per row, one column per POS,
    plus an AVERAGE summary row across all rows.
    """
    parsed_rows = []
    pos_tags = set()

    for idx in range(len(df_pos_props)):
        row = df_pos_props.iloc[idx]
        abs_props = parse_pos_proportions_string(row.get("Abstract_POS_Proportions", ""))
        gen_props = parse_pos_proportions_string(row.get("Generated_Abstract_POS_Proportions", ""))

        tags = set(abs_props.keys()) | set(gen_props.keys())
        pos_tags.update(tags)

        diff_row = {}
        for tag in tags:
            diff_row[tag] = round(abs_props.get(tag, 0.0) - gen_props.get(tag, 0.0), ndigits)
        parsed_rows.append(diff_row)

    sorted_tags = sorted(pos_tags)
    out_rows = []
    for idx, diff_row in enumerate(parsed_rows, start=1):
        out = {"Row": idx}
        for tag in sorted_tags:
            out[f"POS_Diff_{tag}"] = diff_row.get(tag, 0.0)
        out_rows.append(out)

    # Summary row: average signed difference per POS across all files.
    avg_row = {"Row": "AVERAGE"}
    if out_rows:
        df_tmp = pd.DataFrame(out_rows)
        for tag in sorted_tags:
            col = f"POS_Diff_{tag}"
            avg_row[col] = round(float(df_tmp[col].mean()), ndigits)
    else:
        for tag in sorted_tags:
            avg_row[f"POS_Diff_{tag}"] = 0.0

    out_rows.append(avg_row)
    pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows to {out_path}")


def _fast_path(out, min_tokens):
    """Derive POS proportions + differences from an existing counts file (no spaCy re-run)."""
    df_counts = pd.read_csv(out["out_pos_counts"], encoding="utf-8")
    rows_pos_props = []
    skipped = 0
    for idx in range(len(df_counts)):
        row = df_counts.iloc[idx]
        abs_counts = parse_pos_counts_string(row.get("Abstract_POS_Counts", ""))
        gen_counts = parse_pos_counts_string(row.get("Generated_Abstract_POS_Counts", ""))
        abs_len = token_length_from_counts(abs_counts)
        gen_len = token_length_from_counts(gen_counts)
        if (0 < abs_len < min_tokens) or (0 < gen_len < min_tokens):
            skipped += 1
            continue
        rows_pos_props.append({
            "Abstract_POS_Proportions": pos_proportions_string(abs_counts),
            "Generated_Abstract_POS_Proportions": pos_proportions_string(gen_counts),
            "Abstract_Word_Length": abs_len,
            "Generated_Abstract_Word_Length": gen_len,
        })
    if skipped:
        print(f"  Skipped {skipped} rows with fewer than {min_tokens} tokens in either text.")
    df_pos_props = pd.DataFrame(rows_pos_props)
    df_pos_props.to_csv(out["out_pos_proportions"], index=False, encoding="utf-8")
    print(f"  Wrote {len(rows_pos_props)} rows to {out['out_pos_proportions']}")
    write_pos_differences_csv(df_pos_props, out["out_pos_differences"])


def _process_condition(df, nlp, out, human_col, gen_col, min_tokens):
    """Full spaCy analysis for one (human, generated) column pairing; writes all nine CSVs."""
    rows_out, rows_pos, rows_pos_counts, rows_pos_props = [], [], [], []
    rows_dep, rows_dep_counts, rows_entities, rows_entity_counts = [], [], [], []

    for idx in range(len(df)):
        row = df.iloc[idx]

        doc_abs = process_text(nlp, row.get(human_col))
        abs_tokens = TOKEN_SEP.join(t.text for t in doc_abs) if doc_abs is not None else ""
        abs_pos = TOKEN_SEP.join(f"{t.text}_{t.pos_}" for t in doc_abs) if doc_abs is not None else ""
        abs_pos_counts = pos_counts_string(doc_abs)
        abs_len = 0 if doc_abs is None else len(doc_abs)
        abs_props = pos_proportions_string(Counter(t.pos_ for t in doc_abs)) if doc_abs is not None else ""

        doc_gen = process_text(nlp, row.get(gen_col))
        gen_tokens = TOKEN_SEP.join(t.text for t in doc_gen) if doc_gen is not None else ""
        gen_pos = TOKEN_SEP.join(f"{t.text}_{t.pos_}" for t in doc_gen) if doc_gen is not None else ""
        gen_pos_counts = pos_counts_string(doc_gen)
        gen_len = 0 if doc_gen is None else len(doc_gen)
        gen_props = pos_proportions_string(Counter(t.pos_ for t in doc_gen)) if doc_gen is not None else ""

        if (0 < abs_len < min_tokens) or (0 < gen_len < min_tokens):
            continue

        rows_out.append({"Abstract_Tokens": abs_tokens, "Generated_Abstract_Tokens": gen_tokens})
        rows_pos.append({"Abstract_POS": abs_pos, "Generated_Abstract_POS": gen_pos})
        rows_pos_counts.append({"Abstract_POS_Counts": abs_pos_counts, "Generated_Abstract_POS_Counts": gen_pos_counts})
        rows_pos_props.append({
            "Abstract_POS_Proportions": abs_props,
            "Generated_Abstract_POS_Proportions": gen_props,
            "Abstract_Word_Length": abs_len,
            "Generated_Abstract_Word_Length": gen_len,
        })
        rows_dep.append({"Abstract_Dep_Tree": dep_tree_string(doc_abs), "Generated_Abstract_Dep_Tree": dep_tree_string(doc_gen)})
        rows_dep_counts.append({"Abstract_Dep_Counts": dep_counts_string(doc_abs), "Generated_Abstract_Dep_Counts": dep_counts_string(doc_gen)})
        rows_entities.append({"Abstract_Entities": entities_string(doc_abs), "Generated_Abstract_Entities": entities_string(doc_gen)})
        rows_entity_counts.append({"Abstract_Entity_Counts": entity_counts_string(doc_abs), "Generated_Abstract_Entity_Counts": entity_counts_string(doc_gen)})

    pd.DataFrame(rows_out).to_csv(out["out_tokens"], index=False, encoding="utf-8")
    pd.DataFrame(rows_pos).to_csv(out["out_pos"], index=False, encoding="utf-8")
    pd.DataFrame(rows_pos_counts).to_csv(out["out_pos_counts"], index=False, encoding="utf-8")
    df_pos_props = pd.DataFrame(rows_pos_props)
    df_pos_props.to_csv(out["out_pos_proportions"], index=False, encoding="utf-8")
    write_pos_differences_csv(df_pos_props, out["out_pos_differences"])
    pd.DataFrame(rows_dep).to_csv(out["out_dep"], index=False, encoding="utf-8")
    pd.DataFrame(rows_dep_counts).to_csv(out["out_dep_counts"], index=False, encoding="utf-8")
    pd.DataFrame(rows_entities).to_csv(out["out_entities"], index=False, encoding="utf-8")
    pd.DataFrame(rows_entity_counts).to_csv(out["out_entity_counts"], index=False, encoding="utf-8")
    print(f"  Wrote 9 linguistic files ({len(rows_pos_counts)} paired rows each) to {os.path.dirname(out['out_tokens'])}")


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    min_tokens = MIN_TOKENS_BY_DATASET.get(args.dataset, 30)
    df = None
    nlp = None

    for cond, gen_col in resolve_conditions(args.dataset, args.condition):
        # each formal condition (baseline/human_like/detector_aware) -> its own subdir;
        # informal (cond=None, single generated column) -> the register dir directly.
        output_dir = os.path.join(cfg["results_dir"], cond) if cond else cfg["results_dir"]
        out = build_out_paths(output_dir)
        print(f"\n=== condition: {cond or 'generated'}  ->  {output_dir} ===")

        # Fast path (opt-in): reuse an existing counts file to derive proportions without spaCy.
        if args.use_cache and os.path.exists(out["out_pos_counts"]):
            print("  using cached POS counts (--use-cache)")
            _fast_path(out, min_tokens)
            continue

        if df is None:
            df = pd.read_csv(cfg["csv_path"], encoding="utf-8")
        if gen_col not in df.columns:
            print(f"  skipping condition '{cond}': column '{gen_col}' not found")
            continue
        if nlp is None:
            nlp = load_nlp()

        _process_condition(df, nlp, out, cfg["human_col"], gen_col, min_tokens)


if __name__ == "__main__":
    main()
