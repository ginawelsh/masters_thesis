"""Per-document named-entity rate features for significance testing.

Converts the spaCy entity-count CSVs (results/<dataset>/<condition>/) into the
paired human_/llm_ per-document format that significance_tests.py and
plot_condition_comparison.py consume, written to:

  csv_files/ner_<dataset>_<condition>.csv

One column pair per entity type as a rate per 100 tokens (length-normalised) —
ner_<TYPE>_rate — plus ner_total_rate. Row i is a matched pair: the human
document and its LLM regeneration. A side with 0 tokens contributes NaN so the
paired Wilcoxon test drops that pair for that feature.

Rates (not raw counts) are used because the two corpora differ in length;
per-100-token rates are the length-fair paired measure.

Run:  python src/2_text_analysis_scripts/scripts/ner_features.py --dataset both
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd

from data_utils import add_condition_arg, resolve_conditions, condition_tag

_HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_2TAS, "results")
CSV_DIR = os.path.join(_2TAS, "csv_files")

ENT_FILE = "linguistic_analysis_entity_counts.csv"
TOK_FILE = "linguistic_analysis_tokens.csv"
TOKEN_SEP = ","  # tokens are joined by "," inside each (quoted) cell

# Fixed column order so every output CSV has the same schema (Språkbanken SUC tags).
ENTITY_TYPES = ["LOC", "PRS", "ORG", "TME", "WRK", "MSR", "OBJ", "EVN"]


def parse_entity_cell(cell):
    """'LOC:1 | PRS:5' -> Counter({'LOC':1, 'PRS':5}); blank/NaN -> empty Counter."""
    c = Counter()
    if not isinstance(cell, str):
        return c
    for part in cell.split("|"):
        tag, sep, num = part.strip().rpartition(":")
        if not sep:
            continue
        try:
            c[tag.strip()] += int(num)
        except ValueError:
            continue
    return c


def token_length(cell):
    """Number of spaCy tokens in a comma-joined token cell."""
    if not isinstance(cell, str) or not cell.strip():
        return 0
    return sum(1 for t in cell.split(TOKEN_SEP) if t != "")


def doc_rates(ent_cell, tok_cell, prefix):
    """{f'{prefix}_ner_<TYPE>_rate': rate, ...} for one document.

    All-NaN if the document has 0 tokens (dropped downstream by the paired test).
    """
    length = token_length(tok_cell)
    if length == 0:
        row = {f"{prefix}_ner_{t}_rate": np.nan for t in ENTITY_TYPES}
        row[f"{prefix}_ner_total_rate"] = np.nan
        return row
    counts = parse_entity_cell(ent_cell)
    row = {f"{prefix}_ner_{t}_rate": counts.get(t, 0) / length * 100.0 for t in ENTITY_TYPES}
    row[f"{prefix}_ner_total_rate"] = sum(counts.values()) / length * 100.0
    return row


def build_condition(dataset, cond):
    folder = os.path.join(RESULTS_DIR, dataset, cond)
    ent_path, tok_path = os.path.join(folder, ENT_FILE), os.path.join(folder, TOK_FILE)
    if not (os.path.exists(ent_path) and os.path.exists(tok_path)):
        print(f"  [skip] missing entity/token CSVs for {dataset}/{cond}")
        return
    ent = pd.read_csv(ent_path, encoding="utf-8")
    tok = pd.read_csv(tok_path, encoding="utf-8")
    n = min(len(ent), len(tok))
    if len(ent) != len(tok):
        print(f"  [warn] {dataset}/{cond}: entity rows ({len(ent)}) != token rows "
              f"({len(tok)}); aligning to first {n}")

    rows = []
    for i in range(n):
        row = doc_rates(ent.iloc[i, 0], tok.iloc[i, 0], "human")   # col 0 = human side
        row.update(doc_rates(ent.iloc[i, 1], tok.iloc[i, 1], "llm"))  # col 1 = generated side
        rows.append(row)

    out_df = pd.DataFrame(rows)
    os.makedirs(CSV_DIR, exist_ok=True)
    out = os.path.join(CSV_DIR, f"ner_{condition_tag(dataset, cond)}.csv")
    out_df.to_csv(out, index=False, encoding="utf-8")
    print(f"  [{cond}] wrote {len(out_df)} rows -> {out}")


def parse_args():
    p = argparse.ArgumentParser(description="Per-document NER rate features (paired human/llm)")
    p.add_argument("--dataset", choices=["formal", "informal", "both"], default="both")
    add_condition_arg(p)
    return p.parse_args()


def main():
    args = parse_args()
    datasets = ["formal", "informal"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        print(f"[{ds}]")
        for cond, _ in resolve_conditions(ds, args.condition):
            build_condition(ds, cond)


if __name__ == "__main__":
    main()
