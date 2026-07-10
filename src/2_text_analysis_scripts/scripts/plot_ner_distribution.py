"""Named-entity distribution figure: Human vs the LLM prompt conditions.

Rebuilds the NER result plot inside the current 4-group pipeline (the old
all_ling_analysis_plot.py that produced fig_ner_distribution.png is archived).

Reads the per-document spaCy entity-count CSVs written by
spacy_linguistic_analysis.py, from results/<dataset>/<condition>/:

  linguistic_analysis_entity_counts.csv   <prefix>_Entity_Counts | Generated_<prefix>_Entity_Counts
  linguistic_analysis_tokens.csv          <prefix>_Tokens        | Generated_<prefix>_Tokens

and renders, matching plot_condition_comparison.py's style, to figures/:

  ner_distribution_<dataset>.png   grouped bars — entity type x group, mean
                                   entities per 100 tokens (Human + 3 LLM conditions)

Metric: mean over documents of (type count / doc token length) * 100 — the same
"mean entities per 100 tokens" measure used by the previous NER plot. Human text
is identical across conditions, so it is read once (from the first condition folder).

A tidy data table is also written to csv_files/ner_distribution_<dataset>.csv.

Run:  python src/2_text_analysis_scripts/scripts/plot_ner_distribution.py --dataset formal
      python src/2_text_analysis_scripts/scripts/plot_ner_distribution.py --dataset both
"""
import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display; write PNGs
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from data_utils import resolve_conditions

_HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(_HERE)
RESULTS_DIR = os.path.join(_2TAS, "results")
CSV_DIR = os.path.join(_2TAS, "csv_files")
FIG_DIR = os.path.join(_2TAS, "figures")

# palette + labels shared with plot_condition_comparison.py
PALETTE = {
    "Human":           "#0F9E84",
    "baseline":        "#3B6FB0",
    "human_like":      "#C77F2C",
    "detector_aware":  "#9A56B8",
    "detector_evasive": "#C6495B",
}
COND_LABEL = {
    "Human": "Human", "baseline": "LLM baseline",
    "human_like": "LLM human-like", "detector_aware": "LLM detector-aware",
    "detector_evasive": "LLM detector-evasive",
}
GROUP_ORDER = ["Human", "baseline", "human_like", "detector_aware", "detector_evasive"]
INK, MUTED, GRID = "#1b1f27", "#667085", "#e3e7ec"

# Swedish SUC / Språkbanken entity labels -> readable names (see main.tex NER section)
ENTITY_NAME = {
    "LOC": "Location", "ORG": "Organisation", "PRS": "Person",
    "TME": "Time", "WRK": "Work/Artefact", "MSR": "Measure",
    "OBJ": "Object", "EVN": "Event", "PER": "Person", "MISC": "Misc",
}

ENT_FILE = "linguistic_analysis_entity_counts.csv"
TOK_FILE = "linguistic_analysis_tokens.csv"
TOKEN_SEP = ","  # tokens are joined by "," inside each (quoted) cell


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


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


def rate_per_100(ent_cells, tok_cells, min_tokens=0):
    """(rates, n_docs): mean over kept docs of (type count / token length) * 100.

    Documents shorter than max(1, min_tokens) spaCy tokens are dropped — with
    min_tokens=0 this only skips empty docs (identical to the unfiltered figure).
    """
    floor = max(1, min_tokens)
    sums = Counter()
    n_docs = 0
    for ent, tok in zip(ent_cells, tok_cells):
        length = token_length(tok)
        if length < floor:
            continue
        for tag, n in parse_entity_cell(ent).items():
            sums[tag] += (n / length) * 100.0
        n_docs += 1
    rates = {tag: v / n_docs for tag, v in sums.items()} if n_docs else {}
    return rates, n_docs


def load_dataset(dataset, min_tokens=0):
    """Return (rates, counts): {group: {entity_type: rate}} and {group: n_docs}."""
    rates, counts = {}, {}
    for cond, _ in resolve_conditions(dataset, "all"):
        folder = os.path.join(RESULTS_DIR, dataset, cond)
        ent_path, tok_path = os.path.join(folder, ENT_FILE), os.path.join(folder, TOK_FILE)
        if not (os.path.exists(ent_path) and os.path.exists(tok_path)):
            print(f"  [skip] missing entity/token CSVs for {dataset}/{cond}")
            continue
        ent = pd.read_csv(ent_path, encoding="utf-8")
        tok = pd.read_csv(tok_path, encoding="utf-8")
        # column 0 = human side, column 1 = generated side (prefix varies by corpus)
        rates[cond], counts[cond] = rate_per_100(ent.iloc[:, 1], tok.iloc[:, 1], min_tokens)
        if "Human" not in rates:  # human text is shared across conditions -> read once
            rates["Human"], counts["Human"] = rate_per_100(ent.iloc[:, 0], tok.iloc[:, 0], min_tokens)
    return rates, counts


def entity_order(rates):
    """Entity types present, ordered by total rate across groups (largest first)."""
    totals = Counter()
    for grp in rates.values():
        for tag, v in grp.items():
            totals[tag] += v
    return [tag for tag, _ in totals.most_common()]


def plot_ner(dataset, rates, counts, min_tokens=0, keep=None, tag=""):
    groups = [g for g in GROUP_ORDER
              if g in rates and (g == "Human" or keep is None or g in keep)]
    if not groups:
        print(f"  [skip] no groups loaded for {dataset}")
        return
    tags = entity_order({g: rates[g] for g in groups})
    if not tags:
        print(f"  [skip] no entities found for {dataset}")
        return

    suffix = tag + (f"_min{min_tokens}" if min_tokens else "")
    note = f"  (≥{min_tokens} tokens)" if min_tokens else ""

    x = np.arange(len(tags))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(7.5, 1.15 * len(tags) + 2), 4.8))
    _style(ax)
    for i, g in enumerate(groups):
        offset = (i - (len(groups) - 1) / 2) * width
        vals = [rates[g].get(t, 0.0) for t in tags]
        ax.bar(x + offset, vals, width, color=PALETTE[g], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([ENTITY_NAME.get(t, t) for t in tags], fontsize=9, color=INK)
    ax.set_ylabel("Mean entities per 100 tokens", fontsize=9, color=MUTED)
    ax.set_title(f"Named-entity type rate by group ({dataset}){note}",
                 fontsize=11, fontweight="700", color=INK, pad=10)
    handles = [Patch(facecolor=PALETTE[g], label=f"{COND_LABEL[g]} (n={counts[g]})")
               for g in groups]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"ner_distribution_{dataset}{suffix}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")

    # tidy data table: rows = entity type, columns = groups
    os.makedirs(CSV_DIR, exist_ok=True)
    table = pd.DataFrame(
        {COND_LABEL[g]: [round(rates[g].get(t, 0.0), 4) for t in tags] for g in groups},
        index=[f"{ENTITY_NAME.get(t, t)} ({t})" for t in tags],
    )
    csv_out = os.path.join(CSV_DIR, f"ner_distribution_{dataset}{suffix}.csv")
    table.to_csv(csv_out, encoding="utf-8")
    print(f"  wrote {csv_out}")
    print(f"  docs kept per group: " + ", ".join(f"{g}={counts[g]}" for g in groups))
    print(table.to_string())


def main():
    ap = argparse.ArgumentParser(description="Named-entity distribution figure by condition")
    ap.add_argument("--dataset", choices=["formal", "informal", "both"], default="both")
    ap.add_argument("--min-tokens", type=int, default=0,
                    help="Drop documents shorter than this many spaCy tokens (per group); "
                         "output gets a _min<N> suffix. 0 = no filter.")
    ap.add_argument("--conditions", default="all",
                    help="comma-separated condition subset (e.g. baseline,human_like,detector_evasive); default all")
    ap.add_argument("--tag", default="",
                    help="filename suffix so a subset view does not overwrite the canonical figure")
    args = ap.parse_args()
    keep = None if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    tag = f"_{args.tag}" if args.tag else ""
    datasets = ["formal", "informal"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        print(f"[{ds}]")
        rates, counts = load_dataset(ds, args.min_tokens)
        plot_ner(ds, rates, counts, args.min_tokens, keep, tag)


if __name__ == "__main__":
    main()
