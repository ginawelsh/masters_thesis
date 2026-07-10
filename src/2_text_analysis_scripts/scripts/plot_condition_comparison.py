"""Comparison figures across Human + the LLM prompt conditions.

Reads the per-condition feature CSVs, significance_tests_results.csv and the embedding
two-sample CSVs (all produced by the other analysis scripts) and renders, to figures/:

  condition_boxplots_<dataset>.png     distributions of key features across the 4 groups
                                       (Human, LLM baseline, LLM human-like, LLM detector-aware)
  condition_effect_sizes_<dataset>.png Cohen's dz heatmap, feature x condition (LLM vs Human)
  condition_detectability_<dataset>.png embedding classifier ROC AUC per condition

Run:  python src/2_text_analysis_scripts/scripts/plot_condition_comparison.py --dataset formal
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display; write PNGs
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from data_utils import read_csv_robust, resolve_conditions, condition_tag

_HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(_HERE)
CSV_DIR = os.path.join(_2TAS, "csv_files")
FIG_DIR = os.path.join(_2TAS, "figures")

# validated categorical palette (node scripts/validate_palette.js — all checks pass, CVD ΔE≈18)
PALETTE = {
    "Human":           "#0F9E84",
    "baseline":        "#3B6FB0",
    "human_like":      "#C77F2C",
    "detector_aware":  "#9A56B8",
    "detector_evasive": "#C6495B",  # NB: extends the validated 4-colour set; re-run scripts/validate_palette.js
}
COND_LABEL = {
    "Human": "Human", "baseline": "LLM baseline",
    "human_like": "LLM human-like", "detector_aware": "LLM detector-aware",
    "detector_evasive": "LLM detector-evasive",
    None: "LLM",
}
INK, MUTED, GRID = "#1b1f27", "#667085", "#e3e7ec"

FEATURE_GROUPS = ["syntactic_complexity", "stylometric_surface", "pragmatic_markers"]
# curated, interpretable features for the distribution grid (only those present are drawn)
CURATED = [
    "dep_distance_mean", "tree_depth_mean", "subordinate_clause_rate",
    "mtld", "mattr", "ttr", "function_word_rate", "bigram_repetition_rate",
    "punct_comma", "punct_period", "hedge_rate",
    "hedge_formal_rate", "hedge_neutral_rate", "hedge_informal_rate",
    "negation_rate",
]
PRETTY = {
    "dep_distance_mean": "Dependency distance", "tree_depth_mean": "Parse-tree depth",
    "subordinate_clause_rate": "Subordinate clauses/sent", "mtld": "MTLD (lexical diversity)",
    "mattr": "MATTR", "ttr": "Type–token ratio", "function_word_rate": "Function-word rate",
    "bigram_repetition_rate": "Bigram repetition", "punct_comma": "Commas /100 tok",
    "punct_period": "Periods /100 tok", "hedge_rate": "Hedges /100 tok",
    "negation_rate": "Negation /100 tok",
    # remaining features (used in the effect-size heatmap)
    "trigram_repetition_rate": "Trigram repetition", "tree_depth_max": "Parse-tree depth (max)",
    "n_alpha_tokens": "Length (alpha tokens)", "n_sentences": "Sentence count",
    "punct_dash": "Dashes /100 tok", "punct_colon": "Colons /100 tok",
    "punct_semicolon": "Semicolons /100 tok", "hedge_count": "Hedges (count)",
    "negation_count": "Negation (count)", "negation_sentence_rate": "Negation-sentence rate",
    "epistemic_rate": "Epistemic /100 tok", "epistemic_count": "Epistemic (count)",
    # hedges split by register (F/N/I) — see pragmatic_markers.HEDGE_LEXICON
    "hedge_formal_rate": "Hedges: formal /100 tok", "hedge_formal_count": "Hedges: formal (count)",
    "hedge_neutral_rate": "Hedges: neutral /100 tok", "hedge_neutral_count": "Hedges: neutral (count)",
    "hedge_informal_rate": "Hedges: informal /100 tok", "hedge_informal_count": "Hedges: informal (count)",
    # named-entity rates per 100 tokens (see ner_features.py / plot_ner_distribution.py)
    "ner_total_rate": "NER: total /100 tok",
    "ner_LOC_rate": "NER: Location /100 tok", "ner_PRS_rate": "NER: Person /100 tok",
    "ner_ORG_rate": "NER: Organisation /100 tok", "ner_TME_rate": "NER: Time /100 tok",
    "ner_WRK_rate": "NER: Work/Artefact /100 tok", "ner_MSR_rate": "NER: Measure /100 tok",
    "ner_OBJ_rate": "NER: Object /100 tok", "ner_EVN_rate": "NER: Event /100 tok",
}


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def load_groups(dataset, keep=None):
    """Return (order, {feature: {group: np.array}}). Human is shared across conditions.
    keep: optional list of condition names to include (default all)."""
    conds = resolve_conditions(dataset, "all")          # baseline/human_like/detector_aware/detector_evasive
    if keep is not None:
        conds = [(c, col) for c, col in conds if c in keep]
    order = ["Human"] + [c for c, _ in conds if c] if conds and conds[0][0] else ["Human"] + [None]
    data = {}
    human_done = set()
    for grp in FEATURE_GROUPS:
        for cond, _ in conds:
            tag = condition_tag(dataset, cond)
            path = os.path.join(CSV_DIR, f"{grp}_{tag}.csv")
            if not os.path.exists(path):
                continue
            df = read_csv_robust(path)
            for hcol in [c for c in df.columns if c.startswith("human_")]:
                feat = hcol[len("human_"):]
                lcol = "llm_" + feat
                if lcol not in df.columns:
                    continue
                try:
                    lvals = pd.to_numeric(df[lcol], errors="raise").dropna().to_numpy()
                    hvals = pd.to_numeric(df[hcol], errors="raise").dropna().to_numpy()
                except (ValueError, TypeError):
                    continue
                d = data.setdefault(feat, {})
                d[cond if cond else None] = lvals
                if feat not in human_done:
                    d["Human"] = hvals
                    human_done.add(feat)
    return order, data


# ---------------------------------------------------------------------------
# Figure 1: distribution box plots across the 4 groups
# ---------------------------------------------------------------------------

def plot_boxplots(dataset, order, data, tag=""):
    feats = [f for f in CURATED if f in data and "Human" in data[f]]
    if not feats:
        print("  [skip] no curated features found for boxplots")
        return
    ncol = 4
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, feats):
        _style(ax)
        groups = [g for g in order if g in data[feat]]
        series = [data[feat][g] for g in groups]
        bp = ax.boxplot(series, positions=range(len(groups)), widths=0.62,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.4),
                        whiskerprops=dict(color=MUTED, linewidth=1.0),
                        capprops=dict(color=MUTED, linewidth=1.0),
                        boxprops=dict(linewidth=0))
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(PALETTE[g if g else "baseline"])
            patch.set_alpha(0.85)
        ax.set_title(PRETTY.get(feat, feat), fontsize=9, color=INK, fontweight="600")
        ax.set_xticks([])
    for ax in axes[len(feats):]:
        ax.axis("off")

    handles = [Patch(facecolor=PALETTE[g if g else "baseline"], label=COND_LABEL[g])
               for g in order]
    fig.legend(handles=handles, loc="lower center", ncol=len(order), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Feature distributions — Human vs LLM conditions ({dataset})",
                 fontsize=12, fontweight="700", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    out = os.path.join(FIG_DIR, f"condition_boxplots_{dataset}{tag}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 2: Cohen's dz heatmap (feature x condition)
# ---------------------------------------------------------------------------

def plot_effect_sizes(dataset, keep=None, tag=""):
    path = os.path.join(CSV_DIR, "significance_tests_results.csv")
    if not os.path.exists(path):
        print("  [skip] significance_tests_results.csv not found")
        return
    s = pd.read_csv(path)
    s = s[s["dataset"] == dataset]
    if s.empty:
        print(f"  [skip] no {dataset} rows in significance results")
        return
    cond_order = [c for c in ["baseline", "human_like", "detector_aware", "detector_evasive"]
                  if c in s["condition"].unique() and (keep is None or c in keep)]
    dz = s.pivot_table(index="feature", columns="condition", values="cohens_dz", aggfunc="first")
    sig = s.pivot_table(index="feature", columns="condition",
                        values="significant_fdr_0.05", aggfunc="first")
    dz = dz.reindex(columns=cond_order)
    sig = sig.reindex(columns=cond_order)
    # order rows by mean dz (most llm>human at top)
    dz = dz.loc[dz.mean(axis=1).sort_values(ascending=False).index]
    sig = sig.reindex(index=dz.index)

    vmax = float(np.nanmax(np.abs(dz.values))) or 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    fig, ax = plt.subplots(figsize=(1.15 * len(cond_order) + 4.2, 0.32 * len(dz) + 1.6))
    im = ax.imshow(dz.values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(cond_order)))
    ax.set_xticklabels([COND_LABEL[c] for c in cond_order], fontsize=9, color=INK)
    ax.set_yticks(range(len(dz)))
    ax.set_yticklabels([PRETTY.get(f, f) for f in dz.index], fontsize=8, color=INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(dz)):
        for j in range(len(cond_order)):
            v = dz.values[i, j]
            if np.isnan(v):
                continue
            star = "*" if bool(sig.values[i, j]) else ""
            ax.text(j, i, f"{v:+.2f}{star}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(norm(v) - 0.5) > 0.28 else INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Cohen's dz  (+ = LLM > Human)", fontsize=8, color=MUTED)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    ax.set_title(f"Effect size vs Human by condition ({dataset})   * = sig. (BH-FDR<0.05)",
                 fontsize=11, fontweight="700", color=INK, pad=10)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"condition_effect_sizes_{dataset}{tag}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Figure 3: embedding detectability (ROC AUC) per condition
# ---------------------------------------------------------------------------

def plot_detectability(dataset, keep=None, tag=""):
    conds = [c for c, _ in resolve_conditions(dataset, "all") if (keep is None or c in keep)]
    aucs = []
    for cond in conds:
        ctag = condition_tag(dataset, cond)
        path = os.path.join(CSV_DIR, f"embedding_twosample_{ctag}.csv")
        if not os.path.exists(path):
            continue
        m = dict(zip(*[pd.read_csv(path)[c] for c in ("metric", "value")]))
        try:
            aucs.append((cond, float(m["classifier_cv_roc_auc"])))
        except (KeyError, ValueError):
            continue
    if not aucs:
        print("  [skip] no embedding_twosample CSVs found")
        return
    fig, ax = plt.subplots(figsize=(1.4 * len(aucs) + 2.0, 3.4))
    _style(ax)
    labels = [COND_LABEL[c] for c, _ in aucs]
    vals = [v for _, v in aucs]
    bars = ax.bar(range(len(aucs)), vals, width=0.6,
                  color=[PALETTE[c if c else "baseline"] for c, _ in aucs])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, color=INK, fontweight="600")
    ax.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(len(aucs) - 0.5, 0.5, " chance", va="center", ha="left", fontsize=8, color=MUTED)
    ax.set_xticks(range(len(aucs)))
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(0.4, 1.12)  # headroom so near-1.0 bar labels don't collide with the title
    ax.set_ylabel("classifier ROC AUC (human vs LLM)", fontsize=9, color=MUTED)
    ax.set_title(f"Detectability by condition ({dataset})", fontsize=11,
                 fontweight="700", color=INK, pad=12)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, f"condition_detectability_{dataset}{tag}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Comparison figures across Human + LLM conditions")
    ap.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    ap.add_argument("--conditions", default="all",
                    help="comma-separated condition subset (e.g. baseline,human_like,detector_evasive); default all")
    ap.add_argument("--tag", default="",
                    help="filename suffix so a subset view does not overwrite the canonical figures")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)

    keep = None if args.conditions == "all" else [c.strip() for c in args.conditions.split(",")]
    tag = f"_{args.tag}" if args.tag else ""
    order, data = load_groups(args.dataset, keep)
    print(f"[{args.dataset}] groups: {[COND_LABEL[g] for g in order]}")
    plot_boxplots(args.dataset, order, data, tag)
    plot_effect_sizes(args.dataset, keep, tag)
    plot_detectability(args.dataset, keep, tag)


if __name__ == "__main__":
    main()
