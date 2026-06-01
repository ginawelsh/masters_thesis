"""
Visualise outputs from the four new feature-extraction scripts:
  stylometric_surface, syntactic_complexity, affective_analysis, pragmatic_markers

Reads CSVs from features/ and saves publication-ready figures to figures/.

Figures produced:
  fig_stylometric_vocab.png     — TTR, MATTR, MTLD, n-gram repetition, function words
  fig_stylometric_punct.png     — punctuation rates per 100 tokens
  fig_syntactic_complexity.png  — dependency distance, tree depth, subordinate clauses
  fig_affective_distributions.png — affective extremity, sentiment score, subjectivity
  fig_affective_sentiment.png   — stacked bar of POSITIVE / NEGATIVE / NEUTRAL
  fig_affective_emotions.png    — emotion category means (if emotion data was collected)
  fig_pragmatic_markers.png     — hedge, booster, negation, epistemic, discourse particle rates

Usage:
  python src/2_text_analysis_scripts/visualize_new_features.py
  python src/2_text_analysis_scripts/visualize_new_features.py --dataset formal
  python src/2_text_analysis_scripts/visualize_new_features.py --dataset informal
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

_script_dir = os.path.dirname(os.path.abspath(__file__))
FEAT_DIR    = os.path.join(_script_dir, "features")
FIG_DIR     = os.path.join(_script_dir, "figures")

PALETTE = {
    "Formal Human":   "#1565C0",
    "Formal LLM":     "#B71C1C",
    "Informal Human": "#64B5F6",
    "Informal LLM":   "#EF9A9A",
}
GROUP_ORDER = list(PALETTE)

sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 150


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_one(analysis, dataset):
    path = os.path.join(FEAT_DIR, f"{analysis}_{dataset}.csv")
    if not os.path.exists(path):
        return None
    df       = pd.read_csv(path)
    h_cols   = [c for c in df.columns if c.startswith("human_")]
    l_cols   = [c for c in df.columns if c.startswith("llm_")]
    strip    = lambda cols, pfx: {c: c[len(pfx):] for c in cols}
    label    = dataset.capitalize()
    hdf      = df[h_cols].rename(columns=strip(h_cols, "human_"))
    hdf["group"] = f"{label} Human"
    ldf      = df[l_cols].rename(columns=strip(l_cols, "llm_"))
    ldf["group"] = f"{label} LLM"
    return pd.concat([hdf, ldf], ignore_index=True)


def load_analysis(analysis, datasets):
    frames = [_load_one(analysis, ds) for ds in datasets]
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Generic box + strip grid
# ---------------------------------------------------------------------------

def box_grid(df, features, subtitles, suptitle, outname, ncols=3):
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]
    pal    = {g: PALETTE[g] for g in groups}

    valid = [(f, t) for f, t in zip(features, subtitles)
             if f in df.columns and df[f].notna().any()]
    if not valid:
        print(f"  [skip] no data for {outname}")
        return

    feats, titles = zip(*valid)
    nrows = -(-len(feats) // ncols)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.6, nrows * 3.4),
                             squeeze=False)
    flat = axes.flatten()

    for ax, feat, title in zip(flat, feats, titles):
        sub = df[["group", feat]].dropna()
        sns.boxplot(data=sub, x="group", y=feat, order=groups,
                    palette=pal, linewidth=0.8, fliersize=2,
                    width=0.55, ax=ax)
        sns.stripplot(data=sub, x="group", y=feat, order=groups,
                      palette=pal, size=2.5, alpha=0.35, jitter=True, ax=ax)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(groups, rotation=35, ha="right", fontsize=7.5)

    for ax in flat[len(feats):]:
        ax.set_visible(False)

    handles = [mpatches.Patch(color=PALETTE[g], label=g) for g in groups]
    fig.legend(handles=handles, loc="lower center", ncol=len(groups),
               fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)

    _save(fig, outname)


# ---------------------------------------------------------------------------
# Plot functions per analysis
# ---------------------------------------------------------------------------

def plot_stylometric(df):
    box_grid(
        df,
        features=["ttr", "mattr", "mtld",
                   "bigram_repetition_rate", "trigram_repetition_rate",
                   "function_word_rate"],
        subtitles=["TTR", "MATTR", "MTLD",
                   "Bigram repetition", "Trigram repetition",
                   "Function word rate"],
        suptitle="Vocabulary richness & repetition",
        outname="fig_stylometric_vocab.png",
        ncols=3,
    )
    box_grid(
        df,
        features=["punct_period", "punct_comma", "punct_semicolon",
                   "punct_colon", "punct_question", "punct_exclaim",
                   "punct_dash", "punct_ellipsis"],
        subtitles=["Period", "Comma", "Semicolon", "Colon",
                   "Question mark", "Exclamation", "Dash", "Ellipsis"],
        suptitle="Punctuation rates (per 100 tokens)",
        outname="fig_stylometric_punct.png",
        ncols=4,
    )


def plot_syntactic(df):
    box_grid(
        df,
        features=["dep_distance_mean", "tree_depth_mean",
                   "tree_depth_max", "subordinate_clause_rate"],
        subtitles=["Mean dep. distance", "Mean tree depth",
                   "Max tree depth", "Subordinate clause rate"],
        suptitle="Syntactic complexity",
        outname="fig_syntactic_complexity.png",
        ncols=2,
    )


def plot_affective(df):
    # Continuous features
    box_grid(
        df,
        features=["affective_extremity", "sentiment_score", "is_subjective"],
        subtitles=["Affective extremity", "Sentiment confidence", "Subjectivity (0/1)"],
        suptitle="Affective features",
        outname="fig_affective_distributions.png",
        ncols=3,
    )

    # Sentiment label stacked bar
    if "sentiment_label" in df.columns:
        _sentiment_bar(df)

    # Emotion category means (optional)
    emotion_cols = [c for c in df.columns if c.startswith("emotion_")]
    if emotion_cols:
        _emotion_bar(df, emotion_cols)


def _sentiment_bar(df):
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]
    rows   = []
    for g in groups:
        sub    = df.loc[df["group"] == g, "sentiment_label"].dropna()
        counts = sub.value_counts(normalize=True).rename(g)
        rows.append(counts)
    label_df = (pd.DataFrame(rows)
                  .fillna(0)
                  .T
                  .reindex(columns=groups, fill_value=0))

    label_colors = {"POSITIVE": "#43A047", "NEGATIVE": "#E53935", "NEUTRAL": "#90A4AE"}
    labels       = [l for l in ["POSITIVE", "NEGATIVE", "NEUTRAL"]
                    if l in label_df.index]

    fig, ax = plt.subplots(figsize=(max(4, len(groups) * 1.5), 4))
    bottoms = np.zeros(len(groups))
    for lbl in labels:
        vals = label_df.loc[lbl, groups].values
        ax.bar(groups, vals, bottom=bottoms,
               color=label_colors[lbl], label=lbl, width=0.55)
        bottoms += vals

    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion", fontsize=9)
    ax.set_title("Sentiment label distribution", fontsize=11, fontweight="bold")
    ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=8.5)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fig_affective_sentiment.png")


def _emotion_bar(df, emotion_cols):
    groups = [g for g in GROUP_ORDER if g in df["group"].unique()]
    means  = (df.groupby("group")[emotion_cols]
                .mean()
                .reindex([g for g in groups if g in df["group"].unique()]))
    means.columns = [c.replace("emotion_", "") for c in means.columns]

    x     = np.arange(len(means.columns))
    width = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(max(6, len(means.columns) * 1.3), 4))
    for k, (g, row) in enumerate(means.iterrows()):
        ax.bar(x + k * width, row.values, width=width,
               label=g, color=PALETTE.get(g, "#888"), alpha=0.85)

    ax.set_xticks(x + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(means.columns, fontsize=9)
    ax.set_ylabel("Mean probability", fontsize=9)
    ax.set_title("Emotion category distributions", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    _save(fig, "fig_affective_emotions.png")


def plot_pragmatic(df):
    box_grid(
        df,
        features=["hedge_rate", "booster_rate", "negation_rate",
                   "epistemic_rate", "discourse_particle_rate",
                   "negation_sentence_rate"],
        subtitles=["Hedge rate", "Booster rate", "Negation rate",
                   "Epistemic rate", "Discourse particle rate",
                   "Negation sentence rate"],
        suptitle="Pragmatic marker rates (per 100 alpha tokens; negation sentence rate: 0–1)",
        outname="fig_pragmatic_markers.png",
        ncols=3,
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Visualise new feature analyses")
    p.add_argument("--dataset", choices=["formal", "informal", "both"],
                   default="both")
    return p.parse_args()


def main():
    args     = parse_args()
    datasets = ["formal", "informal"] if args.dataset == "both" else [args.dataset]
    os.makedirs(FIG_DIR, exist_ok=True)

    analyses = {
        "stylometric_surface":  plot_stylometric,
        "syntactic_complexity": plot_syntactic,
        "affective_analysis":   plot_affective,
        "pragmatic_markers":    plot_pragmatic,
    }

    for name, plot_fn in analyses.items():
        df = load_analysis(name, datasets)
        if df is None:
            print(f"\n[{name}] No feature CSVs found in {FEAT_DIR} — run the analysis script first.")
            continue
        groups = df["group"].unique().tolist()
        print(f"\n[{name}] {len(df)} rows across {groups}")
        plot_fn(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
