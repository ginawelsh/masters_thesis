"""
Word frequency analysis for Swedish formal (abstracts) and informal (Reddit) registers.
Compares human vs. AI-generated text using spaCy lemmatization.

Usage:
  python src/2_text_analysis_scripts/word_frequency_analysis.py --dataset formal
  python src/2_text_analysis_scripts/word_frequency_analysis.py --dataset informal
  python src/2_text_analysis_scripts/word_frequency_analysis.py --dataset formal --top-n 30
"""
import argparse
import os
from collections import Counter

import pandas as pd
import spacy
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Ellipse

MODEL = "sv_core_news_lg"

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HUMAN_FORMAL_DIR = os.path.join(_root, "1_data_collection", "human_formal")
LLM_FORMAL_DIR = os.path.join(_root, "1_data_collection", "llm_formal", "abstracts")
LLM_INFORMAL_DIR = os.path.join(_root, "1_data_collection", "llm_informal")
OUT_DIR = os.path.join(_root, "2_text_analysis_scripts")

DEFAULT_TOP_N = 20


def parse_args():
    parser = argparse.ArgumentParser(description="Word frequency analysis for formal/informal datasets")
    parser.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Number of top words to show")
    return parser.parse_args()


def resolve_config(dataset: str) -> dict:
    if dataset == "informal":
        return {
            "csv_path": os.path.join(_root, "1_data_collection", "llm_comments", "consolidated_informal_comments.csv"),
            "human_col": "human_comment",
            "ai_col": "generated_comment",
            "human_label": "Human comment",
            "ai_label": "AI comment",
            "out_csv": os.path.join(OUT_DIR, "word_freq_informal.csv"),
            "out_png": os.path.join(OUT_DIR, "word_freq_informal.png"),
        }
    return {
        "csv_path": os.path.join(LLM_FORMAL_DIR, "sv_ai_generated_abstracts.csv"),
        "human_col": "Human_Abstract",
        "ai_col": "AI_Abstract",
        "human_label": "Human abstract",
        "ai_label": "AI abstract",
        "out_csv": os.path.join(OUT_DIR, "word_freq_formal.csv"),
        "out_png": os.path.join(OUT_DIR, "word_freq_formal.png"),
    }


def load_nlp():
    try:
        return spacy.load(MODEL)
    except OSError:
        print(f"Model '{MODEL}' not found. Run: python -m spacy download {MODEL}")
        raise


def process_text(nlp, text: str):
    if pd.isna(text) or not str(text).strip():
        return None
    return nlp(str(text).strip())


def word_freq(nlp, texts: list[str], top_n: int) -> pd.DataFrame:
    counts: Counter = Counter()
    for text in texts:
        doc = process_text(nlp, text)
        if doc is None:
            continue
        for token in doc:
            if token.is_alpha and not token.is_stop and not token.is_space:
                counts[token.lemma_.lower()] += 1
    top = counts.most_common(top_n)
    return pd.DataFrame(top, columns=["word", "count"])


def plot_comparison(human_df: pd.DataFrame, ai_df: pd.DataFrame,
                    human_label: str, ai_label: str, out_path: str, top_n: int) -> None:
    human_words = set(human_df["word"])
    ai_words = set(ai_df["word"])
    human_only = human_words - ai_words   # in Human top-N but not AI top-N
    ai_only = ai_words - human_words       # in AI top-N but not Human top-N

    # Colour scheme: base colour per chart, distinct highlight for exclusive words
    HUMAN_BASE = "#4C72B0"    # steel blue
    HUMAN_EXCL = "#DD3333"    # red — words only in the Human chart
    AI_BASE    = "#DD8800"    # amber/orange
    AI_EXCL    = "#229944"    # green — words only in the AI chart

    def bar_colours(df: pd.DataFrame, base: str, excl_set: set, excl_colour: str):
        return [excl_colour if w in excl_set else base for w in df["word"]]

    human_colours = bar_colours(human_df, HUMAN_BASE, human_only, HUMAN_EXCL)
    ai_colours    = bar_colours(ai_df,    AI_BASE,    ai_only,    AI_EXCL)

    x_max = max(human_df["count"].max(), ai_df["count"].max())
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, top_n // 3)))

    for ax, df, label, colours, base, excl_col, excl_label in [
        (axes[0], human_df, human_label, human_colours, HUMAN_BASE, HUMAN_EXCL, "Only in human"),
        (axes[1], ai_df,    ai_label,    ai_colours,    AI_BASE,    AI_EXCL,    "Only in AI"),
    ]:
        ax.barh(df["word"][::-1], df["count"][::-1], color=colours[::-1])
        ax.set_title(f"Top {top_n} words — {label}")
        ax.set_xlabel("Count")
        ax.set_xlim(0, x_max)
        legend_handles = [
            Patch(facecolor=base,    label="Shared with other chart"),
            Patch(facecolor=excl_col, label=excl_label),
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()


def plot_venn(human_df: pd.DataFrame, ai_df: pd.DataFrame,
              human_label: str, ai_label: str, out_path: str, top_n: int) -> None:
    human_words = list(human_df["word"])   # frequency-ordered
    ai_words    = list(ai_df["word"])
    human_set   = set(human_words)
    ai_set      = set(ai_words)

    human_only = [w for w in human_words if w not in ai_set]
    ai_only    = [w for w in ai_words    if w not in human_set]
    shared     = [w for w in human_words if w in ai_set]   # ordered by human frequency

    HUMAN_COL  = "#4C72B0"
    AI_COL     = "#DD8800"
    SHARED_COL = "#444444"

    max_col   = max(len(human_only), len(shared), len(ai_only), 1)
    font_size = max(6, min(9, int(90 / max_col)))
    fig_h     = max(7, max_col * 0.35 + 2)

    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    cy = 0.50
    e_h = 0.80

    ax.add_patch(Ellipse((0.36, cy), width=0.50, height=e_h,
                         facecolor=HUMAN_COL, alpha=0.15,
                         edgecolor=HUMAN_COL, linewidth=2, zorder=1))
    ax.add_patch(Ellipse((0.64, cy), width=0.50, height=e_h,
                         facecolor=AI_COL, alpha=0.15,
                         edgecolor=AI_COL, linewidth=2, zorder=1))

    for x, label, count, colour in [
        (0.20, human_label, len(human_only), HUMAN_COL),
        (0.50, "Shared",    len(shared),     SHARED_COL),
        (0.80, ai_label,    len(ai_only),    AI_COL),
    ]:
        ax.text(x, 0.95, label,        ha="center", va="center",
                fontsize=10, fontweight="bold", color=colour)
        ax.text(x, 0.90, f"({count})", ha="center", va="center",
                fontsize=8, color=colour)

    y_top, y_bot = 0.85, 0.07

    def place_words(words, x, colour):
        if not words:
            return
        n = len(words)
        step = (y_top - y_bot) / max(n, 1)
        for i, word in enumerate(words):
            ax.text(x, y_top - i * step, word,
                    ha="center", va="center",
                    fontsize=font_size, color=colour, zorder=2)

    place_words(human_only, 0.20, HUMAN_COL)
    place_words(shared,     0.50, SHARED_COL)
    place_words(ai_only,    0.80, AI_COL)

    ax.set_title(f"Word overlap — top {top_n} words per source", fontsize=13, pad=12)

    venn_path = out_path.replace(".png", "_venn.png")
    plt.savefig(venn_path, dpi=150, bbox_inches="tight")
    print(f"Saved Venn diagram to {venn_path}")
    plt.show()


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    top_n = args.top_n

    df = pd.read_csv(cfg["csv_path"], encoding="utf-8", engine="python", on_bad_lines="warn")
    print(f"Loaded {len(df)} rows from {cfg['csv_path']}")

    nlp = load_nlp()

    human_texts = df[cfg["human_col"]].tolist()
    ai_texts = df[cfg["ai_col"]].tolist()

    print("Computing word frequencies...")
    human_freq = word_freq(nlp, human_texts, top_n)
    ai_freq = word_freq(nlp, ai_texts, top_n)

    merged = human_freq.rename(columns={"count": "human_count"}).merge(
        ai_freq.rename(columns={"count": "ai_count"}),
        on="word", how="outer"
    ).fillna(0).astype({"human_count": int, "ai_count": int})
    merged["diff"] = merged["human_count"] - merged["ai_count"]
    merged = merged.sort_values("human_count", ascending=False)

    merged.to_csv(cfg["out_csv"], index=False, encoding="utf-8")
    print(f"Saved frequency table to {cfg['out_csv']}")

    plot_comparison(human_freq, ai_freq, cfg["human_label"], cfg["ai_label"], cfg["out_png"], top_n)
    plot_venn(human_freq, ai_freq, cfg["human_label"], cfg["ai_label"], cfg["out_png"], top_n)


if __name__ == "__main__":
    main()
