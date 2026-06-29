"""
Linguistic analysis plots comparing four groups across POS, dependency parsing,
named entity recognition, token length, and sentiment.

Groups
------
  Formal Human    — sv_human_collection_with_kws.tsv     (Abstract)
  Formal LLM      — sv_ai_generated_abstracts.csv         (AI_Abstract)
  Informal Human  — LLM_consolidated_reddit_comments.csv  (real_comment)
  Informal LLM    — LLM_consolidated_reddit_comments.csv  (generated_comment)

Swedish spaCy model: python -m spacy download sv_core_news_lg
Sentiment model:     pip install transformers torch
                     KBLab/robust-swedish-sentiment-multiclass (downloaded automatically)

Output: src/2_text_analysis_scripts/figures/
"""
print("importing modules...")
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import spacy
from transformers import pipeline as hf_pipeline

print("Modules imported.")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_2tas_dir = os.path.dirname(_script_dir)
_root = os.path.dirname(_2tas_dir)

FORMAL_HUMAN_CSV = os.path.join(_root, "1_data_collection", "human_formal", "sv_human_collection_with_kws.csv")
FORMAL_LLM_CSV = os.path.join(_root, "1_data_collection", "llm_abstracts", "abstracts", "sv_ai_generated_abstracts.csv")
INFORMAL_CSV = os.path.join(_root, "1_data_collection", "llm_comments", "consolidated_informal_comments.csv")

FIGURES_DIR = os.path.join(_2tas_dir, "figures")
INFORMAL_FIGURES_DIR = os.path.join(_2tas_dir, "figures", "informal_comparison")
FORMAL_FIGURES_DIR = os.path.join(_2tas_dir, "figures", "formal_comparison")
MODEL = "sv_core_news_lg"

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
GROUP_ORDER = ["Formal Human", "Formal LLM", "Informal Human", "Informal LLM"]
GROUP_COLORS = {
    "Formal Human":   "#D06148",
    "Formal LLM":     "#4A52EE",
    "Informal Human": "#D06148",
    "Informal LLM":   "#4A52EE",
}

# Top POS and dep tags to show in charts (others bundled as "OTHER")
TOP_N_POS = 10
TOP_N_DEP = 12
INFORMAL_MIN_TOKENS = 3  # rows where either informal text has fewer words are excluded
NER_MIN_TOKENS = 20      # secondary NER analysis threshold — filters informal pairs where either text is shorter

SENTIMENT_MODEL = "KBLab/robust-swedish-sentiment-multiclass"
SENTIMENT_LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
SENTIMENT_COLORS = {"POSITIVE": "#0C8E10", "NEUTRAL": "#9E9E9E", "NEGATIVE": "#D12115"}

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_texts() -> dict[str, list[str]]:
    """Load raw text lists for each group."""
    groups: dict[str, list[str]] = {}

    # Formal — both columns from same paired file so counts always match
    df_fl = pd.read_csv(FORMAL_LLM_CSV, encoding="utf-8", on_bad_lines="warn")
    df_fl = df_fl.dropna(subset=["Human_Abstract", "AI_Abstract"])
    df_fl["Human_Abstract"] = df_fl["Human_Abstract"].astype(str).str.strip()
    df_fl["AI_Abstract"] = df_fl["AI_Abstract"].astype(str).str.strip()
    groups["Formal Human"] = df_fl["Human_Abstract"].tolist()
    groups["Formal LLM"] = df_fl["AI_Abstract"].tolist()

    # Informal — both columns from same file; filter rows where either text is too short
    df_inf = pd.read_csv(INFORMAL_CSV, encoding="utf-8")
    df_inf = df_inf.dropna(subset=["human_comment", "generated_comment"])
    df_inf["human_comment"] = df_inf["human_comment"].astype(str).str.strip()
    df_inf["generated_comment"] = df_inf["generated_comment"].astype(str).str.strip()
    before = len(df_inf)
    df_inf = df_inf[
        (df_inf["human_comment"].str.split().str.len() >= INFORMAL_MIN_TOKENS) &
        (df_inf["generated_comment"].str.split().str.len() >= INFORMAL_MIN_TOKENS)
    ]
    print(f"  Informal: removed {before - len(df_inf)} rows under {INFORMAL_MIN_TOKENS} words, {len(df_inf)} remaining")
    groups["Informal Human"] = df_inf["human_comment"].tolist()
    groups["Informal LLM"] = df_inf["generated_comment"].tolist()

    for name, texts in groups.items():
        print(f"  {name}: {len(texts)} texts")

    return groups


# ---------------------------------------------------------------------------
# spaCy processing
# ---------------------------------------------------------------------------

def process_group(nlp, texts: list[str]):
    """
    Returns per-doc counters for POS, dep, entity types, and token lengths.

    Returns
    -------
    pos_counts    : list[Counter]   POS counts per doc
    dep_counts    : list[Counter]   dep-relation counts per doc
    ent_counts    : list[Counter]   entity-type counts per doc
    token_lengths : list[int]       non-punct non-space token count per doc
    """
    pos_counts, dep_counts, ent_counts, token_lengths = [], [], [], []
    for doc in nlp.pipe(texts, batch_size=64, disable=[]):
        pos_counts.append(Counter(t.pos_ for t in doc))
        dep_counts.append(Counter(t.dep_ for t in doc))
        ent_counts.append(Counter(e.label_ for e in doc.ents))
        # "content" token count: exclude SPACE and PUNCT
        token_lengths.append(sum(1 for t in doc if t.pos_ not in ("PUNCT", "SPACE")))
    return pos_counts, dep_counts, ent_counts, token_lengths


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

# load pipeline from kblab
def load_sentiment_pipeline():
    """Load KBLab Swedish sentiment model. Install: pip install transformers torch"""
    try:
        return hf_pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
            batch_size=32,
        )
    except Exception as e:
        print(f"Could not load sentiment model '{SENTIMENT_MODEL}': {e}")
        print("Install with: pip install transformers torch")
        raise


def process_sentiment(sentiment_pipe, texts: list[str]) -> Counter:
    """
    Run sentiment over a list of texts.
    Returns a Counter of {label: count} using POSITIVE / NEUTRAL / NEGATIVE.
    """
    results = sentiment_pipe(texts)
    counts: Counter = Counter()
    for r in results:
        label = r["label"].upper()
        # Normalise any label variants the model may emit
        if "POS" in label:
            counts["POSITIVE"] += 1
        elif "NEG" in label:
            counts["NEGATIVE"] += 1
        else:
            counts["NEUTRAL"] += 1
    return counts


def plot_sentiment(group_sentiment: dict[str, Counter], out_dir: str) -> None:
    """Bar chart of sentiment proportions for all four groups."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(GROUP_ORDER))
    width = 0.25

    for i, label in enumerate(SENTIMENT_LABELS):
        proportions = []
        for group in GROUP_ORDER:
            counts = group_sentiment[group]
            total = sum(counts.values())
            proportions.append(counts.get(label, 0) / total if total else 0)
        offset = (i - 1) * width
        ax.bar(x + offset, proportions, width, label=label,
               color=SENTIMENT_COLORS[label], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Proportion of texts")
    ax.set_title("Sentiment distribution by group")
    ax.legend(title="Sentiment")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_sentiment_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    _save_sentiment_csv(group_sentiment, GROUP_ORDER, os.path.join(out_dir, "data_sentiment_distribution.csv"))


def plot_sentiment_pair(group_sentiment: dict[str, Counter], groups: list[str],
                        prefix: str, out_dir: str) -> None:
    """Bar chart of sentiment proportions for a two-group comparison."""
    g1, g2 = groups
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(SENTIMENT_LABELS))
    width = 0.35

    for i, group in enumerate(groups):
        counts = group_sentiment[group]
        total = sum(counts.values())
        proportions = [counts.get(lbl, 0) / total if total else 0 for lbl in SENTIMENT_LABELS]
        offset = (i - 0.5) * width
        color = GROUP_COLORS[group]
        ax.bar(x + offset, proportions, width, label=group, color=color, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(SENTIMENT_LABELS)
    ax.set_ylabel("Proportion of texts")
    ax.set_title(f"Sentiment distribution — {g1} vs {g2}")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_sentiment_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    _save_sentiment_csv(group_sentiment, groups, os.path.join(out_dir, f"{prefix}_sentiment_distribution.csv"))


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_proportions(per_doc_counts: list[Counter]) -> dict[str, float]:
    """Mean proportion of each tag across documents."""
    total_per_doc: list[float] = [sum(c.values()) for c in per_doc_counts]
    tag_sums: Counter = Counter()
    for c, total in zip(per_doc_counts, total_per_doc):
        if total == 0:
            continue
        for tag, n in c.items():
            tag_sums[tag] += n / total
    n_docs = sum(1 for t in total_per_doc if t > 0)
    return {tag: val / n_docs for tag, val in tag_sums.items()} if n_docs else {}


def aggregate_rate_per_100(per_doc_counts: list[Counter], per_doc_lengths: list[int]) -> dict[str, float]:
    """Mean entity-type occurrence rate per 100 content tokens."""
    tag_sums: Counter = Counter()
    n_docs = 0
    for c, length in zip(per_doc_counts, per_doc_lengths):
        if length == 0:
            continue
        for tag, n in c.items():
            tag_sums[tag] += (n / length) * 100
        n_docs += 1
    return {tag: val / n_docs for tag, val in tag_sums.items()} if n_docs else {}


def top_n_with_other(proportions: dict[str, float], top_n: int) -> dict[str, float]:
    """Keep top_n tags by mean proportion; merge the rest into 'OTHER'."""
    sorted_tags = sorted(proportions, key=proportions.get, reverse=True)
    top = {t: proportions[t] for t in sorted_tags[:top_n]}
    other_sum = sum(proportions[t] for t in sorted_tags[top_n:])
    if other_sum > 0:
        top["OTHER"] = other_sum
    return top


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def _save_tag_data_csv(data: dict[str, dict[str, float]], path: str) -> None:
    """Save {group: {tag: value}} to CSV with tags as rows, groups as columns."""
    all_tags = sorted({tag for d in data.values() for tag in d})
    rows = [{"tag": tag, **{g: data[g].get(tag, 0.0) for g in data}} for tag in all_tags]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {path}")


def _save_token_lengths_csv(groups_order: list[str], group_results: dict, path: str) -> None:
    """Save per-document token lengths (long form) to CSV."""
    rows = [{"group": g, "token_length": length}
            for g in groups_order for length in group_results[g]["lengths"]]
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {path}")


def _save_sentiment_csv(group_sentiment: dict[str, Counter], groups_order: list[str], path: str) -> None:
    """Save sentiment counts and proportions to CSV."""
    rows = []
    for g in groups_order:
        counts = group_sentiment[g]
        total = sum(counts.values())
        row = {"group": g}
        for lbl in SENTIMENT_LABELS:
            row[f"{lbl}_count"] = counts.get(lbl, 0)
            row[f"{lbl}_proportion"] = counts.get(lbl, 0) / total if total else 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def grouped_bar(ax, data: dict[str, dict[str, float]], ylabel: str, title: str,
                y_fmt: str = "{:.0%}") -> None:
    """
    Draw a grouped bar chart.

    data: {group_name: {tag: value}}
    """
    all_tags = []
    seen = set()
    for group in GROUP_ORDER:
        for tag in data.get(group, {}):
            if tag not in seen:
                all_tags.append(tag)
                seen.add(tag)

    n_tags = len(all_tags)
    n_groups = len(GROUP_ORDER)
    width = 0.7 / n_groups
    x = np.arange(n_tags)

    for i, group in enumerate(GROUP_ORDER):
        vals = [data.get(group, {}).get(tag, 0.0) for tag in all_tags]
        offset = (i - n_groups / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=group, color=GROUP_COLORS[group], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(all_tags, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: y_fmt.format(v)))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_pos(group_results: dict, out_dir: str) -> None:
    """Fig 1 — POS proportion comparison."""
    # Collect global top-N tags across all groups
    global_pos: Counter = Counter()
    for group in GROUP_ORDER:
        props = aggregate_proportions(group_results[group]["pos"])
        for tag, v in props.items():
            global_pos[tag] += v
    top_tags = [t for t, _ in global_pos.most_common(TOP_N_POS)]

    data = {}
    for group in GROUP_ORDER:
        props = aggregate_proportions(group_results[group]["pos"])
        # Keep only top tags, bundle rest as OTHER
        top = {t: props.get(t, 0.0) for t in top_tags}
        other = sum(v for t, v in props.items() if t not in top_tags)
        if other > 0:
            top["OTHER"] = other
        data[group] = top

    _save_tag_data_csv(data, os.path.join(out_dir, "data_pos_distribution.csv"))
    fig, ax = plt.subplots(figsize=(12, 5))
    grouped_bar(ax, data, "Mean proportion per document", "POS tag distribution by group")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_pos_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_dep(group_results: dict, out_dir: str) -> None:
    """Fig 2 — Dependency relation proportion comparison."""
    global_dep: Counter = Counter()
    for group in GROUP_ORDER:
        props = aggregate_proportions(group_results[group]["dep"])
        for tag, v in props.items():
            global_dep[tag] += v
    top_tags = [t for t, _ in global_dep.most_common(TOP_N_DEP)]

    data = {}
    for group in GROUP_ORDER:
        props = aggregate_proportions(group_results[group]["dep"])
        top = {t: props.get(t, 0.0) for t in top_tags}
        other = sum(v for t, v in props.items() if t not in top_tags)
        if other > 0:
            top["OTHER"] = other
        data[group] = top

    _save_tag_data_csv(data, os.path.join(out_dir, "data_dep_distribution.csv"))
    fig, ax = plt.subplots(figsize=(13, 5))
    grouped_bar(ax, data, "Mean proportion per document", "Dependency relation distribution by group")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_dep_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_ner(group_results: dict, out_dir: str) -> None:
    """Fig 3 — NER entity-type rate per 100 content tokens (full sample + filtered)."""
    all_labels: set[str] = set()
    for group in GROUP_ORDER:
        all_labels.update(
            aggregate_rate_per_100(group_results[group]["ent"], group_results[group]["lengths"]).keys()
        )
    sorted_labels = sorted(all_labels)

    # Full-sample rates
    data_full = {}
    for group in GROUP_ORDER:
        rates = aggregate_rate_per_100(group_results[group]["ent"], group_results[group]["lengths"])
        data_full[group] = {label: rates.get(label, 0.0) for label in sorted_labels}

    # Filtered rates: keep only informal pairs where BOTH texts have >= NER_MIN_TOKENS content tokens.
    # Informal Human and Informal LLM share the same index (loaded from the same paired dataframe row).
    inf_h_lengths = group_results["Informal Human"]["lengths"]
    inf_l_lengths = group_results["Informal LLM"]["lengths"]
    keep_idx = [i for i, (lh, ll) in enumerate(zip(inf_h_lengths, inf_l_lengths))
                if lh >= NER_MIN_TOKENS and ll >= NER_MIN_TOKENS]
    n_kept = len(keep_idx)
    n_total_inf = len(inf_h_lengths)
    print(f"  NER filter: {n_kept}/{n_total_inf} informal pairs retained (both >= {NER_MIN_TOKENS} tokens)")

    data_filtered = {}
    for group in GROUP_ORDER:
        if group in INFORMAL_GROUPS:
            ent_f = [group_results[group]["ent"][i] for i in keep_idx]
            lengths_f = [group_results[group]["lengths"][i] for i in keep_idx]
        else:
            ent_f = group_results[group]["ent"]
            lengths_f = group_results[group]["lengths"]
        rates = aggregate_rate_per_100(ent_f, lengths_f)
        data_filtered[group] = {label: rates.get(label, 0.0) for label in sorted_labels}

    _save_tag_data_csv(data_full, os.path.join(out_dir, "data_ner_distribution.csv"))
    _save_tag_data_csv(data_filtered, os.path.join(out_dir, f"data_ner_distribution_min{NER_MIN_TOKENS}.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_bar(ax, data_full, "Mean entities per 100 tokens",
                "Named entity type rate by group (all comments)", y_fmt="{:.2f}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_ner_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    grouped_bar(ax, data_filtered, "Mean entities per 100 tokens",
                f"Named entity type rate by group (informal >= {NER_MIN_TOKENS} tokens, n={n_kept})",
                y_fmt="{:.2f}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, f"fig_ner_distribution_min{NER_MIN_TOKENS}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_token_length(group_results: dict, out_dir: str) -> None:
    """Fig 4 — Token length distribution (violin + strip)."""
    _save_token_lengths_csv(GROUP_ORDER, group_results, os.path.join(out_dir, "data_token_lengths.csv"))
    fig, ax = plt.subplots(figsize=(9, 5))

    positions = range(1, len(GROUP_ORDER) + 1)
    data_lists = [group_results[g]["lengths"] for g in GROUP_ORDER]

    parts = ax.violinplot(data_lists, positions=positions, showmedians=True,
                          showextrema=True, widths=0.6)

    # Colour each violin
    for pc, group in zip(parts["bodies"], GROUP_ORDER):
        pc.set_facecolor(GROUP_COLORS[group])
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)

    # Add strip of raw points (jittered)
    rng = np.random.default_rng(42)
    for i, (lengths, group) in enumerate(zip(data_lists, GROUP_ORDER), start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(lengths))
        ax.scatter(np.full(len(lengths), i) + jitter, lengths,
                   s=5, alpha=0.3, color=GROUP_COLORS[group], zorder=3)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(GROUP_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Content token count (excl. punct/space)")
    ax.set_title("Token length distribution by group")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Median annotation
    for i, lengths in enumerate(data_lists, start=1):
        med = float(np.median(lengths))
        ax.text(i, med + 1, f"med={med:.0f}", ha="center", va="bottom",
                fontsize=7.5, color="black")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_token_length.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_dep_formal_vs_informal(group_results: dict, out_dir: str) -> None:
    """
    Fig 5 — Side-by-side subplots: formal register vs informal register,
    showing human vs LLM dep-relation proportions within each register.
    """
    global_dep: Counter = Counter()
    for group in GROUP_ORDER:
        props = aggregate_proportions(group_results[group]["dep"])
        for tag, v in props.items():
            global_dep[tag] += v
    top_tags = [t for t, _ in global_dep.most_common(TOP_N_DEP)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    pair_groups = [("Formal Human", "Formal LLM"), ("Informal Human", "Informal LLM")]
    subtitles = ["Formal register", "Informal register"]

    for ax, (g1, g2), subtitle in zip(axes, pair_groups, subtitles):
        sub_data = {}
        for g in (g1, g2):
            props = aggregate_proportions(group_results[g]["dep"])
            top = {t: props.get(t, 0.0) for t in top_tags}
            other = sum(v for t, v in props.items() if t not in top_tags)
            if other > 0:
                top["OTHER"] = other
            sub_data[g] = top
        # Use only these two groups on this subplot
        n_tags = len(top_tags) + (1 if any("OTHER" in d for d in sub_data.values()) else 0)
        x = np.arange(n_tags)
        all_tags = top_tags + (["OTHER"] if any("OTHER" in d for d in sub_data.values()) else [])
        width = 0.35
        for j, g in enumerate((g1, g2)):
            vals = [sub_data[g].get(t, 0.0) for t in all_tags]
            offset = (j - 0.5) * width
            ax.bar(x + offset, vals, width, label=g, color=GROUP_COLORS[g], alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(all_tags, rotation=35, ha="right")
        ax.set_title(subtitle)
        ax.set_ylabel("Mean proportion per document")
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for (g1, g2), subtitle in zip(pair_groups, subtitles):
        sub_data = {}
        for g in (g1, g2):
            props = aggregate_proportions(group_results[g]["dep"])
            top = {t: props.get(t, 0.0) for t in top_tags}
            other = sum(v for t, v in props.items() if t not in top_tags)
            if other > 0:
                top["OTHER"] = other
            sub_data[g] = top
        _save_tag_data_csv(sub_data, os.path.join(out_dir, f"data_dep_by_register_{subtitle.split()[0].lower()}.csv"))
    fig.suptitle("Dependency relation distribution: human vs LLM by register", fontsize=12)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_dep_by_register.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_pos_heatmap(group_results: dict, out_dir: str) -> None:
    """Fig 6 — Heatmap of POS proportions (groups × tags)."""
    global_pos: Counter = Counter()
    for group in GROUP_ORDER:
        props = aggregate_proportions(group_results[group]["pos"])
        for tag, v in props.items():
            global_pos[tag] += v
    top_tags = [t for t, _ in global_pos.most_common(TOP_N_POS)]

    matrix = np.zeros((len(GROUP_ORDER), len(top_tags)))
    for i, group in enumerate(GROUP_ORDER):
        props = aggregate_proportions(group_results[group]["pos"])
        for j, tag in enumerate(top_tags):
            matrix[i, j] = props.get(tag, 0.0)

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(matrix, aspect="auto", cmap="Blues")
    ax.set_xticks(range(len(top_tags)))
    ax.set_xticklabels(top_tags, rotation=30, ha="right")
    ax.set_yticks(range(len(GROUP_ORDER)))
    ax.set_yticklabels(GROUP_ORDER)
    plt.colorbar(im, ax=ax, label="Mean proportion")
    # Annotate cells
    for i in range(len(GROUP_ORDER)):
        for j in range(len(top_tags)):
            ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center",
                    fontsize=7.5,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    heatmap_data = {group: {top_tags[j]: matrix[i, j] for j in range(len(top_tags))}
                    for i, group in enumerate(GROUP_ORDER)}
    _save_tag_data_csv(heatmap_data, os.path.join(out_dir, "data_pos_heatmap.csv"))
    ax.set_title("POS proportion heatmap")
    fig.tight_layout()
    path = os.path.join(out_dir, "fig_pos_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Informal-only comparison plots
# ---------------------------------------------------------------------------

INFORMAL_GROUPS = ["Informal Human", "Informal LLM"]
INFORMAL_COLORS = {g: GROUP_COLORS[g] for g in INFORMAL_GROUPS}

FORMAL_GROUPS = ["Formal Human", "Formal LLM"]
FORMAL_COLORS = {g: GROUP_COLORS[g] for g in FORMAL_GROUPS}


def _two_group_bar(ax, data: dict[str, dict[str, float]], ylabel: str, title: str,
                   groups: list[str], colors: dict[str, str],
                   y_fmt: str = "{:.0%}") -> None:
    """Grouped bar chart for exactly two groups."""
    all_tags: list[str] = []
    seen: set[str] = set()
    for g in groups:
        for tag in data.get(g, {}):
            if tag not in seen:
                all_tags.append(tag)
                seen.add(tag)

    x = np.arange(len(all_tags))
    width = 0.35
    for i, g in enumerate(groups):
        vals = [data.get(g, {}).get(tag, 0.0) for tag in all_tags]
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=g, color=colors[g], alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(all_tags, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: y_fmt.format(v)))
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_pair_comparison(group_results: dict, groups: list[str], colors: dict[str, str],
                          prefix: str, label: str, heatmap_cmap: str, out_dir: str) -> None:
    """
    Generic two-group comparison: saves 5 figures to out_dir.
      {prefix}_pos_distribution.png
      {prefix}_dep_distribution.png
      {prefix}_ner_distribution.png
      {prefix}_token_length.png
      {prefix}_pos_heatmap.png
    """
    os.makedirs(out_dir, exist_ok=True)
    g1, g2 = groups[0], groups[1]

    # --- POS distribution ---
    global_pos: Counter = Counter()
    for g in groups:
        for tag, v in aggregate_proportions(group_results[g]["pos"]).items():
            global_pos[tag] += v
    top_pos = [t for t, _ in global_pos.most_common(TOP_N_POS)]

    pos_data = {}
    for g in groups:
        props = aggregate_proportions(group_results[g]["pos"])
        top = {t: props.get(t, 0.0) for t in top_pos}
        other = sum(v for t, v in props.items() if t not in top_pos)
        if other > 0:
            top["OTHER"] = other
        pos_data[g] = top

    _save_tag_data_csv(pos_data, os.path.join(out_dir, f"{prefix}_pos_distribution.csv"))
    fig, ax = plt.subplots(figsize=(11, 5))
    _two_group_bar(ax, pos_data, "Mean proportion per document",
                   f"POS distribution — {g1} vs {g2}",
                   groups=groups, colors=colors)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_pos_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # --- Dependency relation distribution ---
    global_dep: Counter = Counter()
    for g in groups:
        for tag, v in aggregate_proportions(group_results[g]["dep"]).items():
            global_dep[tag] += v
    top_dep = [t for t, _ in global_dep.most_common(TOP_N_DEP)]

    dep_data = {}
    for g in groups:
        props = aggregate_proportions(group_results[g]["dep"])
        top = {t: props.get(t, 0.0) for t in top_dep}
        other = sum(v for t, v in props.items() if t not in top_dep)
        if other > 0:
            top["OTHER"] = other
        dep_data[g] = top

    _save_tag_data_csv(dep_data, os.path.join(out_dir, f"{prefix}_dep_distribution.csv"))
    fig, ax = plt.subplots(figsize=(12, 5))
    _two_group_bar(ax, dep_data, "Mean proportion per document",
                   f"Dependency relation distribution — {g1} vs {g2}",
                   groups=groups, colors=colors)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_dep_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # --- NER entity type rate ---
    all_labels: set[str] = set()
    for g in groups:
        all_labels.update(
            aggregate_rate_per_100(group_results[g]["ent"], group_results[g]["lengths"]).keys()
        )
    sorted_labels = sorted(all_labels)

    ner_data = {}
    for g in groups:
        rates = aggregate_rate_per_100(group_results[g]["ent"], group_results[g]["lengths"])
        ner_data[g] = {lbl: rates.get(lbl, 0.0) for lbl in sorted_labels}

    _save_tag_data_csv(ner_data, os.path.join(out_dir, f"{prefix}_ner_distribution.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    _two_group_bar(ax, ner_data, "Mean entities per 100 tokens",
                   f"NER entity type rate — {g1} vs {g2}",
                   groups=groups, colors=colors, y_fmt="{:.2f}")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_ner_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # Filtered NER (informal only) — keep pairs where BOTH texts have >= NER_MIN_TOKENS tokens
    if set(groups) == set(INFORMAL_GROUPS):
        lengths_a = group_results[groups[0]]["lengths"]
        lengths_b = group_results[groups[1]]["lengths"]
        keep_idx = [i for i, (la, lb) in enumerate(zip(lengths_a, lengths_b))
                    if la >= NER_MIN_TOKENS and lb >= NER_MIN_TOKENS]
        n_kept = len(keep_idx)
        n_total = len(lengths_a)
        print(f"  NER filter ({prefix}): {n_kept}/{n_total} pairs retained (both >= {NER_MIN_TOKENS} tokens)")

        ner_data_f = {}
        for g in groups:
            ent_f = [group_results[g]["ent"][i] for i in keep_idx]
            lengths_f = [group_results[g]["lengths"][i] for i in keep_idx]
            rates = aggregate_rate_per_100(ent_f, lengths_f)
            ner_data_f[g] = {lbl: rates.get(lbl, 0.0) for lbl in sorted_labels}

        _save_tag_data_csv(ner_data_f, os.path.join(out_dir, f"{prefix}_ner_distribution_min{NER_MIN_TOKENS}.csv"))
        fig, ax = plt.subplots(figsize=(7, 5))
        _two_group_bar(ax, ner_data_f, "Mean entities per 100 tokens",
                       f"NER entity type rate — {g1} vs {g2} (>= {NER_MIN_TOKENS} tokens, n={n_kept})",
                       groups=groups, colors=colors, y_fmt="{:.2f}")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.2f}"))
        fig.tight_layout()
        path = os.path.join(out_dir, f"{prefix}_ner_distribution_min{NER_MIN_TOKENS}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"Saved {path}")

    # --- Token length violin ---
    fig, ax = plt.subplots(figsize=(7, 5))
    data_lists = [group_results[g]["lengths"] for g in groups]
    positions = [1, 2]

    parts = ax.violinplot(data_lists, positions=positions, showmedians=True,
                          showextrema=True, widths=0.5)
    for pc, g in zip(parts["bodies"], groups):
        pc.set_facecolor(colors[g])
        pc.set_alpha(0.7)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("black")
        parts[key].set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for i, (lengths, g) in enumerate(zip(data_lists, groups), start=1):
        jitter = rng.uniform(-0.07, 0.07, size=len(lengths))
        ax.scatter(np.full(len(lengths), i) + jitter, lengths,
                   s=6, alpha=0.35, color=colors[g], zorder=3)
        med = float(np.median(lengths))
        ax.text(i, med + 1, f"med={med:.0f}", ha="center", va="bottom",
                fontsize=8, color="black")

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Content token count (excl. punct/space)")
    ax.set_title(f"Token length — {g1} vs {g2}")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_token_length.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    _save_token_lengths_csv(groups, group_results, os.path.join(out_dir, f"{prefix}_token_lengths.csv"))

    # --- POS heatmap (2-row) ---
    matrix = np.zeros((len(groups), len(top_pos)))
    for i, g in enumerate(groups):
        props = aggregate_proportions(group_results[g]["pos"])
        for j, tag in enumerate(top_pos):
            matrix[i, j] = props.get(tag, 0.0)

    fig, ax = plt.subplots(figsize=(11, 3))
    im = ax.imshow(matrix, aspect="auto", cmap=heatmap_cmap)
    ax.set_xticks(range(len(top_pos)))
    ax.set_xticklabels(top_pos, rotation=30, ha="right")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    plt.colorbar(im, ax=ax, label="Mean proportion")
    for i in range(len(groups)):
        for j in range(len(top_pos)):
            ax.text(j, i, f"{matrix[i, j]:.1%}", ha="center", va="center",
                    fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() * 0.6 else "black")
    ax.set_title(f"POS proportion heatmap — {g1} vs {g2}")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{prefix}_pos_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")
    heatmap_data = {groups[i]: {top_pos[j]: matrix[i, j] for j in range(len(top_pos))}
                    for i in range(len(groups))}
    _save_tag_data_csv(heatmap_data, os.path.join(out_dir, f"{prefix}_pos_heatmap.csv"))


def plot_informal_comparison(group_results: dict, out_dir: str) -> None:
    _plot_pair_comparison(group_results, INFORMAL_GROUPS, INFORMAL_COLORS,
                          prefix="inf", label="Informal", heatmap_cmap="Greens", out_dir=out_dir)


def plot_formal_comparison(group_results: dict, out_dir: str) -> None:
    _plot_pair_comparison(group_results, FORMAL_GROUPS, FORMAL_COLORS,
                          prefix="frm", label="Formal", heatmap_cmap="Blues", out_dir=out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip the HuggingFace sentiment model (faster reruns)")
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading texts...")
    groups = load_texts()

    print(f"\nLoading spaCy model '{MODEL}'...")
    try:
        nlp = spacy.load(MODEL)
    except OSError:
        print(f"Model not found. Run: python -m spacy download {MODEL}")
        raise

    print("\nRunning spaCy pipeline...")
    group_results: dict = {}
    for group in GROUP_ORDER:
        print(f"  Processing {group} ({len(groups[group])} texts)...")
        pos, dep, ent, lengths = process_group(nlp, groups[group])
        group_results[group] = {
            "pos": pos,
            "dep": dep,
            "ent": ent,
            "lengths": lengths,
        }

    print("\nGenerating figures...")
    plot_pos(group_results, FIGURES_DIR)
    plot_dep(group_results, FIGURES_DIR)
    plot_ner(group_results, FIGURES_DIR)
    plot_token_length(group_results, FIGURES_DIR)
    plot_dep_formal_vs_informal(group_results, FIGURES_DIR)
    plot_pos_heatmap(group_results, FIGURES_DIR)

    print("\nGenerating informal comparison figures...")
    plot_informal_comparison(group_results, INFORMAL_FIGURES_DIR)

    print("\nGenerating formal comparison figures...")
    plot_formal_comparison(group_results, FORMAL_FIGURES_DIR)

    if args.skip_sentiment:
        print("\nSkipping sentiment analysis (--skip-sentiment).")
    else:
        print(f"\nLoading sentiment model '{SENTIMENT_MODEL}'...")
        sentiment_pipe = load_sentiment_pipeline()

        print("\nRunning sentiment analysis...")
        group_sentiment: dict[str, Counter] = {}
        for group in GROUP_ORDER:
            print(f"  Sentiment: {group} ({len(groups[group])} texts)...")
            group_sentiment[group] = process_sentiment(sentiment_pipe, groups[group])
            counts = group_sentiment[group]
            total = sum(counts.values())
            print(f"    {', '.join(f'{lbl}: {counts.get(lbl,0)/total:.0%}' for lbl in SENTIMENT_LABELS)}")

        print("\nGenerating sentiment figures...")
        plot_sentiment(group_sentiment, FIGURES_DIR)
        plot_sentiment_pair(group_sentiment, INFORMAL_GROUPS, prefix="inf", out_dir=INFORMAL_FIGURES_DIR)
        plot_sentiment_pair(group_sentiment, FORMAL_GROUPS, prefix="frm", out_dir=FORMAL_FIGURES_DIR)

    print(f"\nDone.")
    print(f"  All-group figures  → {FIGURES_DIR}")
    print(f"  Informal comparison → {INFORMAL_FIGURES_DIR}")
    print(f"  Formal comparison   → {FORMAL_FIGURES_DIR}")


if __name__ == "__main__":
    main()
