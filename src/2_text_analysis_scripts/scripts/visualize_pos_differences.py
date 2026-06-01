"""
Visualize POS proportion differences (Human - Generated).

Input:
  src/1_data_collection/human_formal/linguistic_analysis_pos_differences.csv

Outputs (PNG):
  src/1_data_collection/human_formal/plots/pos_diff_average_bar.png
  src/1_data_collection/human_formal/plots/pos_diff_heatmap.png
  src/1_data_collection/human_formal/plots/pos_proportions_side_by_side_bar.png

Usage:
  python src/2_text_analysis_scripts/visualize_pos_differences.py
"""

import os
import argparse

import pandas as pd

POS_SEP = " | "  # matches linguistic_analysis_pos_proportions.csv cells


def get_paths(dataset: str):
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(root, "1_data_collection", f"human_{dataset}")
    csv_path = os.path.join(data_dir, "linguistic_analysis_pos_differences.csv")
    proportions_csv = os.path.join(data_dir, "linguistic_analysis_pos_proportions.csv")
    plot_dir = os.path.join(data_dir, "plots")
    return csv_path, proportions_csv, plot_dir


def get_pos_columns(df: pd.DataFrame):
    return [c for c in df.columns if c.startswith("POS_Diff_")]


def build_average_series(df: pd.DataFrame, pos_cols):
    avg_row = df[df["Row"].astype(str) == "AVERAGE"]
    if not avg_row.empty:
        avg = avg_row.iloc[0][pos_cols].astype(float)
    else:
        non_avg = df[df["Row"].astype(str) != "AVERAGE"]
        avg = non_avg[pos_cols].astype(float).mean(axis=0)
    avg.index = [c.replace("POS_Diff_", "") for c in avg.index]
    return avg.sort_values(ascending=False)


def build_heatmap_matrix(df: pd.DataFrame, pos_cols):
    non_avg = df[df["Row"].astype(str) != "AVERAGE"].copy()
    non_avg["Row"] = pd.to_numeric(non_avg["Row"], errors="coerce")
    non_avg = non_avg.dropna(subset=["Row"]).sort_values("Row")
    matrix = non_avg[pos_cols].astype(float)
    matrix.columns = [c.replace("POS_Diff_", "") for c in matrix.columns]
    matrix.index = non_avg["Row"].astype(int)
    return matrix


def plot_average_bar(avg_series, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in avg_series.values]
    avg_series.plot(kind="bar", ax=ax, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Average POS Difference (Human - Generated)")
    ax.set_ylabel("Average Decimal Difference")
    ax.set_xlabel("POS Tag")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_heatmap(matrix, out_path):
    import matplotlib.pyplot as plt

    vmax = float(matrix.abs().to_numpy().max()) if not matrix.empty else 0.1
    vmax = max(vmax, 0.1)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    ax.set_title("POS Difference Heatmap by Row (Human - Generated)")
    ax.set_xlabel("POS Tag")
    ax.set_ylabel("Row")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")

    # Keep row tick labels readable for larger datasets.
    step = max(1, len(matrix.index) // 20)
    y_ticks = list(range(0, len(matrix.index), step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(matrix.index[i]) for i in y_ticks])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Decimal Difference")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_proportions_cell(s) -> dict:
    """Parse 'TAG:0.xxxx | TAG:0.xxxx | ...' into {TAG: float}."""
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
        tag, p_str = tag.strip(), p_str.strip()
        if not tag or not p_str:
            continue
        try:
            props[tag] = float(p_str)
        except ValueError:
            continue
    return props


def build_average_proportions(df_props: pd.DataFrame):
    """Mean proportion per POS across rows for Human (Abstract) and Generated."""
    human_sums: dict[str, float] = {}
    gen_sums: dict[str, float] = {}
    n_rows = 0
    for _, row in df_props.iterrows():
        h = parse_proportions_cell(row.get("Abstract_POS_Proportions", ""))
        g = parse_proportions_cell(row.get("Generated_Abstract_POS_Proportions", ""))
        if not h and not g:
            continue
        n_rows += 1
        tags = set(h.keys()) | set(g.keys())
        for tag in tags:
            human_sums[tag] = human_sums.get(tag, 0.0) + h.get(tag, 0.0)
            gen_sums[tag] = gen_sums.get(tag, 0.0) + g.get(tag, 0.0)
    if n_rows == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    human_avg = pd.Series({t: human_sums[t] / n_rows for t in sorted(human_sums.keys())})
    gen_avg = pd.Series({t: gen_sums[t] / n_rows for t in sorted(gen_sums.keys())})
    all_tags = sorted(set(human_avg.index) | set(gen_avg.index))
    human_avg = human_avg.reindex(all_tags, fill_value=0.0)
    gen_avg = gen_avg.reindex(all_tags, fill_value=0.0)
    return human_avg, gen_avg


def plot_proportions_side_by_side(human_avg: pd.Series, gen_avg: pd.Series, out_path):
    """Grouped bar chart: average proportion as % for Human vs AI-generated per POS."""
    import matplotlib.pyplot as plt
    import numpy as np

    tags = human_avg.index.tolist()
    x = np.arange(len(tags))
    width = 0.36
    human_pct = human_avg.values * 100.0
    gen_pct = gen_avg.values * 100.0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, human_pct, width, label="Human", color="#2ca02c")
    ax.bar(x + width / 2, gen_pct, width, label="AI generated", color="#ff7f0e")
    ax.set_ylabel("Average proportion (%)")
    ax.set_xlabel("POS tag")
    ax.set_title("Average POS proportion: Human vs AI-generated abstracts")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, max(human_pct.max(), gen_pct.max(), 1.0) * 1.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize POS differences/proportions")
    parser.add_argument("--dataset", choices=["formal", "informal"], default="formal")
    args = parser.parse_args()

    csv_path, proportions_csv, plot_dir = get_paths(args.dataset)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    os.makedirs(plot_dir, exist_ok=True)
    df = pd.read_csv(csv_path, encoding="utf-8")
    pos_cols = get_pos_columns(df)
    if not pos_cols:
        raise ValueError("No POS_Diff_* columns found in the input CSV.")

    avg_series = build_average_series(df, pos_cols)
    matrix = build_heatmap_matrix(df, pos_cols)

    avg_out = os.path.join(plot_dir, "pos_diff_average_bar.png")
    heatmap_out = os.path.join(plot_dir, "pos_diff_heatmap.png")
    side_by_side_out = os.path.join(plot_dir, "pos_proportions_side_by_side_bar.png")

    try:
        plot_average_bar(avg_series, avg_out)
        plot_heatmap(matrix, heatmap_out)
        if os.path.exists(proportions_csv):
            df_props = pd.read_csv(proportions_csv, encoding="utf-8")
            h_avg, g_avg = build_average_proportions(df_props)
            if len(h_avg) > 0:
                plot_proportions_side_by_side(h_avg, g_avg, side_by_side_out)
                print(f"Wrote: {side_by_side_out}")
            else:
                print(f"Skipped proportions chart: no data in {proportions_csv}")
        else:
            print(f"Skipped proportions chart: missing {proportions_csv}")
    except ImportError as exc:
        raise ImportError(
            "This script requires matplotlib for plotting. "
            "Install with: pip install matplotlib"
        ) from exc

    print(f"Wrote: {avg_out}")
    print(f"Wrote: {heatmap_out}")


if __name__ == "__main__":
    main()
