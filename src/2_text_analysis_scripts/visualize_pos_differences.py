"""
Visualize POS proportion differences (Human - Generated).

Input:
  src/1_data_collection/human_formal/linguistic_analysis_pos_differences.csv

Outputs (PNG):
  src/1_data_collection/human_formal/plots/pos_diff_average_bar.png
  src/1_data_collection/human_formal/plots/pos_diff_heatmap.png

Usage:
  python src/2_text_analysis_scripts/visualize_pos_differences.py
"""

import os

import pandas as pd


def get_paths():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(root, "1_data_collection", "human_formal")
    csv_path = os.path.join(data_dir, "linguistic_analysis_pos_differences.csv")
    plot_dir = os.path.join(data_dir, "plots")
    return csv_path, plot_dir


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


def main():
    csv_path, plot_dir = get_paths()
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

    try:
        plot_average_bar(avg_series, avg_out)
        plot_heatmap(matrix, heatmap_out)
    except ImportError as exc:
        raise ImportError(
            "This script requires matplotlib for plotting. "
            "Install with: pip install matplotlib"
        ) from exc

    print(f"Wrote: {avg_out}")
    print(f"Wrote: {heatmap_out}")


if __name__ == "__main__":
    main()
