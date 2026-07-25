"""Compare LLM detectors (Claude / Gemini / DeepSeek) on the shared quiz corpus.

Reads the three detection_results_<provider>.csv files produced by
ai_detection_llm.py and renders a two-panel accuracy figure:
  left  — overall + by register (formal vs informal)
  right — by prompt condition (human control, baseline, human_like, detector_evasive)

Saves detection_comparison.png (300 dpi) and .pdf for direct use in the thesis.
"""
import os
import glob

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

_dir = os.path.dirname(os.path.abspath(__file__))

# First three validated categorical slots (blue / orange / aqua) — clear the
# all-pairs CVD floors, so safe for three series.
COLORS = {"claude": "#2a78d6", "gemini": "#eb6834", "deepseek": "#1baf7a"}
LABELS = {"claude": "Claude opus-4-8", "gemini": "Gemini 3.1 pro", "deepseek": "DeepSeek reasoner"}
# ordered best -> worst so the legend / bar order tells the story
PROVIDERS = ["claude", "gemini", "deepseek"]

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def load():
    data = {}
    for f in glob.glob(os.path.join(_dir, "detection_results_*.csv")):
        name = os.path.basename(f).replace("detection_results_", "").replace(".csv", "")
        d = pd.read_csv(f, encoding="utf-8")
        d["is_human"] = d["is_human"].astype(str).str.lower().isin(("true", "1", "yes"))
        data[name] = d
    return data


def acc(d, **filt):
    for k, v in filt.items():
        d = d[d[k] == v]
    return d["correct"].mean() if len(d) else float("nan")


def grouped_bars(ax, categories, series_vals, title):
    """series_vals: {provider: [val per category]}."""
    n = len(PROVIDERS)
    width = 0.8 / n
    x = range(len(categories))
    for i, prov in enumerate(PROVIDERS):
        offs = [xi - 0.4 + width / 2 + i * width for xi in x]
        bars = ax.bar(offs, series_vals[prov], width * 0.92, color=COLORS[prov],
                      label=LABELS[prov], zorder=3)
        for b, v in zip(bars, series_vals[prov]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.0%}",
                    ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(categories, fontsize=9, color=INK)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axhline(0.5, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_axisbelow(True)


def main():
    data = load()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE,
                                   gridspec_kw={"width_ratios": [3, 4]})
    for ax in (axL, axR):
        ax.set_facecolor(SURFACE)

    # left: overall + by register
    cats_L = ["Overall", "Formal\n(abstracts)", "Informal\n(comments)"]
    vals_L = {p: [acc(data[p]), acc(data[p], register="formal"),
                  acc(data[p], register="informal")] for p in PROVIDERS}
    grouped_bars(axL, cats_L, vals_L, "Accuracy by register")
    axL.set_ylabel("Accuracy", fontsize=9, color=INK)

    # right: by prompt condition (control first, then increasing evasion)
    conds = ["human", "baseline", "human_like", "detector_evasive"]
    cats_R = ["Human\n(control)", "Baseline", "Human-like", "Detector-\nevasive"]
    vals_R = {p: [acc(data[p], condition=c) for c in conds] for p in PROVIDERS}
    grouped_bars(axR, cats_R, vals_R, "Accuracy by AI-generation prompt condition")

    axR.legend(loc="upper right", frameon=False, fontsize=8.5)
    # annotate the chance line once
    axL.text(len(cats_L) - 0.5, 0.5, "chance", fontsize=7, color=MUTED,
             va="bottom", ha="right")

    fig.suptitle("LLM detectors on Swedish human-vs-AI quiz (n=56)",
                 fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("png", "pdf"):
        out = os.path.join(_dir, f"detection_comparison.{ext}")
        fig.savefig(out, dpi=300, facecolor=SURFACE, bbox_inches="tight")
        print("saved ->", out)


if __name__ == "__main__":
    main()
