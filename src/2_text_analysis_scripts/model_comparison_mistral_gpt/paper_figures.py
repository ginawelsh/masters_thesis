"""EACL paper figures from model_comparison_results.csv (GPT-5.2 vs Mistral vs human).

Fig 1  distance-from-human (median |Cohen's dz|) x condition x register x model
       -> RQ1 (register), RQ2 (adversarial evasion: baseline vs the 3 conditions), RQ3 (model)
Fig 2  mean |dz| per feature-family x condition, 2x2 (register x model)
       -> where the divergence lives and whether adversarial prompting shrinks it

Matches the repo figure style (plot_condition_comparison.py). Writes PNG (dpi 200) + PDF
(vector, for LaTeX). Run:
  python src/2_text_analysis_scripts/model_comparison_mistral_gpt/paper_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "model_comparison_results.csv")

INK, MUTED, GRID = "#1b1f27", "#667085", "#e3e7ec"
MODEL_COLOR = {"GPT-5.2": "#0072B2", "Mistral": "#D55E00"}   # validated CVD-safe pair
COMP_MODEL = {"human_vs_gpt": "GPT-5.2", "human_vs_mistral": "Mistral"}
COND_ORDER = ["baseline", "human_like", "detector_aware", "detector_evasive"]
COND_LABEL = {"baseline": "Baseline", "human_like": "Human-like",
              "detector_aware": "Detector-\naware", "detector_evasive": "Detector-\nevasive"}
REG_LABEL = {"formal": "Formal (abstracts)", "informal": "Informal (comments)"}
FAM_ORDER = ["stylometric_surface", "syntactic_complexity", "pragmatic_markers",
             "ner", "textstat", "pos_proportions"]
FAM_LABEL = {"stylometric_surface": "Stylometric", "syntactic_complexity": "Syntactic",
             "pragmatic_markers": "Pragmatic/hedge", "ner": "NER",
             "textstat": "Readability", "pos_proportions": "POS"}


PRETTY = {
    "n_alpha_tokens": "Length (alpha tokens)", "word_count": "Word count",
    "char_count": "Char count", "rough_syllable_count_sv": "Syllable count",
    "n_sentences": "Sentence count", "textstat_flesch_reading_ease": "Flesch reading-ease",
    "textstat_gunning_fog": "Gunning fog", "textstat_smog_index": "SMOG index",
    "textstat_flesch_kincaid_grade": "Flesch–Kincaid", "function_word_rate": "Function-word rate",
    "punct_period": "Periods /100tok", "punct_comma": "Commas /100tok",
    "mtld": "MTLD", "mattr": "MATTR", "ttr": "Type–token ratio",
    "bigram_repetition_rate": "Bigram repetition", "trigram_repetition_rate": "Trigram repetition",
    "dep_distance_mean": "Dependency distance", "tree_depth_mean": "Parse-tree depth",
    "subordinate_clause_rate": "Subordinate clauses", "hedge_rate": "Hedges /100tok",
    "epistemic_rate": "Epistemic /100tok",
}
REG_SHORT = {"formal": "Form", "informal": "Inf"}
COND_SHORT = {"baseline": "base", "human_like": "HL", "detector_aware": "DA", "detector_evasive": "DE"}


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _save(fig, stem):
    for ext in ("png", "pdf"):
        out = os.path.join(HERE, f"{stem}.{ext}")
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
        print("  wrote", out)
    plt.close(fig)


def fig1_distance(df):
    hv = df[df.comparison.isin(COMP_MODEL)].copy()
    hv["absdz"] = hv.cohens_dz.abs()
    hv["model"] = hv.comparison.map(COMP_MODEL)
    # median |dz| across features, per (register, model, condition)
    med = hv.groupby(["register", "model", "condition"])["absdz"].median().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    x = np.arange(len(COND_ORDER)); w = 0.38
    for ax, reg in zip(axes, ["formal", "informal"]):
        _style(ax)
        for k, model in enumerate(["GPT-5.2", "Mistral"]):
            vals = [med[(med.register == reg) & (med.model == model) & (med.condition == c)]["absdz"].mean()
                    for c in COND_ORDER]
            bars = ax.bar(x + (k - 0.5) * w, vals, width=w, color=MODEL_COLOR[model],
                          label=model, zorder=3)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.2, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels([COND_LABEL[c] for c in COND_ORDER], fontsize=8, color=INK)
        ax.set_title(REG_LABEL[reg], fontsize=10, color=INK, fontweight="600")
    axes[0].set_ylabel("Median |Cohen's dz| vs human\n(distance from human; lower = more human-like)",
                       fontsize=8.5, color=MUTED)
    handles = [Patch(facecolor=MODEL_COLOR[m], label=m) for m in ["GPT-5.2", "Mistral"]]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _save(fig, "fig1_distance_from_human")


def fig2_family(df):
    hv = df[df.comparison.isin(COMP_MODEL)].copy()
    hv["absdz"] = hv.cohens_dz.abs()
    hv["model"] = hv.comparison.map(COMP_MODEL)
    agg = hv.groupby(["register", "model", "feature_group", "condition"])["absdz"].mean().reset_index()
    vmax = float(agg["absdz"].max())

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.9), constrained_layout=True)
    im = None
    for r, reg in enumerate(["formal", "informal"]):
        for c, model in enumerate(["GPT-5.2", "Mistral"]):
            ax = axes[r][c]
            M = np.full((len(FAM_ORDER), len(COND_ORDER)), np.nan)
            for i, fam in enumerate(FAM_ORDER):
                for j, cond in enumerate(COND_ORDER):
                    sub = agg[(agg.register == reg) & (agg.model == model) &
                              (agg.feature_group == fam) & (agg.condition == cond)]
                    if len(sub):
                        M[i, j] = sub["absdz"].iloc[0]
            im = ax.imshow(M, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
            for i in range(len(FAM_ORDER)):
                for j in range(len(COND_ORDER)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if M[i, j] > 0.55 * vmax else INK)
            ax.set_title(f"{REG_LABEL[reg].split(' ')[0]} · {model}", fontsize=9.5,
                         color=INK, fontweight="600")
            ax.set_xticks(range(len(COND_ORDER)))
            ax.set_xticklabels(
                [COND_LABEL[c].replace("\n", " ") for c in COND_ORDER] if r == 1 else [],
                fontsize=7.2, color=INK, rotation=25, ha="right")
            ax.set_yticks(range(len(FAM_ORDER)))
            ax.set_yticklabels([FAM_LABEL[f] for f in FAM_ORDER] if c == 0 else [],
                               fontsize=8, color=INK)
            ax.tick_params(length=0)
            for sp in ax.spines.values():
                sp.set_visible(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.03)
    cbar.set_label("Mean |Cohen's dz| vs human", fontsize=8, color=MUTED)
    cbar.ax.tick_params(labelsize=7, colors=MUTED)
    fig.suptitle("Feature-family divergence from human by condition", fontsize=12,
                 fontweight="700", color=INK)
    _save(fig, "fig2_family_effectsizes")


def fig3_model_divergence(df, top_n=16):
    # mistral_vs_gpt: dz>0 => GPT > Mistral, dz<0 => Mistral > GPT (test_feature(mistral, gpt))
    mg = df[df.comparison == "mistral_vs_gpt"].copy()
    mg["absdz"] = mg.cohens_dz.abs()
    mg = mg.sort_values("absdz").tail(top_n)  # largest at top after barh
    labels = [f"{PRETTY.get(r.feature, r.feature)}  ({REG_SHORT[r.register]}·{COND_SHORT[r.condition]})"
              for _, r in mg.iterrows()]
    vals = mg.cohens_dz.values
    colors = [MODEL_COLOR["GPT-5.2"] if v > 0 else MODEL_COLOR["Mistral"] for v in vals]

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * top_n + 1.0))
    ax.set_facecolor("white")
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)

    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, height=0.72, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1.0)
    for yi, v in zip(y, vals):
        ax.text(v + (0.05 if v > 0 else -0.05), yi, f"{v:+.2f}",
                va="center", ha="left" if v > 0 else "right", fontsize=7.2, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    m = float(np.abs(vals).max())
    ax.set_xlim(-m * 1.25, m * 1.25)
    ax.set_xlabel("Cohen's dz   (← Mistral > GPT     |     GPT > Mistral →)",
                  fontsize=8.5, color=MUTED)
    handles = [Patch(facecolor=MODEL_COLOR["Mistral"], label="Mistral higher"),
               Patch(facecolor=MODEL_COLOR["GPT-5.2"], label="GPT-5.2 higher")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    ax.set_title(f"Where GPT-5.2 and Mistral differ most (top {top_n} features)",
                 fontsize=11, fontweight="700", color=INK, pad=10)
    fig.tight_layout()
    _save(fig, "fig3_model_divergence")


def fig4_sentiment():
    p = os.path.join(HERE, "affective_results.csv")
    if not os.path.exists(p):
        print("  (skip fig4: affective_results.csv not found)")
        return
    a = pd.read_csv(p)
    sp = a[a.feature == "signed_polarity"]
    def series(comp):
        m = {r.condition: r.median_B for _, r in sp[sp.comparison == comp].iterrows()}
        return [m.get(c, np.nan) for c in COND_ORDER]
    gpt, mis = series("human_vs_gpt"), series("human_vs_mistral")
    human = float(sp[sp.comparison == "human_vs_gpt"]["median_A"].median())

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _style(ax)
    x = list(range(len(COND_ORDER)))
    ax.axhline(0, color=GRID, lw=1, zorder=1)
    ax.axhline(human, color="#0F9E84", ls=(0, (5, 3)), lw=1.7, zorder=2, label="Human (reference)")
    ax.text(len(x) - 1, human, " human", va="bottom", ha="right", fontsize=8, color="#0F9E84")
    for vals, m in [(gpt, "GPT-5.2"), (mis, "Mistral")]:
        ax.plot(x, vals, "-o", color=MODEL_COLOR[m], lw=2, ms=7, label=m, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.008, f"{v:+.2f}", ha="center", va="bottom", fontsize=7.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([COND_LABEL[c].replace("\n", " ") for c in COND_ORDER], fontsize=8.5, color=INK)
    ax.set_ylabel("Median signed sentiment polarity\n(+ positive  ·  − negative)", fontsize=8.5, color=MUTED)
    ax.set_title("Sentiment polarity vs human by condition (informal)", fontsize=11,
                 fontweight="700", color=INK, pad=8)
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    _save(fig, "fig4_sentiment_polarity")


if __name__ == "__main__":
    df = pd.read_csv(RESULTS, encoding="utf-8")
    print("loaded", len(df), "rows")
    fig1_distance(df)
    fig2_family(df)
    fig3_model_divergence(df)
    fig4_sentiment()
