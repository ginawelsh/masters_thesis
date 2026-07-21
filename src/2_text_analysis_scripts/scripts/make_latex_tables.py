"""
make_latex_tables.py

Emit booktabs LaTeX tables for the paper straight from the analysis output CSVs
(so the numbers can't drift from a hand-transcription). Writes everything to
csv_files/../latex_tables.txt. Requires \\usepackage{booktabs} in the preamble.

Covers: embedding detectability, NER distributions, per-condition linguistic
feature significance, and (if present) informal sentiment.

Run: python src/2_text_analysis_scripts/scripts/make_latex_tables.py
"""
import math
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_2TAS = os.path.dirname(HERE)
CSV = os.path.join(_2TAS, "csv_files")
OUT = os.path.join(_2TAS, "latex_tables.txt")

REPORTED = ["baseline", "human_like", "detector_evasive"]  # detector_aware excluded everywhere
COND_LABEL = {"baseline": "Baseline", "human_like": "Human-like",
              "detector_evasive": "Detector-evasive"}
TOP_N = 12


def tex_feat(f):
    return "\\texttt{" + f.replace("_", "\\_") + "}"


def tex_dir(d):
    if d == "llm > human":
        return "LLM $>$ H"
    if d == "human > llm":
        return "H $>$ LLM"
    return "$\\approx$"


def qfmt(q):
    q = float(q)
    if q <= 0:
        return "$<\\!10^{-300}$"
    if q >= 1e-3:
        return f"${q:.3g}$"
    e = math.floor(math.log10(q))
    m = q / 10 ** e
    return f"${m:.1f}\\times10^{{{e}}}$"


def num(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def read_twosample(dataset, cond):
    p = os.path.join(CSV, f"embedding_twosample_{dataset}_{cond}.csv")
    if not os.path.exists(p):
        return None
    d = dict(zip(*[pd.read_csv(p)[c] for c in ("metric", "value")]))
    return d


# ---------------------------------------------------------------------------
def table_embedding():
    rows = []
    for cond in REPORTED:
        f = read_twosample("formal", cond)
        i = read_twosample("informal", cond)
        if not f or not i:
            continue
        rows.append((COND_LABEL[cond],
                     num(f["classifier_cv_roc_auc"], 3), num(f["mean_pair_cosine"], 3),
                     num(i["classifier_cv_roc_auc"], 3), num(i["mean_pair_cosine"], 3)))
    body = "\n".join(f"    {r[0]:<26} & {r[1]} & {r[2]} & {r[3]} & {r[4]} \\\\" for r in rows)
    return f"""% ---- Embedding detectability ----
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Human--LLM separability by prompt condition: cross-validated ROC AUC
  of a logistic-regression two-sample test on multilingual sentence embeddings
  (\\texttt{{paraphrase-multilingual-mpnet-base-v2}}, 5-fold CV, 1000 label
  permutations), and mean per-pair cosine similarity. AUC${{}}=0.5$ is chance;
  all conditions separate at permutation $p<0.001$.
  $n=170$ (formal) / $1{{,}}149$ (informal) pairs per class.}}
  \\label{{tab:embedding-detectability}}
  \\begin{{tabular}}{{lcccc}}
    \\toprule
    & \\multicolumn{{2}}{{c}}{{Formal}} & \\multicolumn{{2}}{{c}}{{Informal}} \\\\
    \\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}
    Condition & ROC AUC & Mean cos. & ROC AUC & Mean cos. \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""


def table_ner(dataset, label):
    p = os.path.join(CSV, f"ner_distribution_{dataset}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0)
    heads = {"Human": "Human", "LLM baseline": "Baseline", "LLM human-like": "Human-like",
             "LLM detector-evasive": "Det-evasive"}
    cols = [c for c in heads if c in df.columns]
    header = " & ".join(["Entity"] + [heads[c] for c in cols])
    lines = []
    for ent, row in df.iterrows():
        lines.append(f"    {ent:<22} & " + " & ".join(num(row[c], 3) for c in cols) + " \\\\")
    body = "\n".join(lines)
    colspec = "l" + "c" * len(cols)
    return f"""% ---- NER {dataset} ----
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Named-entity rate (mean entities per 100 content tokens),
  {label} register, by source. Swedish SUC tagset (\\texttt{{sv\\_core\\_news\\_lg}}).}}
  \\label{{tab:ner-{dataset}}}
  \\begin{{tabular}}{{{colspec}}}
    \\toprule
    {header} \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""


def table_significance(sig, dataset, cond):
    sub = sig[(sig.dataset == dataset) & (sig.condition == cond) & (sig["significant_fdr_0.05"])].copy()
    if sub.empty:
        return None
    sub["absdz"] = sub["cohens_dz"].abs()
    sub = sub.sort_values("absdz", ascending=False).head(TOP_N)
    lines = []
    for _, r in sub.iterrows():
        dz = f"{r['cohens_dz']:+.2f}"
        lines.append(f"    {tex_feat(r['feature']):<40} & {num(r['median_human'])} & "
                     f"{num(r['median_llm'])} & {dz} & {tex_dir(r['direction'])} & {qfmt(r['p_fdr_bh'])} \\\\")
    body = "\n".join(lines)
    ntot = len(sig[(sig.dataset == dataset) & (sig.condition == cond)])
    nsig = len(sig[(sig.dataset == dataset) & (sig.condition == cond) & (sig["significant_fdr_0.05"])])
    return f"""% ---- Significance: {dataset} / {cond} ----
\\begin{{table}}[htbp]
  \\centering
  \\caption{{{dataset.capitalize()} register, {COND_LABEL[cond].replace('$^{\\dagger}$','')} condition:
  top {min(TOP_N,len(sub))} features by $|d_z|$ (paired Wilcoxon signed-rank).
  $d_z>0$ means LLM $>$ human. {nsig} of {ntot} features significant at BH-FDR $q<0.05$.
  $n$={'170' if dataset=='formal' else '1{,}149'} pairs.}}
  \\label{{tab:sig-{dataset}-{cond}}}
  \\begin{{tabular}}{{lccccc}}
    \\toprule
    Feature & Med.\\,H & Med.\\,LLM & $d_z$ & Dir. & $q$ \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""


def table_sentiment():
    p = os.path.join(CSV, "inf_sentiment_distribution_by_condition.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df = df[~df["group"].str.contains("detector_aware")]  # not in reported set
    lines = []
    for _, r in df.iterrows():
        lines.append(f"    {r['group']:<20} & {int(r['n'])} & {r['POSITIVE_proportion']*100:.1f} & "
                     f"{r['NEUTRAL_proportion']*100:.1f} & {r['NEGATIVE_proportion']*100:.1f} & "
                     f"{r['median_signed_polarity']:+.3f} & {r['subjective_rate']*100:.1f} \\\\")
    body = "\n".join(lines)
    return f"""% ---- Informal sentiment ----
\\begin{{table}}[htbp]
  \\centering
  \\caption{{Sentiment-label mix (\\%), median signed polarity ($p_{{pos}}-p_{{neg}}$)
  and subjectivity rate for the informal register, by source
  (\\texttt{{KBLab/robust-swedish-sentiment-multiclass}}).}}
  \\label{{tab:sentiment-informal}}
  \\begin{{tabular}}{{lrrrrrr}}
    \\toprule
    Source & $n$ & POS & NEU & NEG & Med.\\,pol. & Subj.\\,\\% \\\\
    \\midrule
{body}
    \\bottomrule
  \\end{{tabular}}
\\end{{table}}"""


def main():
    parts = ["% =====================================================================",
             "% LaTeX tables for the CLUU thesis results. Requires \\usepackage{booktabs}.",
             "% Generated by make_latex_tables.py from the analysis output CSVs.",
             "% =====================================================================", ""]

    parts.append(table_embedding())
    for ds, lab in (("formal", "formal"), ("informal", "informal")):
        t = table_ner(ds, lab)
        if t:
            parts.append("\n" + t)

    sig = pd.read_csv(os.path.join(CSV, "significance_tests_results.csv"))
    for ds in ("formal", "informal"):
        for cond in REPORTED:
            t = table_significance(sig, ds, cond)
            if t:
                parts.append("\n" + t)

    st = table_sentiment()
    if st:
        parts.append("\n" + st)
    else:
        parts.append("\n% [sentiment table pending — inf_sentiment_distribution_by_condition.csv "
                     "not yet written; re-run this script once affective_analysis.py finishes]")

    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    print(f"wrote {OUT}")
    print(f"  tables: embedding, NER (formal+informal), significance "
          f"({len(REPORTED)} conds x 2 datasets), sentiment={'yes' if st else 'PENDING'}")


if __name__ == "__main__":
    main()
