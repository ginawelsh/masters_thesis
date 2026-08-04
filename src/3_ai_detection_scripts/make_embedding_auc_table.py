"""make_embedding_auc_table.py

Emit Table 1 of detection_results_tables.tex -- embedding CV ROC AUC by
generator x register x condition -- straight from the embedding_twosample_*.csv
files, so the cells cannot drift from a hand-transcription.

Source files (written by 2_text_analysis_scripts/scripts/embedding_comparison.py):
  GPT-5.2      embedding_twosample_<register>_<condition>.csv
  GPT-5.6      embedding_twosample_<register>_<condition>_gpt56.csv
  Mistral 3.2  embedding_twosample_<register>_<condition>_mistral_t1.0.csv

Cells are colour-shaded by AUC. Requires \\usepackage[table]{xcolor} in the preamble.

  --palette redgreen  (default) red = lower AUC, green = higher, as requested.
  --palette blue      single-hue sequential fallback; use this one if the document
                      may be printed in greyscale or read by a red-green
                      colour-deficient reader, since it encodes magnitude by
                      lightness alone.

Both ramps hold lightness strictly monotone in AUC, so the shading still reads when
hue is lost; the printed number is always the primary channel.

Writes embedding_auc_table.tex next to this script and prints the same block.

Run: python src/3_ai_detection_scripts/make_embedding_auc_table.py
     python src/3_ai_detection_scripts/make_embedding_auc_table.py --palette blue
"""
import argparse
import math
import os
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(HERE)
CSV = os.path.join(SRC, "2_text_analysis_scripts", "csv_files")
OUT = os.path.join(HERE, "embedding_auc_table.tex")

# detector_aware is excluded everywhere in the reported results, and the GPT-5.6
# quiz corpus was never generated under it.
CONDITIONS = [("baseline", "Baseline"),
              ("human_like", "Human-mimicking"),
              ("detector_evasive", "Detector-evasive")]
REGISTERS = [("formal", "Formal"), ("informal", "Informal")]
GENERATORS = [("GPT-5.2", ""), ("GPT-5.6", "_gpt56"), ("Mistral 3.2", "_mistral_t1.0")]


def read_cell(register, cond, suffix):
    p = os.path.join(CSV, f"embedding_twosample_{register}_{cond}{suffix}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return dict(zip(df["metric"], df["value"]))


def r3(x):
    """3 dp, half-up (matches make_latex_tables.py; .7895 -> .790, not .789)."""
    return str(Decimal(str(float(x))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# Cell shading.
#
# Both ramps run light -> dark as AUC rises, in OKLCH, with the endpoints chosen so
# that (a) nothing clips the sRGB gamut -- clipping silently flattens lightness and
# breaks the monotonicity the ramp depends on -- and (b) black body text clears
# 4.5:1 on the darkest cell (measured 4.89:1), so no cell needs white text.
# Sampled at 5 steps the redgreen ramp gives adjacent OKLab dL = 0.080 (>= 0.06).
#
# Known, deliberate deviations from the house data-viz rules, both inherent to the
# request rather than defects: the redgreen ramp spans two hues (124 deg) where the
# rule wants one, and its lightest step sits at 1.44:1 against white, under the 2:1
# floor that applies to discrete ordinal marks. A continuous heatmap's lightest step
# is allowed to recede toward the surface, and the value is printed in every cell,
# so colour is never the only channel. --palette blue clears the single-hue rule.
RAMPS = {                 # (L_lo, L_hi, C_lo, C_hi, H_lo, H_hi)
    "redgreen": (0.885, 0.565, 0.060, 0.140, 28.0, 152.0),
    "blue":     (0.885, 0.565, 0.045, 0.130, 250.0, 250.0),
}


def shade(t, ramp):
    """t in [0,1] (0 = lowest AUC) -> uppercase RRGGBB."""
    L_lo, L_hi, C_lo, C_hi, H_lo, H_hi = RAMPS[ramp]
    L = L_lo + (L_hi - L_lo) * t
    C = C_lo + (C_hi - C_lo) * t
    H = H_lo + (H_hi - H_lo) * t
    a, b = C * math.cos(math.radians(H)), C * math.sin(math.radians(H))
    l_, m_, s_ = (L + 0.3963377774 * a + 0.2158037573 * b,
                  L - 0.1055613458 * a - 0.0638541728 * b,
                  L - 0.0894841775 * a - 1.2914855480 * b)
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    rgb = (4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
           -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
           -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s)

    def enc(x):
        x = max(0.0, min(1.0, x))
        x = 1.055 * x ** (1 / 2.4) - 0.055 if x > 0.0031308 else 12.92 * x
        return round(max(0.0, min(1.0, x)) * 255)

    return "%02X%02X%02X" % tuple(enc(v) for v in rgb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--palette", choices=sorted(RAMPS), default="redgreen")
    args = ap.parse_args()

    # pass 1: collect every AUC so the ramp can be scaled to the observed range
    cells = {}
    for gen, suffix in GENERATORS:
        for register, _ in REGISTERS:
            for cond, _ in CONDITIONS:
                d = read_cell(register, cond, suffix)
                if d is not None:
                    cells[(gen, register, cond)] = d
    # Shade from the ROUNDED value that the cell actually prints, not the raw float:
    # two cells both showing 0.965 (0.9652 / 0.9647) must not carry different shades.
    aucs = [float(r3(d["classifier_cv_roc_auc"])) for d in cells.values()]
    lo, hi = min(aucs), max(aucs)
    span = (hi - lo) or 1.0

    # pass 2: emit
    blocks, ns = [], {}
    for gen, suffix in GENERATORS:
        reg_blocks = []
        for register, reg_label in REGISTERS:
            lines = []
            for ci, (cond, cond_label) in enumerate(CONDITIONS):
                d = cells.get((gen, register, cond))
                if d is None:
                    val = "\\textit{pending}"
                else:
                    txt = r3(d["classifier_cv_roc_auc"])
                    val = f"\\cellcolor[HTML]{{{shade((float(txt) - lo) / span, args.palette)}}} {txt}"
                    ns.setdefault((gen, reg_label), set()).add(int(d["n_per_class"]))
                lead = f"      & \\multirow{{3}}{{*}}{{{reg_label}}}\n        " if ci == 0 else "      & "
                lines.append(f"{lead}& {cond_label:<16} & {val} \\\\")
            reg_blocks.append("\n".join(lines))
        block = f"    \\multirow{{6}}{{*}}{{{gen}}}\n" + "\n    \\cmidrule(l){2-4}\n".join(reg_blocks)
        blocks.append(block)

    body = "\n    \\midrule\n".join(blocks)
    legend = " ".join(
        f"\\colorbox[HTML]{{{shade(i / 4, args.palette)}}}{{\\strut~{lo + span * i / 4:.2f}~}}"
        for i in range(5))
    table = (
        "\\begin{table*}[t]\n  \\centering\n  \\small\n"
        "  \\caption{CAPTION}\n  \\label{tab:embedding-auc}\n"
        "  \\begin{tabular}{lllc}\n    \\toprule\n"
        "    Generator & Register & Condition & Embedding CV ROC AUC \\\\\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n  \\end{{tabular}}\n\n"
        f"  \\vspace{{2pt}}\n  {{\\footnotesize Shading tracks the printed AUC on a single scale"
        f" across all cells, darkening monotonically so the ordering survives greyscale"
        f" printing; the unequal-$n$ caveat above applies to the colours as much as to the"
        f" numbers. {legend}}}\n"
        "\\end{table*}\n"
    )
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(table)
    print(table)
    print(f"% palette={args.palette}  range {lo:.3f}-{hi:.3f} mapped light->dark")
    print("% n per class actually used:")
    for k in sorted(ns):
        print(f"%   {k[0]:12s} {k[1]:9s} n={sorted(ns[k])}")
    print(f"\nwrote {os.path.relpath(OUT, SRC)}")


if __name__ == "__main__":
    main()
