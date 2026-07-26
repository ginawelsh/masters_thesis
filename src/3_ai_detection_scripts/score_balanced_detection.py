"""Score the balanced, matched LLM-judge detection run (quiz_master_balanced.csv).

Reads every  detection_results_<judge>.csv  produced by ai_detection_llm.py on the
balanced quiz, joins each back to the quiz on `item_id` to recover the matched
`pair_id`, and reports properly-powered, paired statistics:

  * per (judge x register x condition) detection recall with Wilson 95% CIs;
  * per-judge overall + balanced accuracy, human/AI recall, per-register accuracy,
    and confidence calibration;
  * McNemar exact tests for the condition contrasts (baseline vs human_like,
    baseline vs detector_evasive, human_like vs detector_evasive), paired by
    document within each judge x register -- the powered test of "does the
    evasive prompt significantly change detectability?".

Recall convention: for an AI condition, recall = P(judge says AI | text is that
condition's AI). For human, recall = P(judge says human | text is human). Because
the pooled file is 1 human : 3 AI by construction, always read balanced accuracy
and per-condition recall, never pooled accuracy alone.

Outputs (into --outdir, default this folder):
  balanced_detection_summary.csv   long-format recall + Wilson CIs + calibration
  balanced_detection_mcnemar.csv   paired condition contrasts per judge x register
  balanced_detection_tables.tex    heatmap LaTeX (summary + by-condition tables)

Usage
  python score_balanced_detection.py
  python score_balanced_detection.py --quiz quiz_master_balanced.csv --results-dir .
"""
import argparse
import glob
import math
import os

import pandas as pd

CONDITIONS_AI = ["baseline", "human_like", "detector_evasive"]
COND_LABEL = {"baseline": "Baseline", "human_like": "Human-mimicking",
              "detector_evasive": "Detector-evasive"}
JUDGE_LABEL = {"claude": "Claude", "deepseek": "DeepSeek", "gemini": "Gemini"}
Z = 1.959963984540054   # 95%

_here = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# stats helpers (no scipy dependency required for Wilson; McNemar uses scipy if
# available, else a normal-approximation fallback)
# ---------------------------------------------------------------------------
def wilson(k, n, z=Z):
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(b, c):
    """Exact (binomial) McNemar two-sided p for discordant counts b, c.

    b = #(A correct, B wrong), c = #(A wrong, B correct). Returns (p, method).
    """
    n = b + c
    if n == 0:
        return 1.0, "no-discordance"
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue), "exact"
    except Exception:
        pass
    try:
        from scipy.stats import binom_test  # older scipy
        return float(binom_test(min(b, c), n, 0.5, alternative="two-sided")), "exact"
    except Exception:
        # normal approximation with continuity correction
        chi = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
        # survival of chi-square df=1 via erfc
        p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
        return p, "normal-approx"


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


# ---------------------------------------------------------------------------
# load + validate
# ---------------------------------------------------------------------------
def load_quiz(path):
    q = pd.read_csv(path, encoding="utf-8")
    need = {"item_id", "register", "condition", "is_human", "pair_id"}
    missing = need - set(q.columns)
    if missing:
        raise SystemExit(f"quiz {path} missing columns: {missing}. "
                         f"Use quiz_master_balanced.csv from make_balanced_quiz.py.")
    q["is_human"] = _coerce_bool(q["is_human"])
    return q[["item_id", "register", "condition", "is_human", "pair_id"]].copy()


def _coerce_bool(s):
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def load_results(path, quiz):
    df = pd.read_csv(path, encoding="utf-8")
    if "item_id" not in df.columns:
        raise SystemExit(f"{path}: no item_id column; cannot pair.")
    name = os.path.basename(path)
    # ground truth comes from the quiz (single source of truth); results provide
    # only the judgement. Keep the results' OWN register/condition/is_human (if
    # present) so we can detect a mismatched run. Join on item_id (1:1).
    keep = ["item_id", "predicted", "confidence"] + \
        [c for c in ("register", "condition", "is_human") if c in df.columns]
    g = df[keep].copy()
    merged = g.merge(quiz, on="item_id", how="left", validate="one_to_one",
                     suffixes=("_res", ""))

    unmatched = merged["register"].isna().sum()
    if unmatched:
        raise SystemExit(
            f"{name}: {unmatched}/{len(merged)} item_ids are not in the balanced "
            f"quiz. Results are from a DIFFERENT run (e.g. the old n=56 "
            f"quiz_master.csv). Re-run the judges on quiz_master_balanced.csv, or "
            f"point --quiz at the matching quiz.")

    # CRITICAL: item_id ranges can OVERLAP across runs (old ids 1-56 are also in
    # 1-800), so matching ids alone is not enough. Verify the results' own labels
    # agree with the quiz; a mismatch means the ids point at different texts.
    disagree = pd.Series(False, index=merged.index)
    if "register_res" in merged:
        disagree |= merged["register_res"].astype(str) != merged["register"].astype(str)
    if "condition_res" in merged:
        disagree |= merged["condition_res"].astype(str) != merged["condition"].astype(str)
    if "is_human_res" in merged:
        disagree |= _coerce_bool(merged["is_human_res"]) != merged["is_human"]
    if disagree.mean() > 0.02:
        raise SystemExit(
            f"{name}: {disagree.sum()}/{len(merged)} items disagree with the quiz on "
            f"register/condition/is_human. These results were produced on a DIFFERENT "
            f"corpus (item_ids collide but point at different texts — most likely the "
            f"old n=56 run). Move the old detection_results_*.csv aside and re-run the "
            f"judges on quiz_master_balanced.csv.")

    # recompute correctness from quiz truth so a stale/renamed file can't lie;
    # UNKNOWN (unparseable) is always wrong.
    pred = merged["predicted"].astype(str).str.strip().str.upper()
    pred_human = pred.isin(("MÄNNISKA", "MANNISKA", "HUMAN"))
    is_unknown = pred.eq("UNKNOWN")
    merged["correct"] = (~is_unknown) & (pred_human == merged["is_human"])
    merged["confidence"] = pd.to_numeric(merged["confidence"], errors="coerce")
    return merged


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def recall_row(judge, register, condition, sub):
    k = int(sub["correct"].sum())
    n = int(len(sub))
    p, lo, hi = wilson(k, n)
    return {"judge": judge, "register": register, "condition": condition,
            "n": n, "n_correct": k, "recall": p, "wilson_lo": lo, "wilson_hi": hi}


def summarise(judge, m):
    """Long-format recall rows + a per-judge headline row."""
    rows = []
    for reg in sorted(m["register"].unique()):
        for cond in ["human"] + CONDITIONS_AI:
            sub = m[(m["register"] == reg) & (m["condition"] == cond)]
            if len(sub):
                rows.append(recall_row(judge, reg, cond, sub))
        # per-register pooled accuracy
        rsub = m[m["register"] == reg]
        rows.append(recall_row(judge, reg, "_ALL_", rsub))

    hum = m[m["is_human"]]
    ai = m[~m["is_human"]]
    hp, _, _ = wilson(int(hum["correct"].sum()), len(hum))
    ap, _, _ = wilson(int(ai["correct"].sum()), len(ai))
    ov, ov_lo, ov_hi = wilson(int(m["correct"].sum()), len(m))
    bal = (hp + ap) / 2 if len(hum) and len(ai) else float("nan")
    cor, wr = m[m["correct"]], m[~m["correct"]]
    headline = {
        "judge": judge, "register": "ALL", "condition": "_HEADLINE_",
        "n": len(m), "n_correct": int(m["correct"].sum()),
        "overall_acc": ov, "overall_lo": ov_lo, "overall_hi": ov_hi,
        "balanced_acc": bal, "human_recall": hp, "ai_recall": ap,
        "conf_correct": cor["confidence"].mean(), "conf_wrong": wr["confidence"].mean(),
    }
    return rows, headline


def mcnemar_contrasts(judge, m):
    """Paired condition contrasts within each register (aligned by pair_id)."""
    out = []
    pairs = [("baseline", "human_like"), ("baseline", "detector_evasive"),
             ("human_like", "detector_evasive")]
    for reg in sorted(m["register"].unique()):
        sub = m[m["register"] == reg]
        wide = sub.pivot_table(index="pair_id", columns="condition",
                               values="correct", aggfunc="first")
        for a, cond_b in pairs:
            if a not in wide or cond_b not in wide:
                continue
            both = wide[[a, cond_b]].dropna()
            ca, cb = both[a].astype(bool), both[cond_b].astype(bool)
            b = int((ca & ~cb).sum())   # A right, B wrong
            c = int((~ca & cb).sum())   # A wrong, B right
            p, method = mcnemar_exact(b, c)
            out.append({
                "judge": judge, "register": reg, "contrast": f"{a}_vs_{cond_b}",
                "n_pairs": int(len(both)),
                f"recall_{a}": float(ca.mean()), f"recall_{cond_b}": float(cb.mean()),
                "b_only_first_correct": b, "c_only_second_correct": c,
                "p_value": p, "method": method, "sig": stars(p),
            })
    return out


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def _cg(pct):
    return f"\\cg{{{max(0, min(50, round(pct / 2)))}}}"


def _pctci(p, lo, hi):
    half = (hi - lo) / 2 * 100
    return f"{_cg(p*100)}${p*100:.1f}_{{\\pm{half:.0f}}}$"


def write_latex(headlines, long_rows, mcn, path):
    hl = {h["judge"]: h for h in headlines}
    L = pd.DataFrame(long_rows)
    judges = [j for j in ["claude", "deepseek", "gemini"] if j in hl] \
        + [j for j in hl if j not in ("claude", "deepseek", "gemini")]

    def reg_acc(j, reg):
        r = L[(L.judge == j) & (L.register == reg) & (L.condition == "_ALL_")]
        return r.iloc[0] if len(r) else None

    lines = [
        "% Auto-generated by score_balanced_detection.py — do not edit by hand.",
        "% REQUIRES: \\usepackage{booktabs}, \\usepackage[table]{xcolor}",
        "% Shares \\cg / hmgreen with feature_analysis_tables.tex.",
        "\\providecommand{\\cg}[1]{\\cellcolor{hmgreen!#1}}",
        "% \\definecolor{hmgreen}{HTML}{2E8B57} % uncomment if used standalone",
        "",
        "% ---- Table 1: per-judge headline performance ----",
        "\\begin{table*}[t]", "  \\centering", "  \\small",
        "  \\begin{tabular}{lcccccc l}", "  \\toprule",
        "    Judge & Overall & Balanced$^{\\dagger}$ & Human & AI & Formal & Informal & Conf.\\ \\\\",
        "          & acc.\\   & acc.\\               & recall & recall & acc.\\ & acc.\\    & calib. \\\\",
        "    \\midrule",
    ]
    for j in judges:
        h = hl[j]
        fa, ia = reg_acc(j, "formal"), reg_acc(j, "informal")
        calib = ("Calibrated" if h["conf_correct"] > h["conf_wrong"] + 2
                 else "Inverted" if h["conf_correct"] < h["conf_wrong"] - 2 else "Flat")
        cells = [
            f"{_cg(h['overall_acc']*100)}${h['overall_acc']*100:.1f}$",
            f"{_cg(h['balanced_acc']*100)}${h['balanced_acc']*100:.1f}$",
            f"{_cg(h['human_recall']*100)}${h['human_recall']*100:.1f}$",
            f"{_cg(h['ai_recall']*100)}${h['ai_recall']*100:.1f}$",
            f"{_cg(fa['recall']*100)}${fa['recall']*100:.1f}$" if fa is not None else "---",
            f"{_cg(ia['recall']*100)}${ia['recall']*100:.1f}$" if ia is not None else "---",
        ]
        lines.append(f"    {JUDGE_LABEL.get(j, j)} & " + " & ".join(cells) + f" & {calib} \\\\")
    lines += [
        "    \\bottomrule", "  \\end{tabular}",
        "  \\caption{LLM-as-judge AI-text detection on the balanced matched corpus "
        "(GPT-5.2; 100 documents per register, each contributing a human original "
        "and its three aligned regenerations). Cells shaded by accuracy. "
        "$^{\\dagger}$Balanced accuracy = mean(human recall, AI recall). "
        "Confidence calibration compares mean confidence on correct vs.\\ incorrect "
        "judgements.}",
        "  \\label{tab:llm-judge-balanced}", "\\end{table*}", "",
        "% ---- Table 2: AI-detection recall by condition (Wilson 95% CI); "
        "superscripts = McNemar vs baseline ----",
        "\\begin{table*}[t]", "  \\centering", "  \\small",
        "  \\begin{tabular}{ll ccc}", "  \\toprule",
        "    Judge & Register & Baseline & Human-mimicking & Detector-evasive \\\\",
        "    \\midrule",
    ]
    # significance vs baseline lookup
    sigmap = {(r["judge"], r["register"], r["contrast"]): r["sig"] for r in mcn}
    for j in judges:
        for reg in ["formal", "informal"]:
            cells = []
            for cond in CONDITIONS_AI:
                r = L[(L.judge == j) & (L.register == reg) & (L.condition == cond)]
                if not len(r):
                    cells.append("---"); continue
                row = r.iloc[0]
                sup = ""
                if cond != "baseline":
                    sup = sigmap.get((j, reg, f"baseline_vs_{cond}"), "")
                cells.append(_pctci(row["recall"], row["wilson_lo"], row["wilson_hi"])
                             + (f"$^{{{sup}}}$" if sup else ""))
            reglab = reg.capitalize()
            jlab = JUDGE_LABEL.get(j, j) if reg == "formal" else ""
            lines.append(f"    {jlab} & {reglab} & " + " & ".join(cells) + " \\\\")
        lines.append("    \\addlinespace")
    lines += [
        "    \\bottomrule", "  \\end{tabular}",
        "  \\caption{AI-detection recall (\\%, Wilson 95\\% CI half-width in subscript) "
        "by prompt condition, per judge and register ($n=100$ AI items per cell, "
        "paired by source document). Superscripts give the McNemar exact-test "
        "significance of each condition vs.\\ baseline "
        "($^{*}q<.05$, $^{**}q<.01$, $^{***}q<.001$).}",
        "  \\label{tab:llm-judge-balanced-condition}", "\\end{table*}", "",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiz", default=os.path.join(_here, "quiz_master_balanced.csv"))
    ap.add_argument("--results-dir", default=_here,
                    help="folder holding detection_results_<judge>.csv (default: here)")
    ap.add_argument("--outdir", default=_here)
    args = ap.parse_args()

    quiz = load_quiz(args.quiz)
    files = sorted(glob.glob(os.path.join(args.results_dir, "detection_results_*.csv")))
    if not files:
        raise SystemExit(f"No detection_results_*.csv in {args.results_dir}. "
                         f"Run ai_detection_llm.py on quiz_master_balanced.csv first.")

    long_rows, headlines, mcn = [], [], []
    print(f"Quiz: {os.path.basename(args.quiz)}  ({len(quiz)} items, "
          f"{quiz['pair_id'].nunique()} documents)\n")
    for f in files:
        judge = os.path.basename(f)[len("detection_results_"):-len(".csv")]
        m = load_results(f, quiz)
        if len(m) < len(quiz):
            print(f"[note] {judge}: only {len(m)}/{len(quiz)} items scored — "
                  f"partial run; figures are provisional.")
        rows, hd = summarise(judge, m)
        long_rows += rows
        headlines.append(hd)
        mcn += mcnemar_contrasts(judge, m)

        print(f"=== {JUDGE_LABEL.get(judge, judge)}  (n={hd['n']}) ===")
        print(f"  overall {hd['overall_acc']*100:5.1f}%  "
              f"balanced {hd['balanced_acc']*100:5.1f}%  "
              f"human-recall {hd['human_recall']*100:5.1f}%  "
              f"ai-recall {hd['ai_recall']*100:5.1f}%")
        for reg in ["formal", "informal"]:
            parts = []
            for cond in CONDITIONS_AI:
                r = [x for x in rows if x["register"] == reg and x["condition"] == cond]
                if r:
                    parts.append(f"{COND_LABEL[cond]} {r[0]['recall']*100:.0f}%")
            if parts:
                print(f"    {reg:8s} AI recall: " + " | ".join(parts))
        for r in [x for x in mcn if x["judge"] == judge
                  and x["contrast"] == "baseline_vs_detector_evasive"]:
            print(f"    McNemar {r['register']:8s} baseline vs evasive: "
                  f"p={r['p_value']:.2e} {r['sig']}")
        print()

    # ---- write outputs
    sum_path = os.path.join(args.outdir, "balanced_detection_summary.csv")
    mcn_path = os.path.join(args.outdir, "balanced_detection_mcnemar.csv")
    tex_path = os.path.join(args.outdir, "balanced_detection_tables.tex")

    # merge headline fields into a wide per-judge frame + keep long recall rows
    pd.DataFrame(long_rows).to_csv(sum_path, index=False, encoding="utf-8")
    pd.DataFrame(headlines).to_csv(
        os.path.join(args.outdir, "balanced_detection_headline.csv"),
        index=False, encoding="utf-8")
    pd.DataFrame(mcn).to_csv(mcn_path, index=False, encoding="utf-8")
    write_latex(headlines, long_rows, mcn, tex_path)

    print("Wrote:")
    print(f"  {sum_path}")
    print(f"  {os.path.join(args.outdir, 'balanced_detection_headline.csv')}")
    print(f"  {mcn_path}")
    print(f"  {tex_path}")


if __name__ == "__main__":
    main()
