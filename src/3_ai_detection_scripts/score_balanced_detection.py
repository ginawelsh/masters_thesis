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

Multiple generators
-------------------
`generator` (which system wrote the text) is a factor alongside register and
condition. Recall is reported per generator in SEPARATE tables, and a dedicated
contrast table gives the paired generator-vs-generator tests within each
(register, condition) -- the powered test of "does detectability change across
model generations?". Human items carry generator "human" and are shared across
generators (one human set, reused), so human recall is not per-generator.

GENERATOR RELABELLING (on by default, --no-relabel to disable)
The original quiz labels every AI item `gpt`, but that label is not uniform: the
informal `baseline` column (`generated_comment`) is gpt-4o-mini text, while every
other GPT cell is GPT-5.2. See ../../BASELINE_PROVENANCE.md. By default `gpt` is
therefore resolved to `gpt-4o-mini` for informal+baseline and `gpt-5.2` elsewhere,
and the substitution is printed. Quizzes that already carry explicit labels
(gpt-5.2 / gpt-5.6, from make_followup_quiz.py) pass through untouched.

Outputs (into --outdir, default this folder):
  balanced_detection_summary.csv    long-format recall + Wilson CIs + calibration
  balanced_detection_mcnemar.csv    paired condition contrasts per judge x register
                                    x generator
  balanced_detection_generators.csv paired generator contrasts per judge x register
                                    x condition
  balanced_detection_tables.tex     heatmap LaTeX (headline + one recall table per
                                    generator + generator-contrast table)

Usage
  python score_balanced_detection.py
  python score_balanced_detection.py --quiz quiz_master_balanced.csv --results-dir .
  # pool the original run and the GPT-5.6 follow-up:
  python score_balanced_detection.py --quiz quiz_master_balanced.csv quiz_followup.csv
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
GEN_LABEL = {"gpt-4o-mini": "GPT-4o-mini", "gpt-5.2": "GPT-5.2",
             "gpt-5.6": "GPT-5.6", "gpt": "GPT (unresolved)", "human": "Human"}
# display order; anything unrecognised is appended alphabetically
GEN_ORDER = ["gpt-4o-mini", "gpt-5.2", "gpt-5.6"]
Z = 1.959963984540054   # 95%

_here = os.path.dirname(os.path.abspath(__file__))


def gen_sort(gens):
    known = [g for g in GEN_ORDER if g in gens]
    return known + sorted(g for g in gens if g not in GEN_ORDER)


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
def load_quiz(paths, relabel=True):
    """Load one or more quiz files and concatenate them.

    Multiple quizzes are supported so the original run and a follow-up (e.g.
    quiz_followup.csv, GPT-5.6) can be scored together; `pair_id` is shared across
    them by construction, which is what makes the generator contrasts paired.
    """
    if isinstance(paths, str):
        paths = [paths]
    frames = []
    for path in paths:
        q = pd.read_csv(path, encoding="utf-8")
        need = {"item_id", "register", "condition", "is_human", "pair_id"}
        missing = need - set(q.columns)
        if missing:
            raise SystemExit(f"quiz {path} missing columns: {missing}. "
                             f"Use quiz_master_balanced.csv from make_balanced_quiz.py.")
        if "generator" not in q.columns:
            raise SystemExit(
                f"quiz {path} has no `generator` column, so per-generator reporting is "
                f"impossible. Rebuild it with make_balanced_quiz.py / make_followup_quiz.py.")
        q["is_human"] = _coerce_bool(q["is_human"])
        keep = ["item_id", "register", "condition", "is_human", "pair_id", "generator"]
        if "supersedes" in q.columns:
            keep.append("supersedes")
        frames.append(q[keep].copy())
    quiz = pd.concat(frames, ignore_index=True)
    if "supersedes" not in quiz.columns:
        quiz["supersedes"] = pd.NA

    dup = quiz["item_id"].duplicated()
    if dup.any():
        raise SystemExit(
            f"{int(dup.sum())} item_id values appear in more than one quiz file "
            f"(e.g. {sorted(quiz.loc[dup, 'item_id'])[:5]}). Results join on item_id, so "
            f"overlapping ids would be scored against the wrong text. Rebuild the "
            f"follow-up quiz with a non-overlapping --offset.")

    # A follow-up item may SUPERSEDE an already-judged one (e.g. the same text with
    # markdown stripped). Drop the superseded originals so the document is counted once
    # and the pivot in the contrast functions stays 1:1.
    sup = pd.to_numeric(quiz["supersedes"], errors="coerce").dropna().astype(int)
    superseded = set()
    if len(sup):
        missing = set(sup) - set(quiz["item_id"])
        if missing:
            raise SystemExit(
                f"follow-up quiz supersedes item_id(s) {sorted(missing)} that are not in "
                f"any loaded quiz. Load the original quiz alongside the follow-up, or the "
                f"replacement would silently be counted as an extra observation.")
        superseded = set(sup)
        drop = quiz["item_id"].isin(superseded)
        print(f"[supersede] dropping {int(drop.sum())} superseded item(s) "
              f"{sorted(quiz.loc[drop, 'item_id'])} — replaced by re-judged versions")
        quiz = quiz[~drop].reset_index(drop=True)

    quiz["generator"] = quiz["generator"].astype(str).str.strip()
    quiz.loc[quiz["is_human"], "generator"] = "human"
    if relabel:
        bare = quiz["generator"].eq("gpt")
        if bare.any():
            inf_base = bare & quiz["register"].eq("informal") & quiz["condition"].eq("baseline")
            quiz.loc[inf_base, "generator"] = "gpt-4o-mini"
            quiz.loc[bare & ~inf_base, "generator"] = "gpt-5.2"
            print(f"[relabel] resolved {int(bare.sum())} items labelled 'gpt': "
                  f"{int(inf_base.sum())} informal+baseline -> gpt-4o-mini, "
                  f"{int((bare & ~inf_base).sum())} -> gpt-5.2 "
                  f"(see BASELINE_PROVENANCE.md; --no-relabel to disable)")
            if inf_base.any():
                print("          NOTE: the gpt-4o-mini label is verified for ~75% of the "
                      "informal baseline column (863/1149 corpus-wide, 71/100 in the "
                      "quiz); the rest is inferred, not traced. Report it with that "
                      "caveat -- see BASELINE_PROVENANCE.md D3.")
    return quiz, superseded


def _coerce_bool(s):
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))


def load_results(path, quiz, superseded=frozenset()):
    df = pd.read_csv(path, encoding="utf-8")
    if "item_id" not in df.columns:
        raise SystemExit(f"{path}: no item_id column; cannot pair.")
    name = os.path.basename(path)
    # Judgements for superseded items are stale by construction (the text changed), so
    # drop them here rather than letting the corpus-mismatch guard below fire on them.
    if superseded:
        stale = df["item_id"].isin(superseded)
        if stale.any():
            print(f"  [{name}] dropping {int(stale.sum())} stale judgement(s) for "
                  f"superseded item(s)")
            df = df[~stale].copy()
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
def recall_row(judge, generator, register, condition, sub):
    k = int(sub["correct"].sum())
    n = int(len(sub))
    p, lo, hi = wilson(k, n)
    return {"judge": judge, "generator": generator, "register": register,
            "condition": condition, "n": n, "n_correct": k,
            "recall": p, "wilson_lo": lo, "wilson_hi": hi}


def summarise(judge, m):
    """Long-format recall rows + one headline row per (judge, generator).

    Human items are shared across generators (a single human set, reused), so they
    are emitted once under generator "human" and are NOT duplicated per generator.
    Each generator's headline pairs that generator's AI recall with the shared human
    recall, which is what makes balanced accuracy comparable across generators.
    """
    rows = []
    hum_all = m[m["is_human"]]

    # human recall, once per register
    for reg in sorted(hum_all["register"].unique()):
        sub = hum_all[hum_all["register"] == reg]
        if len(sub):
            rows.append(recall_row(judge, "human", reg, "human", sub))

    ai_all = m[~m["is_human"]]
    headlines = []
    for gen in gen_sort(set(ai_all["generator"].unique())):
        g = ai_all[ai_all["generator"] == gen]
        for reg in sorted(g["register"].unique()):
            for cond in CONDITIONS_AI:
                sub = g[(g["register"] == reg) & (g["condition"] == cond)]
                if len(sub):
                    rows.append(recall_row(judge, gen, reg, cond, sub))
            rows.append(recall_row(judge, gen, reg, "_ALL_", g[g["register"] == reg]))

        # headline: this generator's AI items + the shared human items
        hp, _, _ = wilson(int(hum_all["correct"].sum()), len(hum_all))
        ap, _, _ = wilson(int(g["correct"].sum()), len(g))
        both = pd.concat([hum_all, g], ignore_index=True)
        ov, ov_lo, ov_hi = wilson(int(both["correct"].sum()), len(both))
        bal = (hp + ap) / 2 if len(hum_all) and len(g) else float("nan")
        cor, wr = both[both["correct"]], both[~both["correct"]]
        headlines.append({
            "judge": judge, "generator": gen, "register": "ALL",
            "condition": "_HEADLINE_",
            "n": len(both), "n_correct": int(both["correct"].sum()),
            "n_ai": len(g), "n_human": len(hum_all),
            "overall_acc": ov, "overall_lo": ov_lo, "overall_hi": ov_hi,
            "balanced_acc": bal, "human_recall": hp, "ai_recall": ap,
            "conf_correct": cor["confidence"].mean(),
            "conf_wrong": wr["confidence"].mean(),
        })
    return rows, headlines


def _paired(wide, a, b_key):
    """Shared McNemar body for a two-column wide frame indexed by pair_id."""
    both = wide[[a, b_key]].dropna()
    ca, cb = both[a].astype(bool), both[b_key].astype(bool)
    b = int((ca & ~cb).sum())   # A right, B wrong
    c = int((~ca & cb).sum())   # A wrong, B right
    p, method = mcnemar_exact(b, c)
    return both, ca, cb, b, c, p, method


def mcnemar_contrasts(judge, m):
    """Paired condition contrasts within each (register, generator), by pair_id."""
    out = []
    pairs = [("baseline", "human_like"), ("baseline", "detector_evasive"),
             ("human_like", "detector_evasive")]
    ai = m[~m["is_human"]]
    for gen in gen_sort(set(ai["generator"].unique())):
        for reg in sorted(ai["register"].unique()):
            sub = ai[(ai["generator"] == gen) & (ai["register"] == reg)]
            if not len(sub):
                continue
            wide = sub.pivot_table(index="pair_id", columns="condition",
                                   values="correct", aggfunc="first")
            for a, cond_b in pairs:
                if a not in wide or cond_b not in wide:
                    continue
                both, ca, cb, b, c, p, method = _paired(wide, a, cond_b)
                out.append({
                    "judge": judge, "generator": gen, "register": reg,
                    "contrast": f"{a}_vs_{cond_b}", "n_pairs": int(len(both)),
                    f"recall_{a}": float(ca.mean()),
                    f"recall_{cond_b}": float(cb.mean()),
                    "b_only_first_correct": b, "c_only_second_correct": c,
                    "p_value": p, "method": method, "sig": stars(p),
                })
    return out


def generator_contrasts(judge, m):
    """Paired GENERATOR contrasts within each (register, condition), by pair_id.

    This is the "has detectability changed across model generations?" test. It is
    paired because make_followup_quiz.py pins every new item to the pair_id of an
    already-judged item, so both generators are scored on the SAME source document.
    A generator present for a document under one condition but not the other simply
    drops out of that contrast (dropna), so unequal coverage is safe.
    """
    out = []
    ai = m[~m["is_human"]]
    gens = gen_sort(set(ai["generator"].unique()))
    if len(gens) < 2:
        return out
    for reg in sorted(ai["register"].unique()):
        for cond in CONDITIONS_AI:
            sub = ai[(ai["register"] == reg) & (ai["condition"] == cond)]
            if not len(sub):
                continue
            wide = sub.pivot_table(index="pair_id", columns="generator",
                                   values="correct", aggfunc="first")
            present = [g for g in gens if g in wide]
            for i, ga in enumerate(present):
                for gb in present[i + 1:]:
                    both, ca, cb, b, c, p, method = _paired(wide, ga, gb)
                    if not len(both):
                        continue
                    out.append({
                        "judge": judge, "register": reg, "condition": cond,
                        "contrast": f"{ga}_vs_{gb}",
                        "generator_a": ga, "generator_b": gb,
                        "n_pairs": int(len(both)),
                        "recall_a": float(ca.mean()), "recall_b": float(cb.mean()),
                        "delta_b_minus_a": float(cb.mean() - ca.mean()),
                        "b_only_a_correct": b, "c_only_b_correct": c,
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


def write_latex(headlines, long_rows, mcn, gcon, path):
    H = pd.DataFrame(headlines)
    L = pd.DataFrame(long_rows)
    all_j = list(dict.fromkeys(H["judge"]))
    judges = [j for j in ["claude", "deepseek", "gemini"] if j in all_j] \
        + [j for j in all_j if j not in ("claude", "deepseek", "gemini")]
    generators = gen_sort(set(H["generator"]))

    def reg_acc(j, gen, reg):
        r = L[(L.judge == j) & (L.generator == gen) & (L.register == reg)
              & (L.condition == "_ALL_")]
        return r.iloc[0] if len(r) else None

    lines = [
        "% Auto-generated by score_balanced_detection.py — do not edit by hand.",
        "% REQUIRES: \\usepackage{booktabs}, \\usepackage[table]{xcolor}",
        "% Shares \\cg / hmgreen with feature_analysis_tables.tex.",
        "\\providecommand{\\cg}[1]{\\cellcolor{hmgreen!#1}}",
        "% \\definecolor{hmgreen}{HTML}{2E8B57} % uncomment if used standalone",
        "",
        "% ---- Table 1: headline performance per judge x generator ----",
        "\\begin{table*}[t]", "  \\centering", "  \\small",
        "  \\begin{tabular}{llcccccc l}", "  \\toprule",
        "    Judge & Generator & Overall & Balanced$^{\\dagger}$ & Human & AI & Formal "
        "& Informal & Conf.\\ \\\\",
        "          &           & acc.\\   & acc.\\               & recall & recall "
        "& acc.\\ & acc.\\    & calib. \\\\",
        "    \\midrule",
    ]
    for j in judges:
        first = True
        for gen in generators:
            hr = H[(H.judge == j) & (H.generator == gen)]
            if not len(hr):
                continue
            h = hr.iloc[0]
            fa, ia = reg_acc(j, gen, "formal"), reg_acc(j, gen, "informal")
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
            jlab = JUDGE_LABEL.get(j, j) if first else ""
            lines.append(f"    {jlab} & {GEN_LABEL.get(gen, gen)} & "
                         + " & ".join(cells) + f" & {calib} \\\\")
            first = False
        lines.append("    \\addlinespace")
    lines += [
        "    \\bottomrule", "  \\end{tabular}",
        "  \\caption{LLM-as-judge AI-text detection on the balanced matched corpus, "
        "per generator (100 documents per register; each document contributes a human "
        "original and its aligned regenerations). The human item set is shared across "
        "generators, so human recall is identical down each judge's block. Cells shaded "
        "by accuracy. $^{\\dagger}$Balanced accuracy = mean(human recall, AI recall). "
        "Confidence calibration compares mean confidence on correct vs.\\ incorrect "
        "judgements.}",
        "  \\label{tab:llm-judge-balanced}", "\\end{table*}", "",
    ]

    # ---- one recall-by-condition table per generator ----
    sigmap = {(r["judge"], r["generator"], r["register"], r["contrast"]): r["sig"]
              for r in mcn}
    for gi, gen in enumerate(generators, 1):
        sub_any = L[(L.generator == gen) & (L.condition.isin(CONDITIONS_AI))]
        if not len(sub_any):
            continue
        glab = GEN_LABEL.get(gen, gen)
        # the legacy informal baseline is only ~75% traced to gpt-4o-mini output; flag
        # that in the caption rather than presenting the label as fully verified
        caveat = ("" if gen != "gpt-4o-mini" else
                  " The gpt-4o-mini attribution is verified for 863 of 1{,}149 corpus "
                  "items (71 of 100 sampled here); the remainder is inferred from the "
                  "absence of any other baseline-generating script.")
        lines += [
            f"% ---- Table 1.{gi}: {glab} recall by condition ----",
            "\\begin{table}[t]", "  \\centering", "  \\small",
            "  \\begin{tabular}{ll ccc}", "  \\toprule",
            "    Judge & Register & Baseline & Human-mimicking & Detector-evasive \\\\",
            "    \\midrule",
        ]
        for j in judges:
            wrote = False
            for reg in ["formal", "informal"]:
                cells = []
                for cond in CONDITIONS_AI:
                    r = L[(L.judge == j) & (L.generator == gen)
                          & (L.register == reg) & (L.condition == cond)]
                    if not len(r):
                        cells.append("---")
                        continue
                    row = r.iloc[0]
                    sup = "" if cond == "baseline" else \
                        sigmap.get((j, gen, reg, f"baseline_vs_{cond}"), "")
                    cells.append(_pctci(row["recall"], row["wilson_lo"], row["wilson_hi"])
                                 + (f"$^{{{sup}}}$" if sup else ""))
                if set(cells) == {"---"}:
                    continue
                jlab = JUDGE_LABEL.get(j, j) if not wrote else ""
                lines.append(f"    {jlab} & {reg.capitalize()} & "
                             + " & ".join(cells) + " \\\\")
                wrote = True
            if wrote:
                lines.append("    \\addlinespace")
        lines += [
            "    \\bottomrule", "  \\end{tabular}",
            f"  \\caption{{AI-detection recall (\\%, Wilson 95\\% CI half-width in "
            f"subscript) for \\textbf{{{glab}}} text, by prompt condition, per judge and "
            f"register (paired by source document). Superscripts give the McNemar "
            f"exact-test significance of each condition vs.\\ baseline "
            f"($^{{*}}q<.05$, $^{{**}}q<.01$, $^{{***}}q<.001$).{caveat}}}",
            f"  \\label{{tab:llm-judge-{gen.replace('.', '-')}}}", "\\end{table}", "",
        ]

    # ---- generator contrast table ----
    if gcon:
        G = pd.DataFrame(gcon)
        contrasts = list(dict.fromkeys(G["contrast"]))
        lines += [
            "% ---- Table 2: paired generator contrasts (McNemar, same source documents) ----",
            "\\begin{table*}[t]", "  \\centering", "  \\small",
            "  \\begin{tabular}{lll rrr r l}", "  \\toprule",
            "    Judge & Register & Condition & \\multicolumn{2}{c}{Recall (\\%)} "
            "& $\\Delta$ & $n$ & $p$ \\\\",
            "    \\midrule",
        ]
        for con in contrasts:
            ga, gb = G[G.contrast == con].iloc[0][["generator_a", "generator_b"]]
            lines.append(f"    \\multicolumn{{8}}{{l}}{{\\textit{{"
                         f"{GEN_LABEL.get(ga, ga)} $\\rightarrow$ {GEN_LABEL.get(gb, gb)}"
                         f"}}}} \\\\")
            for _, r in G[G.contrast == con].iterrows():
                lines.append(
                    f"    {JUDGE_LABEL.get(r['judge'], r['judge'])} & "
                    f"{r['register'].capitalize()} & {COND_LABEL.get(r['condition'], r['condition'])} & "
                    f"${r['recall_a']*100:.0f}$ & ${r['recall_b']*100:.0f}$ & "
                    f"${r['delta_b_minus_a']*100:+.0f}$ & ${int(r['n_pairs'])}$ & "
                    f"${r['p_value']:.1g}$ {r['sig']} \\\\")
            lines.append("    \\addlinespace")
        lines += [
            "    \\bottomrule", "  \\end{tabular}",
            "  \\caption{Paired generator contrasts (McNemar exact test). Each pair of "
            "generators is compared on the SAME source documents, within register and "
            "prompt condition, so the test is paired rather than two-sample. $\\Delta$ is "
            "the second generator's recall minus the first: positive means the newer "
            "generator is MORE detectable. "
            "($^{*}q<.05$, $^{**}q<.01$, $^{***}q<.001$).}",
            "  \\label{tab:llm-judge-generator-contrast}", "\\end{table*}", "",
        ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiz", nargs="+",
                    default=[os.path.join(_here, "quiz_master_balanced.csv")],
                    help="one or more quiz CSVs; pass the follow-up quiz alongside the "
                         "original to score both generators together")
    ap.add_argument("--results-dir", default=_here,
                    help="folder holding detection_results_<judge>.csv (default: here)")
    ap.add_argument("--outdir", default=_here)
    ap.add_argument("--no-relabel", action="store_true",
                    help="keep the bare 'gpt' generator label instead of resolving it to "
                         "gpt-4o-mini / gpt-5.2 (see BASELINE_PROVENANCE.md)")
    args = ap.parse_args()

    quiz, superseded = load_quiz(args.quiz, relabel=not args.no_relabel)
    files = sorted(glob.glob(os.path.join(args.results_dir, "detection_results_*.csv")))
    if not files:
        raise SystemExit(f"No detection_results_*.csv in {args.results_dir}. "
                         f"Run ai_detection_llm.py on the quiz first.")

    long_rows, headlines, mcn, gcon = [], [], [], []
    print(f"Quiz: {', '.join(os.path.basename(p) for p in args.quiz)}  "
          f"({len(quiz)} items, {quiz['pair_id'].nunique()} documents)")
    ai_gens = gen_sort(set(quiz.loc[~quiz['is_human'], 'generator']))
    print(f"Generators: {', '.join(ai_gens)}\n")
    for f in files:
        judge = os.path.basename(f)[len("detection_results_"):-len(".csv")]
        m = load_results(f, quiz, superseded)
        if len(m) < len(quiz):
            print(f"[note] {judge}: only {len(m)}/{len(quiz)} items scored — "
                  f"partial run; figures are provisional.")
        rows, hds = summarise(judge, m)
        long_rows += rows
        headlines += hds
        mcn += mcnemar_contrasts(judge, m)
        gcon += generator_contrasts(judge, m)

        print(f"=== {JUDGE_LABEL.get(judge, judge)} ===")
        for hd in hds:
            print(f"  [{hd['generator']}] n={hd['n']} "
                  f"overall {hd['overall_acc']*100:5.1f}%  "
                  f"balanced {hd['balanced_acc']*100:5.1f}%  "
                  f"human-recall {hd['human_recall']*100:5.1f}%  "
                  f"ai-recall {hd['ai_recall']*100:5.1f}%")
            for reg in ["formal", "informal"]:
                parts = []
                for cond in CONDITIONS_AI:
                    r = [x for x in rows if x["generator"] == hd["generator"]
                         and x["register"] == reg and x["condition"] == cond]
                    if r:
                        parts.append(f"{COND_LABEL[cond]} {r[0]['recall']*100:.0f}%")
                if parts:
                    print(f"      {reg:8s} AI recall: " + " | ".join(parts))
        for r in [x for x in gcon if x["judge"] == judge]:
            print(f"    generator {r['register']:8s} {r['condition']:17s} "
                  f"{r['contrast']}: {r['recall_a']*100:.0f}% -> {r['recall_b']*100:.0f}% "
                  f"({r['delta_b_minus_a']*100:+.0f}pp) p={r['p_value']:.2e} {r['sig']}")
        print()

    # ---- write outputs
    sum_path = os.path.join(args.outdir, "balanced_detection_summary.csv")
    hd_path = os.path.join(args.outdir, "balanced_detection_headline.csv")
    mcn_path = os.path.join(args.outdir, "balanced_detection_mcnemar.csv")
    gen_path = os.path.join(args.outdir, "balanced_detection_generators.csv")
    tex_path = os.path.join(args.outdir, "balanced_detection_tables.tex")

    pd.DataFrame(long_rows).to_csv(sum_path, index=False, encoding="utf-8")
    pd.DataFrame(headlines).to_csv(hd_path, index=False, encoding="utf-8")
    pd.DataFrame(mcn).to_csv(mcn_path, index=False, encoding="utf-8")
    if gcon:
        pd.DataFrame(gcon).to_csv(gen_path, index=False, encoding="utf-8")
    write_latex(headlines, long_rows, mcn, gcon, tex_path)

    print("Wrote:")
    for p in [sum_path, hd_path, mcn_path] + ([gen_path] if gcon else []) + [tex_path]:
        print(f"  {p}")
    if not gcon:
        print(f"  [note] no generator-contrast table: of the generators present "
              f"({', '.join(ai_gens)}), no (register, condition) cell holds two of them, "
              f"so there is nothing to pair. Score the follow-up quiz alongside this one: "
              f"--quiz quiz_master_balanced.csv quiz_followup.csv")


if __name__ == "__main__":
    main()
