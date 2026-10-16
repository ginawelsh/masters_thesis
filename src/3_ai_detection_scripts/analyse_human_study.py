"""Experiment 3 (human detection study): class-balance-aware scoring and mixed-effects model.

Reads the raw `AI eller inte` export (one row per judgement) and reproduces the thesis
sample: the 37 participants recorded 6-25 August 2026 (the main study; responses before
6 August come from a pilot with a different item set, and one later participant who took
the quiz after the analysis was frozen is not included).

Reports, per register:
  - class distribution of the responses (the quiz is 25% human / 75% AI by design)
  - accuracy, the always-"AI" baseline, human recall, AI recall, balanced accuracy,
    human precision, and the share of texts labelled human
  - a one-sample t-test of per-participant balanced accuracy against 0.5
and fits  correct ~ register + (1 | participant) + (1 | item)  (and a version that also
adjusts for whether the item is human-written) as a Bayesian binomial mixed GLM.

It also writes a long-format CSV that `human_study_glmer.R` fits with lme4::glmer.

Usage:
  python src/3_ai_detection_scripts/analyse_human_study.py \
      --results "../4_ai_eller_inte/ai_eller_inte_results - Sheet1.csv"
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent

# Participant-id prefixes of the 37 participants in the thesis sample.
THESIS_SAMPLE = """
7a8d562e 296c32da a5d4f4fb 5e1eab5a 844ceac7 92748037 6efa276d 1b4fbcc4 d9e1e47a
3a18a8d6 fd90281c 6b7fa1e1 158f6965 33cc8c24 5f7d58d7 2b86c248 b8d75ed8 ea21535a
fad9652e d4d286a7 442a4511 e273f6b8 5281af06 82b04249 f71c8368 868757b4 c8f0ab6b
1ab7c640 93d8ba54 abbf464a d3c318bf 1ab13775 23c13baf f6fde72a 978dbd57 e2bf86b3
bfdbe752
""".split()


def load(results_csv):
    d = pd.read_csv(results_csv)
    d = d[d.timestamp != "timestamp"]  # the export contains a repeated header row
    d = d[d.questionType.isin(["comment", "abstract"])].copy()
    d["ts"] = pd.to_datetime(d.timestamp, format="mixed")
    d["pid"] = d.participantId.str[:8]
    d = d[d.pid.isin(THESIS_SAMPLE) & (d.ts >= "2026-08-06")].copy()
    as_bool = lambda s: s.astype(str).str.upper() == "TRUE"
    d["is_human"] = as_bool(d.isHumanLabel)
    d["guess_human"] = as_bool(d.userGuessIsHuman)
    d["correct"] = as_bool(d.isCorrect).astype(int)
    d["item"] = d.questionText.str.split().str.join(" ").astype("category").cat.codes
    d["register"] = d.questionType
    return d


def scores(g):
    h, a = g[g.is_human], g[~g.is_human]
    rec_h, rec_ai = h.correct.mean(), a.correct.mean()
    return pd.Series({
        "participants": g.pid.nunique(),
        "responses": len(g),
        "human_share": g.is_human.mean(),
        "accuracy": g.correct.mean(),
        "always_ai_baseline": 1 - g.is_human.mean(),
        "recall_human": rec_h,
        "recall_ai": rec_ai,
        "balanced_accuracy": (rec_h + rec_ai) / 2,
        "precision_human": (g.guess_human & g.is_human).sum() / g.guess_human.sum(),
        "labelled_human": g.guess_human.mean(),
    })


def participant_ba_test(g):
    per = g.groupby("pid").apply(
        lambda x: (x[x.is_human].correct.mean() + x[~x.is_human].correct.mean()) / 2,
        include_groups=False).dropna()  # needs at least one human and one AI item
    t = stats.ttest_1samp(per, 0.5)
    return len(per), per.mean(), t.statistic, t.pvalue


def mixed_models(d):
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    d = d.assign(abstract=(d.register == "abstract").astype(int), human=d.is_human.astype(int))
    vc = {"participant": "0 + C(pid)", "item": "0 + C(item)"}
    for formula in ["correct ~ abstract", "correct ~ abstract * human"]:
        fit = BinomialBayesMixedGLM.from_formula(formula, vc, d).fit_vb()
        print(f"\n{formula}  + (1|participant) + (1|item)   [variational Bayes]")
        print(pd.DataFrame({"post_mean": fit.fe_mean, "post_sd": fit.fe_sd},
                           index=fit.model.exog_names).round(3).to_string())
        print("random-intercept SDs:",
              dict(zip(fit.model.vcp_names, np.exp(fit.vcp_mean).round(3))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="raw AI eller inte export (CSV)")
    ap.add_argument("--out", default=HERE / "human_study_long.csv")
    args = ap.parse_args()

    d = load(args.results)
    print(d.groupby("register").apply(scores, include_groups=False).T.round(3).to_string())
    for reg, g in d.groupby("register"):
        n, m, t, p = participant_ba_test(g)
        print(f"{reg}: per-participant balanced accuracy {m:.3f} (n={n}), t={t:.2f}, p={p:.4f}")
    mixed_models(d)

    d[["pid", "item", "register", "is_human", "correct"]].to_csv(args.out, index=False)
    print(f"\nWrote {args.out} for human_study_glmer.R")


if __name__ == "__main__":
    main()
