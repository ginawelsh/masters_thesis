# EACL paper skeleton (draft)

> Abstract, Introduction (+ contributions + RQs), section stubs, Limitations, and Ethics.
> Draft prose to adapt to the paper's voice / co-author framing. Results prose + numbers are
> in `results_draft.md`; figures in this folder; data/method in `METHODS.md`.

**Working title:**
*Register Makes the Tell: LLM-Generated Swedish Resists Adversarial Detection-Evasion in Abstracts but Blends In in Comments*

Alternatives:
- *Can LLMs Evade Stylometric Detection in Swedish? Register Effects and the Failure of Adversarial Prompting*
- *Prompting to Pass: Why "Don't Sound Like AI" Fails against Feature-Based Detection of Swedish LLM Text*

---

## Abstract

Detecting machine-generated text is increasingly important, yet most work targets English
and a single register, and it is unclear how robust interpretable, feature-based detection is
to (i) register and (ii) deliberate evasion by prompting. We study LLM-generated **Swedish**
across two contrasting registers — formal thesis abstracts and informal forum comments — for
two generators (GPT-5.2 and Mistral-small) under a baseline prompt and three adversarial
"sound human / evade detection" prompts. Using a paired, effect-size-based stylometric analysis
over six feature families, we find that (1) LLM text is far more distinguishable from human
text in the **formal** register than the informal one, where both models blend in; (2)
adversarial prompting **does not** reduce stylometric distance from human text and frequently
**increases** it — a signal-targeted "detector-aware" prompt is the most counterproductive,
via over-compliance; and (3) **model identity matters more than any prompt manipulation** —
Mistral is more human-like than GPT-5.2 in the formal register, and the two diverge most,
by up to a very large effect, precisely under the adversarial conditions. We release the
register-contrastive corpus and analysis pipeline.

---

## 1 Introduction

*(Motivation.)* Large language models now produce fluent text at scale, raising the practical
need to tell machine-generated text from human writing — for academic integrity, provenance,
and information ecosystems. Two properties make this hard in the wild but are under-examined
together: **register** (the same model writes very differently in an academic abstract vs a
forum comment) and **adversarial evasion** (a user can simply *ask* the model to "write so it
can't be detected as AI"). Both bear directly on interpretable, feature-based detectors —
stylometric and linguistic features that are attractive because they are transparent and do not
require a trained black-box classifier — yet their robustness to register and to prompt-level
evasion is largely untested, especially for languages other than English.

*(Gap.)* Prior detection work is overwhelmingly English and single-register, and evaluations of
"can prompting evade detection" typically target neural detectors on English. Whether the same
adversarial prompts move *interpretable* features, whether they behave differently across
registers, and whether findings hold across models, are open — as is the picture for a
moderately-resourced language such as Swedish.

*(This work.)* We assemble a **register-contrastive Swedish corpus** — human formal abstracts
and informal forum comments — and, over identical prompts, generate matched text from two LLMs
(GPT-5.2; Mistral-small-2506) under four conditions: a neutral **baseline**, a naive
**human-like** prompt, a **detector-aware** prompt targeting known AI stylometric signals, and a
**detector-evasive** prompt tuned to invert the specific features we measure. We compare human
vs each model, and the two models against each other, with paired Wilcoxon tests and matched-
pairs effect sizes (Cohen's *dz*, rank-biserial) under BH-FDR, across six feature families
(stylometric surface, syntactic complexity, pragmatic/hedge markers, named entities,
readability, POS proportions).

*(Findings / contributions.)* We contribute:
1. A **register-contrastive Swedish human/LLM corpus** with two generators and four
   prompt conditions, plus a reproducible paired feature-analysis pipeline.
2. Evidence that **detectability is strongly register-dependent**: LLM Swedish diverges from
   human writing at a small-to-medium effect in formal abstracts but only negligibly-to-small in
   informal comments, consistently across both models.
3. Evidence that **adversarial prompting fails to evade feature-based detection and often
   backfires** — of twelve model×register×strategy cells, ten are neutral or *more* detectable
   than baseline, with the "detector-aware" strategy the worst, driven by over-compliance.
4. Evidence that **model identity dominates prompt manipulation**: Mistral is more human-like
   than GPT-5.2 in the formal register, and the two models diverge most (up to very large
   effects) exactly under the adversarial conditions, chiefly in output length and its
   downstream readability/structure effects.

### Research questions
- **RQ1 — Register.** Does the stylometric distinguishability of LLM-generated Swedish from
  human text depend on register (formal vs informal)?
- **RQ2 — Adversarial evasion.** Can adversarial prompting ("write human-like" / "evade
  detection") reduce that distinguishability, and does its effect differ by register or strategy?
- **RQ3 — Model.** Do different LLMs (GPT-5.2 vs Mistral) differ in how human-like their Swedish
  is, and in how they respond to adversarial prompting?

---

## 2 Related work *(stub — to write)*
Cover: machine-text detection (feature/stylometric vs neural/zero-shot, e.g. DetectGPT-style);
adversarial/paraphrase evasion of detectors; register/genre variation in detection;
non-English & Swedish NLP resources; LLM stylometry. Position our gap: interpretable features ×
register × prompt-evasion × Swedish × two models.

## 3 Data and method *(stub — see METHODS.md)*
Corpus (registers, sources, sizes: 1,149 informal comment rows / 170 formal abstracts; human +
2 models × 4 conditions), byte-identical prompt contract (temp 1.0, no system prompt), feature
families + extractors, paired Wilcoxon + *dz* + rank-biserial + BH-FDR, effect-size-first reporting.

## 4 Results *(see results_draft.md — Table 1, Figs 1–3, RQ1/RQ2/RQ3)*

---

## Limitations

- **Detector scope.** Our "detectability" is defined w.r.t. *interpretable stylometric/linguistic
  features*, not neural or zero-shot detectors (e.g. DetectGPT-style, fine-tuned classifiers).
  Adversarial prompting could behave differently against those; our claims are about
  feature-based detection.
- **Prompt-only evasion.** We test *prompt-level* evasion with fixed prompt formulations of our
  own design. More sophisticated, iterative, or optimisation-based evasion (paraphrase attacks,
  RL) is out of scope and might succeed where prompting fails.
- **One language, two registers, two models.** Swedish, formal abstracts + informal forum
  comments, GPT-5.2 + Mistral-small. Generalisation to other languages, registers, and models
  (notably a **Swedish-native model such as GPT-SW3**, left to future work due to gated access)
  is open. Model versions are snapshots and may change.
- **Sampling.** One generation per prompt per model; we do not model per-prompt sampling
  variance. Temperature is fixed at 1.0 for all backends (a robustness check found sampling
  temperature has negligible effect on the features, justifying the choice).
- **Corpus imbalance & power.** The informal set (1,149) is much larger than the formal (170);
  at large *n* nearly all features reach significance, so we lead with effect sizes. The smaller
  formal set has lower power for *tiny* effects (adequate for small-and-larger).
- **Readability metrics.** `textstat` indices are English-calibrated; we use them as *relative*
  signals across matched conditions, not as absolute Swedish readability.
- **Possible pretraining exposure.** The human texts are public (theses, forum posts) and may
  appear in model pretraining, which could bias human–LLM similarity in either direction.
- **Excluded layers.** Affective/sentiment, sentence-embedding, and topic (BERTopic) analyses are
  out of scope here.

## Ethics / Broader impact *(stub)*

- **Dual use.** AI-text detection research can inform both detection and evasion; we report an
  evasion *failure*, which we judge low-risk, but note the dual-use framing.
- **Data.** Informal data is from public forums (incl. Flashback), which can contain sensitive or
  offensive content; we analyse aggregate linguistic features, not individuals, and release
  derived features / generated text under [licence TBD] with any necessary redaction.
- **Environmental / cost.** Generation used hosted APIs (GPT-5.2, Mistral) at modest cost; report
  approximate compute.
