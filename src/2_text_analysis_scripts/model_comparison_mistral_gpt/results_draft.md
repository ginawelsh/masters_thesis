# Results (draft) — EACL

> Draft prose + numbers for the results section, generated from
> `model_comparison_results.csv`. All figures referenced are in this folder
> (`fig1_distance_from_human.pdf`, `fig2_family_effectsizes.pdf`). Numbers are exact;
> phrasing is a starting point to adapt to the paper's voice / co-author framing.

## Setup (one paragraph for the reader)

We compare human text against two LLM generators (GPT-5.2 and Mistral-small-2506) over
identical prompts, in two registers (formal thesis abstracts, informal forum comments) and
four prompt conditions (baseline, human-like, detector-aware, detector-evasive). Each of the
three pairings — human↔GPT, human↔Mistral, Mistral↔GPT — is **paired by prompt** and tested
per feature with a Wilcoxon signed-rank test, matched-pairs effect sizes (Cohen's *dz*,
rank-biserial), and Benjamini–Hochberg FDR correction within each (register × condition)
family. Because *n* is large (up to 1,149 paired comments), we report and interpret **effect
sizes**, not *p*-values: at this *n*, 77–79% of features clear FDR in every comparison, so
significance alone is uninformative. We summarise a generator's divergence from human as the
median |*dz*| over all features ("distance from human"; lower = more human-like).

## Table 1 — Distance from human (median |*dz*|) by register, model, condition

| Register | Model | Baseline | Human-like | Detector-aware | Detector-evasive |
|---|---|--:|--:|--:|--:|
| Formal | GPT-5.2 | 0.48 | 0.52 | **0.56** | 0.45 |
| Formal | Mistral | 0.32 | 0.29 | 0.40 | 0.29 |
| Informal | GPT-5.2 | 0.16 | 0.20 | 0.17 | 0.20 |
| Informal | Mistral | 0.16 | 0.21 | **0.24** | 0.20 |

(Also Fig. 1.) Bold = largest divergence in that register.

## RQ1 — Detectability is strongly register-dependent

LLM Swedish diverges from human writing **far more in the formal register than the informal
one**. Median distance-from-human is 0.29–0.56 (a *small-to-medium* effect) for abstracts but
only 0.16–0.24 (*negligible-to-small*) for comments (Table 1, Fig. 1). At the feature level
(Fig. 2) the gap is widest in **readability** and **syntactic-complexity** families in formal
text, whereas **NER** rates are near-human everywhere (|*dz*| ≤ 0.24 formal, ≤ 0.07 informal) —
both models reproduce human named-entity density. The register effect is consistent across
both generators, so it is a property of the task, not of one model: short, conversational
comments are intrinsically hard to distinguish from human writing on surface/structural
features, while structured academic prose exposes LLM tells.

## RQ2 — Adversarial "evade detection" prompting fails, and often backfires

Prompting the model to sound human or to evade detection does **not** reduce its stylometric
distance from human — and frequently increases it. Change in distance-from-human vs the
baseline prompt (Δ; negative = more human = successful evasion):

| Register · Model | Human-like | Detector-aware | Detector-evasive |
|---|--:|--:|--:|
| Formal · GPT | +0.04 | **+0.08** | −0.03 |
| Formal · Mistral | −0.03 | **+0.08** | −0.03 |
| Informal · GPT | +0.04 | +0.00 | +0.04 |
| Informal · Mistral | +0.05 | **+0.08** | +0.04 |

Of the twelve model×register×strategy cells, only two show any reduction (both −0.03,
negligible); the rest are neutral or **worse than baseline**. The `detector-aware` strategy is
the most counterproductive (+0.08 in both models, formal). The mechanism is *over-compliance*:
Fig. 2 shows the formal/GPT `detector-evasive` cell spiking to |*dz*| = 1.87 (readability) and
1.14 (syntactic) — instructed to "use short sentences and more periods," the model produces
abnormally choppy prose that is *more* distinguishable from human abstracts, not less.
Take-away: naive and signal-targeted anti-detection prompting is not a reliable route to
evading feature-based detection of LLM Swedish.

## RQ3 — Model identity matters more than any prompt manipulation

The two generators are **not interchangeable**: across 532 feature-tests they differ at
medium-or-larger effect on 183 (|*dz*| ≥ 0.5) and large effect on 99 (≥ 0.8), median |*dz*|
0.30. **Mistral is the more human-like generator in the formal register** — closer to human on
60.7% of formal features and at lower distance in all four formal conditions (Table 1); in the
informal register the two are effectively tied (Mistral closer on 48.2% of features). The
largest model differences are concentrated in the adversarial conditions (Fig. 3): instructed
to evade, **Mistral writes far shorter than GPT** (word/character/token/sentence counts, |*dz*|
up to 2.4), with more function words and more periods — it follows the "short and dense"
instruction much more aggressively. Where each model *does* diverge from human: GPT via readability indices
and sentence structure (over-formal, over-choppy under adversarial prompts); Mistral via
over-punctuation (period/comma/PUNCT rates) and, in informal text, over-hedging and reduced
lexical diversity.

## Effect-size distribution (for the methods/limitations note)

Per comparison (of 532 feature-tests): human↔GPT median |*dz*| 0.28 (195 features ≥ 0.5, 112 ≥
0.8); human↔Mistral 0.26 (159 ≥ 0.5, 84 ≥ 0.8); Mistral↔GPT 0.30 (183 ≥ 0.5, 99 ≥ 0.8). ~78% of
features are FDR-significant in each — hence the effect-size-first reporting.

## RQ4 — Affective sentiment (informal): the tell the stylometry missed

We additionally scored every informal text for sentiment with a Swedish-specific model
(`KBLab/robust-swedish-sentiment-multiclass`), paired human vs each generator per condition.
Formal is excluded (near-uniformly neutral). This directly probes the affective/tonal tell that
surface stylometry cannot capture.

**Top-label sentiment is uninformative** — all three sources are ~77% neutral / ~16% negative /
~7% positive. A categorical sentiment comparison would find nothing. But the **continuous signed
polarity** (P(positive) − P(negative)) tells a different story:

| condition | human | GPT | dz(H,GPT) | Mistral | dz(H,Mistral) |
|---|--:|--:|--:|--:|--:|
| baseline | −0.03 | **+0.12** | **+0.57** | −0.02 | −0.01 (n.s.) |
| human_like | −0.03 | −0.01 | +0.04 | −0.00 | +0.19 |
| detector_aware | −0.03 | −0.08 | −0.20 | −0.05 | −0.13 |
| detector_evasive | −0.03 | −0.11 | −0.23 | −0.07 | −0.13 |

Two findings:

1. **GPT has a baseline positivity bias** — its default comments lean positive while humans (and
   Mistral) lean slightly negative/neutral: dz **+0.57** vs human, and **+0.74 vs Mistral** (the
   single largest affective effect). This is a genuine informal-register tell — *larger than any
   structural feature there* (which topped out ~0.2) — and it is **invisible to the top-label
   sentiment and to the surface stylometry**; only the continuous polarity exposes it.
2. **Adversarial prompting moves affect — but overshoots.** Unlike the structural features (where
   the adversarial prompts failed to move anything toward human, RQ2), on sentiment the
   `human_like` prompt neutralises GPT's positivity (dz +0.57 → +0.04 ≈ human), and
   `detector_aware`/`detector_evasive` push it *past* human into more-negative-than-human
   (dz −0.20 / −0.23). So the prompts have real leverage on tone, but they **can't calibrate it** —
   they swing GPT from too-positive to too-negative rather than onto the human distribution.
3. **Mistral shows no positivity bias** and stays close to human throughout (all |dz| ≤ 0.19),
   reinforcing that it is the more human-like generator (RQ3) on the affective axis too.

**Method note:** the categorical sentiment label misses this entirely; the *continuous* signed
polarity is what surfaces it — a caution for sentiment-based detection work.

## Figures
- **Fig. 1** (`fig1_distance_from_human.pdf`): distance-from-human by condition, register, model
  — the RQ1/RQ2/RQ3 overview.
- **Fig. 2** (`fig2_family_effectsizes.pdf`): mean |*dz*| per feature family × condition, 2×2 by
  register × model — where divergence lives and how (little) adversarial prompting moves it.
- **Fig. 3** (`fig3_model_divergence.pdf`): top-16 features where GPT and Mistral differ most
  (signed *dz*, coloured by which model is higher) — RQ3. Shows the divergence is length-driven
  under adversarial prompts: GPT stays long, Mistral over-compresses (more function words/periods).
- **Fig. 4** (`fig4_sentiment_polarity.pdf`): median signed sentiment polarity by condition vs a
  human reference line — RQ4. GPT's baseline positivity (+0.12) and the overshoot below human
  under adversarial prompts; Mistral tracks human throughout.

## Notes / caveats to carry into the paper
- One generation per prompt per model; large *n* → lead with effect sizes.
- Feature families: stylometric surface, syntactic complexity, pragmatic/hedge markers, NER,
  readability (textstat), POS proportions, **plus affective sentiment** (KBLab Swedish model,
  informal register only; RQ4). Zero-shot **emotion** (7 categories) was deferred for time, and
  sentence-embedding / topic (BERTopic) layers are excluded. Sentiment rests on a single
  classifier — validate on a sample and report the model.
- GPT-SW3 (Swedish-native) is future work (gated access; see model_comparison METHODS).
