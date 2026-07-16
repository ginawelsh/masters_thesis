# Mistral temperature A/B (0.15 vs 1.0) — sub-study

**Question.** Does sampling temperature materially change Mistral's Swedish output on the
stylometric/linguistic features used in the main human-vs-generated analysis? The pipeline
generates at temperature 1.0 (explicit, cross-backend invariant); Mistral's model card
recommends ~0.15. This checks whether that deviation matters.

**Verdict: no — negligible to small, and only on lexical diversity.** Across 532
feature-tests, 522 show a negligible effect (|Cohen's dz| < 0.2) and 10 a small effect
(0.2–0.5); none reach medium (≥0.5). Max effect anywhere: dz = 0.28. The only features
temperature nudges are the lexical-diversity/repetition metrics (MTLD, MATTR up at 1.0;
bi/tri-gram repetition down) — the direct mechanical signature of temperature — plus a tiny
ellipsis bump. Syntax, pragmatics/hedging, NER, POS proportions and readability are flat.

## Data & generation

- **Model:** `mistral-small-2506` via Mistral La Plateforme (`https://api.mistral.ai/v1`),
  called through `src/1_data_collection/generation_pipeline.py`.
- **Invariant preserved:** single user message, **no system prompt**, temperature the ONLY
  thing varied (0.15 vs 1.0). Mistral's recommended system prompt was deliberately not applied.
- **Paired design:** for every human document (row) the same prompt is generated at both
  temperatures, so 0.15 and 1.0 are compared pair-by-pair (by `register, doc_id, condition`).
- **Scope:** 1,149 informal comment rows + 170 formal abstract rows = 1,319 base docs ×
  4 conditions (baseline, human_like, detector_aware, detector_evasive) × 2 temps = **10,552
  generations**; **5,276 fully paired** docs (4,596 informal + 680 formal).
- Informal prompts come from `src/4_archive/consolidated_informal_comments_JUN26.csv`
  (`question` column); formal from `.../llm_abstracts/abstracts/sv_abstracts_openai_2.csv`
  (`Title` + `Keywords`).

## Feature extraction (faithful reuse of the main pipeline)

Each generated text is scored with the **actual** `scripts/` extractors, so numbers are
computed exactly as in the human-vs-generated study (not re-implemented):

| family | source function | notes |
|---|---|---|
| stylometric surface | `stylometric_surface.compute_metrics` | TTR, MATTR, MTLD, bi/tri-gram repetition, function-word rate, punctuation rates (`drop_expressive=True` for formal) |
| syntactic complexity | `syntactic_complexity.compute_metrics` | dep distance, tree depth, subordinate-clause rate (`min_tokens=30` formal / 0 informal) |
| pragmatic markers | `pragmatic_markers.compute_metrics` | hedge (+register split), booster, negation, epistemic, discourse particles (`FORMAL_MATCHERS` for formal) |
| NER rates | `spacy_linguistic_analysis.entity_counts_string` → `ner_features.doc_rates` | per-100-token rates by entity type |
| textstat | replicated from `multi_layer_analysis` | char/word/syllable counts + readability indices |
| POS proportions | UPOS counts / total | per-tag proportions |

Analyzed the **generated text alone** (question not prepended for informal): the prompt is
identical across temperatures, so prepending it would only dilute the temperature signal.
→ 70 features/doc informal, 63/doc formal (formal drops the 3 expressive-punctuation features
and booster/discourse-particle pragmatics). spaCy model: `sv_core_news_lg`.

## Statistics

Same machinery as the main study (`significance_tests.py`), comparing temp 0.15 vs 1.0 as the
two paired groups:

- **Paired Wilcoxon signed-rank** per feature (`test_feature`).
- **Cohen's dz** (paired) and **matched-pairs rank-biserial** effect sizes.
- **Benjamini–Hochberg FDR** applied within each (register × condition) family.

## Results

**Effect-size distribution (all 532 tests):** negligible (|dz|<0.2) = 522; small (0.2–0.5) = 10;
medium/large = 0. **Max |dz| = 0.28.** Of 92 FDR-significant features, only 8 reach |dz| ≥ 0.2.

**Top movers (all in stylometric surface, direction = 1.0 vs 0.15):**
MATTR/MTLD +0.21…+0.28 (more lexical variety at 1.0); bi/tri-gram repetition −0.19…−0.26
(less repetition at 1.0); function-word rate −0.18; informal ellipsis +0.22.

**Registers:** the effect is the same small magnitude in both; formal shows only 2 significant
features (n=170, low power) vs ~20/condition informal (n=1,149) — a sample-size artifact, not a
larger effect.

**Large-n caveat (important for reporting):** at n=1,149 many features clear FDR while being
practically meaningless (|dz| < 0.1). Lead with effect sizes, not p-values. Consistent with the
project's existing effect-size heatmap approach.

## Caveats

- One draw per row per temperature (temperature variance is captured across the 5,276 pairs,
  not by repeated draws of one prompt).
- `affective_analysis` family excluded (external script; no data file produced here).
- Sentence-embedding PCA and BERTopic excluded by scope (semantic/content layers, topic-matched,
  not temperature-relevant).

## Reproduce

```bash
# 1. generate (needs MISTRAL_API_KEY etc. in project-root .env); resumable
python src/2_text_analysis_scripts/mistral_temperature_ab/generate_mistral_temps.py
# 2. extract features + paired significance
python src/2_text_analysis_scripts/mistral_temperature_ab/extract_and_test.py
```

## Files
- `generate_mistral_temps.py` — generation (paired, resumable).
- `extract_and_test.py` — feature extraction + paired BH-FDR significance.
- `generated_corpus_mistral_temps.csv` — raw generated corpus (10,552 rows: register, doc_id,
  condition, temp, text).
- `significance_results.csv` — per-feature results (532 rows: register, condition, feature_group,
  feature, n_pairs, medians, direction, cohens_dz, rank_biserial, p_value, p_fdr_bh,
  significant_fdr_0.05).
