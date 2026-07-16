# Model comparison: GPT-5.2 vs Mistral (mistral-small-2506) — sub-study

**Question.** Adding Mistral as a second generator: how does its Swedish output compare to
GPT-5.2 on the stylometric/linguistic feature suite — (a) is it more or less human-like than
GPT, and (b) where do the two models differ from each other?

**Design.** All three sources answered the SAME prompts (same rows), so every comparison is
**paired by row**. Three paired comparisons per register × condition:
`human_vs_gpt`, `human_vs_mistral`, `mistral_vs_gpt`. Mistral is taken at **temperature 1.0**
to match GPT's pinned temperature (justified by the temperature sub-study, which showed Mistral
is temperature-stable). **No new generation** — GPT + human text from the existing adversarial
CSVs; Mistral@1.0 from `../mistral_temperature_ab/generated_corpus_mistral_temps.csv`.

**Features & stats.** Identical to the temperature sub-study / main pipeline: `stylometric_surface`,
`syntactic_complexity`, `pragmatic_markers`, `ner`, `textstat`, `pos_proportions` (raw text, no
question prepend). Paired Wilcoxon signed-rank + Cohen's dz + matched-pairs rank-biserial + BH-FDR,
per (register × condition × comparison) family. 532 feature-tests per comparison.

## Findings

### 1. Which model is more human-like? (median |dz| from human; lower = more human-like)

| register | condition | human↔GPT | human↔Mistral | more human-like |
|---|---|--:|--:|:--|
| formal | baseline | 0.48 | **0.32** | Mistral |
| formal | detector_aware | 0.56 | **0.40** | Mistral |
| formal | detector_evasive | 0.45 | **0.29** | Mistral |
| formal | human_like | 0.52 | **0.29** | Mistral |
| informal | baseline | 0.16 | 0.16 | ~tie (Mistral) |
| informal | detector_aware | 0.17 | 0.24 | GPT |
| informal | detector_evasive | 0.20 | 0.20 | ~tie |
| informal | human_like | 0.20 | 0.21 | ~tie (GPT) |

- **Formal (abstracts): Mistral is clearly more human-like in all four conditions** — GPT diverges
  from human at a *medium* effect (~0.5), Mistral only *small* (~0.3). Head-to-head, Mistral is
  closer to human on **61%** of formal features.
- **Informal (comments): essentially tied**, both *small* divergence (0.16–0.24). Marginal:
  Mistral closer in baseline/detector_evasive, GPT closer in detector_aware/human_like. Head-to-head
  48% Mistral-closer. Both models mimic short forum comments well.

### 2. How each model diverges from human (top significant features)

- **GPT** (worst in formal/detector_evasive): readability indices (SMOG, gunning-fog, flesch-kincaid)
  and sentence/period counts far from human — GPT over-complies with "short sentences / more periods",
  producing very choppy abstracts unlike human writing.
- **Mistral**: over-punctuation vs human (PUNCT proportion, periods, commas) in formal; in informal
  detector_aware, more hedges and *lower* lexical diversity (TTR/MATTR) than human.

### 3. Where the two models differ from each other (`mistral_vs_gpt`)

They are **not interchangeable**: median |dz| = 0.30, but **183/532 features reach medium+ (|dz|≥0.5)
and 99 reach large (≥0.8)**. Largest differences are under the adversarial conditions, especially formal:
**Mistral writes much shorter than GPT** (word/char/token/sentence counts, |dz| up to 2.4),
with **more function words and more periods**. Mistral follows the "short, dense, more periods"
evasive instructions more aggressively than GPT — a genuine instruction-following difference.

## Takeaways
- Adding Mistral is worthwhile: it produces genuinely different text from GPT and is **more human-like
  in the formal register** — directly relevant to a detection thesis (a more-human-like generator is
  harder to detect).
- Model differences are largest exactly where the adversarial prompts bite hardest — so the
  evasion effect is model-dependent, not universal.
- Effect sizes here are large and real (unlike the temperature sub-study, where all effects were
  negligible) — as expected, model identity matters far more than sampling temperature.

## Caveats
- One generation per row per model; large n (1,149 informal / 170 formal) → lead with effect sizes.
- `affective_analysis`, sentence-embedding PCA, and BERTopic excluded (as in the temperature sub-study).
- dz magnitude used for "distance"; direction read from human vs model medians.

## Files
- `model_comparison.py` — extraction + 3-way paired significance (reproduces this).
- `model_comparison_results.csv` — comparison, register, condition, feature_group,
  feature, n_pairs, median_A, median_B, cohens_dz, rank_biserial, p_value, p_fdr_bh, significant_fdr_0.05.
  For a comparison labelled `A_vs_B`, `median_A` is the first generator and `median_B` the second
  (e.g. `human_vs_gpt` → A=human, B=GPT; `sw3_vs_mistral` → A=GPT-SW3, B=Mistral).

## Adding a third generator: GPT-SW3 (Swedish-native, open)

`model_comparison.py` is generator-agnostic. Drop a GPT-SW3 corpus into this folder as
`generated_corpus_gpt-sw3.csv` (schema: `register, doc_id, condition, temp, text`, temp=1.0,
same `doc_id` row-indexing as the Mistral corpus) and re-run `model_comparison.py`; it
auto-detects the file and adds `human_vs_sw3`, `sw3_vs_gpt`, `sw3_vs_mistral` to the existing
comparisons. If the file is absent it silently runs the GPT/Mistral comparisons only.

**Producing that corpus (Colab):** GPT-SW3 isn't a hosted API, so generate it on a GPU via
`gpt_sw3_generate_colab.ipynb`:
1. Colab Runtime → GPU (T4 for `gpt-sw3-6.7b-v2-instruct`; L4/A100 on Colab Pro for
   `gpt-sw3-20b-instruct` — better instruction-following and closer in scale to Mistral-24B,
   which reduces the capability confound).
1b. GPT-SW3 is a **gated** HF model: accept the terms on the model page
   (huggingface.co/AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct) and create a read token; the
   notebook's auth cell logs in (store the token in Colab Secrets as `HF_TOKEN`).
2. Upload just the two CSVs — `consolidated_informal_comments_JUN26.csv` and
   `sv_abstracts_openai_2.csv` — to a Drive folder; run the notebook top to bottom. (The prompt
   builder is inlined in the notebook, verified byte-identical to `generation_pipeline.build_prompt`,
   so no `.py` upload is needed.)
3. It uses the inlined `build_prompt` for byte-identical prompts and the same invariant (temp 1.0, no system
   prompt; GPT-SW3's `User:`/`Bot:` chat template is the model's required format, not an injected
   system prompt). Generation is **resumable** via a Drive-persisted cache (survives Colab
   disconnects) — important, as a full run is several hours on a free T4.
4. Smoke-test first (the notebook has a cell): the decision gate is whether GPT-SW3 actually
   follows the `detector_evasive` prompt. If not, keep it but scope to baseline/human_like and
   report the instruction-following gap as the finding.
5. Download `generated_corpus_gpt-sw3.csv` → this folder → re-run `model_comparison.py`.

**Caveat:** GPT-SW3 is older/smaller than GPT-5.2 and Mistral, so "Swedish-native" and "weaker
model" are partly entangled — using the 20b (scale-matched to Mistral-24B) mitigates this; name
it explicitly in the writeup.
