# CLUU Thesis — Computational Linguistics Analysis

Thesis project comparing human-authored and LLM-generated Swedish text across formal
(academic abstracts) and informal (Reddit/Flashback comments) registers.

The LLM side is generated under **three prompt conditions** so the analysis can separate a
neutral baseline from two *adversarial* prompts designed to evade AI-text detection:

| Condition | Prompt intent |
|---|---|
| `baseline` | Neutral instruction — write an abstract/comment from the title/keywords/question. Control. |
| `human_like` | Adversarial — "write as human as possible, indistinguishable from a person." |
| `detector_evasive` | Adversarial — tuned to *invert this study's measured signals*: reuse key terms verbatim, avoid nominal compression, more/shorter sentences, minimal hedging, no dashes, concrete names. |

The full prompts (Swedish + English) for both registers are in the thesis appendix (`thesis/appendix_prompts.tex`).

> A fourth condition, `detector_aware` (literature-driven evasion), remains in the code and
> data (`data_utils.py`, the `*_detector_aware` columns) but is **excluded from the reported
> analysis**.

## Repository structure

```
src/
├── 1_data_collection/
│   ├── human_abstracts/        Raw human abstracts (master_human_theses.csv) + source spreadsheet
│   ├── llm_abstracts/abstracts/ LLM abstracts + generator (sv_abstracts_adversarial.csv)
│   ├── human_comments/         Human Reddit & Flashback comments + scrapers
│   ├── llm_comments/           LLM comment generation (consolidated_informal_comments_JUN26.csv)
│   └── analysis_runs/          Corpus manifests + multi-layer outputs (formal/, informal/)
├── 2_text_analysis_scripts/
│   ├── scripts/                Analysis scripts (see below)
│   ├── csv_files/              Per-document feature CSVs + distribution summaries
│   ├── results/                spaCy linguistic outputs (formal/, informal/, + condition subdirs)
│   └── figures/                Generated plots
├── 3_ai_detection_scripts/     AI detection experiments
└── 4_archive/                  Superseded scripts, data, and backups (not part of the pipeline)
```

Raw inputs live under `1_data_collection/`; all analysis outputs are written under
`2_text_analysis_scripts/` (they are not mixed into the raw-data folders).

## Data

| Group | Source | n |
|---|---|---|
| Formal Human | Swedish academic abstracts | 170 |
| Formal LLM | GPT abstracts × 3 reported conditions | 170 each |
| Informal Human | Reddit + Flashback comments | 1,149 |
| Informal LLM | GPT comment responses × 3 reported conditions | 1,149 each |

Informal data (post-exclusion): 1,149 comments = 197 Reddit + 952 Flashback, across 136
question threads (59 Reddit, 77 Flashback). 11 link/image question threads (73 comments) were
excluded.

**Canonical files**

- Formal: `1_data_collection/llm_abstracts/abstracts/sv_abstracts_adversarial.csv` — one row per
  thesis with `Abstract` (human) plus `Abstract_baseline`, `Abstract_human_like`,
  `Abstract_detector_aware`, `Abstract_detector_evasive`. Human source of truth:
  `human_abstracts/master_human_theses.csv`.
- Informal: `1_data_collection/llm_comments/consolidated_informal_comments_adversarial.csv` —
  `question`, `human_comment`, `generated_comment` (baseline), plus the adversarial columns
  `comment_human_like`, `comment_detector_aware`, `comment_detector_evasive`.

All generated text is normalized to a single block (newlines collapsed) to match the
single-paragraph human abstracts, so paragraph formatting is not a confound.

Emoji are also stripped from the generated text (the human corpus, collected 2005–2017,
predates the mobile-era rise in emoji use and contains none), so emoji presence is treated
as a temporal confound rather than a human/LLM difference. Human text is left verbatim.
Stripping had negligible effect (informal baseline detectability AUC 0.963 → 0.960; no
change in any feature's significance), confirming the human/LLM separation is not an emoji
artifact.

## LLM data generation

`generate_adversarial_openai.py` (one in `llm_abstracts/abstracts/`, one in `llm_comments/`)
produces the prompt conditions as **independent one-shot generations** (no chaining) over the
same inputs. Each caches per (row, condition) and resumes after interruption.

- Formal: model `gpt-5.2`, writes `sv_abstracts_adversarial.csv`.
- Informal: model `gpt-5.2`, writes `consolidated_informal_comments_adversarial.csv`.

```
python src/1_data_collection/llm_abstracts/abstracts/generate_adversarial_openai.py
python src/1_data_collection/llm_comments/generate_adversarial_openai.py
```

Requires `OPENAI_API_KEY` (environment variable or `.env` in the repo root).

## Analysis scripts

Every analysis script accepts `--dataset {formal,informal}` and
`--condition {baseline,human_like,detector_aware,detector_evasive,all}` (default `all`).
Both registers carry all conditions in the code/data; the analysis **reported** in the
thesis and paper uses `baseline`, `human_like` and `detector_evasive` (`detector_aware`
is generated but not reported).

Naming/layout by condition:
- Per-document feature CSVs (`csv_files/`) are named `<feature>_<dataset>_<condition>.csv`
  for every condition (e.g. `stylometric_surface_informal_detector_evasive.csv`).
- spaCy linguistic outputs: one directory per condition — `results/<dataset>/<condition>/`.
- Aggregate stats (`distribution_distances`, `significance_tests`) run per condition and tag
  each output row with a `condition` column.

| Script | Output | Features |
|---|---|---|
| `spacy_linguistic_analysis.py` | `results/formal/<condition>/linguistic_analysis_*.csv` | POS, dependency & NER distributions, tokens, POS proportions/differences |
| `syntactic_complexity.py` | `csv_files/syntactic_complexity_*.csv` | Dependency distance, parse-tree depth, subordinate-clause rate |
| `stylometric_surface.py` | `csv_files/stylometric_surface_*.csv` | N-gram repetition, punctuation, function-word rate, vocabulary richness (TTR/MATTR/MTLD) |
| `pragmatic_markers.py` | `csv_files/pragmatic_markers_*.csv` | Hedges, negation, epistemic markers (+ boosters & discourse particles for informal) |
| `word_frequency_analysis.py` | `word_freq_*.csv/.png` | Top-N lemma frequencies, vocabulary overlap, Venn diagrams |
| `embedding_comparison.py` | `csv_files/embedding_*_*.csv` | Per-pair cosine similarity + classifier two-sample test (ROC AUC, permutation p) |
| `distribution_distances.py` | `csv_files/distribution_distances_*.csv` | Jensen–Shannon divergence, Wasserstein distance (per condition) |
| `significance_tests.py` | `csv_files/significance_tests_results.csv` | Paired Wilcoxon + effect sizes (rank-biserial, Cohen's dz), BH-FDR within each (dataset, condition) |
| `multi_layer_analysis.py` | `analysis_runs/` | Morphology (spaCy), sentence-transformer embeddings + PCA, readability. BERTopic is opt-in via `--bertopic`. |
| `plot_condition_comparison.py` | `figures/` | Comparison PNGs across Human + the 3 LLM conditions (see Visualization) |

### Formal-register feature scope

Some features have near-zero variance in academic abstracts (register floor effects) and are
**excluded a priori** from the formal analysis, while retained for the informal register where
they vary:

- **Affective analysis** (sentiment/emotion) — informal only (script archived from the formal pipeline).
- **Pragmatic markers** — boosters and discourse particles are dropped for formal (interpersonal/
  spoken-register markers); hedges, negation and epistemic markers are kept.
- **Stylometric punctuation** — expressive marks (`!`, `?`, `…`) are dropped for formal.

## Running the analysis

Install dependencies:
```
pip install -r src/2_text_analysis_scripts/requirements.txt
python -m spacy download sv_core_news_lg
```

Run any script from the repo root (defaults to all three formal conditions):
```
python src/2_text_analysis_scripts/scripts/spacy_linguistic_analysis.py --dataset formal
python src/2_text_analysis_scripts/scripts/syntactic_complexity.py --dataset formal
python src/2_text_analysis_scripts/scripts/stylometric_surface.py --dataset formal
python src/2_text_analysis_scripts/scripts/pragmatic_markers.py --dataset formal
python src/2_text_analysis_scripts/scripts/word_frequency_analysis.py --dataset formal
python src/2_text_analysis_scripts/scripts/embedding_comparison.py --dataset formal
python src/2_text_analysis_scripts/scripts/distribution_distances.py --dataset formal
python src/2_text_analysis_scripts/scripts/significance_tests.py --dataset formal
python src/2_text_analysis_scripts/scripts/multi_layer_analysis.py --register formal
```

Restrict to one condition with e.g. `--condition detector_aware`. Use `--dataset informal`
for the comment register. The extended `multi_layer_analysis.py` stack is in
`requirements_analysis.txt` (Stanza is no longer required; BERTopic only if you pass `--bertopic`).

## Visualization

`plot_condition_comparison.py` renders PNGs that compare **Human** against all three LLM
conditions (baseline, human-like, detector-aware) on one canvas, reading the per-condition
feature CSVs and `significance_tests_results.csv`:

```
python src/2_text_analysis_scripts/scripts/plot_condition_comparison.py --dataset formal
```

Outputs to `figures/`: grouped box/bar panels per feature (4 groups side by side), an
effect-size (Cohen's dz) heatmap of feature × condition, and an embedding-detectability
(ROC AUC) bar chart across conditions.

## NER notes

NER uses the Swedish SUC tagset via `sv_core_news_lg`: PRS (person), ORG (organisation),
LOC (location), TME (time), MSR (measure), WRK (work of art), OBJ (object).

Values are **rates per 100 content tokens**, not proportions — values above 1.0 are expected.
Two outputs are generated: full sample and a filtered version (`min20`) keeping only informal
pairs where both texts have ≥ 20 content tokens, to reduce inflation from very short comments
(median informal comment: 31 tokens, Q1: 15 tokens).

## Archive

`src/4_archive/` holds superseded material kept for provenance but not part of the pipeline:
- `stale_scripts/` — `affective_analysis.py`, `sentiment_analysis.py`, `all_ling_analysis_plot.py`,
  `generate_text_openai.py`.
- Old data backups, human-collection intermediates, and duplicate linguistic outputs.
