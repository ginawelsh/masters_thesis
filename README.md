# CLUU Thesis — Computational Linguistics Analysis

Thesis project comparing human-authored and LLM-generated Swedish text across formal (academic abstracts) and informal (Reddit/Flashback comments) registers.

## Repository structure

```
src/
├── 1_data_collection/
│   ├── human_abstracts/        Human-authored academic abstracts
│   ├── llm_abstracts/          LLM-generated abstracts (OpenAI)
│   ├── human_comments/         Human Reddit & Flashback comments
│   └── llm_comments/           LLM-generated comment responses
├── 2_text_analysis_scripts/
│   ├── scripts/                Analysis scripts (see below)
│   ├── figures/                Generated plots and CSVs
│   └── csv_files/              Summary distribution CSVs
└── 3_ai_detection_scripts/     AI detection experiments
```

## Data

Four groups compared throughout:

| Group | Source | n |
|---|---|---|
| Formal Human | Swedish academic abstracts | ~200 |
| Formal LLM | GPT-generated abstracts | ~200 |
| Informal Human | Reddit + Flashback comments | ~1,222 |
| Informal LLM | GPT-generated comment responses | ~1,222 |

Informal data: 208 Reddit comments + 1,014 Flashback comments (84 threads).

## Analysis scripts

### Feature coverage (Table 3.3)

Organised by output CSV. All scripts accept `--dataset formal` or `--dataset informal`.

**`syntactic_complexity_{dataset}.csv`** — `syntactic_complexity.py`

| Feature |
|---|
| Dependency distance |
| Parse tree depth |
| Subordinate-clause rate |

**`stylometric_surface_{dataset}.csv`** — `stylometric_surface.py`

| Feature |
|---|
| N-gram repetition rate |
| Word length distribution |
| Punctuation patterns |
| Function-word frequencies |
| Vocabulary richness (TTR, MATTR, MTLD) |

**`pragmatic_markers_{dataset}.csv`** — `pragmatic_markers.py`

| Feature |
|---|
| Hedging markers |
| Boosters/intensifiers |
| Negation rate |
| Epistemic markers |

**`affective_analysis_{dataset}.csv`** — `affective_analysis.py`

| Feature |
|---|
| Affective extremity / polarity magnitude |
| Subjectivity vs objectivity |
| Emotion categories |

**`distribution_distances_{dataset}.csv`** — `distribution_distances.py`

| Feature |
|---|
| Jensen–Shannon divergence |
| Wasserstein distance |

**`figures/data_{pos,dep,ner,token}*.csv`** — `all_ling_analysis_plot.py`

| Feature |
|---|
| POS distribution |
| Dependency distribution |
| NER distribution |
| Sentence length distribution |

### Additional scripts

| Script | Purpose |
|---|---|
| `word_frequency_analysis.py` | Top-N lemma frequencies, vocabulary overlap, Venn diagrams |
| `sentiment_analysis.py` | Standalone KBLab pos/neutral/neg labelling (earlier version of affective analysis) |
| `multi_layer_analysis.py` | Stanza morphology, sentence-transformer embeddings + PCA, BERTopic topic modelling, textstat readability |
| `distribution_distances.py` | Jensen–Shannon divergence and Wasserstein distance across all feature distributions |
| `all_ling_analysis_plot.py` | Master plotting script — aggregates spaCy POS/DEP/NER/token-length across all four groups |

## Running the analysis

Install dependencies:
```
pip install -r src/2_text_analysis_scripts/requirements.txt
python -m spacy download sv_core_news_lg
```

Main linguistic analysis (POS, DEP, NER, token length):
```
python src/2_text_analysis_scripts/scripts/all_ling_analysis_plot.py
python src/2_text_analysis_scripts/scripts/all_ling_analysis_plot.py --skip-sentiment
```

Individual feature scripts (run from repo root):
```
python src/2_text_analysis_scripts/scripts/syntactic_complexity.py
python src/2_text_analysis_scripts/scripts/stylometric_surface.py
python src/2_text_analysis_scripts/scripts/pragmatic_markers.py
python src/2_text_analysis_scripts/scripts/affective_analysis.py
python src/2_text_analysis_scripts/scripts/distribution_distances.py
```

## NER notes

NER uses the Swedish SUC tagset via `sv_core_news_lg`: PRS (person), ORG (organisation), LOC (location), TME (time), MSR (measure), WRK (work of art), OBJ (object).

Values are **rates per 100 content tokens**, not proportions — values above 1.0 are expected. Two outputs are generated: full sample and a filtered version (`min20`) keeping only informal pairs where both texts have ≥ 20 content tokens, to reduce inflation from very short comments (median informal comment: 31 tokens, Q1: 15 tokens).
