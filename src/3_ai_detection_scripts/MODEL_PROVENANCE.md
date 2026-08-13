# Detection run provenance

Records which model produced each `detection_results_<provider>.csv` and when it
was run. Needed because two of the three providers are called via **rolling
aliases** that do not pin a fixed model version — re-running later can silently hit
a different model. Dates are the results file's last-written (mtime) timestamp.

## Runs and their corpora

There are now four distinct runs in different directories. **The earlier version of
this file documented only the n=56 pilot, which feeds no reported table.** The
figures in the thesis tables come from `gpt52_run_results/` and the top-level files.

| Run | Corpus | n | Output location | Date |
|---|---|---|---|---|
| Pilot | `quiz_master.csv` | 56 | `old_n56_results/` | 2026-07-22..24 |
| **GPT-5.2 main** | `quiz_master_balanced.csv` | 800 | `gpt52_run_results/` | 2026-07-26 |
| **GPT-5.6 / Mistral informal** | `quiz_followup.csv` | 706 | top level | 2026-08-01..03 (complete) |
| **Formal normalized** | `formal_cleaned_quiz.csv` | 998 | `formal_cleaned_results/` | 2026-08-02 (incomplete) |

Model strings called (all runs use the same `PROVIDERS` config):

| Provider | Model string | Reproducible? |
|---|---|---|
| Claude | `claude-opus-4-8` | **Yes** — dated/versioned model ID, stable. |
| Gemini | `gemini-3.1-pro-preview` | Partly — `-preview` is a moving pointer, not frozen. |

**Gemini informal run spans three days (quota).** `gemini-3.1-pro-preview` is capped at
250 requests/day on this key, so the 706-item informal run was completed across
several quota-limited sessions: 235 items in the first, +231 to reach 466 (the state
committed in `ba15359`), +208 to reach 674 (file mtime 2026-08-02 11:27), and finally
+32 to reach 706 on 2026-08-03. Every session called the same model string, and a probe on
2026-08-03 echoed `resp.model == "gemini-3.1-pro-preview"`, so the served name did not
change over the window — but because `-preview` is a moving pointer this does not
guarantee identical weights across the three days. The 32 items finished on 08-03 are
all Mistral informal, so if the pointer did move, the drift would land inside a single
generator's cells rather than being spread evenly across the design. No `--model`
override was used at any point (a fallback to `gemini-3.6-flash` would have mixed two
models within one run; it was deliberately avoided).
| DeepSeek | `deepseek-reasoner` | **No** — rolling alias; see below. |

## DeepSeek caveat (important)

`deepseek-reasoner` is an alias DeepSeek repoints to their current reasoning model.

- **2026-07-22 (pilot):** resolved to a V3-generation reasoner. Exact snapshot
  **not recoverable**.
- **2026-07-24:** alias moved to **`deepseek-v4-flash`**; `/models` listed only
  `deepseek-v4-flash` and `deepseek-v4-pro`.
- **2026-08-02 (re-probed):** still **`deepseek-v4-flash`**; `/models` unchanged.
  So the 2026-07-26 n=800 run and the 2026-08-02 formal-normalized run most likely
  both used `deepseek-v4-flash` — but note `deepseek-v4-flash` is itself a label
  DeepSeek can update in place, so this is not a guarantee of identical weights.

**Cite as:** "deepseek-reasoner — DeepSeek's hosted reasoning model as accessed
&lt;date&gt;" (do not claim a specific V-number for the pilot run; for the 07-26 and
later runs `deepseek-v4-flash` is supported by the probes above).

### Correction: DeepSeek *does* echo the resolved model

The earlier version of this file stated that "the API only ever echoes the literal
string `deepseek-reasoner`, so the underlying version is not logged". **That is
false as of 2026-08-02.** A probe call sent with `model="deepseek-reasoner"`
returned `resp.model == "deepseek-v4-flash"` — the resolved name, not the alias.

Consequence: the "make future runs reproducible" fix below applies to **all three**
providers, DeepSeek included. There is no technical obstacle to self-documenting
DeepSeek runs.

## To make future runs reproducible

Capture the served model per item: `resp.model` (OpenAI SDK, works for DeepSeek and
Gemini) and the Anthropic response `model` field, written into a `served_model`
column by `ai_detection_llm.py`. None of the existing results CSVs carry this, so
the model behind every completed run is inferred from dated probes rather than
recorded — which is why the table above has to hedge.

## Open question this provenance bears on

DeepSeek's formal GPT-5.2 detection recall is **58%** in the 2026-07-26 n=800 run
but **~88%** in the 2026-08-02 formal-normalized run. Text normalization does not
explain it (it changes only 13% of those documents, and 86.7% are byte-identical),
and the judge prompt is unchanged. Provider-side drift was the leading hypothesis,
but both runs post-date the 07-24 move to `deepseek-v4-flash`, which weakens it
unless `deepseek-v4-flash` was itself updated in place. **The control that settles
it:** re-score the original *uncleaned* GPT-5.2 formal items with today's DeepSeek.
Same model string, same prompt, different corpus version — if recall returns ~88%,
the difference is drift, not normalization. Not yet run.
