# Detection run provenance

Records which model produced each `detection_results_<provider>.csv` and when it
was run. Needed because two of the three providers are called via **rolling
aliases** that do not pin a fixed model version — re-running later can silently hit
a different model. Dates are the results file's last-written (mtime) timestamp.

Corpus: `quiz_master.csv` (n=56), last built 2026-07-21.

| Provider | Model string called | Run date | Reproducible? |
|---|---|---|---|
| Claude | `claude-opus-4-8` | 2026-07-23 | **Yes** — dated/versioned model ID, stable. |
| Gemini | `gemini-3.1-pro-preview` | 2026-07-24 | Partly — `-preview` is a moving pointer, not frozen. |
| DeepSeek | `deepseek-reasoner` | 2026-07-22 | **No** — rolling alias; exact snapshot not captured. |

## DeepSeek caveat (important)

`deepseek-reasoner` is an alias DeepSeek repoints to their current reasoning model;
the API only ever echoes the literal string `deepseek-reasoner`, so the underlying
version is not logged in the response or the results CSV.

- **At run time (2026-07-22):** resolved to DeepSeek's then-current reasoning model
  (a V3-generation reasoner). Exact snapshot **not recoverable**.
- **Checked 2026-07-24:** the alias now resolves to **`deepseek-v4-flash`** — the
  `/models` endpoint lists only `deepseek-v4-flash` and `deepseek-v4-pro`, and a
  probe call echoed `deepseek-v4-flash`. This is a *newer, different* model than the
  2026-07-22 run used.

**Cite as:** "deepseek-reasoner — DeepSeek's hosted reasoning model as accessed
2026-07-22" (do not claim a specific V-number for the run).

## To make future runs reproducible

Log the resolved model per item. DeepSeek won't expose a dated ID behind the alias,
but for Gemini/Claude the response carries the served model — capture
`resp.model` (OpenAI SDK) / the Anthropic response `model` field into a column when
scoring, so the CSV is self-documenting.
