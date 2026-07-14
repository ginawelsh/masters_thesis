"""Generate the four prompt conditions for the formal corpus under identical settings.

All four are independent one-shot generations (no chaining) over the same title+keywords,
so length/style differences reflect the prompt, not model/run drift:

  baseline         - neutral control prompt, no instruction about sounding human
  human_like       - naive adversarial: "write as human-like as possible"
  detector_aware   - adversarial targeting the stylometric/detection signals
                     (sentence-length variation, no formulaic connectives, less hedging)
  detector_evasive - adversarial tuned to invert this study's measured signals
                     (reuse key terms verbatim, short sentences / more periods / fewer
                     commas, no dashes, minimal hedging, concrete named entities)

Prompts + model access come from the shared ../../generation_pipeline.py, so this
script, the comments script, and smoke_test.py build byte-identical prompts and can
swap models the same way. Pick the backend with GEN_MODEL (default "gpt-5.2"; e.g.
"mistral").

Input:  sv_abstracts_openai_2.csv  (has Title, Keywords, Abstract).
Output: generated_three_prompt_formal_14_07_26.csv, with four generated columns added:
  Abstract_baseline, Abstract_human_like, Abstract_detector_aware, Abstract_detector_evasive
The older-run Generated_OpenAI_Abstract column is dropped on write; Abstract_baseline
(freshly generated under identical settings) is the sole baseline. Generated text is
normalized to a single block to match the human abstracts' formatting.

Output/cache names are model-aware so a second model never clobbers the first (the
default gpt-5.2 keeps the existing _gen_cache_adversarial.csv so its completed run
resumes for free). Generation is cached per (row, condition) so the run can resume
after an interruption without paying for completed cells again.
"""
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# force UTF-8 stdout so printing Swedish / special chars can't crash the run on a cp1252 console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# import the shared pipeline from src/1_data_collection/ (three levels up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import generation_pipeline as pipe  # noqa: E402

REGISTER = "formal"
MODEL = os.environ.get("GEN_MODEL", "gpt-5.2")  # e.g. "mistral" to use Mistral-Small-3.2
CONCURRENCY = 8  # parallel API calls

# script is at src/1_data_collection/llm_abstracts/abstracts/ -> project root is 5 levels up
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_dir = os.path.join(_root, "src", "1_data_collection", "llm_abstracts", "abstracts")
_in_path = os.path.join(_dir, "sv_abstracts_openai_2.csv")

# model-aware output/cache names (default gpt-5.2 reuses the existing cache to resume free)
_safe_model = MODEL.replace("/", "-").replace(":", "-")
_stem = "generated_three_prompt_formal_14_07_26"
_out_name = f"{_stem}.csv" if MODEL == "gpt-5.2" else f"{_stem}_{_safe_model}.csv"
_out_path = os.path.join(_dir, _out_name)
_cache_name = "_gen_cache_adversarial.csv" if MODEL == "gpt-5.2" else f"_gen_cache_adversarial_{_safe_model}.csv"
_cache_path = os.path.join(_dir, _cache_name)

# all four conditions are generated here under identical settings (independent one-shot calls)
CONDITIONS = ["baseline", "human_like", "detector_aware", "detector_evasive"]

_cache_lock = threading.Lock()


def _norm(text):
    """Collapse newlines/whitespace to a single block, matching the human abstracts' format."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _load_cache():
    if not os.path.exists(_cache_path):
        return {}
    df = pd.read_csv(_cache_path, encoding="utf-8")
    return {(int(r["row"]), r["condition"]): r["abstract"] for _, r in df.iterrows()}


def _append_cache(row_i, condition, abstract):
    line = pd.DataFrame([{"row": row_i, "condition": condition, "abstract": abstract}])
    with _cache_lock:  # serialize appends from worker threads
        line.to_csv(_cache_path, mode="a", header=not os.path.exists(_cache_path),
                    index=False, encoding="utf-8")


if __name__ == "__main__":
    data = pd.read_csv(_in_path, encoding="utf-8")
    n = len(data)
    print(f"Loaded {n} rows from {_in_path}")
    print(f"Model: {MODEL} | register: {REGISTER}")

    pipe.get_client(MODEL)  # fail fast if the model is unknown or its API key is missing

    cache = _load_cache()
    if cache:
        print(f"Resuming: {len(cache)} cached cells found")

    for cond in CONDITIONS:
        data[f"Abstract_{cond}"] = pd.NA

    # collect the (row, condition) cells that still need generating (cached cells are skipped)
    tasks = []
    for i in range(n):
        t, k = data["Title"].iloc[i], data["Keywords"].iloc[i]
        if pd.isna(t) or not str(t).strip():
            continue
        for cond in CONDITIONS:
            if (i, cond) not in cache:
                tasks.append((i, cond, t, k))

    print(f"{len(tasks)} cells to generate with {CONCURRENCY} parallel workers "
          f"({len(cache)} already cached)")

    def _work(task):
        i, cond, t, k = task
        rec = pipe.generate(MODEL, cond, REGISTER, {"id": i, "title": t, "keywords": k})
        _append_cache(i, cond, rec.output)  # persist immediately so progress survives interruption
        return i, cond, rec.output

    done = 0
    if tasks:
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futures = [ex.submit(_work, t) for t in tasks]
            for fut in as_completed(futures):
                try:
                    i, cond, text = fut.result()
                except Exception as e:
                    print(f"  [error] a cell failed after retries: {e}", flush=True)
                    continue
                cache[(i, cond)] = text
                done += 1
                if done % 25 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} generated", flush=True)

    # fill output columns from cache; normalize to a single block to match the human abstracts
    for (i, cond), abstract in cache.items():
        col = f"Abstract_{cond}"
        if 0 <= i < n and col in data.columns:
            data.loc[data.index[i], col] = _norm(abstract)

    # drop the superseded older-run baseline; Abstract_baseline is now the sole baseline
    data = data.drop(columns=["Generated_OpenAI_Abstract"], errors="ignore")

    data.to_csv(_out_path, index=False, encoding="utf-8")
    print(f"\nFinished. Saved {n} rows (+{len(CONDITIONS)} generated cols) to {_out_path}")
