"""Add three ADVERSARIAL LLM-comment columns to the informal corpus.

The existing neutral baseline (generated_comment, produced by the basic forum prompt)
is kept as the control. This script only generates the adversarial conditions and
appends them, so the four prompts can be compared over identical inputs:

  baseline         - existing generated_comment column (control; not regenerated)
  human_like       - naive adversarial: "write as human-like as possible"
  detector_aware   - adversarial targeting the stylometric/detection signals
                     (informal tone, length variation, no AI politeness/hedging)
  detector_evasive - adversarial tuned to invert this study's measured signals
                     (reword rather than repeat, dense content, short sentences / more
                     periods / fewer commas, no dashes, minimal hedging, concrete names)

Prompts + model access come from the shared ../generation_pipeline.py, so this
script and smoke_test.py build byte-identical prompts and can swap models the
same way. Pick the backend with GEN_MODEL (default "gpt-5.2"; e.g. "mistral").

Input:  ../../4_archive/consolidated_informal_comments_JUN26.csv  (has question,
        human_comment, generated_comment). Three columns are appended:
  comment_human_like, comment_detector_aware, comment_detector_evasive
Output: generated_three_prompt_14_07_26.csv (input left untouched).

Calls run concurrently (CONCURRENCY workers) with retry/backoff (in the pipeline).
Generation is cached per (row, condition) in _gen_cache_three_prompt_14_07_26.csv
so an interrupted run resumes without paying for completed cells again. NOTE: the
cache is keyed by row index, so it is tied to THIS input file/ordering -- a fresh
cache name is used here because the input corpus changed (link/image questions were
removed), which would otherwise misalign the old _gen_cache_adversarial.csv rows.
"""
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# force UTF-8 stdout so printing Swedish / special chars can't crash the run on a cp1252 console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# import the shared pipeline from src/1_data_collection/ (one level up)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import generation_pipeline as pipe  # noqa: E402

REGISTER = "informal"
MODEL = os.environ.get("GEN_MODEL", "gpt-5.2")  # e.g. "mistral" to use Mistral-Small-3.2
CONCURRENCY = 16  # parallel API calls (retry/backoff absorbs any rate-limit 429s)

# script is at src/1_data_collection/llm_comments/ -> project root is 4 levels up
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_dir = os.path.join(_root, "src", "1_data_collection", "llm_comments")
_archive = os.path.join(_root, "src", "4_archive")
_in_path = os.path.join(_archive, "consolidated_informal_comments_JUN26.csv")

# output/cache names are model-aware so a second model never clobbers the first.
# The default gpt-5.2 run writes exactly generated_three_prompt_14_07_26.csv; any
# other model (e.g. mistral) gets a "_<model>" suffix. Cache is always per-model
# (the key is only (row, condition)), matching the existing per-model cache files.
_safe_model = MODEL.replace("/", "-").replace(":", "-")
_stem = "generated_three_prompt_14_07_26"
_out_name = f"{_stem}.csv" if MODEL == "gpt-5.2" else f"{_stem}_{_safe_model}.csv"
_out_path = os.path.join(_dir, _out_name)
_cache_path = os.path.join(_dir, f"_gen_cache_three_prompt_14_07_26_{_safe_model}.csv")

# only the ADVERSARIAL conditions are generated here (baseline is reused as-is)
CONDITIONS = ["human_like", "detector_aware", "detector_evasive"]

_cache_lock = threading.Lock()


def _load_cache():
    if not os.path.exists(_cache_path):
        return {}
    df = pd.read_csv(_cache_path, encoding="utf-8")
    return {(int(r["row"]), r["condition"]): r["comment"] for _, r in df.iterrows()}


def _append_cache(row_i, condition, comment):
    line = pd.DataFrame([{"row": row_i, "condition": condition, "comment": comment}])
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
        data[f"comment_{cond}"] = pd.NA

    # collect the (row, condition) cells that still need generating
    tasks = []
    for i in range(n):
        q = data["question"].iloc[i]
        if pd.isna(q) or not str(q).strip():
            continue
        for cond in CONDITIONS:
            if (i, cond) not in cache:
                tasks.append((i, cond, str(q)))

    print(f"{len(tasks)} cells to generate with {CONCURRENCY} parallel workers "
          f"({len(cache)} already cached)")

    def _work(task):
        i, cond, q = task
        rec = pipe.generate(MODEL, cond, REGISTER, {"id": i, "question": q})
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
                if done % 50 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} generated", flush=True)

    # fill output columns from cache verbatim -- line breaks preserved to match the human
    # and baseline comment columns (forum multi-line structure is a genuine register signal)
    for (i, cond), text in cache.items():
        col = f"comment_{cond}"
        if 0 <= i < n and col in data.columns:
            data.loc[data.index[i], col] = text

    data.to_csv(_out_path, index=False, encoding="utf-8")
    print(f"\nFinished. Saved {n} rows (+{len(CONDITIONS)} adversarial cols) to {_out_path}")
