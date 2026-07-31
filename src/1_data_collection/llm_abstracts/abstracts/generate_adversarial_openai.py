"""Generate the three reported prompt conditions for the formal corpus.

All three are independent one-shot generations (no chaining) over the same title+keywords,
so length/style differences reflect the prompt, not model/run drift:

  baseline         - neutral control prompt, no instruction about sounding human
  human_like       - naive adversarial: "write as human-like as possible"
  detector_evasive - adversarial tuned to invert this study's measured signals
                     (reuse key terms verbatim, short sentences / more periods / fewer
                     commas, no dashes, minimal hedging, concrete named entities)

detector_aware is NOT generated: it is excluded from the reported analysis and from
the detection quiz. The already-saved GPT-5.2 corpus (sv_abstracts_adversarial.csv)
keeps its Abstract_detector_aware column; nothing existing is lost, but re-running
this script will not reproduce it.

Prompts + model access come from the shared ../../generation_pipeline.py, so this
script, the comments script, and smoke_test.py build byte-identical prompts and can
swap models the same way. Pick the backend with GEN_MODEL (default "gpt-5.2"; e.g.
"gpt-5.6", "mistral").

Input:  sv_abstracts_openai_2.csv  (has Title, Keywords, Abstract).
Output: generated_three_prompt_formal_14_07_26.csv, with three generated columns added:
  Abstract_baseline, Abstract_human_like, Abstract_detector_evasive
The older-run Generated_OpenAI_Abstract column is dropped on write; Abstract_baseline
(freshly generated under identical settings) is the sole baseline. Generated text is
normalized to a single block to match the human abstracts' formatting.

Scoping a run (all four knobs are folded into the output AND cache filenames, so no
two differently-scoped runs can ever share a cache):
  GEN_MODEL=gpt-5.6        which backend (default gpt-5.2)
  GEN_CONDITIONS=baseline  restrict to a subset of the three conditions
  GEN_INPUT=<path>         read a different corpus than the default input
  QUIZ_SUBSET=1            only the 100 detection-quiz documents

Set QUIZ_SUBSET=1 to generate ONLY the 100 documents sampled into
3_ai_detection_scripts/quiz_master_balanced.csv, instead of all 170 rows -- the cheap
path for a detection-only model comparison (300 calls instead of 510). Rows are matched
on Abstract text, not row number (see quiz_subset() for why), and the output/cache names
gain a "_quiz100" tag so subset and full runs never share a cache. Leave it unset to
regenerate the whole corpus, which is what the feature analysis needs.

Output/cache names are model-aware so a second model never clobbers the first (the
default full gpt-5.2 run keeps the existing _gen_cache_adversarial.csv so its completed
run resumes for free). Generation is cached per (row, condition) so the run can resume
after an interruption without paying for completed cells again. NOTE: the cache key is
a row index, so it is tied to the input ordering AND to whether QUIZ_SUBSET is set --
both are encoded in the cache filename to keep them apart.
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
MODEL = os.environ.get("GEN_MODEL", "gpt-5.2")  # e.g. "gpt-5.6", or "mistral"
CONCURRENCY = 8  # parallel API calls
REFUSAL_RETRIES = 2  # extra attempts when a cell comes back as a refusal
# QUIZ_SUBSET=1 -> only the 100 documents in the detection quiz (see module docstring)
QUIZ_SUBSET = os.environ.get("QUIZ_SUBSET", "").strip() not in ("", "0", "false", "False")

# script is at src/1_data_collection/llm_abstracts/abstracts/ -> project root is 5 levels up
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_dir = os.path.join(_root, "src", "1_data_collection", "llm_abstracts", "abstracts")
_in_path = os.path.join(_dir, "sv_abstracts_openai_2.csv")

# the three REPORTED conditions, generated under identical settings (independent
# one-shot calls); detector_aware is excluded (see module docstring)
ALL_CONDITIONS = ["baseline", "human_like", "detector_evasive"]
CONDITIONS, _cond_tag = pipe.conditions_from_env(ALL_CONDITIONS)
_in_path, _input_tag = pipe.input_from_env(_in_path)

_safe_model = MODEL.replace("/", "-").replace(":", "-")
# every scoping choice (model, quiz subset, condition set, input file) is folded into
# the output AND cache names. The cache key is only (row index, condition), so two runs
# that differ in any of these must never share a cache file.
_tag = ("_quiz100" if QUIZ_SUBSET else "") + _cond_tag + _input_tag
_is_canonical = (MODEL == "gpt-5.2" and not _tag)
_stem = "generated_three_prompt_formal_14_07_26"
# the bare filenames are reserved for the FULL default gpt-5.2 corpus run
_out_name = f"{_stem}.csv" if _is_canonical else f"{_stem}_{_safe_model}{_tag}.csv"
_out_path = os.path.join(_dir, _out_name)
_cache_name = ("_gen_cache_adversarial.csv" if _is_canonical
               else f"_gen_cache_adversarial_{_safe_model}{_tag}.csv")
_cache_path = os.path.join(_dir, _cache_name)

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
    print(f"Loaded {len(data)} rows from {_in_path}")
    if QUIZ_SUBSET:
        data = pipe.quiz_subset(data, REGISTER, "Abstract")
        print(f"QUIZ_SUBSET: restricted to {len(data)} quiz documents "
              f"(matched on Abstract, not row number)")
    n = len(data)
    print(f"Model: {MODEL} | register: {REGISTER} | conditions: {', '.join(CONDITIONS)}")
    print(f"Output: {_out_path}")

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

    refusals = []

    def _work(task):
        i, cond, t, k = task
        # Retry on refusal. A refusal is NOT cached, so a later re-run retries it
        # naturally instead of silently inheriting the refusal text forever. GPT-5.2
        # already put 5 refusal/meta-preamble abstracts into the shipped corpus.
        for attempt in range(REFUSAL_RETRIES + 1):
            rec = pipe.generate(MODEL, cond, REGISTER,
                                {"id": i, "title": t, "keywords": k})
            out = pipe.strip_markdown(rec.output)
            why = pipe.looks_like_refusal(out)
            if why is None:
                _append_cache(i, cond, out)  # persist so progress survives interruption
                return i, cond, out
            if attempt == REFUSAL_RETRIES:
                refusals.append((i, cond, why))
                print(f"  [refusal] row {i} / {cond}: {why} — giving up after "
                      f"{REFUSAL_RETRIES + 1} attempts; cell left EMPTY (not cached)",
                      flush=True)
                return i, cond, None
            print(f"  [refusal] row {i} / {cond}: {why} — retrying", flush=True)
        return i, cond, None

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
                if text is None:      # refusal after retries; leave the cell empty
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
    if refusals:
        print(f"\n[WARNING] {len(refusals)} cell(s) left EMPTY after "
              f"{REFUSAL_RETRIES + 1} refused attempts:")
        for i, cond, why in refusals[:10]:
            print(f"    row {i} / {cond}: {why}")
        print("  Refusals are never cached, so re-running this command retries only "
              "those cells.\n  If the rate is high, measure it with refusal_probe.py "
              "rather than retrying blindly —\n  a condition this model won't produce "
              "is a finding, not a transient error.")
