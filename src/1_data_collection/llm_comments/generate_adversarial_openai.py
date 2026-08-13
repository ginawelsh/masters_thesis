"""Generate the three reported prompt conditions for the informal corpus.

All three are independent one-shot generations over the same thread question, so
length/style differences reflect the prompt, not model/run drift:

  baseline         - neutral control prompt, no instruction about sounding human
  human_like       - naive adversarial: "write as human-like as possible"
  detector_evasive - adversarial tuned to invert this study's measured signals
                     (reword rather than repeat, dense content, short sentences / more
                     periods / fewer commas, no dashes, minimal hedging, concrete names)

detector_aware is NOT generated: it is excluded from the reported analysis and from
the detection quiz. The already-saved GPT-5.2 corpora keep their detector_aware
columns; nothing existing is lost, but re-running this script will not reproduce them.

BASELINE IS NOW GENERATED PER MODEL, into a new `comment_baseline` column. It used to
be inherited from the input's `generated_comment`, which holds GPT-5.2 text -- so a
run under any other model produced a corpus whose baseline cell was still GPT-5.2,
silently contaminating exactly the cell where detection is strongest. `generated_comment`
is left byte-identical as the published GPT-5.2 control.

Prompts + model access come from the shared ../generation_pipeline.py, so this
script and smoke_test.py build byte-identical prompts and can swap models the
same way. Pick the backend with GEN_MODEL (default "gpt-5.2"; e.g. "gpt-5.6", "mistral").

Input:  ../../4_archive/consolidated_informal_comments_JUN26.csv  (has question,
        human_comment, generated_comment). Three columns are appended:
  comment_baseline, comment_human_like, comment_detector_evasive
Output: generated_three_prompt_14_07_26.csv (input left untouched).

Scoping a run (all four knobs are folded into the output AND cache filenames, so no
two differently-scoped runs can ever share a cache):
  GEN_MODEL=gpt-5.6        which backend (default gpt-5.2)
  GEN_CONDITIONS=baseline  restrict to a subset of the three conditions
  GEN_INPUT=<path>         read a different corpus than the default input
  QUIZ_SUBSET=1            only the 100 detection-quiz documents

To replace the legacy gpt-4o-mini baseline with genuine GPT-5.2 text across the whole
informal corpus, in one pass, with nothing else regenerated:

  GEN_CONDITIONS=baseline GEN_INPUT=<...>/consolidated_informal_comments_adversarial.csv

That reads the corpus the feature analysis and the quiz builder already use, generates
1,149 baseline comments, and writes that file back out with one column ADDED
(`comment_baseline`). Every other column, including the old gpt-4o-mini
`generated_comment`, passes through byte-identical -- so the gpt-4o-mini condition stays
available as a third generator rather than being destroyed. Point
data_utils.INFORMAL_CONDITIONS["baseline"] and make_balanced_quiz.INFORMAL_COL["baseline"]
at `comment_baseline` to consume it.

Set QUIZ_SUBSET=1 to generate ONLY the 100 documents sampled into
3_ai_detection_scripts/quiz_master_balanced.csv, instead of all ~1149 rows -- the
cheap path for a detection-only model comparison (300 calls instead of ~3447). Rows
are matched on human_comment text, not row number (see quiz_subset() for why), and
the output/cache names gain a "_quiz100" tag so subset and full runs never share a
cache. Leave it unset to regenerate the whole corpus, which is what the feature
analysis in 2_text_analysis_scripts needs.

Calls run concurrently (CONCURRENCY workers) with retry/backoff (in the pipeline).
Generation is cached per (row, condition) in _gen_cache_three_prompt_14_07_26_*.csv
so an interrupted run resumes without paying for completed cells again. NOTE: the
cache is keyed by row index, so it is tied to THIS input file/ordering AND to whether
QUIZ_SUBSET is set -- both are encoded in the cache filename to keep them apart.
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
MODEL = os.environ.get("GEN_MODEL", "gpt-5.2")  # e.g. "gpt-5.6", or "mistral"
CONCURRENCY = 16  # parallel API calls (retry/backoff absorbs any rate-limit 429s)
REFUSAL_RETRIES = 2  # extra attempts when a cell comes back as a refusal
# QUIZ_SUBSET=1 -> only the 100 documents in the detection quiz (see module docstring)
QUIZ_SUBSET = os.environ.get("QUIZ_SUBSET", "").strip() not in ("", "0", "false", "False")

# script is at src/1_data_collection/llm_comments/ -> project root is 4 levels up
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_dir = os.path.join(_root, "src", "1_data_collection", "llm_comments")
_archive = os.path.join(_root, "src", "4_archive")
_in_path = os.path.join(_archive, "consolidated_informal_comments_JUN26.csv")

# the three REPORTED conditions; detector_aware is excluded (see module docstring).
# baseline is generated per model into comment_baseline -- generated_comment (the old
# gpt-4o-mini text) is left untouched.
ALL_CONDITIONS = ["baseline", "human_like", "detector_evasive"]
CONDITIONS, _cond_tag = pipe.conditions_from_env(ALL_CONDITIONS)
_in_path, _input_tag = pipe.input_from_env(_in_path)

_safe_model = MODEL.replace("/", "-").replace(":", "-")
# every scoping choice (model, quiz subset, condition set, input file) is folded into
# the output AND cache names. The cache key is only (row index, condition), so two runs
# that differ in any of these must never share a cache file.
_tag = ("_quiz100" if QUIZ_SUBSET else "") + _cond_tag + _input_tag
_stem = "generated_three_prompt_14_07_26"
# the bare filename is reserved for the FULL default gpt-5.2 corpus run
_out_name = (f"{_stem}.csv" if (MODEL == "gpt-5.2" and not _tag)
             else f"{_stem}_{_safe_model}{_tag}.csv")
_out_path = os.path.join(_dir, _out_name)
_cache_path = os.path.join(_dir, f"_gen_cache_three_prompt_14_07_26_{_safe_model}{_tag}.csv")

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
    print(f"Loaded {len(data)} rows from {_in_path}")
    if QUIZ_SUBSET:
        data = pipe.quiz_subset(data, REGISTER, "human_comment")
        print(f"QUIZ_SUBSET: restricted to {len(data)} quiz documents "
              f"(matched on human_comment, not row number)")
    n = len(data)
    print(f"Model: {MODEL} | register: {REGISTER} | conditions: {', '.join(CONDITIONS)}")
    print(f"Output: {_out_path}")

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

    refusals = []

    def _work(task):
        i, cond, q = task
        # Retry on refusal. A refusal is NOT cached, so a later re-run retries it
        # naturally instead of silently inheriting the refusal text forever.
        last = ""
        for attempt in range(REFUSAL_RETRIES + 1):
            rec = pipe.generate(MODEL, cond, REGISTER, {"id": i, "question": q})
            last = pipe.strip_markdown(rec.output)
            why = pipe.looks_like_refusal(last)
            if why is None:
                _append_cache(i, cond, last)  # persist so progress survives interruption
                return i, cond, last
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
                if done % 50 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)} generated", flush=True)

    # fill output columns from cache verbatim -- line breaks preserved to match the human
    # and baseline comment columns (forum multi-line structure is a genuine register signal)
    for (i, cond), text in cache.items():
        col = f"comment_{cond}"
        if 0 <= i < n and col in data.columns:
            data.loc[data.index[i], col] = text

    data.to_csv(_out_path, index=False, encoding="utf-8")
    print(f"\nFinished. Saved {n} rows (+{len(CONDITIONS)} generated cols) to {_out_path}")
    if refusals:
        print(f"\n[WARNING] {len(refusals)} cell(s) left EMPTY after "
              f"{REFUSAL_RETRIES + 1} refused attempts:")
        for i, cond, why in refusals[:10]:
            print(f"    row {i} / {cond}: {why}")
        print(f"  Refusals are never cached, so re-running this command retries only "
              f"those cells.\n  If the rate is high, measure it with refusal_probe.py "
              f"rather than retrying blindly —\n  a condition this model won't produce "
              f"is a finding, not a transient error.")
