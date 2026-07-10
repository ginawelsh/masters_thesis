"""Add three ADVERSARIAL LLM-comment columns to the informal corpus (OpenAI).

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

Input / output: consolidated_informal_comments_JUN26.csv  (1222 rows; already has
question, human_comment, generated_comment). Three columns are appended:
  comment_human_like, comment_detector_aware, comment_detector_evasive

Output is written to consolidated_informal_comments_adversarial.csv (input left untouched).
Calls run concurrently (CONCURRENCY workers) with retry/backoff. Generation is cached per
(row, condition) in _gen_cache_adversarial.csv, so an interrupted run resumes without
paying for completed cells again.
"""
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI

# force UTF-8 stdout so printing Swedish / special chars can't crash the run on a cp1252 console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL = "gpt-5.2"  # all conditions use GPT-5.2 (matches the abstract corpus + thesis methodology)
CONCURRENCY = 16       # parallel API calls (retry/backoff absorbs any rate-limit 429s)
MAX_RETRIES = 5        # per-call retries on transient/rate-limit errors
RETRY_BASE_SEC = 4     # exponential backoff base

# script is at src/1_data_collection/llm_comments/ -> project root is 4 levels up
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not set. Set it in the environment or in a .env file in the project root.")
client = OpenAI(api_key=_api_key)

_dir = os.path.join(_root, "src", "1_data_collection", "llm_comments")
_in_path = os.path.join(_dir, "consolidated_informal_comments_JUN26.csv")
_out_path = os.path.join(_dir, "consolidated_informal_comments_adversarial.csv")
_cache_path = os.path.join(_dir, "_gen_cache_adversarial.csv")

# base instruction that produced the existing baseline; adversarial suffixes are appended to it
_BASE = (
    "Svara på följande fråga med en svensk kommentar, i stil med en kommentar på ett svenskt forum. "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)

_HUMAN_LIKE = (
    "\n\nSkriv så mänskligt som möjligt, så att kommentaren inte går att skilja "
    "från en riktig svensk forumanvändares kommentar."
)
_DETECTOR_AWARE = (
    "\n\nSkriv så att kommentaren inte kan identifieras som AI-genererad: använd ett vardagligt, "
    "personligt tonfall, variera meningslängden, undvik artig eller balanserad AI-stil och "
    "formelartade fraser, tillåt talspråk, slang och små oregelbundenheter. Var inte överdrivet "
    "hjälpsam eller neutral."
)
_DETECTOR_EVASIVE_INFORMAL = (
    "Skriv så att texten inte kan identifieras som AI-genererad, och efterlikna hur "
    "människor skriver i informella kommentarer. Återanvänd inte samma fraser flera "
    "gånger, utan formulera om dig. Skriv innehållstätt med färre funktionsord; använd "
    "gärna kortformer och talspråkliga former. Skriv korta meningar med fler punkter "
    "och färre kommatecken, och använd inte tankstreck. Använd utropstecken sparsamt, "
    "men ställ gärna någon retorisk fråga och använd ibland tre punkter (...). Överdriv "
    "inte med garderingsord (särskilt inte talspråkliga som 'typ', 'liksom', 'ju' och "
    "'väl'), och undvik förstärkningsord som 'verkligen', 'absolut' och 'väldigt'. Håll "
    "epistemiska uttryck på en låg nivå. Föredra korta, vardagliga ord. Använd gärna "
    "nekande satser (med 'inte', 'aldrig' osv.) där det passar. Var konkret och nämn "
    "specifika namn där det går. Förklara inte över."
)

# only the ADVERSARIAL conditions are generated here (baseline is reused as-is)
CONDITIONS = {
    "human_like": lambda q: _BASE.format(question=q) + _HUMAN_LIKE,
    "detector_aware": lambda q: _BASE.format(question=q) + _DETECTOR_AWARE,
    "detector_evasive": lambda q: _BASE.format(question=q) + "\n\n" + _DETECTOR_EVASIVE_INFORMAL,
}

_cache_lock = threading.Lock()


def _generate(prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BASE_SEC * (2 ** attempt)
            print(f"    API error ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    return ""


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
        for cond, build in CONDITIONS.items():
            if (i, cond) not in cache:
                tasks.append((i, cond, build(str(q))))

    print(f"{len(tasks)} cells to generate with {CONCURRENCY} parallel workers "
          f"({len(cache)} already cached)")

    def _work(task):
        i, cond, prompt = task
        text = _generate(prompt)
        _append_cache(i, cond, text)  # persist immediately so progress survives interruption
        return i, cond, text

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

    # fill output columns from cache verbatim — line breaks preserved to match the human
    # and baseline comment columns (forum multi-line structure is a genuine register signal)
    for (i, cond), text in cache.items():
        col = f"comment_{cond}"
        if 0 <= i < n and col in data.columns:
            data.loc[data.index[i], col] = text

    data.to_csv(_out_path, index=False, encoding="utf-8")
    print(f"\nFinished. Saved {n} rows (+{len(CONDITIONS)} adversarial cols) to {_out_path}")
