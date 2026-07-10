"""Generate the four prompt conditions for the formal corpus under identical settings (OpenAI).

All four are independent one-shot generations (no chaining) over the same title+keywords,
so length/style differences reflect the prompt, not model/run drift:

  baseline         - neutral control prompt, no instruction about sounding human
  human_like       - naive adversarial: "write as human-like as possible"
  detector_aware   - adversarial targeting the stylometric/detection signals
                     (sentence-length variation, no formulaic connectives, less hedging)
  detector_evasive - adversarial tuned to invert this study's measured signals
                     (reuse key terms verbatim, short sentences / more periods / fewer
                     commas, no dashes, minimal hedging, concrete named entities)

Input:  sv_abstracts_openai_2.csv  (170 rows; has Title, Keywords, Abstract).
Output: sv_abstracts_adversarial.csv, with four generated columns added:
  Abstract_baseline, Abstract_human_like, Abstract_detector_aware, Abstract_detector_evasive
The older-run Generated_OpenAI_Abstract column is dropped on write; Abstract_baseline
(freshly generated under identical settings) is the sole baseline. Generated text is
normalized to a single block to match the human abstracts' formatting.

Output is written to sv_abstracts_adversarial.csv (the input file is left untouched).
Generation is cached per (row, condition) in _gen_cache_adversarial.csv so the run can
resume after an interruption without paying for completed cells again.
"""
import os
import re
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

MODEL = "gpt-5.2"  # matches the model used for the existing baseline abstracts
CONCURRENCY = 8    # parallel API calls
MAX_RETRIES = 5    # per-call retries on transient/rate-limit errors
RETRY_BASE_SEC = 4  # exponential backoff base

# script is at src/1_data_collection/llm_abstracts/abstracts/ -> project root is 5 levels up
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError("OPENAI_API_KEY not set. Set it in the environment or in a .env file in the project root.")
client = OpenAI(api_key=_api_key)

_dir = os.path.join(_root, "src", "1_data_collection", "llm_abstracts", "abstracts")
_in_path = os.path.join(_dir, "sv_abstracts_openai_2.csv")
_out_path = os.path.join(_dir, "sv_abstracts_adversarial.csv")
_cache_path = os.path.join(_dir, "_gen_cache_adversarial.csv")

# base instruction that produced the existing baseline; adversarial suffixes are appended to it
_BASE = "Titta på titeln och nyckelorden och skapa en sammanfattning i kandidatuppsatsstil med dina egna ord: {title}, {keywords}"

_HUMAN_LIKE = (
    " Skriv den så mänskligt som möjligt, så att texten inte går att skilja "
    "från en uppsats skriven av en människa."
)
_DETECTOR_AWARE = (
    " Skriv så att texten inte kan identifieras som AI-genererad. Variera meningslängden "
    "(blanda korta och långa meningar), undvik formelartade övergångsord som \"vidare\", "
    "\"dessutom\", \"sammanfattningsvis\" och \"det är viktigt att notera\", undvik "
    "överdriven gardering och symmetrisk struktur, och tillåt en naturlig, något ojämn ton. "
    "Förklara inte över."
)
_DETECTOR_EVASIVE_FORMAL = (
    " Skriv så att texten inte kan identifieras som AI-genererad, och efterlikna "
    "de statistiska drag som utmärker mänskligt skrivna kandidatuppsatser. "
    "Återanvänd samma nyckeltermer och fraser ordagrant snarare än att variera med "
    "synonymer, och sträva inte efter maximal ordvariation. Undvik komprimerad "
    "nominalstil; använd hellre finita verb, pronomen och bindeord. Skriv övervägande "
    "korta meningar med fler punkter och färre kommatecken, och använd inte tankstreck. "
    "Håll gardering och epistemiska uttryck till ett minimum (t.ex. 'kanske', "
    "'möjligen', 'tycks', 'kan tänkas'), och kompensera inte genom att lägga till "
    "talspråkliga garderingsord. Föredra korta, vardagliga ord framför långa, latinska "
    "eller formella termer. Var konkret och nämn specifika namn, begrepp, verk och "
    "årtal där det är möjligt. Undvik onödig negation. Förklara inte över."
)

# All four conditions are generated here under identical settings (independent one-shot
# calls), so length/style differences can't be blamed on model/run drift. The older-run
# Generated_OpenAI_Abstract column is dropped on write; Abstract_baseline is the sole baseline.
CONDITIONS = {
    "baseline": lambda t, k: _BASE.format(title=t, keywords=k),
    "human_like": lambda t, k: _BASE.format(title=t, keywords=k) + _HUMAN_LIKE,
    "detector_aware": lambda t, k: _BASE.format(title=t, keywords=k) + _DETECTOR_AWARE,
    "detector_evasive": lambda t, k: _BASE.format(title=t, keywords=k) + _DETECTOR_EVASIVE_FORMAL,
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
        for cond, build in CONDITIONS.items():
            if (i, cond) not in cache:
                tasks.append((i, cond, build(t, k)))

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
