"""Generate the three prompt conditions for the formal corpus under identical settings (OpenAI).

All three are independent one-shot generations (no chaining) over the same title+keywords,
so length/style differences reflect the prompt, not model/run drift:

  baseline        - neutral control prompt, no instruction about sounding human
  human_like      - naive adversarial: "write as human-like as possible"
  detector_aware  - adversarial targeting the stylometric/detection signals
                    (sentence-length variation, no formulaic connectives, less hedging)

Input:  sv_abstracts_openai_2.csv  (170 rows; has Title, Keywords, Abstract).
Output: sv_abstracts_adversarial.csv, with three generated columns added:
  Abstract_baseline, Abstract_human_like, Abstract_detector_aware
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
import pandas as pd
from openai import OpenAI

# force UTF-8 stdout so printing Swedish / special chars can't crash the run on a cp1252 console
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

MODEL = "gpt-5.2"  # matches the model used for the existing baseline abstracts

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

# All three conditions are generated here under identical settings (independent one-shot
# calls), so length/style differences can't be blamed on model/run drift. The older-run
# Generated_OpenAI_Abstract column is dropped on write; Abstract_baseline is the sole baseline.
CONDITIONS = {
    "baseline": lambda t, k: _BASE.format(title=t, keywords=k),
    "human_like": lambda t, k: _BASE.format(title=t, keywords=k) + _HUMAN_LIKE,
    "detector_aware": lambda t, k: _BASE.format(title=t, keywords=k) + _DETECTOR_AWARE,
}


def _generate(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


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
    line.to_csv(_cache_path, mode="a", header=not os.path.exists(_cache_path), index=False, encoding="utf-8")


if __name__ == "__main__":
    data = pd.read_csv(_in_path, encoding="utf-8")
    n = len(data)
    print(f"Loaded {n} rows from {_in_path}")

    cache = _load_cache()
    if cache:
        print(f"Resuming: {len(cache)} cached cells found")

    for cond in CONDITIONS:
        data[f"Abstract_{cond}"] = pd.NA

    for i in range(n):
        t, k = data["Title"].iloc[i], data["Keywords"].iloc[i]
        if pd.isna(t) or not str(t).strip():
            continue
        print(f"[{i + 1}/{n}] {str(t)[:60]}")
        for cond, build in CONDITIONS.items():
            if (i, cond) in cache:
                abstract = cache[(i, cond)]
                print(f"    {cond:<15} (cached)")
            else:
                abstract = _generate(build(t, k))
                _append_cache(i, cond, abstract)
                print(f"    {cond:<15} {abstract[:60]}...")
            # cache keeps the raw text (line above); output is normalized to single-block
            data.loc[data.index[i], f"Abstract_{cond}"] = _norm(abstract)

    # drop the superseded older-run baseline; Abstract_baseline is now the sole baseline
    data = data.drop(columns=["Generated_OpenAI_Abstract"], errors="ignore")

    data.to_csv(_out_path, index=False, encoding="utf-8")
    print(f"\nFinished. Saved {n} rows (+{len(CONDITIONS)} generated cols) to {_out_path}")
