"""Add two ADVERSARIAL LLM-comment columns to the informal corpus (OpenAI).

The existing neutral baseline (generated_comment, produced by the basic forum prompt)
is kept as the control. This script only generates the two adversarial conditions and
appends them, so the three prompts can be compared over identical inputs:

  baseline        - existing generated_comment column (control; not regenerated)
  human_like      - naive adversarial: "write as human-like as possible"
  detector_aware  - adversarial targeting the stylometric/detection signals
                    (informal tone, length variation, no AI politeness/hedging)

Input / output: consolidated_informal_comments_JUN26.csv  (1222 rows; already has
question, human_comment, generated_comment). Two columns are appended:
  comment_human_like, comment_detector_aware

Output is written to consolidated_informal_comments_adversarial.csv (input left untouched).
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

MODEL = "gpt-4o-mini"  # matches the model used for the existing baseline comments

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

# only the ADVERSARIAL conditions are generated here (baseline is reused as-is)
CONDITIONS = {
    "human_like": lambda q: _BASE.format(question=q) + _HUMAN_LIKE,
    "detector_aware": lambda q: _BASE.format(question=q) + _DETECTOR_AWARE,
}


def _generate(prompt):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.choices[0].message.content or "").strip()


def _norm(text):
    """Collapse newlines/whitespace to a single block, matching the human comments' format."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _load_cache():
    if not os.path.exists(_cache_path):
        return {}
    df = pd.read_csv(_cache_path, encoding="utf-8")
    return {(int(r["row"]), r["condition"]): r["comment"] for _, r in df.iterrows()}


def _append_cache(row_i, condition, comment):
    line = pd.DataFrame([{"row": row_i, "condition": condition, "comment": comment}])
    line.to_csv(_cache_path, mode="a", header=not os.path.exists(_cache_path), index=False, encoding="utf-8")


if __name__ == "__main__":
    data = pd.read_csv(_in_path, encoding="utf-8")
    n = len(data)
    print(f"Loaded {n} rows from {_in_path}")

    cache = _load_cache()
    if cache:
        print(f"Resuming: {len(cache)} cached cells found")

    for cond in CONDITIONS:
        data[f"comment_{cond}"] = pd.NA

    for i in range(n):
        q = data["question"].iloc[i]
        if pd.isna(q) or not str(q).strip():
            continue
        print(f"[{i + 1}/{n}] {str(q)[:60]}")
        for cond, build in CONDITIONS.items():
            if (i, cond) in cache:
                comment = cache[(i, cond)]
                print(f"    {cond:<15} (cached)")
            else:
                comment = _generate(build(str(q)))
                _append_cache(i, cond, comment)
                print(f"    {cond:<15} {comment[:60]}...")
            # cache keeps the raw text (line above); output is normalized to single-block
            data.loc[data.index[i], f"comment_{cond}"] = _norm(comment)

    data.to_csv(_out_path, index=False, encoding="utf-8")
    print(f"\nFinished. Saved {n} rows (+{len(CONDITIONS)} adversarial cols) to {_out_path}")
