"""
Sentiment analysis on Abstract and Generated_Abstract from the same CSV as linguistic_analysis.
Uses KBLab Swedish multiclass sentiment (positive / neutral / negative).
Install: pip install transformers torch
"""
import os

import pandas as pd
from transformers import pipeline

# Same input as linguistic_analysis.py
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_root, "data_collection", "formal", "sv_abstracts_generated.csv")
OUT_CSV_PATH = os.path.join(_root, "data_collection", "formal", "sentiment_analysis.csv")

SENTIMENT_MODEL = "KBLab/robust-swedish-sentiment-multiclass"
# BERT-style models have max length; truncate long abstracts to avoid errors
MAX_LENGTH = 512


def get_sentiment(pipe, text: str) -> str:
    """Return sentiment as 'label (score)' or '' if empty."""
    if pd.isna(text) or not str(text).strip():
        return ""
    text = str(text).strip()
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]
    try:
        out = pipe(text, truncation=True, max_length=MAX_LENGTH)
        if out and isinstance(out, list):
            item = out[0]
            label = item.get("label", "")
            score = item.get("score", 0)
            return f"{label} ({score:.2f})"
        return ""
    except Exception as e:
        return f"error: {e}"


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    print("Loading sentiment model...", flush=True)
    pipe = pipeline("text-classification", model=SENTIMENT_MODEL)

    rows_out = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        abs_text = row.get("Abstract")
        gen_text = row.get("Generated_Abstract") or row.get("Generated Abstract")

        print(f"Processing row {idx + 1}/{len(df)}...", flush=True)
        abs_sent = get_sentiment(pipe, abs_text)
        gen_sent = get_sentiment(pipe, gen_text)

        rows_out.append({
            "Abstract_Sentiment": abs_sent,
            "Generated_Abstract_Sentiment": gen_sent,
        })

    pd.DataFrame(rows_out).to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_out)} rows to {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()
