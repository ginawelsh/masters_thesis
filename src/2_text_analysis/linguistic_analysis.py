"""
Linguistic analysis (spaCy) on Abstract and Generated Abstract from formal CSV.
Install: pip install -r requirements.txt
Swedish: python -m spacy download sv_core_news_sm
"""
import os
from collections import Counter

import pandas as pd
import spacy

# Use Swedish small model for thesis data; change to "en_core_web_sm" for English
MODEL = "sv_core_news_sm"

# CSV with Abstract and Generated Abstract columns (sv_abstracts_generated.csv has both filled)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_root, "data_collection", "formal", "sv_abstracts_generated.csv")
OUT_CSV_PATH = os.path.join(_root, "data_collection", "formal", "linguistic_analysis_tokens.csv")
OUT_POS_CSV_PATH = os.path.join(_root, "data_collection", "formal", "linguistic_analysis_pos.csv")
OUT_POS_COUNTS_CSV_PATH = os.path.join(_root, "data_collection", "formal", "linguistic_analysis_pos_counts.csv")
TOKEN_SEP = " | "  # separator for tokens/POS in CSV cells


def load_nlp(model_name: str = MODEL):
    """Load spaCy pipeline. Download with: python -m spacy download <model_name>"""
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Model '{model_name}' not found. Run: python -m spacy download {model_name}")
        raise


def process_text(nlp, text: str):
    """Return a Doc for the given text."""
    if pd.isna(text) or not str(text).strip():
        return None
    return nlp(str(text).strip())


def pos_counts_string(doc) -> str:
    """Return POS counts as 'TAG:n | TAG:n | ...' sorted by tag."""
    if doc is None:
        return ""
    counts = Counter(t.pos_ for t in doc)
    return TOKEN_SEP.join(f"{tag}:{n}" for tag, n in sorted(counts.items()))


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    nlp = load_nlp()

    rows_out = []
    rows_pos = []
    rows_pos_counts = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        abs_text = row.get("Abstract")
        gen_text = row.get("Generated_Abstract") or row.get("Generated Abstract")

        print(f"\n--- Row {idx + 1} ---")
        doc_abs = process_text(nlp, abs_text)
        if doc_abs is not None:
            abs_tokens = TOKEN_SEP.join(t.text for t in doc_abs)
            abs_pos = TOKEN_SEP.join(f"{t.text}_{t.pos_}" for t in doc_abs)
            print("[Abstract] Tokens:", [t.text for t in doc_abs], "...")
            print("[Abstract] POS:", [(t.text, t.pos_) for t in doc_abs])
            if doc_abs.ents:
                print("[Abstract] Entities:", [(e.text, e.label_) for e in doc_abs.ents])
        else:
            abs_tokens = ""
            abs_pos = ""
        abs_pos_counts = pos_counts_string(doc_abs)

        doc_gen = process_text(nlp, gen_text)
        if doc_gen is not None:
            gen_tokens = TOKEN_SEP.join(t.text for t in doc_gen)
            gen_pos = TOKEN_SEP.join(f"{t.text}_{t.pos_}" for t in doc_gen)
            print("[Generated Abstract] Tokens:", [t.text for t in doc_gen], "...")
            print("[Generated Abstract] POS:", [(t.text, t.pos_) for t in doc_gen])
            if doc_gen.ents:
                print("[Generated Abstract] Entities:", [(e.text, e.label_) for e in doc_gen.ents])
        else:
            gen_tokens = ""
            gen_pos = ""
        gen_pos_counts = pos_counts_string(doc_gen)

        rows_out.append({
            "Abstract_Tokens": abs_tokens,
            "Generated_Abstract_Tokens": gen_tokens,
        })
        rows_pos.append({
            "Abstract_POS": abs_pos,
            "Generated_Abstract_POS": gen_pos,
        })
        rows_pos_counts.append({
            "Abstract_POS_Counts": abs_pos_counts,
            "Generated_Abstract_POS_Counts": gen_pos_counts,
        })

    pd.DataFrame(rows_out).to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {len(rows_out)} rows to {OUT_CSV_PATH}")

    pd.DataFrame(rows_pos).to_csv(OUT_POS_CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos)} rows to {OUT_POS_CSV_PATH}")

    pd.DataFrame(rows_pos_counts).to_csv(OUT_POS_COUNTS_CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos_counts)} rows to {OUT_POS_COUNTS_CSV_PATH}")


if __name__ == "__main__":
    main()
