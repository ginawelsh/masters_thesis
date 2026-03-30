"""
Linguistic analysis (spaCy) on Abstract and Generated Abstract from formal CSV.
Install: pip install -r requirements.txt
Swedish: python -m spacy download sv_core_news_sm
"""
import os
from collections import Counter

import pandas as pd
import spacy

# Use Swedish small model for thesis data
MODEL = "sv_core_news_sm"

# NOTE: This repo currently stores outputs under `src/1_data_collection/...`.
# Some older paths in this script referenced `src/data_collection/...`.
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUMAN_FORMAL_DIR = os.path.join(_root, "1_data_collection", "human_formal")
CSV_PATH = os.path.join(HUMAN_FORMAL_DIR, "sv_human_collection_with_kws.csv")
OUT_CSV_PATH = os.path.join(HUMAN_FORMAL_DIR, "linguistic_analysis_tokens.csv")
OUT_POS_CSV_PATH = os.path.join(HUMAN_FORMAL_DIR, "linguistic_analysis_pos.csv")
OUT_POS_COUNTS_CSV_PATH = os.path.join(HUMAN_FORMAL_DIR, "linguistic_analysis_pos_counts.csv")
OUT_POS_PROPORTIONS_CSV_PATH = os.path.join(HUMAN_FORMAL_DIR, "linguistic_analysis_pos_proportions.csv")
OUT_POS_DIFFERENCES_CSV_PATH = os.path.join(HUMAN_FORMAL_DIR, "linguistic_analysis_pos_differences.csv")

TOKEN_SEP = ","  # separator for tokens/POS in CSV cells
POS_SEP = " | "  # separator for TAG:n lists in CSV cells (matches existing checked-in counts file)


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
    return POS_SEP.join(f"{tag}:{n}" for tag, n in sorted(counts.items()))


def parse_pos_counts_string(s: str) -> Counter:
    """Parse a 'TAG:n | TAG:n | ...' string into a Counter."""
    counts: Counter = Counter()
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return counts
    text = str(s).strip()
    if not text:
        return counts
    for part in text.split(POS_SEP):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        tag, n_str = part.split(":", 1)
        tag = tag.strip()
        n_str = n_str.strip()
        if not tag or not n_str:
            continue
        try:
            counts[tag] += int(n_str)
        except ValueError:
            continue
    return counts


def pos_proportions_string(counts: Counter, ndigits: int = 4) -> str:
    """Return POS proportions as 'TAG:0.xxxx | TAG:0.xxxx | ...' sorted by tag."""
    total = sum(counts.values())
    if total <= 0:
        return ""
    return POS_SEP.join(
        f"{tag}:{round(n / total, ndigits):.{ndigits}f}" for tag, n in sorted(counts.items())
    )


def token_length_from_counts(counts: Counter) -> int:
    """Token length using option B: spaCy tokens including everything (sum of POS counts)."""
    return int(sum(counts.values()))


def parse_pos_proportions_string(s: str) -> dict:
    """Parse a 'TAG:0.xxxx | TAG:0.xxxx | ...' string into a dict."""
    props = {}
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return props
    text = str(s).strip()
    if not text:
        return props
    for part in text.split(POS_SEP):
        part = part.strip()
        if not part or ":" not in part:
            continue
        tag, p_str = part.split(":", 1)
        tag = tag.strip()
        p_str = p_str.strip()
        if not tag or not p_str:
            continue
        try:
            props[tag] = float(p_str)
        except ValueError:
            continue
    return props


def write_pos_differences_csv(df_pos_props: pd.DataFrame, ndigits: int = 4) -> None:
    """
    Write POS proportion differences (Human - Generated) per row, one column per POS,
    plus an AVERAGE summary row across all rows.
    """
    parsed_rows = []
    pos_tags = set()

    for idx in range(len(df_pos_props)):
        row = df_pos_props.iloc[idx]
        abs_props = parse_pos_proportions_string(row.get("Abstract_POS_Proportions", ""))
        gen_props = parse_pos_proportions_string(row.get("Generated_Abstract_POS_Proportions", ""))

        tags = set(abs_props.keys()) | set(gen_props.keys())
        pos_tags.update(tags)

        diff_row = {}
        for tag in tags:
            diff_row[tag] = round(abs_props.get(tag, 0.0) - gen_props.get(tag, 0.0), ndigits)
        parsed_rows.append(diff_row)

    sorted_tags = sorted(pos_tags)
    out_rows = []
    for idx, diff_row in enumerate(parsed_rows, start=1):
        out = {"Row": idx}
        for tag in sorted_tags:
            out[f"POS_Diff_{tag}"] = diff_row.get(tag, 0.0)
        out_rows.append(out)

    # Summary row: average signed difference per POS across all files.
    avg_row = {"Row": "AVERAGE"}
    if out_rows:
        df_tmp = pd.DataFrame(out_rows)
        for tag in sorted_tags:
            col = f"POS_Diff_{tag}"
            avg_row[col] = round(float(df_tmp[col].mean()), ndigits)
    else:
        for tag in sorted_tags:
            avg_row[f"POS_Diff_{tag}"] = 0.0

    out_rows.append(avg_row)
    pd.DataFrame(out_rows).to_csv(OUT_POS_DIFFERENCES_CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows to {OUT_POS_DIFFERENCES_CSV_PATH}")


def main():
    # Fast path: if POS counts file already exists, derive proportions + lengths directly from it.
    # This guarantees we match the checked-in counts file format and avoids rerunning spaCy.
    if os.path.exists(OUT_POS_COUNTS_CSV_PATH):
        df_counts = pd.read_csv(OUT_POS_COUNTS_CSV_PATH, encoding="utf-8")
        rows_pos_props = []
        for idx in range(len(df_counts)):
            row = df_counts.iloc[idx]
            abs_counts = parse_pos_counts_string(row.get("Abstract_POS_Counts", ""))
            gen_counts = parse_pos_counts_string(row.get("Generated_Abstract_POS_Counts", ""))
            rows_pos_props.append(
                {
                    "Abstract_POS_Proportions": pos_proportions_string(abs_counts),
                    "Generated_Abstract_POS_Proportions": pos_proportions_string(gen_counts),
                    "Abstract_Word_Length": token_length_from_counts(abs_counts),
                    "Generated_Abstract_Word_Length": token_length_from_counts(gen_counts),
                }
            )

        df_pos_props = pd.DataFrame(rows_pos_props)
        df_pos_props.to_csv(OUT_POS_PROPORTIONS_CSV_PATH, index=False, encoding="utf-8")
        print(f"Wrote {len(rows_pos_props)} rows to {OUT_POS_PROPORTIONS_CSV_PATH}")

        write_pos_differences_csv(df_pos_props)
        return

    df = pd.read_csv(CSV_PATH, encoding="utf-8")
    nlp = load_nlp()

    rows_out = []
    rows_pos = []
    rows_pos_counts = []
    rows_pos_props = []

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
        abs_len = 0 if doc_abs is None else len(doc_abs)
        abs_props = pos_proportions_string(Counter(t.pos_ for t in doc_abs)) if doc_abs is not None else ""

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
        gen_len = 0 if doc_gen is None else len(doc_gen)
        gen_props = pos_proportions_string(Counter(t.pos_ for t in doc_gen)) if doc_gen is not None else ""

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
        rows_pos_props.append(
            {
                "Abstract_POS_Proportions": abs_props,
                "Generated_Abstract_POS_Proportions": gen_props,
                "Abstract_Word_Length": abs_len,
                "Generated_Abstract_Word_Length": gen_len,
            }
        )

    pd.DataFrame(rows_out).to_csv(OUT_CSV_PATH, index=False, encoding="utf-8")
    print(f"\nWrote {len(rows_out)} rows to {OUT_CSV_PATH}")

    pd.DataFrame(rows_pos).to_csv(OUT_POS_CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos)} rows to {OUT_POS_CSV_PATH}")

    pd.DataFrame(rows_pos_counts).to_csv(OUT_POS_COUNTS_CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos_counts)} rows to {OUT_POS_COUNTS_CSV_PATH}")

    df_pos_props = pd.DataFrame(rows_pos_props)
    df_pos_props.to_csv(OUT_POS_PROPORTIONS_CSV_PATH, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos_props)} rows to {OUT_POS_PROPORTIONS_CSV_PATH}")

    write_pos_differences_csv(df_pos_props)


if __name__ == "__main__":
    main()
