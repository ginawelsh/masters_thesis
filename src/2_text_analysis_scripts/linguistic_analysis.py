"""
Linguistic analysis (spaCy) on Abstract and Generated Abstract from formal CSV.
Install: pip install -r requirements.txt
Swedish: python -m spacy download sv_core_news_sm
"""
import argparse
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
HUMAN_INFORMAL_DIR = os.path.join(_root, "1_data_collection", "human_informal")
LLM_INFORMAL_DIR = os.path.join(_root, "1_data_collection", "llm_informal")

TOKEN_SEP = ","  # separator for tokens/POS in CSV cells
POS_SEP = " | "  # separator for TAG:n lists in CSV cells (matches existing checked-in counts file)
VERBOSE = False  # set True to print per-row token/POS debug output


def parse_args():
    parser = argparse.ArgumentParser(description="POS analysis for formal/informal paired datasets")
    parser.add_argument(
        "--dataset",
        choices=["formal", "informal"],
        default="formal",
        help="Preset input/output config.",
    )
    return parser.parse_args()


def resolve_config(dataset: str) -> dict:
    if dataset == "informal":
        input_csv = os.path.join(LLM_INFORMAL_DIR, "reddit_comments_openai.csv")
        output_dir = HUMAN_INFORMAL_DIR
        return {
            "csv_path": input_csv,
            "out_tokens": os.path.join(output_dir, "linguistic_analysis_tokens.csv"),
            "out_pos": os.path.join(output_dir, "linguistic_analysis_pos.csv"),
            "out_pos_counts": os.path.join(output_dir, "linguistic_analysis_pos_counts.csv"),
            "out_pos_proportions": os.path.join(output_dir, "linguistic_analysis_pos_proportions.csv"),
            "out_pos_differences": os.path.join(output_dir, "linguistic_analysis_pos_differences.csv"),
            "human_col": "comment",
            "generated_col_candidates": ["Generated_OpenAI_Comment"],
            "human_label": "Comment",
            "generated_label": "Generated Comment",
        }

    input_csv = os.path.join(HUMAN_FORMAL_DIR, "sv_human_collection_with_kws.csv")
    output_dir = HUMAN_FORMAL_DIR
    return {
        "csv_path": input_csv,
        "out_tokens": os.path.join(output_dir, "linguistic_analysis_tokens.csv"),
        "out_pos": os.path.join(output_dir, "linguistic_analysis_pos.csv"),
        "out_pos_counts": os.path.join(output_dir, "linguistic_analysis_pos_counts.csv"),
        "out_pos_proportions": os.path.join(output_dir, "linguistic_analysis_pos_proportions.csv"),
        "out_pos_differences": os.path.join(output_dir, "linguistic_analysis_pos_differences.csv"),
        "human_col": "Abstract",
        "generated_col_candidates": ["Generated_Abstract", "Generated Abstract"],
        "human_label": "Abstract",
        "generated_label": "Generated Abstract",
    }


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


def write_pos_differences_csv(df_pos_props: pd.DataFrame, out_path: str, ndigits: int = 4) -> None:
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
    pd.DataFrame(out_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_rows)} rows to {out_path}")


def main():
    args = parse_args()
    cfg = resolve_config(args.dataset)
    csv_path = cfg["csv_path"]
    out_csv_path = cfg["out_tokens"]
    out_pos_csv_path = cfg["out_pos"]
    out_pos_counts_csv_path = cfg["out_pos_counts"]
    out_pos_proportions_csv_path = cfg["out_pos_proportions"]
    out_pos_differences_csv_path = cfg["out_pos_differences"]
    human_col = cfg["human_col"]
    generated_col_candidates = cfg["generated_col_candidates"]
    human_label = cfg["human_label"]
    generated_label = cfg["generated_label"]

    # Fast path: if POS counts file already exists, derive proportions + lengths directly from it.
    # This guarantees we match the checked-in counts file format and avoids rerunning spaCy.
    if os.path.exists(out_pos_counts_csv_path):
        df_counts = pd.read_csv(out_pos_counts_csv_path, encoding="utf-8")
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
        df_pos_props.to_csv(out_pos_proportions_csv_path, index=False, encoding="utf-8")
        print(f"Wrote {len(rows_pos_props)} rows to {out_pos_proportions_csv_path}")

        write_pos_differences_csv(df_pos_props, out_pos_differences_csv_path)
        return

    df = pd.read_csv(csv_path, encoding="utf-8")
    nlp = load_nlp()

    rows_out = []
    rows_pos = []
    rows_pos_counts = []
    rows_pos_props = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        abs_text = row.get(human_col)
        gen_text = None
        for c in generated_col_candidates:
            gen_text = row.get(c)
            if gen_text is not None and str(gen_text).strip():
                break

        if VERBOSE:
            print(f"\n--- Row {idx + 1} ---")
        doc_abs = process_text(nlp, abs_text)
        if doc_abs is not None:
            abs_tokens = TOKEN_SEP.join(t.text for t in doc_abs)
            abs_pos = TOKEN_SEP.join(f"{t.text}_{t.pos_}" for t in doc_abs)
            if VERBOSE:
                print(f"[{human_label}] Tokens:", [t.text for t in doc_abs], "...")
                print(f"[{human_label}] POS:", [(t.text, t.pos_) for t in doc_abs])
                if doc_abs.ents:
                    print(f"[{human_label}] Entities:", [(e.text, e.label_) for e in doc_abs.ents])
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
            if VERBOSE:
                print(f"[{generated_label}] Tokens:", [t.text for t in doc_gen], "...")
                print(f"[{generated_label}] POS:", [(t.text, t.pos_) for t in doc_gen])
                if doc_gen.ents:
                    print(f"[{generated_label}] Entities:", [(e.text, e.label_) for e in doc_gen.ents])
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

    pd.DataFrame(rows_out).to_csv(out_csv_path, index=False, encoding="utf-8")
    print(f"\nWrote {len(rows_out)} rows to {out_csv_path}")

    pd.DataFrame(rows_pos).to_csv(out_pos_csv_path, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos)} rows to {out_pos_csv_path}")

    pd.DataFrame(rows_pos_counts).to_csv(out_pos_counts_csv_path, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos_counts)} rows to {out_pos_counts_csv_path}")

    df_pos_props = pd.DataFrame(rows_pos_props)
    df_pos_props.to_csv(out_pos_proportions_csv_path, index=False, encoding="utf-8")
    print(f"Wrote {len(rows_pos_props)} rows to {out_pos_proportions_csv_path}")

    write_pos_differences_csv(df_pos_props, out_pos_differences_csv_path)


if __name__ == "__main__":
    main()
