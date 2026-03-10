"""
Inspect generated vs human-written Reddit comments, organised by question.
Uses pandas to load both CSVs and print a side-by-side view per question.
"""

import os
import sys

import pandas as pd

# Paths
_data_dir = os.path.dirname(os.path.abspath(__file__))
_informal_dir = os.path.join(_data_dir, "informal")
DEFAULT_HUMAN_PATH = os.path.join(_informal_dir, "reddit_comments.csv")
DEFAULT_GENERATED_PATH = os.path.join(_data_dir, "reddit_comments_openai_100.csv")

# Display
QUESTION_TRUNCATE = 200
COMMENT_TRUNCATE = 300
SEP = "-" * 80


def load_and_label(human_path: str, generated_path: str):
    """Load both CSVs and add a 'source' column."""
    human = pd.read_csv(human_path, encoding="utf-8")
    human["source"] = "human"
    gen = pd.read_csv(generated_path, encoding="utf-8")
    gen["source"] = "generated"
    return human, gen


def _truncate(s: str, max_len: int, suffix: str = "…") -> str:
    if pd.isna(s) or not isinstance(s, str):
        return ""
    s = s.strip().replace("\n", " ")
    return s[:max_len] + suffix if len(s) > max_len else s


def inspect_by_question(
    human_path: str = DEFAULT_HUMAN_PATH,
    generated_path: str = DEFAULT_GENERATED_PATH,
    question_truncate: int = QUESTION_TRUNCATE,
    comment_truncate: int = COMMENT_TRUNCATE,
    link_id_filter=None,
):
    """
    Print human vs generated comments grouped by question.
    If link_id_filter is set, only show that link_id (e.g. '9lfs5').
    """
    human, gen = load_and_label(human_path, generated_path)

    # All unique link_ids from both; get question text from either table
    all_link_ids = pd.Index(human["link_id"].dropna().unique()).union(
        gen["link_id"].dropna().unique()
    )
    q_from_h = human.groupby("link_id")["question"].first()
    q_from_g = gen.groupby("link_id")["question"].first()
    questions = pd.DataFrame({"link_id": all_link_ids})
    questions["question"] = questions["link_id"].map(
        lambda lid: q_from_h.get(lid) or q_from_g.get(lid) or ""
    )
    questions = questions.sort_values("link_id")

    if link_id_filter:
        questions = questions[questions["link_id"] == link_id_filter]
        if questions.empty:
            print(f"No data for link_id={link_id_filter}")
            return

    for _, row in questions.iterrows():
        link_id = row["link_id"]
        q = row["question"]
        q_short = _truncate(str(q), question_truncate)
        print(SEP)
        print(f"link_id: {link_id}")
        print(f"Question: {q_short}")
        print(SEP)

        h_comments = human[human["link_id"] == link_id]["comment"]
        g_comments = gen[gen["link_id"] == link_id]["comment"]

        print("  HUMAN comments:")
        for i, c in enumerate(h_comments, 1):
            print(f"    [{i}] {_truncate(str(c), comment_truncate)}")
        if h_comments.empty:
            print("    (none)")

        print("  GENERATED comments:")
        for i, c in enumerate(g_comments, 1):
            print(f"    [{i}] {_truncate(str(c), comment_truncate)}")
        if g_comments.empty:
            print("    (none)")

        print()

    print(SEP)
    print("Summary:")
    print(f"  Human comments total: {len(human)}")
    print(f"  Generated comments total: {len(gen)}")
    print(f"  Questions shown: {len(questions)}")


def get_combined_df(
    human_path: str = DEFAULT_HUMAN_PATH,
    generated_path: str = DEFAULT_GENERATED_PATH,
) -> pd.DataFrame:
    """Return one DataFrame with all comments and columns: link_id, question, comment, source."""
    human, gen = load_and_label(human_path, generated_path)
    cols = [c for c in human.columns if c in gen.columns]
    return pd.concat([human[cols], gen[cols]], ignore_index=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Inspect human vs generated Reddit comments by question.")
    parser.add_argument("--human", default=DEFAULT_HUMAN_PATH, help="Path to human comments CSV")
    parser.add_argument("--generated", default=DEFAULT_GENERATED_PATH, help="Path to generated comments CSV")
    parser.add_argument("--link-id", type=str, default=None, help="Only show this link_id")
    parser.add_argument("--q-len", type=int, default=QUESTION_TRUNCATE, help="Max question length to print")
    parser.add_argument("--c-len", type=int, default=COMMENT_TRUNCATE, help="Max comment length to print")
    parser.add_argument("--csv", type=str, default=None, help="Save combined (human+generated) CSV to this path")
    args = parser.parse_args()

    if args.csv:
        combined = get_combined_df(args.human, args.generated)
        combined.to_csv(args.csv, index=False, encoding="utf-8")
        print(f"Saved combined CSV to {args.csv} ({len(combined)} rows)")
        return

    inspect_by_question(
        human_path=args.human,
        generated_path=args.generated,
        question_truncate=args.q_len,
        comment_truncate=args.c_len,
        link_id_filter=args.link_id,
    )


if __name__ == "__main__":
    main()
