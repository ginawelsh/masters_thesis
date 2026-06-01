"""
Read consolidated_reddit_comments.csv, generate one AI comment per row,
and output LLM_consolidated_reddit_comments_2.csv with columns:
question, human_comment, human_link, generated_comment.
All fields are flattened to a single line (no embedded newlines).
"""

import csv
import os
import sys
from openai import OpenAI

# Project root for .env
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

# Connect using API key
_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. Set it in the environment or in a .env file in the project root."
    )
client = OpenAI(api_key=_api_key)

COMMENT_PROMPT = (
    "Svara på följande fråga med en svensk kommentar, i stil med en kommentar på ett svenskt forum. "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)


def generate_comment(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-5.2",
        messages=[{"role": "user", "content": COMMENT_PROMPT.format(question=question)}],
    )
    return (response.choices[0].message.content or "").strip()


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    human_dir = os.path.join(
        os.path.dirname(script_dir), "human_informal"
    )
    input_path = os.path.join(human_dir, "consolidated_reddit_comments.csv")
    out_path = os.path.join(script_dir, "LLM_consolidated_reddit_comments_2.csv")

    def strip_newlines(text: str) -> str:
        return " ".join(text.split())

    # Read all real comment rows, stripping newlines from fields
    with open(input_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        real_rows = [
            {k: strip_newlines(v) for k, v in row.items()}
            for row in reader
        ]

    print(f"Loaded {len(real_rows)} rows from {input_path}")

    # Generate one AI comment per row (including duplicate questions)
    print(f"Making {len(real_rows)} API calls (one per row)")

    consolidated = []
    for i, row in enumerate(real_rows):
        question = row["question"]
        print(f"[{i+1}/{len(real_rows)}] Generating for: {question[:60]!r}...")
        try:
            generated = strip_newlines(generate_comment(question))
        except Exception as e:
            print(f"  Error: {e}")
            generated = ""
        consolidated.append({
            "question":          question,
            "human_comment":     row["comment"],
            "human_link":        row.get("link", ""),
            "generated_comment": generated,
        })

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "human_comment", "human_link", "generated_comment"])
        writer.writeheader()
        writer.writerows(consolidated)

    print(f"Wrote {len(consolidated)} rows to {out_path}")


if __name__ == "__main__":
    main()
