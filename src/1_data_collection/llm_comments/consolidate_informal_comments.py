"""
Consolidate Reddit and Flashback informal comments (human + AI-generated pairs)
into a single CSV with consistent schema and topic metadata.

Sources:
  Reddit   — LLM_consolidated_reddit_comments_MAY26.csv
  Flashback — flashback_ai_comments.csv

Output columns:
  source            "reddit" or "flashback"
  topic             subreddit name (reddit) or thread_title (flashback)
  question          thread question / OP text
  human_comment     human-written comment
  generated_comment AI-generated comment
  human_link        original Reddit URL  (reddit rows only)
  thread_id         Flashback thread ID  (flashback rows only)
  post_date         comment post date    (flashback rows only)

Usage:
  python consolidate_informal_comments.py [--out PATH]
"""
import argparse
import os
import re

import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
_data = os.path.dirname(_here)

REDDIT_CSV = os.path.join(_here, "LLM_consolidated_reddit_comments_MAY26.csv")
FLASHBACK_CSV = os.path.join(_here, "flashback_ai_comments.csv")
DEFAULT_OUT = os.path.join(_here, "consolidated_informal_comments.csv")


def extract_subreddit(url: str) -> str:
    """Return the subreddit name from a reddit.com URL, e.g. 'sweden'."""
    if not isinstance(url, str):
        return ""
    m = re.search(r"reddit\.com/r/([^/]+)", url, re.IGNORECASE)
    return m.group(1) if m else ""


def load_reddit(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={"human_link": "human_link"})
    df["source"] = "reddit"
    df["topic"] = df["human_link"].apply(extract_subreddit)
    df["thread_id"] = ""
    df["post_date"] = ""
    return df[["source", "topic", "question", "human_comment", "generated_comment",
               "human_link", "thread_id", "post_date"]]


def load_flashback(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    df = df.rename(columns={
        "comment": "human_comment",
        "Generated_OpenAI_Comment": "generated_comment",
        "thread_title": "topic",
    })
    df["source"] = "flashback"
    df["human_link"] = ""
    df = df.rename(columns={"thread_id": "thread_id", "post_date": "post_date"})
    return df[["source", "topic", "question", "human_comment", "generated_comment",
               "human_link", "thread_id", "post_date"]]


def main():
    parser = argparse.ArgumentParser(description="Consolidate Reddit + Flashback comment pairs")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path")
    args = parser.parse_args()

    reddit = load_reddit(REDDIT_CSV)
    flashback = load_flashback(FLASHBACK_CSV)

    combined = pd.concat([reddit, flashback], ignore_index=True)

    combined.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Written {len(combined)} rows to {args.out}")
    print(f"  Reddit rows    : {len(reddit)}  (topics: {sorted(reddit['topic'].unique())})")
    print(f"  Flashback rows : {len(flashback)}  (unique thread topics: {flashback['topic'].nunique()})")


if __name__ == "__main__":
    main()
