"""
Gather Reddit-style comments via OpenAI using the same API and prompt system as
scrape_reddit.py and generate_text_openai.py.

- Fetches Swedish questions from the same comments_url / posts API (r/sweden, before 2017).
- Generates 3 parent comments per question using OpenAI (same prompt as informal).
- Outputs CSV with link_id, question, comment. Aims for 100 comments total.
"""

import csv
import os
import sys

# Project root for .env
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_root, ".env"))
except ImportError:
    pass

from openai import OpenAI

# Reuse Reddit API and question-fetching from scrape_reddit
from scrape_reddit import (
    posts_base,
    send_request,
    _posts_list,
    _is_swedish_question,
    MAX_POSTS_FETCH,
)

TARGET_COMMENTS = 100
COMMENTS_PER_QUESTION = 3
MIN_QUESTIONS_NEEDED = (TARGET_COMMENTS + COMMENTS_PER_QUESTION - 1) // COMMENTS_PER_QUESTION  # 34

_api_key = os.environ.get("OPENAI_API_KEY")
if not _api_key:
    raise ValueError(
        "OPENAI_API_KEY not set. Set it in the environment or in a .env file in the project root."
    )
client = OpenAI(api_key=_api_key)

COMMENT_PROMPT = (
    "Svara på följande fråga med en kort, avslappnad svensk kommentar (som på ett forum). "
    "Skriv bara kommentaren, inget annat.\n\nFråga: {question}"
)


def generate_comment(question: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": COMMENT_PROMPT.format(question=question)}],
    )
    return (response.choices[0].message.content or "").strip()


def fetch_questions(min_questions: int):
    """Fetch Swedish questions from same API as scrape_reddit. Returns list of (link_id, question_text)."""
    post_ids = []
    posts_by_id = {}
    after = None
    while len(post_ids) < MAX_POSTS_FETCH:
        posts_url = f"{posts_base}&sort=asc"
        if after is not None:
            posts_url += f"&after={after}"
        gather_posts = send_request(posts_url)
        posts_list = _posts_list(gather_posts)
        if not posts_list:
            break
        for p in posts_list:
            if not isinstance(p, dict) or "id" not in p:
                continue
            pid = p["id"]
            if pid in posts_by_id:
                continue
            title = p.get("title") or ""
            selftext = p.get("selftext") or ""
            full_text = f"{title}\n{selftext}".strip()
            if not _is_swedish_question(full_text):
                continue
            post_ids.append(pid)
            posts_by_id[pid] = full_text
            if len(posts_by_id) >= min_questions:
                break
        if len(posts_by_id) >= min_questions:
            break
        last = posts_list[-1] if posts_list else {}
        after = last.get("created_utc")
        if after is None:
            break
    question_ids = list(posts_by_id.keys())[:min_questions]
    return [(link_id, posts_by_id[link_id]) for link_id in question_ids]


def main():
    data_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(data_dir, "reddit_comments_openai_100.csv")

    print(f"Fetching at least {MIN_QUESTIONS_NEEDED} questions (same API as scrape_reddit)...")
    questions = fetch_questions(MIN_QUESTIONS_NEEDED)
    if len(questions) < MIN_QUESTIONS_NEEDED:
        print(f"Warning: only found {len(questions)} questions (need {MIN_QUESTIONS_NEEDED})")

    comment_rows = []
    for idx, (link_id, question_text) in enumerate(questions):
        if len(comment_rows) >= TARGET_COMMENTS:
            break
        n_this = min(COMMENTS_PER_QUESTION, TARGET_COMMENTS - len(comment_rows))
        print(f"Question {idx + 1}/{len(questions)} (link_id={link_id}): generating {n_this} comments...")
        for _ in range(n_this):
            try:
                comment = generate_comment(question_text)
                comment_rows.append({
                    "link_id": link_id,
                    "question": question_text,
                    "comment": comment,
                })
            except Exception as e:
                print(f"  Error: {e}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "question", "comment"])
        writer.writeheader()
        writer.writerows(comment_rows)

    print(f"Wrote {len(comment_rows)} comments to {out_path} (target {TARGET_COMMENTS})")


if __name__ == "__main__":
    main()
