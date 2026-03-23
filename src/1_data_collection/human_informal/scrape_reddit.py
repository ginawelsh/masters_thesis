# base url: https://arctic-shift.photon-reddit.com/
# Output: at least 20 Swedish questions, max 5 parent (top-level) Swedish comments per question.
# CSV aligns each comment to the question it replies to.

import csv
import json
import os
import requests

try:
    import langdetect
except ImportError:
    langdetect = None  # pip install langdetect for Swedish filter

MIN_QUESTIONS = 20
MAX_COMMENTS_PER_QUESTION = 5

reddit_base_url = "https://reddit.com"
posts_base = "https://arctic-shift.photon-reddit.com/api/posts/search?subreddit=sweden&before=2017-01-01&limit=100"
comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search?subreddit=sweden&before=2017-01-01"
MAX_POSTS_FETCH = 500  # upper bound when collecting posts to find enough questions


def send_request(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Failed to get data: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
        return None


def _is_swedish(text):
    if not text or not text.strip():
        return False
    if langdetect is None:
        return True
    try:
        return langdetect.detect(text) == "sv"
    except Exception:
        return False


def _is_swedish_question(text):
    """True if text contains a question mark and is detected as Swedish."""
    if not text or "?" not in text:
        return False
    return _is_swedish(text)


def _posts_list(response):
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        return (
            response.get("data")
            or response.get("posts")
            or (list(response.values())[0] if response else [])
        )
    return []


def _comments_list(comments_response):
    if comments_response is None:
        return []
    if isinstance(comments_response, list):
        return comments_response
    if isinstance(comments_response, dict):
        return (
            comments_response.get("data")
            or comments_response.get("comments")
            or (list(comments_response.values())[0] if comments_response else [])
        )
    return []


def _is_parent_comment(comment, post_link_id):
    """True if comment is a direct reply to the post (not to another comment)."""
    data = comment.get("data", comment)
    parent_id = (data.get("parent_id") or comment.get("parent_id") or "").strip()
    link_id = (data.get("link_id") or comment.get("link_id") or "").strip()
    # Post fullname is t3_<id>; top-level comments have parent_id == link_id (post)
    post_fullname = f"t3_{post_link_id}" if not str(post_link_id).startswith("t3_") else post_link_id
    return parent_id == post_fullname or parent_id == post_link_id or link_id == post_fullname


def main():
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
            if len(posts_by_id) >= MIN_QUESTIONS:
                break
        if len(posts_by_id) >= MIN_QUESTIONS:
            break
        last = posts_list[-1] if posts_list else {}
        after = last.get("created_utc")
        if after is None:
            break

    # Use exactly the first MIN_QUESTIONS questions (or all we have)
    question_ids = list(posts_by_id.keys())[:MIN_QUESTIONS]
    print(f"Using {len(question_ids)} questions (Swedish, with '?', target ≥ {MIN_QUESTIONS})")

    if len(question_ids) < MIN_QUESTIONS:
        print(f"Warning: only found {len(question_ids)} Swedish questions (target {MIN_QUESTIONS})")

    comment_rows = []
    for link_id in question_ids:
        question_text = posts_by_id.get(link_id, "")
        query = f"{comments_url}&link_id={link_id}"
        gather_comments = send_request(query)
        comments_list = _comments_list(gather_comments)

        parent_swedish = []
        for c in comments_list:
            if not isinstance(c, dict):
                continue
            if not _is_parent_comment(c, link_id):
                continue
            data = c.get("data", c)
            body = (data.get("body") or c.get("body") or "").strip()
            if not body:
                continue
            if not _is_swedish(body):
                continue
            parent_swedish.append(body)

        for body in parent_swedish[:MAX_COMMENTS_PER_QUESTION]:
            comment_rows.append({
                "link_id": link_id,
                "question": question_text,
                "comment": body,
            })

    out_path = os.path.join(os.path.dirname(__file__), "reddit_comments.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "question", "comment"])
        writer.writeheader()
        writer.writerows(comment_rows)

    print(f"Wrote {len(comment_rows)} comments (max {MAX_COMMENTS_PER_QUESTION} per question) to {out_path}")
    print(f"Questions covered: {len(question_ids)}. Each row aligns one comment to its question.")


if __name__ == "__main__":
    main()
