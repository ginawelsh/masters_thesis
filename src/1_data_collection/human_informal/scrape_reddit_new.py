"""
Swedish Reddit comment scraper — pre-2017 data
Uses Arctic Shift as primary API, falls back to PullPush if unreachable.

Install deps:  pip install requests langdetect
"""

import csv
import os
import time
import requests
from datetime import datetime

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed (pip install langdetect). No language filtering applied.\n")

# ── Config ────────────────────────────────────────────────────────────────────
SUBREDDIT                = "sweden"
BEFORE_DATE              = "2017-01-01"          # collect only data before this date
BEFORE_TS                = int(datetime.strptime(BEFORE_DATE, "%Y-%m-%d").timestamp())
TARGET_QUESTIONS         = 20
MAX_COMMENTS_PER_QUESTION = 5
REQUEST_DELAY            = 0.6                   # seconds between API calls
OUT_FILE                 = "reddit_comments.csv"

# ── API backends ──────────────────────────────────────────────────────────────
# Arctic Shift: best for historical data, supports link_id filtering on comments
ARCTIC = "https://arctic-shift.photon-reddit.com/api"

# PullPush: alternative Pushshift replacement (pullpush.io)
PULLPUSH = "https://api.pullpush.io/reddit/search"

# ── Helpers ───────────────────────────────────────────────────────────────────

def get(url, params=None, timeout=15):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError as e:
        print(f"  Connection error: {e}")
        return None
    except requests.exceptions.Timeout:
        print(f"  Timeout on {url}")
        return None
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def probe_arctic():
    """Return True if Arctic Shift is reachable."""
    r = get(f"{ARCTIC}/posts/search", params={"subreddit": SUBREDDIT, "limit": 1, "before": BEFORE_DATE})
    return r is not None


def extract_list(response, keys=("data", "posts", "comments")):
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        for key in keys:
            if key in response and isinstance(response[key], list):
                return response[key]
        for v in response.values():
            if isinstance(v, list):
                return v
    return []


def is_swedish(text):
    if not text or not text.strip():
        return False
    if not HAS_LANGDETECT:
        return True
    try:
        return detect(text) == "sv"
    except LangDetectException:
        return False


def is_swedish_question(title, selftext=""):
    full = f"{title}\n{selftext}".strip()
    return "?" in full and is_swedish(full)


def is_top_level(comment, post_id):
    parent = comment.get("parent_id", "")
    return parent in (f"t3_{post_id}", post_id)


def ts_before(item):
    """Return True if item's created_utc is strictly before BEFORE_TS."""
    ts = item.get("created_utc", 0)
    try:
        return int(ts) < BEFORE_TS
    except (ValueError, TypeError):
        return False


# ── Arctic Shift backend ──────────────────────────────────────────────────────

def arctic_fetch_questions(target):
    print(f"[Arctic Shift] Fetching posts from r/{SUBREDDIT} before {BEFORE_DATE}...")
    questions = {}
    before = BEFORE_DATE  # moves backwards as a Unix timestamp string after first page

    while len(questions) < target:
        resp = get(f"{ARCTIC}/posts/search", params={
            "subreddit": SUBREDDIT,
            "before":    before,
            "limit":     100,
            "sort":      "desc",
        })
        posts = extract_list(resp)
        if not posts:
            print("  No more posts.")
            break

        for p in posts:
            # Double-check timestamp — API occasionally returns items slightly
            # outside the requested window on page boundaries
            if not ts_before(p):
                continue
            pid      = p.get("id", "")
            title    = p.get("title", "")
            selftext = p.get("selftext", "") or ""
            if pid and is_swedish_question(title, selftext):
                questions[pid] = {"title": title, "selftext": selftext}
                if len(questions) >= target:
                    break

        # Advance cursor: one second before the oldest post in this batch
        oldest_ts = posts[-1].get("created_utc")
        if not oldest_ts:
            break
        before = str(int(oldest_ts) - 1)
        print(f"  {len(questions)}/{target} questions found...")
        time.sleep(REQUEST_DELAY)

    return questions


def arctic_fetch_comments(post_id):
    resp = get(f"{ARCTIC}/comments/search", params={
        "subreddit": SUBREDDIT,
        "link_id":   post_id,
        "before":    BEFORE_DATE,
        "limit":     100,
    })
    all_comments = extract_list(resp)

    swedish_top = []
    for c in all_comments:
        if not ts_before(c):          # extra guard: skip anything post-2017
            continue
        if not is_top_level(c, post_id):
            continue
        body = (c.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]", "[not found in archive]"):
            continue
        if is_swedish(body):
            swedish_top.append(body)
        if len(swedish_top) >= MAX_COMMENTS_PER_QUESTION:
            break

    return swedish_top


# ── PullPush backend ──────────────────────────────────────────────────────────

def pullpush_fetch_questions(target):
    print(f"[PullPush] Fetching posts from r/{SUBREDDIT} before {BEFORE_DATE}...")
    questions = {}
    before = BEFORE_TS

    while len(questions) < target:
        resp = get(f"{PULLPUSH}/submission", params={
            "subreddit": SUBREDDIT,
            "before":    before,
            "size":      100,
            "sort":      "desc",
            "sort_type": "created_utc",
        })
        posts = extract_list(resp, keys=("data",))
        if not posts:
            print("  No more posts.")
            break

        for p in posts:
            if not ts_before(p):
                continue
            pid      = p.get("id", "")
            title    = p.get("title", "")
            selftext = p.get("selftext", "") or ""
            if pid and is_swedish_question(title, selftext):
                questions[pid] = {"title": title, "selftext": selftext}
                if len(questions) >= target:
                    break

        oldest_ts = posts[-1].get("created_utc")
        if not oldest_ts:
            break
        before = int(oldest_ts) - 1
        print(f"  {len(questions)}/{target} questions found...")
        time.sleep(REQUEST_DELAY)

    return questions


def pullpush_fetch_comments(post_id):
    resp = get(f"{PULLPUSH}/comment", params={
        "link_id": f"t3_{post_id}",
        "before":  BEFORE_TS,
        "size":    100,
    })
    all_comments = extract_list(resp, keys=("data",))

    swedish_top = []
    for c in all_comments:
        if not ts_before(c):
            continue
        if not is_top_level(c, post_id):
            continue
        body = (c.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        if is_swedish(body):
            swedish_top.append(body)
        if len(swedish_top) >= MAX_COMMENTS_PER_QUESTION:
            break

    return swedish_top


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Choose backend
    print("Checking API availability...")
    if probe_arctic():
        print("Arctic Shift is reachable — using it as primary.\n")
        fetch_questions = arctic_fetch_questions
        fetch_comments  = arctic_fetch_comments
    else:
        print("Arctic Shift unreachable — falling back to PullPush.\n")
        fetch_questions = pullpush_fetch_questions
        fetch_comments  = pullpush_fetch_comments

    questions = fetch_questions(TARGET_QUESTIONS)
    if not questions:
        print("No questions found. Check your network or try the other API.")
        return

    if len(questions) < TARGET_QUESTIONS:
        print(f"Warning: only found {len(questions)} Swedish questions (target {TARGET_QUESTIONS}).")

    rows = []
    total = len(questions)
    for i, (post_id, post) in enumerate(questions.items(), 1):
        full_question = f"{post['title']}\n{post['selftext']}".strip()
        print(f"[{i}/{total}] Fetching comments for post {post_id}...")
        comments = fetch_comments(post_id)
        print(f"  → {len(comments)} Swedish top-level comments (pre-{BEFORE_DATE})")
        for body in comments:
            rows.append({
                "link_id":  post_id,
                "question": full_question,
                "comment":  body,
            })
        time.sleep(REQUEST_DELAY)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUT_FILE)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "question", "comment"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {len(rows)} rows to {out_path}")
    print(f"Posts covered: {total}, avg {len(rows)/max(total,1):.1f} comments/post")


if __name__ == "__main__":
    main()