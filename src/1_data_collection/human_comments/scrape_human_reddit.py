"""
Swedish Reddit comment scraper — historical data via PullPush API
Targets r/sweden, r/svenskpolitik, r/Stockholm, r/Gothenburg.
Collects top-level comments from BEFORE 2017-01-01 that are in Swedish.

PullPush is a community-maintained Pushshift mirror:
  https://api.pullpush.io

Output: NEW_REAL_reddit_comments.csv  (columns: question, comment, link)

Install deps:  pip install requests langdetect
"""

import csv
import os
import time
import requests

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed (pip install langdetect). No language filtering.\n")

# ── Config ────────────────────────────────────────────────────────────────────
SUBREDDITS            = ["sweden", "svenskpolitik", "Stockholm", "Gothenburg"]
TARGET_COMMENTS_TOTAL = 100
MAX_COMMENTS_PER_POST = 5
REQUEST_DELAY         = 1.2          # seconds between API calls
OUT_FILE              = "NEW_REAL_reddit_comments.csv"

# Column name used in the analysis scripts (resolve_config human_col)
HUMAN_COL             = "real_comment"

# Hard ceiling: ONLY comments strictly before 2017-01-01 00:00:00 UTC
BEFORE_2017_TS        = 1483228800   # unix timestamp for 2017-01-01 00:00:00 UTC

PULLPUSH_BASE         = "https://api.pullpush.io/reddit/search"
HEADERS               = {
    "User-Agent": "thesis-scraper/1.0 (academic research; contact: thesis@example.com)"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_json(url, params=None, retries=3):
    """Fetch JSON from url, returning None on failure."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 15 * attempt
                print(f"  Rate-limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError as e:
            print(f"  Connection error (attempt {attempt}): {e}")
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt}): {url}")
        except Exception as e:
            print(f"  Request failed (attempt {attempt}): {e}")
        time.sleep(3)
    return None


def is_swedish(text):
    """Return True if langdetect identifies the text as Swedish."""
    if not HAS_LANGDETECT:
        return True
    if not text or len(text.strip()) < 10:
        return False
    try:
        return detect(text) == "sv"
    except LangDetectException:
        return False


def is_question_post(title):
    """A post is a question if its title contains '?'."""
    return "?" in title


def build_comment_link(subreddit, post_id, comment_id):
    """Return a direct permalink to a specific comment."""
    return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/_/{comment_id}/"


# ── PullPush fetching ─────────────────────────────────────────────────────────

def fetch_question_posts_historical(subreddit, min_posts=30):
    """
    Fetch question posts (title contains '?') from PullPush,
    strictly before 2017-01-01.

    PullPush paginates via the 'before' timestamp of the last-seen post.
    We walk backwards from the ceiling (2017-01-01) through all available data.
    """
    posts = []
    seen  = set()
    before_cursor = BEFORE_2017_TS  # start at the ceiling and walk backwards

    print(f"  Querying PullPush for r/{subreddit} posts before 2017...")

    while len(posts) < min_posts:
        params = {
            "subreddit": subreddit,
            "before":    before_cursor,
            "size":      100,           # PullPush uses 'size' not 'limit'
            "sort":      "desc",
            "sort_type": "created_utc",
        }
        data = get_json(f"{PULLPUSH_BASE}/submission/", params=params)

        if not data:
            print("  No response from PullPush — stopping post pagination.")
            break

        items = data if isinstance(data, list) else data.get("data", [])
        if not items:
            print("  No more posts returned.")
            break

        for item in items:
            pid   = item.get("id", "")
            title = item.get("title", "")
            ts    = int(item.get("created_utc", 0))

            # Enforce the hard ceiling
            if ts >= BEFORE_2017_TS:
                continue
            if not pid or pid in seen:
                continue
            seen.add(pid)

            if is_question_post(title):
                posts.append({
                    "id":        pid,
                    "title":     title,
                    "selftext":  item.get("selftext") or "",
                    "subreddit": subreddit,
                    "created":   ts,
                })

        # Advance the cursor to just before the oldest post we saw
        oldest_ts = min(int(i.get("created_utc", BEFORE_2017_TS)) for i in items)
        if oldest_ts >= before_cursor:
            break
        before_cursor = oldest_ts

        time.sleep(REQUEST_DELAY)

    return posts


def fetch_swedish_comments_historical(post):
    """
    Fetch top-level comments for a post via PullPush,
    keeping only Swedish comments posted before 2017-01-01.

    PullPush filters comments by parent post using the 'link_id' parameter
    (Reddit fullname: t3_<post_id>).
    """
    subreddit = post["subreddit"]
    post_id   = post["id"]
    results   = []

    params = {
        "link_id": post_id,   # PullPush expects bare post ID, no t3_ prefix
        "size":    200,
    }
    data = get_json(f"{PULLPUSH_BASE}/comment/", params=params)

    if not data:
        return []

    items = data if isinstance(data, list) else data.get("data", [])

    for item in items:
        # Only top-level comments (parent_id == the post's fullname)
        parent_id = item.get("parent_id", "")
        if parent_id != f"t3_{post_id}":
            continue

        ts = int(item.get("created_utc", 0))
        if ts >= BEFORE_2017_TS:
            continue

        body       = item.get("body", "")
        comment_id = item.get("id", "")

        if not body or body in ("[deleted]", "[removed]"):
            continue
        if not is_swedish(body):
            continue

        link = build_comment_link(subreddit, post_id, comment_id)
        results.append({
            "body":       body,
            "comment_id": comment_id,
            "link":       link,
            "created":    ts,
        })

        if len(results) >= MAX_COMMENTS_PER_POST:
            break

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("NOTE: Collecting ONLY Swedish Reddit comments from before 2017-01-01.")
    print(f"      Using PullPush API: {PULLPUSH_BASE}\n")

    all_rows = []

    for subreddit in SUBREDDITS:
        if len(all_rows) >= TARGET_COMMENTS_TOTAL:
            break

        remaining = TARGET_COMMENTS_TOTAL - len(all_rows)
        min_posts = max(10, (remaining // MAX_COMMENTS_PER_POST) + 5)
        print(f"\n=== r/{subreddit} — looking for ~{min_posts} pre-2017 question posts ===")

        posts = fetch_question_posts_historical(subreddit, min_posts=min_posts)
        print(f"  Found {len(posts)} question posts in r/{subreddit} before 2017")

        for i, post in enumerate(posts, 1):
            if len(all_rows) >= TARGET_COMMENTS_TOTAL:
                break

            question_text = post["title"]
            if post["selftext"].strip():
                question_text = question_text + "\n\n" + post["selftext"]

            print(f"  [{i}/{len(posts)}] {post['id']} — {post['title'][:60]!r}")
            comments = fetch_swedish_comments_historical(post)
            print(f"    → {len(comments)} Swedish pre-2017 top-level comment(s)")

            for c in comments:
                all_rows.append({
                    "question":   question_text,
                    HUMAN_COL:    c["body"],
                    "link":       c["link"],
                })

            time.sleep(REQUEST_DELAY)

    # Write output
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path   = os.path.join(script_dir, OUT_FILE)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["question", HUMAN_COL, "link"],
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nDone. Wrote {len(all_rows)} rows to:\n  {out_path}")
    print("\nSample links for manual verification:")
    for row in all_rows[:3]:
        print(f"  {row['link']}")


if __name__ == "__main__":
    main()
