"""
Repair consolidated_reddit_comments.csv:
1. Remove rows whose Reddit post ID has 7+ characters (post-2018 posts that
   slipped through PullPush when it returned created_utc=0 for recent posts).
2. Re-scrape pre-2017 data from Arctic Shift API (more reliable than PullPush)
   for all four target subreddits.
3. Deduplicate against already-valid rows by post ID and comment ID.
4. Write back to consolidated_reddit_comments.csv.
"""

import csv
import os
import re
import time

import requests

try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("Warning: langdetect not installed. No language filtering.\n")

SUBREDDITS         = ["sweden", "svenskpolitik", "Stockholm", "Gothenburg"]
MAX_COMMENTS_POST  = 5
REQUEST_DELAY      = 1.2
TARGET_NEW_ROWS    = 100     # stop once we've collected this many new comment rows
MAX_POSTS_SCANNED  = 150     # max new posts to inspect per subreddit (prevents hour-long runs)

ARCTIC_BASE = "https://arctic-shift.photon-reddit.com/api"
BEFORE_2017 = 1483228800    # 2017-01-01 00:00:00 UTC

HEADERS    = {"User-Agent": "thesis-repair/1.0 (academic research)"}
POST_ID_RE = re.compile(r"/comments/(\w+)/")
CMT_ID_RE  = re.compile(r"/comments/\w+/_/(\w+)/")


# -- helpers ------------------------------------------------------------------

def get_json(url, params=None, retries=3):
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if r.status_code == 429:
                wait = 20 * attempt
                print(f"  Rate-limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError as exc:
            print(f"  Connection error (attempt {attempt}): {exc}")
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt})")
        except Exception as exc:
            print(f"  Request failed (attempt {attempt}): {exc}")
        time.sleep(3)
    return None


def is_swedish(text):
    if not HAS_LANGDETECT:
        return True
    if not text or len(text.strip()) < 10:
        return False
    try:
        return detect(text) == "sv"
    except LangDetectException:
        return False


def build_link(subreddit, post_id, comment_id):
    return f"https://www.reddit.com/r/{subreddit}/comments/{post_id}/_/{comment_id}/"


# -- Arctic Shift fetching ----------------------------------------------------─

def fetch_new_posts(subreddit, skip_post_ids):
    """
    Walk backwards from 2017-01-01 through Arctic Shift, returning question
    posts whose IDs are not already in skip_post_ids.
    Stops after MAX_POSTS_SCANNED new (non-duplicate) posts are inspected.
    """
    posts       = []
    seen        = set(skip_post_ids)
    before_cur  = BEFORE_2017
    scanned     = 0

    print(f"  Querying Arctic Shift for r/{subreddit} before 2017...")

    while scanned < MAX_POSTS_SCANNED:
        params = {
            "subreddit": subreddit,
            "before":    before_cur,   # integer timestamp works for posts endpoint
            "limit":     100,
            "sort":      "desc",
        }
        data = get_json(f"{ARCTIC_BASE}/posts/search", params=params)
        if not data:
            break
        items = data if isinstance(data, list) else data.get("data", [])
        if not items:
            break

        for item in items:
            pid = item.get("id", "")
            if not pid or pid in seen:
                continue
            seen.add(pid)
            scanned += 1

            title = item.get("title", "")
            if "?" not in title:
                continue
            selftext  = item.get("selftext") or ""
            full_text = f"{title}\n{selftext}".strip() if selftext.strip() else title
            if not is_swedish(full_text):
                continue

            posts.append({
                "id":        pid,
                "subreddit": subreddit,
                "question":  full_text,
            })

            if scanned >= MAX_POSTS_SCANNED:
                break

        oldest_ts = min(int(i.get("created_utc", BEFORE_2017)) for i in items)
        if oldest_ts >= before_cur:
            break
        before_cur = oldest_ts
        time.sleep(REQUEST_DELAY)

    print(f"    Scanned {scanned} posts -> {len(posts)} new Swedish question posts")
    return posts


def fetch_comments(post, skip_comment_ids):
    """
    Fetch pre-2017 top-level Swedish comments for a post from Arctic Shift.
    Skips comment IDs in skip_comment_ids.
    """
    subreddit = post["subreddit"]
    post_id   = post["id"]

    # Arctic Shift comments endpoint: link_id alone is sufficient to scope to a
    # post; adding subreddit alongside link_id triggers a 400. We omit 'before'
    # here too (integer timestamps are rejected) and enforce the cutoff below via
    # the created_utc field on each returned comment.
    params = {
        "link_id": post_id,
        "limit":   100,   # Arctic Shift comments endpoint caps at 100
    }
    data = get_json(f"{ARCTIC_BASE}/comments/search", params=params)
    if not data:
        return []

    items   = data if isinstance(data, list) else data.get("data", [])
    results = []

    for item in items:
        cid = item.get("id", "")
        if not cid or cid in skip_comment_ids:
            continue

        parent_id = item.get("parent_id", "")
        if parent_id not in (f"t3_{post_id}", post_id):
            continue

        ts = int(item.get("created_utc", 0))
        if ts >= BEFORE_2017:
            continue

        body = (item.get("body") or "").strip()
        if not body or body in ("[deleted]", "[removed]"):
            continue
        if not is_swedish(body):
            continue

        results.append({
            "body":       body,
            "comment_id": cid,
            "link":       build_link(subreddit, post_id, cid),
        })
        if len(results) >= MAX_COMMENTS_POST:
            break

    return results


# -- main --------------------------------------------------------------------─

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path   = os.path.join(script_dir, "consolidated_reddit_comments.csv")

    # -- Step 1: load & filter ------------------------------------------------─
    print("Step 1: Filtering out recent (post-2018) rows ...")
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    valid_rows         = []
    existing_post_ids  = set()
    existing_cmt_ids   = set()
    removed            = 0

    for row in all_rows:
        link       = row.get("link", "")
        post_match = POST_ID_RE.search(link)
        cmt_match  = CMT_ID_RE.search(link)

        if post_match and len(post_match.group(1)) >= 7:
            removed += 1
            continue

        valid_rows.append(row)
        if post_match:
            existing_post_ids.add(post_match.group(1))
        if cmt_match:
            existing_cmt_ids.add(cmt_match.group(1))

    print(f"  Kept {len(valid_rows)} valid rows, removed {removed} recent rows.")

    # -- Step 2: re-scrape from Arctic Shift ----------------------------------
    print(f"\nStep 2: Re-scraping from Arctic Shift (target {TARGET_NEW_ROWS} new rows) ...")
    new_rows = []

    for subreddit in SUBREDDITS:
        if len(new_rows) >= TARGET_NEW_ROWS:
            break

        posts = fetch_new_posts(subreddit, skip_post_ids=existing_post_ids)

        for post in posts:
            if len(new_rows) >= TARGET_NEW_ROWS:
                break

            comments = fetch_comments(post, skip_comment_ids=existing_cmt_ids)
            for c in comments:
                new_rows.append({
                    "question": post["question"],
                    "comment":  c["body"],
                    "link":     c["link"],
                })
                existing_cmt_ids.add(c["comment_id"])

            existing_post_ids.add(post["id"])

            if comments:
                time.sleep(REQUEST_DELAY)

        print(f"  New rows so far: {len(new_rows)}")

    print(f"  Scraped {len(new_rows)} new comment rows from Arctic Shift.")

    # -- Step 3: write combined result ----------------------------------------─
    combined = valid_rows + new_rows
    print(f"\nStep 3: Writing {len(combined)} total rows to {csv_path} ...")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["question", "comment", "link"], quoting=csv.QUOTE_ALL
        )
        writer.writeheader()
        writer.writerows(combined)

    print("Done.")
    print(f"  Pre-2017 rows retained : {len(valid_rows)}")
    print(f"  New rows from Arctic   : {len(new_rows)}")
    print(f"  Total                  : {len(combined)}")
    print()
    print("Next step: re-run generate_consolidated_reddit_comments.py to rebuild")
    print("LLM_consolidated_reddit_comments_2.csv with the updated human comments.")


if __name__ == "__main__":
    main()
