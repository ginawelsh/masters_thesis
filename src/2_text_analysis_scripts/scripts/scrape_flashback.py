import cloudscraper
from bs4 import BeautifulSoup
import csv
import time
import re
import random
from datetime import datetime

scraper = cloudscraper.create_scraper()

BASE_URL = "https://www.flashback.org"
CUTOFF_DATE = datetime(2017, 1, 1)
TARGET_COMMENTS = 1000
OUTPUT_FILE = "flashback_qa.csv"
MAX_PER_THREAD = 30       # max comments per thread — keeps corpus diverse
DELAY = 2.5               # seconds between requests

# Pre-2017 threads have lower IDs. Thread 474161 was confirmed pre-2017.
# Current threads are ~3.7M. Sampling below 2.5M targets mostly pre-2017 content.
PRE_2017_ID_MIN = 200_000
PRE_2017_ID_MAX = 2_500_000

DEBUG_DATES = False  # set True to print every parsed date

# align month time with swedish months
SWEDISH_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "maj": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "okt": 10, "nov": 11, "dec": 12,
}

def fetch(url, retries=5):
    for attempt in range(retries):
        try:
            # get post
            resp = scraper.get(url, timeout=15)
            # if successful
            if resp.status_code == 200:
                # return text of post
                return resp.text
            if resp.status_code == 429:
                # delay (server rate limit)
                wait = 30 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None  # thread doesn't exist, don't retry
            print(f"  HTTP {resp.status_code}: {url}")
        except Exception as e:
            print(f"  Error: {e}")
        time.sleep(5 * (attempt + 1))
    return None

def parse_date(text):
    text = text.strip()
    # regex to match date
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if m:
        try:
            # convert into datetime proper format
            result = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if DEBUG_DATES:
                print(f"    [date] {result.date()}")
            return result
        except ValueError:
            pass
    # search for swedish text?
    m = re.search(r"(\d{1,2})\s+([a-zåäö]+)\s+(\d{4})", text, re.IGNORECASE)
    if m:
        month = SWEDISH_MONTHS.get(m.group(2).lower()[:3])
        if month:
            try:
                result = datetime(int(m.group(3)), month, int(m.group(1)))
                if DEBUG_DATES:
                    print(f"    [date] {result.date()}")
                return result
            except ValueError:
                pass
    return None  # relative dates ("Igår", "Idag") and unrecognised — skip

def get_post_number(post):
    el = post.select_one("a[id^='postcount']")
    if el:
        try:
            return int(el.get("name", 0))
        except (ValueError, TypeError):
            pass
    return 0

def extract_post_text(post):
    msg = post.select_one("div.post_message")
    if not msg:
        return None
    # Remove quoted blocks — copies of other users' posts, not the commenter's own text.
    # Everything else is preserved exactly as written: spelling, punctuation, spacing.
    for quote in msg.select("div.post-bbcode-quote-wrapper"):
        quote.decompose()
    return msg.get_text(separator="\n", strip=False)

def scan_thread(thread_id):
    """Fetch page 1. Return (title, html) if title is a question, else (None, None)."""
    html = fetch(f"{BASE_URL}/t{thread_id}")
    if not html:
        return None, None
    soup = BeautifulSoup(html, "html.parser")
    title_el = soup.select_one(".page-title h1 a")
    if not title_el:
        return None, None
    title = title_el.get_text(strip=True)
    if "?" not in title:
        return None, None
    return title, html

def get_replies(thread_id, thread_title, first_page_html):
    """Collect pre-2017 replies from a question thread, paired with the OP question text."""
    comments = []
    page = 1
    html = first_page_html  # reuse already-fetched page 1
    question_text = None  # extracted from post #1

    while len(comments) < MAX_PER_THREAD:
        soup = BeautifulSoup(html, "html.parser")
        for post in soup.select("div.post[data-postid]"):
            if len(comments) >= MAX_PER_THREAD:
                break
            if get_post_number(post) == 1 and page == 1:
                # Capture OP body as the question for AI response generation
                question_text = extract_post_text(post)
                continue
            heading = post.select_one("div.post-heading")
            if not heading:
                continue
            post_date = parse_date(heading.get_text())
            if post_date is None or post_date >= CUTOFF_DATE:
                continue  # skip recent or unparseable posts
            text = extract_post_text(post)
            if not text or not text.strip():
                continue
            comments.append({
                "thread_id": thread_id,
                "thread_title": thread_title,
                "question": (question_text or "").strip(),
                "post_date": post_date.strftime("%Y-%m-%d"),
                "comment": text,
            })

        if not soup.select_one("li.next a"):
            break
        page += 1
        time.sleep(DELAY)
        html = fetch(f"{BASE_URL}/t{thread_id}p{page}")
        if not html:
            break

    return comments

def main():
    # Randomly sample thread IDs from the estimated pre-2017 range.
    # Stops as soon as TARGET_COMMENTS is reached — no need to exhaust the list.
    candidate_ids = random.sample(range(PRE_2017_ID_MIN, PRE_2017_ID_MAX + 1), 10_000)
    print(f"Sampled {len(candidate_ids):,} thread IDs from range "
          f"{PRE_2017_ID_MIN:,}–{PRE_2017_ID_MAX:,}\n")

    all_comments = []
    tried = 0

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["thread_id", "thread_title", "question", "post_date", "comment"])
        writer.writeheader()
        try:
            for thread_id in candidate_ids:
                if len(all_comments) >= TARGET_COMMENTS:
                    break
                tried += 1
                title, html = scan_thread(thread_id)
                if not title:
                    time.sleep(DELAY)
                    continue
                print(f"[{tried}] Thread {thread_id}: {title[:70]}")
                replies = get_replies(thread_id, title, html)
                if replies:
                    all_comments.extend(replies)
                    writer.writerows(replies)
                    f.flush()
                    print(f"  -> {len(replies)} pre-2017 comments (total: {len(all_comments)})")
                time.sleep(DELAY)
        except KeyboardInterrupt:
            print(f"\nInterrupted — {len(all_comments)} comments saved to {OUTPUT_FILE}")
            return

    print(f"\nDone. {len(all_comments)} comments saved to {OUTPUT_FILE} "
          f"(checked {tried} threads)")

if __name__ == "__main__":
    main()
