"""
Fetch additional Swedish Bachelor's thesis abstracts from DiVA and append to
an extended copy of the existing human_formal CSV.

Output includes a `new?` flag:
- F for pre-existing rows from sv_human_collection_with_kws.csv
- T for newly fetched rows from DiVA

Usage:
  python src/1_data_collection/human_formal/fetch_more_diva_bachelors.py
  python src/1_data_collection/human_formal/fetch_more_diva_bachelors.py --max-new 25
"""

from __future__ import annotations

import argparse
import html
import os
import re
import time
import urllib.parse

import pandas as pd
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_CSV = os.path.join(BASE_DIR, "sv_human_collection_with_kws.csv")
BASE_XLSX = os.path.join(BASE_DIR, "sv_data_collection.xlsx")
OUT_CSV = os.path.join(BASE_DIR, "sv_human_collection_with_kws_extended.csv")

COLS = [
    "Year",
    "Level",
    "Title",
    "Topic Category",
    "Abstract",
    "Direct Link",
    "Keywords",
    "Permanent Link",
    "Full Text",
    "new?",
]

UA = {"User-Agent": "Mozilla/5.0 (compatible; CLUU-thesis-bot/1.0)"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max-new", type=int, default=20, help="How many new rows to append")
    p.add_argument("--max-pages", type=int, default=10, help="How many result pages to scan")
    p.add_argument("--rows-per-page", type=int, default=50, help="Rows per DiVA search page")
    p.add_argument("--sleep-ms", type=int, default=300, help="Delay between record fetches")
    return p.parse_args()


def build_search_url(start: int, rows: int) -> str:
    # Broad query + strict MODS filtering (basic/bachelor level + Swedish abstract).
    query = urllib.parse.quote("")
    return (
        "https://www.diva-portal.org/smash/resultList.jsf"
        f"?language=en&searchType=SIMPLE&query={query}"
        "&af=%5B%5D&aq=%5B%5B%5D%5D&aq2=%5B%5B%5D%5D&aqe=%5B%5D"
        f"&noOfRows={rows}&sortOrder=dateIssued_sort_desc&sortOrder2=title_sort_asc"
        "&onlyFullText=false&sf=all"
        f"&start={start}"
    )


def extract_pids_from_result_html(result_html: str) -> list[str]:
    txt = urllib.parse.unquote(html.unescape(result_html))
    pids = sorted(set(re.findall(r"pid=(diva2:[0-9]+)", txt)))
    return pids


def record_html_for_pid(pid: str, session: requests.Session) -> str:
    url = f"https://www.diva-portal.org/smash/record.jsf?pid={pid}"
    r = session.get(url, headers=UA, timeout=40)
    r.raise_for_status()
    return r.text


def _first(rx: str, text: str, flags: int = re.IGNORECASE | re.DOTALL) -> str:
    m = re.search(rx, text, flags=flags)
    if not m:
        return ""
    return html.unescape(m.group(1)).strip()


def parse_record_html(record_html: str, pid: str) -> dict | None:
    raw_lower = record_html.lower()
    if ("independent thesis basic level" not in raw_lower) and ("bachelor" not in raw_lower):
        return None
    if "abstract [sv]" not in raw_lower:
        return None

    title = _first(r'<meta\s+property="og:title"\s+content="([^"]+)"', record_html)
    title = re.sub(r"^\d+:\s*", "", title).strip()
    if not title:
        return None

    abstract_sv = _first(r"Abstract\s*\[sv\].*?<span class=\"singleRow\">\s*<p>(.*?)</p>", record_html)
    if not abstract_sv:
        abstract_sv = _first(r"Abstract\s*\[sv\].*?<span class=\"singleRow\">(.*?)</span>", record_html)
    abstract_sv = re.sub(r"<[^>]+>", " ", abstract_sv)
    abstract_sv = re.sub(r"\s+", " ", abstract_sv).strip()
    if not abstract_sv:
        return None

    year_raw = _first(r'<meta\s+name="citation_date"\s+content="([^"]+)"', record_html)
    if not year_raw:
        year_raw = _first(r"<h5>\s*Year\s*</h5>\s*<span class=\"singleRow\">(.*?)</span>", record_html)
    year_match = re.search(r"(19|20)\d{2}", year_raw)
    year = year_match.group(0) if year_match else ""

    keywords_block = _first(r"Keywords\s*\[(?:sv|en)\].*?<span class=\"singleRow\">(.*?)</span>", record_html)
    keywords_block = re.sub(r"<[^>]+>", " ", keywords_block)
    keywords_str = re.sub(r"\s+", " ", keywords_block).strip()

    topic_category = _first(r"Subject\s*/\s*course.*?<span class=\"singleRow\">(.*?)</span>", record_html)
    topic_category = re.sub(r"<[^>]+>", " ", topic_category)
    topic_category = re.sub(r"\s+", " ", topic_category).strip()

    full_text = ""
    m_full = re.search(r'href="([^"]*smash/get/diva2:[^"]*FULLTEXT[^"]*)"', record_html, flags=re.IGNORECASE)
    if m_full:
        full_text = html.unescape(m_full.group(1))
        if full_text.startswith("/"):
            full_text = "https://www.diva-portal.org" + full_text

    perm = _first(r'<meta\s+property="og:url"\s+content="([^"]+)"', record_html)

    return {
        "Year": year,
        "Level": "Bachelor's Thesis",
        "Title": title,
        "Topic Category": topic_category,
        "Abstract": abstract_sv,
        "Direct Link": f"https://www.diva-portal.org/smash/record.jsf?pid={pid}",
        "Keywords": keywords_str,
        "Permanent Link": perm,
        "Full Text": full_text,
        "new?": "T",
    }


def main():
    args = parse_args()

    if os.path.isfile(BASE_XLSX):
        base_df = pd.read_excel(BASE_XLSX)
    elif os.path.isfile(BASE_CSV):
        base_df = pd.read_csv(BASE_CSV, encoding="utf-8")
    else:
        raise FileNotFoundError(f"Missing base source files: {BASE_XLSX} and {BASE_CSV}")

    for c in COLS:
        if c not in base_df.columns:
            base_df[c] = ""
    base_df["new?"] = "F"
    base_df = base_df[COLS]

    existing_links = set(base_df["Direct Link"].fillna("").astype(str).str.strip())
    existing_titles = set(base_df["Title"].fillna("").astype(str).str.strip().str.lower())

    session = requests.Session()
    new_rows: list[dict] = []
    seen_pids: set[str] = set()

    for page_i in range(args.max_pages):
        if len(new_rows) >= args.max_new:
            break
        start = page_i * args.rows_per_page
        url = build_search_url(start=start, rows=args.rows_per_page)
        r = session.get(url, headers=UA, timeout=40)
        if r.status_code == 429:
            print(f"Hit rate limit at start={start}; stopping page scan.")
            break
        r.raise_for_status()
        pids = extract_pids_from_result_html(r.text)
        if not pids:
            continue

        for pid in pids:
            if len(new_rows) >= args.max_new:
                break
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            link = f"https://www.diva-portal.org/smash/record.jsf?pid={pid}"
            if link in existing_links:
                continue

            try:
                rec_html = record_html_for_pid(pid, session)
                row = parse_record_html(rec_html, pid)
            except Exception:
                row = None

            if not row:
                continue
            if row["Title"].strip().lower() in existing_titles:
                continue

            existing_links.add(link)
            existing_titles.add(row["Title"].strip().lower())
            new_rows.append(row)
            time.sleep(args.sleep_ms / 1000.0)

    out_df = pd.concat([base_df, pd.DataFrame(new_rows, columns=COLS)], ignore_index=True)
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"Base rows: {len(base_df)}")
    print(f"New rows added: {len(new_rows)}")
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
