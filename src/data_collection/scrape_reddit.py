# base url: https://arctic-shift.photon-reddit.com/

import csv
import json
import requests


# ENSURE URLS GATHER PRE-2017 DATA

reddit_base_url = "https://reddit.com"
posts_base = "https://arctic-shift.photon-reddit.com/api/posts/search?subreddit=sweden&before=2017-01-01&limit=100"
comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search?subreddit=sweden&before=2017-01-01"
MAX_POSTS = 200 

# gather posts or comments from subreddit
def send_request(url):
    try:
        # send read request
        response = requests.get(url)
        # check if request is successful
        if response.status_code == 200:
            # return in json format
            posts = response.json()
            return posts
        else:
        # otherwise raise exception
            raise Exception(f"Failed to get posts: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error getting posts/comments: {e}")
        return None


def _posts_list(response):
    """Return list of post objects from API response (list or dict)."""
    if response is None:
        return []
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        return (
            response.get("data")
            or response.get("posts")
            or list(response.values())[0]
            if response
            else []
        )
    return []


def main():
    results_dict = {}
    post_ids = []
    posts_by_id = {}  # post_id -> question text (title + selftext)
    # Fetch posts until we have enough (API limit is 100 per request)
    after = None
    while len(post_ids) < MAX_POSTS:
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
            post_ids.append(pid)
            title = p.get("title") or ""
            selftext = p.get("selftext") or ""
            posts_by_id[pid] = f"{title}\n{selftext}".strip() or ""
        # Paginate: after = last post's created_utc
        last = posts_list[-1] if posts_list else {}
        after = last.get("created_utc")
        if after is None:
            break
    post_ids = post_ids[:MAX_POSTS]
    print(f"Using {len(post_ids)} posts (max {MAX_POSTS})")

    for i in post_ids:
        query = f"{comments_url}&link_id={i}"
        gather_comments = send_request(query)
        results_dict[i] = gather_comments

    # Extract comment bodies and write to CSV
    comment_rows = []
    for link_id, comments_response in results_dict.items():
        if comments_response is None:
            continue
        # API may return a list of comments or a dict containing a list
        comments_list = comments_response
        if isinstance(comments_response, dict):
            comments_list = (
                comments_response.get("data")
                or comments_response.get("comments")
                or list(comments_response.values())[0]
                if comments_response
                else []
            )
        if not isinstance(comments_list, list):
            continue
        for comment in comments_list:
            if not isinstance(comment, dict):
                continue
            body = comment.get("body", "")
            subreddit = comment.get("subreddit", "sweden")
            question = posts_by_id.get(link_id, "")
            comment_rows.append({
                "link_id": link_id,
                "subreddit": subreddit,
                "question": question,
                "body": body,
            })

    comments_csv_path = "comments_bodies.csv"
    with open(comments_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["link_id", "subreddit", "question", "body"])
        writer.writeheader()
        writer.writerows(comment_rows)
    print(f"Wrote {len(comment_rows)} comments to {comments_csv_path}")

    print(json.dumps(results_dict, indent=4))
    #print(gather_posts)['id']
    #gather_comments = send_request(comments_url)
    #for i in gather_posts.items():
        #print(i[0])
    #print(gather_posts)
    # put in more readable format
    #prettify_posts = json.dumps(gather_posts, indent=2)
    #print(prettify_posts)
    #prettify_comments = json.dumps(gather_comments, indent=4)
    #print(prettify_comments)

if __name__ == "__main__":
    main()
