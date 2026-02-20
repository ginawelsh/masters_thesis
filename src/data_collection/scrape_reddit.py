# base url: https://arctic-shift.photon-reddit.com/

import requests
import json


# ENSURE URLS GATHER PRE-2017 DATA

reddit_base_url = "https://reddit.com"
posts_url = f"https://arctic-shift.photon-reddit.com/api/posts/search?subreddit=sweden&before=2017-01-01"
comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search?subreddit=sweden&before=2017-01-01" 

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


def main():
    #subreddit = "sweden"
    results_dict = {}
    post_ids = []
    gather_posts = send_request(posts_url) # type dict 
    for id in gather_posts.items():
        for i in id[1]:
            post_ids.append(i['id'])
    #print(post_ids)
    for i in post_ids:
        query = f"{comments_url}&link_id={i}"
        gather_comments = send_request(query)
        results_dict[i] = gather_comments
        #print(f"{comments_url}&link_id={id}")
        #gather_comments = send_request(f"{comments_url}&link_id={id}")
        #results_dict[i] = gather_comments
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
