
import pandas as pd
import requests
import json

COMMENTS_API_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

def fetch_debug(post_id):
    print(f"Fetching {post_id}...")
    params = {
        "link_id": post_id,
        "limit": 100,
        "sort": "asc"
    }
    r = requests.get(COMMENTS_API_URL, params=params)
    data = r.json().get('data', [])
    print(f"Fetched {len(data)} comments")
    if len(data) > 0:
        print(f"Sample Comment 0: {data[0]}")
        print(f"Sample Comment 0 ID: {data[0].get('id')}, Parent: {data[0].get('parent_id')}")
    return data

def process_debug(post_id):
    raw_comments = fetch_debug(post_id)
    
    # Simulate cleaning
    post_id_clean = post_id
    
    roots = []
    adjacency = {}
    
    for c in raw_comments:
        c_id = str(c.get('id', '')).replace('t1_', '').replace('t3_', '')
        p_id = str(c.get('parent_id', '')).replace('t1_', '').replace('t3_', '')
        
        c['id_clean'] = c_id
        c['parent_id_clean'] = p_id
        
        if p_id == post_id_clean:
            roots.append(c)
        else:
            if p_id not in adjacency:
                adjacency[p_id] = []
            adjacency[p_id].append(c)
            
    print(f"Roots found: {len(roots)}")
    if len(roots) > 0:
        print(f"Root 0 ID: {roots[0]['id_clean']}")
        
    # Top 5
    for c in raw_comments:
        c['score_val'] = int(c.get('score', 0))
        
    roots.sort(key=lambda x: x['score_val'], reverse=True)
    top_roots = roots[:5]
    print(f"Top Roots count: {len(top_roots)}")

if __name__ == "__main__":
    process_debug("c6qqyp")
