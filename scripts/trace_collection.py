
import pandas as pd
import requests
import json
import time
import os

INPUT_POSTS_FILE = 'data/processed/nk_posts_final.csv'
COMMENTS_API_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

def fetch_all_comments_for_post(post_id):
    comments = []
    before = None
    print(f"  [Fetch] Start: {post_id}")
    
    while True:
        params = {
            "link_id": post_id,
            "limit": 500,
            "sort": "asc"
        }
        if before:
            params['before'] = before
            
        try:
            resp = requests.get(COMMENTS_API_URL, params=params, timeout=20)
            data = resp.json().get('data', [])
            
            if not data:
                print("  [Fetch] No data returned")
                break
                
            print(f"  [Fetch] Batch size: {len(data)}")
            comments.extend(data)
            before = data[-1]['created_utc']
            
            if len(data) < 100:
                break
                
        except Exception as e:
            print(f"  [Fetch] Error: {e}")
            break
            
    return comments

def process_post(post_row):
    post_id = post_row['id']
    title = post_row.get('title', 'No Title')
    created_utc = float(post_row.get('created_utc', 0))
    
    print(f"\nProcessing Post: {post_id} | Date: {created_utc}")
    
    if created_utc > 1577836799:
        print("  [Filter] Skipped due to date")
        return []

    post_id_clean = str(post_id).replace('t3_', '')
    
    # 1. Fetch
    raw_comments = fetch_all_comments_for_post(post_id_clean)
    print(f"  Raw Comments: {len(raw_comments)}")
    
    if not raw_comments:
        return []

    # 2. Adjacency
    roots = []
    adjacency = {}
    
    for c in raw_comments:
        c_id = str(c.get('id', '')).replace('t1_', '').replace('t3_', '')
        p_id = str(c.get('parent_id', '')).replace('t1_', '').replace('t3_', '')
        
        c['id_clean'] = c_id
        c['parent_id_clean'] = p_id
        
        # Check numeric score
        try:
            c['score_val'] = int(c.get('score', 0))
        except:
            c['score_val'] = 0
            
        if p_id == post_id_clean:
            roots.append(c)
        else:
            if p_id not in adjacency:
                adjacency[p_id] = []
            adjacency[p_id].append(c)
            
    print(f"  Roots: {len(roots)}")
    
    # 3. Top 5
    roots.sort(key=lambda x: x['score_val'], reverse=True)
    top_roots = roots[:5]
    
    processed = []
    
    def get_descendants(pid):
        d = []
        for child in adjacency.get(pid, []):
            d.append(child)
            d.extend(get_descendants(child['id_clean']))
        return d
    
    for r in top_roots:
        processed.append(r)
        d = get_descendants(r['id_clean'])
        processed.extend(d)
        
    print(f"  Selected (Top 5+Desc): {len(processed)}")
    return processed

def main():
    df = pd.read_csv(INPUT_POSTS_FILE, low_memory=False)
    print(f"Loaded {len(df)} posts")
    
    subset = df.head(5)
    
    for _, row in subset.iterrows():
        process_post(row)

if __name__ == "__main__":
    main()
