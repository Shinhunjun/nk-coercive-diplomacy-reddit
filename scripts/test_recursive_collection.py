
import requests
import json
import time
import pandas as pd
import sys

def fetch_and_analyze(post_id):
    url = "https://arctic-shift.photon-reddit.com/api/comments/search"
    comments = []
    before = None
    
    print(f"Fetching comments for {post_id} (Limit 3000 to ensure coverage)...")
    
    while True:
        params = {
            "link_id": post_id,
            "limit": 500, # Increased limit for speed if allowed, usually 100 max but let's try
            "sort": "asc" # Oldest first to catch early top comments and their immediate replies
        }
        if before:
            params['before'] = before
            
        try:
            r = requests.get(url, params=params, timeout=10)
            data = r.json().get('data', [])
            
            if not data:
                break
                
            comments.extend(data)
            before = data[-1]['created_utc']
            
            print(f"  Fetched {len(data)} chunk... Total: {len(comments)}")
            
            if len(comments) >= 5000: # Increase limit to ensure we catch replies
                print("  Reached 5000 comments limit.")
                break
                
            time.sleep(0.5)
                
        except Exception as e:
            print(f"Error: {e}")
            break
    
    if not comments:
        print("No comments found.")
        return

    df = pd.DataFrame(comments)
    df['score'] = pd.to_numeric(df['score'], errors='coerce').fillna(0)
    
    print(f"\nTotal Comments Fetched: {len(df)}")
    
    # helper to normalize IDs for comparison
    def clean_id(x): return str(x).replace('t1_', '').replace('t3_', '')
    
    post_id_clean = clean_id(post_id)
    df['parent_id_clean'] = df['parent_id'].apply(clean_id)
    df['id_clean'] = df['id'].apply(clean_id)
    
    # 1. Identify Root Comments (parent_id == post_id)
    roots = df[df['parent_id_clean'] == post_id_clean].copy()
    print(f"Total Root Comments: {len(roots)}")
    
    if roots.empty:
        # Fallback: maybe parent_id uses full t3_ prefix and our clean didn't match?
        # Check raw parent_ids
        print("Debugging Parent IDs to find roots:")
        print(df['parent_id'].head(10).unique())
        return

    # 2. Get Top 5 Roots by Score
    top_roots = roots.sort_values('score', ascending=False).head(5)
    
    print("\n🏆 TOP 5 ROOT COMMENTS:")
    for i, (_, root) in enumerate(top_roots.iterrows()):
        r_id = root['id_clean']
        score = root['score']
        body = root['body'][:50].replace('\n', ' ')
        
        # 3. Find Replies to this Root
        # Direct replies have parent_id == root_id
        replies = df[df['parent_id_clean'] == r_id]
        reply_count = len(replies)
        
        print(f"#{i+1} [Score: {score}] ID: {r_id} | Body: {body}...")
        print(f"   └── Found {reply_count} direct replies in fetched dataset")
        
        if reply_count > 0:
            top_reply = replies.sort_values('score', ascending=False).iloc[0]
            print(f"       (Top Reply Score: {top_reply['score']}: {top_reply['body'][:30]}...)")

if __name__ == "__main__":
    fetch_and_analyze("7nqzmo")
