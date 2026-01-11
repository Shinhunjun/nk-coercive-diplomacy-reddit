
import pandas as pd
import requests
import json
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Configuration
INPUT_POSTS_FILE = 'data/processed/nk_posts_final.csv'
OUTPUT_FILE = 'data/processed/nk_comments_recursive.csv'
MAX_WORKERS = 5  # Conservative parallelism to respect API limits
MAX_RETRIES = 3
COMMENTS_API_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"

# Rate Limiter to prevent 429s
class RateLimiter:
    def __init__(self, calls_per_second=10):
        self.delay = 1.0 / calls_per_second
        self.lock = threading.Lock()
        self.last_call = 0

    def wait(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self.last_call = time.time()

rate_limiter = RateLimiter(calls_per_second=20) # Arctic Shift is relatively generous

def fetch_all_comments_for_post(post_id):
    """
    Fetch ALL comments for a given post ID using pagination.
    Returns a list of comment dicts.
    """
    comments = []
    before = None
    
    while True:
        rate_limiter.wait()
        
        params = {
            "link_id": post_id,
            "limit": 100, # API Max is 100
            "sort": "asc" # Fetch oldest first often helps with deep threads
        }
        if before:
            params['before'] = before
            
        try:
            resp = requests.get(COMMENTS_API_URL, params=params, timeout=20)
            if resp.status_code == 429:
                time.sleep(5)
                continue
                
            resp.raise_for_status()
            data = resp.json().get('data', [])
            
            if not data:
                break
                
            comments.extend(data)
            before = data[-1]['created_utc']
            
            # If we fetched less than limit, we are likely done (unless exact multiple)
            if len(data) < 100:
                break
                
            # Safety cap for extremely large threads (optional, generally we want all)
            if len(comments) > 20000:
                break
                
        except Exception as e:
            # print(f"Error fetching {post_id}: {e}")
            break
            
    return comments

def get_descendants(parent_id, adjacency_list):
    """
    Recursively find all descendants of a comment.
    """
    descendants = []
    children = adjacency_list.get(parent_id, [])
    for child in children:
        descendants.append(child)
        descendants.extend(get_descendants(child['id'], adjacency_list))
    return descendants

def process_post(post_row):
    """
    Worker function to process a single post:
    1. Fetch all comments
    2. Sort by score
    3. Pick Top 5 Roots
    4. Collect all descendants of these roots
    5. Return list of selected comments
    """
    # Filter by date: Up to 2019-12-31
    # 2019-12-31 23:59:59 UTC = 1577836799
    created_utc = float(post_row.get('created_utc', 0))
    if created_utc > 1577836799:
        return []
        
    post_id = post_row['id']
    if pd.isna(post_id):
        return []
        
    # clean IDs
    post_id_clean = str(post_id).replace('t3_', '')
    
    # 1. Fetch All
    raw_comments = fetch_all_comments_for_post(post_id_clean)
    if not raw_comments:
        return []
        
    # Convert to DF for easier handling? Or just list ops. List is faster for small recursive logic.
    for c in raw_comments:
        c['id_clean'] = str(c['id']).replace('t1_', '').replace('t3_', '')
        c['parent_id_clean'] = str(c['parent_id']).replace('t1_', '').replace('t3_', '')
        # numeric score
        try:
            c['score_val'] = int(c.get('score', 0))
        except:
            c['score_val'] = 0

    # Deduplicate raw comments by id_clean just in case
    # Convert list of dicts to dict by id to ensure uniqueness
    comment_map = {c['id_clean']: c for c in raw_comments}
    unique_comments = list(comment_map.values())

    # 2. Build Adjacency List & Identify Roots
    adjacency = {}
    roots = []
    
    for c in unique_comments:
        pid = c['parent_id_clean']
        if pid == post_id_clean:
            roots.append(c)
        else:
            if pid not in adjacency:
                adjacency[pid] = []
            adjacency[pid].append(c)
            
    # 3. Top 5 Roots
    roots.sort(key=lambda x: x['score_val'], reverse=True)
    top_roots = roots[:5]
    
    selected_comments = []
    
    # 4. Collect Subtrees
    for root in top_roots:
        selected_comments.append(root)
        # Add metadata to track which 'top root' this belongs to (optional but useful)
        root['is_top_root'] = True
        root['root_id'] = root['id_clean']
        
        children_tree = get_descendants(root['id_clean'], adjacency)
        for child in children_tree:
            child['is_top_root'] = False
            child['root_id'] = root['id_clean']
            selected_comments.append(child)
            
            
    # Add post metadata to all selected
    final_unique_map = {}
    for c in selected_comments:
        if c['id_clean'] not in final_unique_map:
            c['parent_post_id'] = post_id_clean
            c['parent_post_title'] = post_row.get('title', '')
            c['parent_post_created_utc'] = post_row.get('created_utc', '')
            final_unique_map[c['id_clean']] = c
            
    return list(final_unique_map.values())

def main():
    print(f"Loading posts from {INPUT_POSTS_FILE}...")
    if not os.path.exists(INPUT_POSTS_FILE):
        print("Input file not found.")
        return
        
    posts_df = pd.read_csv(INPUT_POSTS_FILE, low_memory=False)
    print(f"Total posts to process: {len(posts_df)}")
    
    # Filter for Resume capability
    # If output exists, we can skip processed posts?
    # Ideally we'd log processed IDs using a separate log file
    
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE, usecols=['parent_post_id'])
            processed_ids = set(existing_df['parent_post_id'].astype(str))
            print(f"Found existing output with {len(processed_ids)} processed posts. Skipping them.")
        except Exception as e:
            print(f"Could not read existing file (might be empty or malformed): {e}")
            
    posts_to_process = posts_df[~posts_df['id'].astype(str).isin(processed_ids)]
    print(f"Remaining posts: {len(posts_to_process)}")
    
    batch_size = 100
    current_batch = []
    
    # Write header if new file
    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=['id', 'parent_id', 'body', 'score', 'author', 'created_utc', 'parent_post_id', 'parent_post_title', 'is_top_root', 'root_id']).to_csv(OUTPUT_FILE, index=False)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Map futures
        future_to_post = {executor.submit(process_post, row): row for _, row in posts_to_process.iterrows()}
        
        for future in tqdm(as_completed(future_to_post), total=len(posts_to_process), desc="Processing Posts"):
            try:
                result_comments = future.result()
                if result_comments:
                    current_batch.extend(result_comments)
            except Exception as e:
                # Log error
                pass
            
            # Batch save
            if len(current_batch) >= 100:
                cols = ['id', 'parent_id', 'body', 'score', 'author', 'created_utc', 'parent_post_id', 'parent_post_title', 'is_top_root', 'root_id']
                df_batch = pd.DataFrame(current_batch)
                # Ensure columns align
                df_batch = df_batch.reindex(columns=cols)
                df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
                current_batch = []
                
        # Final save
        if current_batch:
            cols = ['id', 'parent_id', 'body', 'score', 'author', 'created_utc', 'parent_post_id', 'parent_post_title', 'is_top_root', 'root_id']
            df_batch = pd.DataFrame(current_batch)
            df_batch = df_batch.reindex(columns=cols)
            df_batch.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)

    print("Done!")

if __name__ == "__main__":
    main()
