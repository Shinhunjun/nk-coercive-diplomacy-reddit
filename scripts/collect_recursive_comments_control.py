
import pandas as pd
import requests
import json
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import argparse

# Configuration
COMMENTS_API_URL = "https://arctic-shift.photon-reddit.com/api/comments/search"
MAX_WORKERS = 5
BATCH_SIZE = 100

# Rate Limiter
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

rate_limiter = RateLimiter(calls_per_second=20)

def fetch_all_comments_for_post(post_id):
    comments = []
    before = None
    
    while True:
        rate_limiter.wait()
        
        params = {
            "link_id": post_id,
            "limit": 100, # API Max
            "sort": "asc" 
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
            
            if len(data) < 100:
                break
                
            if len(comments) > 20000: # Safety cap
                break
                
        except Exception as e:
            break
            
    return comments

def get_descendants(parent_id, adjacency_list):
    descendants = []
    children = adjacency_list.get(parent_id, [])
    for child in children:
        descendants.append(child)
        descendants.extend(get_descendants(child['id'], adjacency_list))
    return descendants

def process_post(post_row):
    # Filter by date: Up to 2019-12-31
    created_utc = float(post_row.get('created_utc', 0))
    if created_utc > 1577836799:
        return []
        
    post_id = post_row['id']
    if pd.isna(post_id):
        return []
        
    post_id_clean = str(post_id).replace('t3_', '')
    
    raw_comments = fetch_all_comments_for_post(post_id_clean)
    if not raw_comments:
        return []
        
    # Clean and Deduplicate
    for c in raw_comments:
        c['id_clean'] = str(c['id']).replace('t1_', '').replace('t3_', '')
        c['parent_id_clean'] = str(c['parent_id']).replace('t1_', '').replace('t3_', '')
        try:
            c['score_val'] = int(c.get('score', 0))
        except:
            c['score_val'] = 0

    comment_map = {c['id_clean']: c for c in raw_comments}
    unique_comments = list(comment_map.values())

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
            
    roots.sort(key=lambda x: x['score_val'], reverse=True)
    top_roots = roots[:5]
    
    selected_comments = []
    
    for root in top_roots:
        selected_comments.append(root)
        root['is_top_root'] = True
        root['root_id'] = root['id_clean']
        
        children_tree = get_descendants(root['id_clean'], adjacency)
        for child in children_tree:
            child['is_top_root'] = False
            child['root_id'] = root['id_clean']
            selected_comments.append(child)
            
    final_unique_map = {}
    for c in selected_comments:
        if c['id_clean'] not in final_unique_map:
            c['parent_post_id'] = post_id_clean
            c['parent_post_title'] = post_row.get('title', '')
            c['parent_post_created_utc'] = post_row.get('created_utc', '')
            final_unique_map[c['id_clean']] = c
            
    return list(final_unique_map.values())

def collect_for_country(country_name, input_file, output_file):
    print(f"\n[{country_name.upper()}] Loading posts from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
        
    posts_df = pd.read_csv(input_file, low_memory=False)
    print(f"Total posts: {len(posts_df)}")
    
    processed_ids = set()
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_csv(output_file, usecols=['parent_post_id'])
            processed_ids = set(existing_df['parent_post_id'].astype(str))
            print(f"Found existing output with {len(processed_ids)} processed posts.")
        except:
            pass
            
    posts_to_process = posts_df[~posts_df['id'].astype(str).isin(processed_ids)]
    print(f"To Process: {len(posts_to_process)}")
    
    if len(posts_to_process) == 0:
        return

    current_batch = []
    
    if not os.path.exists(output_file):
        cols = ['id', 'parent_id', 'body', 'score', 'author', 'created_utc', 'parent_post_id', 'parent_post_title', 'is_top_root', 'root_id']
        pd.DataFrame(columns=cols).to_csv(output_file, index=False)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_post = {executor.submit(process_post, row): row for _, row in posts_to_process.iterrows()}
        
        for future in tqdm(as_completed(future_to_post), total=len(posts_to_process), desc=f"Collecting {country_name}"):
            try:
                result = future.result()
                if result:
                    current_batch.extend(result)
            except Exception:
                pass
                
            if len(current_batch) >= BATCH_SIZE:
                cols = ['id', 'parent_id', 'body', 'score', 'author', 'created_utc', 'parent_post_id', 'parent_post_title', 'is_top_root', 'root_id']
                df_batch = pd.DataFrame(current_batch)
                df_batch = df_batch.reindex(columns=cols)
                df_batch.to_csv(output_file, mode='a', header=False, index=False)
                current_batch = []
                
        if current_batch:
            cols = ['id', 'parent_id', 'body', 'score', 'author', 'created_utc', 'parent_post_id', 'parent_post_title', 'is_top_root', 'root_id']
            df_batch = pd.DataFrame(current_batch)
            df_batch = df_batch.reindex(columns=cols)
            df_batch.to_csv(output_file, mode='a', header=False, index=False)

def main():
    countries = [
        {'name': 'china', 'input': 'data/processed/china_posts_final.csv', 'output': 'data/processed/china_comments_recursive.csv'},
        {'name': 'iran', 'input': 'data/processed/iran_posts_final.csv', 'output': 'data/processed/iran_comments_recursive.csv'},
        {'name': 'russia', 'input': 'data/processed/russia_posts_final.csv', 'output': 'data/processed/russia_comments_recursive.csv'}
    ]
    
    for c in countries:
        collect_for_country(c['name'], c['input'], c['output'])

if __name__ == "__main__":
    main()
