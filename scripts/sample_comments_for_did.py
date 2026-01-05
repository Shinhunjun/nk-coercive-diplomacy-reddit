
import pandas as pd
import os
import glob

# Configuration
POST_RESULTS_DIR = "data/results/final_framing_v2"
COMMENT_FILES = {
    "nk": "data/nk/nk_comments_full.csv",
    "china": "data/control/china_comments_full.csv",
    "iran": "data/control/iran_comments_full.csv",
    "russia": "data/control/russia_comments_full.csv"
}
OUTPUT_FILE = "data/comments_to_classify_top3.csv"
TOP_K = 3

def load_target_post_ids(country):
    """Load IDs of posts that we have already classified."""
    path = f"{POST_RESULTS_DIR}/{country}_framing_v2.csv"
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, usecols=['id'])
    return set(df['id'].astype(str))

def sample_comments():
    print(f"Sampling Top {TOP_K} comments per post...")
    all_comments = []
    
    for country, comment_path in COMMENT_FILES.items():
        if not os.path.exists(comment_path):
            continue
            
        print(f"Processing {country}...", end='\r')
        
        # 1. Get relevant post IDs
        target_ids = load_target_post_ids(country)
        if not target_ids:
            continue
            
        # 2. Read Comments (chunked for memory)
        # We need: body, score, link_id (or post_id), id
        chunk_iter = pd.read_csv(comment_path, 
                                 usecols=['body', 'score', 'link_id', 'post_id', 'id'], 
                                 chunksize=50000, 
                                 low_memory=False)
        
        country_comments = []
        
        for chunk in chunk_iter:
            # Normalize Post ID linkage
            # prefer 'post_id' col if exists, else 'link_id' stripped of 't3_'
            if 'post_id' in chunk.columns and not chunk['post_id'].isna().all():
                 chunk['parent_post_id'] = chunk['post_id'].astype(str)
            elif 'link_id' in chunk.columns:
                 chunk['parent_post_id'] = chunk['link_id'].astype(str).str.replace('t3_', '')
            else:
                continue
                
            # Filter for our target posts only
            relevant = chunk[chunk['parent_post_id'].isin(target_ids)].copy()
            if relevant.empty:
                continue
            
            # Keep necessary cols
            relevant['country'] = country
            relevant['score'] = pd.to_numeric(relevant['score'], errors='coerce').fillna(0)
            country_comments.append(relevant[['id', 'body', 'score', 'parent_post_id', 'country']])
            
        if not country_comments:
            continue
            
        # 3. Group by Post and Top-K
        full_country_df = pd.concat(country_comments, ignore_index=True)
        # Sort by Post and Score
        full_country_df = full_country_df.sort_values(by=['parent_post_id', 'score'], ascending=[True, False])
        # Group head
        top_k = full_country_df.groupby('parent_post_id').head(TOP_K)
        
        all_comments.append(top_k)
        print(f"✅ {country}: Selected {len(top_k)} comments from {len(full_country_df)} raw.")

    # 4. Save
    if all_comments:
        final_df = pd.concat(all_comments, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"\nSaved total {len(final_df)} comments to {OUTPUT_FILE}")
    else:
        print("\nNo comments found.")

if __name__ == "__main__":
    sample_comments()
