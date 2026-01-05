
import pandas as pd
import os

# Configuration
POST_FILES = {
    "North Korea": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
    "China": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
    "Iran": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
    "Russia": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"]
}

COMMENT_FILES = {
    "North Korea": "data/nk/nk_comments_full.csv",
    "China": "data/control/china_comments_full.csv",
    "Iran": "data/control/iran_comments_full.csv",
    "Russia": "data/control/russia_comments_full.csv"
}

def verify_coverage():
    print(f"{'Country':<15} | {'Posts (Target)':<15} | {'Comments (Total)':<15} | {'Posts w/ Comments':<20} | {'Coverage %':<10}")
    print("-" * 90)

    for country, post_paths in POST_FILES.items():
        # 1. Load Posts
        post_dfs = []
        for p in post_paths:
            if os.path.exists(p):
                post_dfs.append(pd.read_csv(p, usecols=['id']))
        
        if not post_dfs:
            print(f"{country:<15} | {'0':<15} | {'-':<15} | {'-':<20} | 0%")
            continue
            
        posts = pd.concat(post_dfs, ignore_index=True)
        target_ids = set(posts['id'].astype(str))
        target_count = len(target_ids)

        # 2. Load Comments (read chunks if large)
        comment_path = COMMENT_FILES[country]
        if not os.path.exists(comment_path):
            print(f"{country:<15} | {str(target_count):<15} | {'MISSING':<15} | {'-':<20} | 0%")
            continue

        linked_posts = set()
        try:
            # Use chunks to handle large files and find linking columns
            # usually 'link_id' contains 't3_postid' or 'post_id' contains 'postid'
            chunk_iter = pd.read_csv(comment_path, usecols=['link_id', 'post_id'], chunksize=50000, low_memory=False)
            
            total_comments = 0
            for chunk in chunk_iter:
                total_comments += len(chunk)
                
                # Method A: post_id column
                if 'post_id' in chunk.columns:
                    ids = chunk['post_id'].dropna().astype(str)
                    linked_posts.update(ids)
                
                # Method B: link_id column (strip t3_)
                if 'link_id' in chunk.columns:
                    ids = chunk['link_id'].dropna().astype(str).str.replace('t3_', '')
                    linked_posts.update(ids)
                    
        except ValueError:
             # Fallback if specific columns not found
             print(f"{country:<15} | Error reading columns")
             continue

        # 3. Calculate Intersection
        covered = target_ids.intersection(linked_posts)
        coverage_pct = (len(covered) / target_count) * 100 if target_count > 0 else 0
        
        print(f"{country:<15} | {target_count:<15} | {total_comments:<15} | {len(covered):<20} | {coverage_pct:.1f}%")

if __name__ == "__main__":
    verify_coverage()
