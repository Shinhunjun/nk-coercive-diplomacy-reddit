
import pandas as pd
import numpy as np

INPUT_FILE = 'data/processed/nk_comments_recursive.csv'

def analyze_stats():
    print(f"Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Basic Counts
    total_comments = len(df)
    unique_ids = df['id'].nunique()
    unique_posts = df['parent_post_id'].nunique()
    
    print(f"\n=== COLLECTION SUMMARY ===")
    print(f"Total Comments Collected: {total_comments:,}")
    print(f"Unique Comments: {unique_ids:,}")
    print(f"Posts Covered: {unique_posts:,}")
    
    if total_comments == 0:
        return

    # Deduplicate for analysis if needed (though script should have handled it)
    df_unique = df.drop_duplicates(subset=['id'])
    
    # Analyze Structure
    # 1. Top Roots
    top_roots = df_unique[df_unique['is_top_root'].astype(str) == 'True']
    num_top_roots = len(top_roots)
    
    # 2. Replies (Descendants)
    replies = df_unique[df_unique['is_top_root'].astype(str) == 'False']
    num_replies = len(replies)
    
    print(f"\n=== STRUCTURE ===")
    print(f"Top 5 Root Comments: {num_top_roots:,}")
    print(f"Nested Replies (Descendants): {num_replies:,}")
    print(f"Ratio (Replies per Root): {num_replies / num_top_roots:.2f}")

    # 3. Distribution per Post
    comments_per_post = df_unique.groupby('parent_post_id').size()
    print(f"\n=== PER POST STATS ===")
    print(f"Avg Comments per Post: {comments_per_post.mean():.1f}")
    print(f"Max Comments in one Post: {comments_per_post.max():,}")
    print(f"Median Comments per Post: {comments_per_post.median():.1f}")
    
    # 4. Distribution per Root Thread
    # Group by root_id (which identifies the thread)
    comments_per_thread = df_unique.groupby('root_id').size()
    print(f"\n=== PER THREAD STATS (Top Comment + its replies) ===")
    print(f"Avg Thread Size: {comments_per_thread.mean():.1f}")
    print(f"Max Thread Size: {comments_per_thread.max():,}")
    
    # Top 5 biggest threads
    print("\nTop 5 Biggest Discussion Threads:")
    top_threads = comments_per_thread.sort_values(ascending=False).head(5)
    for root_id, count in top_threads.items():
        # Get root body snippet
        root_row = df_unique[df_unique['id'] == root_id]
        if not root_row.empty:
            body = str(root_row.iloc[0]['body'])[:50].replace('\n', ' ')
            post_title = str(root_row.iloc[0]['parent_post_title'])[:40]
            print(f"  [{count:,} comments] Post: {post_title}... | Root: {body}...")

if __name__ == "__main__":
    analyze_stats()
