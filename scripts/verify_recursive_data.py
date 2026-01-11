
import pandas as pd
import sys

INPUT_FILE = 'data/processed/nk_comments_recursive.csv'

def verify_structure():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, low_memory=False)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Total Rows: {len(df)}")
    
    # Check columns
    required_cols = ['id', 'parent_id', 'parent_post_id', 'is_top_root', 'root_id']
    for col in required_cols:
        if col not in df.columns:
            print(f"MISSING COLUMN: {col}")
            return

    # Check Post Groups
    post_counts = df['parent_post_id'].value_counts()
    print(f"\nUnique Posts Collected: {len(post_counts)}")
    
    # Analyze a few random posts
    sample_posts = post_counts.head(5).index.tolist()
    
    for pid in sample_posts:
        print(f"\n--- Post {pid} ---")
        subset = df[df['parent_post_id'] == pid]
        print(f"  Total Comments: {len(subset)}")
        
        # 1. Check Top 5 Roots
        # Roots are where is_top_root == True
        roots = subset[subset['is_top_root'] == True]
        # In my script, is_top_root is string 'True'/'False' or boolean? pandas might infer.
        # Let's check unique values
        # print(f"  is_top_root values: {subset['is_top_root'].unique()}")
        
        # Adjust for possible string/bool type
        roots = subset[subset['is_top_root'].astype(str) == 'True']
        
        print(f"  Top Roots Found: {len(roots)}")
        for _, r in roots.iterrows():
            print(f"    root {r['id']} (score {r.get('score', 'N/A')})")
            
        # 2. Check Descendants
        # Any comment that is NOT a top root but has root_id set
        descendants = subset[subset['is_top_root'].astype(str) == 'False']
        print(f"  Descendants Found: {len(descendants)}")
        
        # Verify linkage
        orphan_count = 0
        for _, d in descendants.iterrows():
            if d['root_id'] not in roots['id'].values:
                # This might happen if adjacency list logic wasn't perfect or root_id matching type mismatch
                # But logic says root_id corresponds to id_clean.
                # Let's strict check
                orphan_count += 1
                
        # print(f"  Orphans (root_id not in roots): {orphan_count}")
        
        # Example Thread
        if len(descendants) > 0:
            sample_child = descendants.iloc[0]
            print(f"  Example Child: {sample_child['id']} -> Parent: {sample_child['parent_id']} -> Root: {sample_child['root_id']}")

if __name__ == "__main__":
    verify_structure()
