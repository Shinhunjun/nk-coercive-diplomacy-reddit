
import pandas as pd
import os
import glob

TOPICS = ['nk', 'china', 'iran', 'russia']

def merge_and_cleanup():
    print("=" * 60)
    print("MERGING POST FILES AND CLEANING UP")
    print("=" * 60)
    
    for topic in TOPICS:
        print(f"\nProcessing {topic.upper()}...")
        
        # Files to merge
        files = [
            f'data/sentiment/{topic}_posts_sentiment.csv',
            f'data/sentiment/{topic}_posts_hanoi_extended_sentiment.csv'
        ]
        
        dfs = []
        existing_files = []
        
        for f in files:
            if os.path.exists(f):
                try:
                    df = pd.read_csv(f, low_memory=False)
                    dfs.append(df)
                    existing_files.append(f)
                    print(f"  Loaded {len(df):,} posts from {os.path.basename(f)}")
                except Exception as e:
                    print(f"  Error loading {f}: {e}")
        
        if not dfs:
            print(f"  Warning: No files found for {topic}")
            continue
            
        # Merge
        merged = pd.concat(dfs, ignore_index=True)
        initial_count = len(merged)
        
        # Deduplicate
        merged = merged.drop_duplicates(subset=['id'])
        final_count = len(merged)
        
        print(f"  Merged: {initial_count:,} -> Unique: {final_count:,}")
        
        # Save final version
        output_dir = 'data/processed'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{topic}_posts_final.csv'
        
        merged.to_csv(output_path, index=False)
        print(f"  ✅ Saved final file to: {output_path}")
        
        # DELETE old files
        print("  🗑️ Deleting old files...")
        for f in existing_files:
            try:
                os.remove(f)
                print(f"    - Deleted {os.path.basename(f)}")
            except Exception as e:
                print(f"    - Failed to delete {os.path.basename(f)}: {e}")

if __name__ == "__main__":
    merge_and_cleanup()
