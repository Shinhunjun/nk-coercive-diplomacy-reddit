
import pandas as pd
import glob
import os

TOPICS = ['nk', 'china', 'iran', 'russia']

def merge_comments():
    print("=" * 60)
    print("MERGING COLLECTED COMMENTS")
    print("=" * 60)
    
    for topic in TOPICS:
        print(f"\nProcessing {topic.upper()}...")
        
        # Define output directory based on topic
        if topic == 'nk':
            base_dir = 'data/processed'
            file_pattern = f'{base_dir}/nk_comments_top3*.csv'
        else:
            base_dir = 'data/control'
            file_pattern = f'{base_dir}/{topic}_comments_top3*.csv'
            
        files = glob.glob(file_pattern)
        # Filter out "final" files to avoid recursion
        files = [f for f in files if '_final.csv' not in f]
        
        extracted_data = []
        for f in sorted(files):
            try:
                df = pd.read_csv(f, low_memory=False)
                extracted_data.append(df)
                print(f"  Loaded {len(df):,} comments from {os.path.basename(f)}")
            except Exception as e:
                print(f"  Error loading {f}: {e}")
                
        if not extracted_data:
            print(f"  Warning: No comment files found for {topic}")
            continue
            
        # Merge
        merged = pd.concat(extracted_data, ignore_index=True)
        initial = len(merged)
        
        # Deduplicate by comment ID
        merged = merged.drop_duplicates(subset=['id'])
        final = len(merged)
        
        print(f"  Merged: {initial:,} -> Unique: {final:,}")
        
        # Save Final
        output_path = f'{base_dir}/{topic}_comments_top3_final.csv'
        merged.to_csv(output_path, index=False)
        print(f"  ✅ Saved merged file to: {output_path}")

if __name__ == "__main__":
    merge_comments()
