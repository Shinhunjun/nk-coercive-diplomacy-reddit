import pandas as pd
import os
import shutil
from pathlib import Path

BASE_DIR = Path('data/anonymized')

def consolidate_folder(folder_name, output_filename, file_pattern=None, country_col=True):
    print(f"\nProcessing {folder_name}...")
    folder_path = BASE_DIR / folder_name
    if not folder_path.exists():
        print(f"Skipping {folder_name} (not found)")
        return

    dfs = []
    files_to_delete = set(os.listdir(folder_path))
    
    # Identify relevant files
    countries = ['nk', 'china', 'iran', 'russia']
    
    for country in countries:
        # Search for main file patterns
        candidates = []
        if file_pattern:
            # Pattern based (e.g. "_posts_framed.csv")
            fname = f"{country}{file_pattern}"
            if (folder_path / fname).exists():
                candidates.append(folder_path / fname)
        else:
            # Heuristic for final/control
            if folder_name == 'control':
                fname = f"{country}_posts.csv" # Base raw file
            elif folder_name == 'final':
                fname = f"{country}_final.csv"
            
            if (folder_path / fname).exists():
                candidates.append(folder_path / fname)
        
        for p in candidates:
            try:
                df = pd.read_csv(p, low_memory=False)
                if country_col:
                    df['country'] = country.upper()
                dfs.append(df)
                print(f"  Loaded {p.name}: {len(df)} rows")
            except Exception as e:
                print(f"  Error loading {p.name}: {e}")

    # Combine
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        # Drop duplicates
        if 'id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['id'], keep='last')
        elif 'post_id' in combined_df.columns:
            combined_df = combined_df.drop_duplicates(subset=['post_id'], keep='last')
            
        output_path = folder_path / output_filename
        combined_df.to_csv(output_path, index=False)
        print(f"  ✅ Created {output_filename}: {len(combined_df)} rows")
        
        # Remove old files
        for f in files_to_delete:
            if f != output_filename and not f.startswith('.'):
                try:
                    os.remove(folder_path / f)
                except:
                    pass
        print(f"  Deleted old files in {folder_name}")
    else:
        print("  No files found to combine")

def main():
    # 1. Final
    # Use existing combined file if possible, or recreate
    final_combined = BASE_DIR / 'final' / 'all_countries_combined.csv'
    if final_combined.exists():
        df = pd.read_csv(final_combined)
        df.to_csv(BASE_DIR / 'final' / 'final_dataset.csv', index=False)
        print("\nRenamed all_countries_combined.csv to final_dataset.csv")
        # Delete others
        for f in os.listdir(BASE_DIR / 'final'):
            if f != 'final_dataset.csv' and not f.startswith('.'):
                os.remove(BASE_DIR / 'final' / f)
    
    # 2. Framing
    consolidate_folder('framing', 'framing_results.csv', '_posts_framed.csv')
    
    # 3. Sentiment
    consolidate_folder('sentiment', 'sentiment_results.csv', '_posts_sentiment.csv')
    
    # 4. Control (Raw data)
    # This might be tricky as filenames vary (_posts.csv vs _final.csv etc)
    # We will try to grab the base collection files
    consolidate_folder('control', 'raw_data_collection.csv')

if __name__ == "__main__":
    main()
