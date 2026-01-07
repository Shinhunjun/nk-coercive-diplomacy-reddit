#!/usr/bin/env python3
"""
Script to prepare anonymized data files for public sharing.
Removes text content (title, selftext, reasoning) while preserving post IDs and labels.
"""

import os
import pandas as pd
import shutil
from pathlib import Path

# Configuration
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "anonymized"

# Columns to remove (text content)
TEXT_COLUMNS = [
    'title', 'selftext', 'body', 'text',  # Post content
    'reason', 'reason_revised', 'frame_reason',  # LLM reasoning
    'reason_original', 'llm_reason',  # Other reasoning columns
]

# Folders to process
FOLDERS_TO_PROCESS = ['annotations', 'final', 'framing', 'sentiment', 'control']


def remove_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove text columns from dataframe."""
    cols_to_drop = [col for col in TEXT_COLUMNS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    return df


def process_csv_file(input_path: Path, output_path: Path) -> dict:
    """Process a single CSV file, removing text columns."""
    try:
        df = pd.read_csv(input_path, low_memory=False)
        original_cols = list(df.columns)
        
        df_anon = remove_text_columns(df)
        removed_cols = [c for c in original_cols if c not in df_anon.columns]
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_anon.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'original_rows': len(df),
            'output_rows': len(df_anon),
            'removed_columns': removed_cols
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


def copy_non_csv_file(input_path: Path, output_path: Path):
    """Copy non-CSV files (like CODEBOOK.md) as-is."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, output_path)


def main():
    print("=" * 60)
    print("Preparing Anonymized Data for Public Sharing")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    total_files = 0
    total_processed = 0
    
    for folder in FOLDERS_TO_PROCESS:
        folder_path = DATA_DIR / folder
        if not folder_path.exists():
            print(f"\n⚠️  Folder not found: {folder}")
            continue
        
        print(f"\n📁 Processing: {folder}/")
        
        for file_path in folder_path.iterdir():
            if file_path.name.startswith('.'):
                continue
                
            output_path = OUTPUT_DIR / folder / file_path.name
            total_files += 1
            
            if file_path.suffix.lower() == '.csv':
                result = process_csv_file(file_path, output_path)
                if result['status'] == 'success':
                    removed = result['removed_columns']
                    if removed:
                        print(f"   ✓ {file_path.name}: removed {removed}")
                    else:
                        print(f"   ✓ {file_path.name}: no text columns")
                    total_processed += 1
                else:
                    print(f"   ✗ {file_path.name}: {result['error']}")
            else:
                # Copy non-CSV files (markdown, txt logs)
                copy_non_csv_file(file_path, output_path)
                print(f"   ✓ {file_path.name}: copied")
                total_processed += 1
    
    # Copy results folder as-is (statistics only, no text)
    results_src = DATA_DIR / "results"
    results_dst = OUTPUT_DIR / "results"
    if results_src.exists():
        print(f"\n📁 Copying results/ (statistics only)")
        if results_dst.exists():
            shutil.rmtree(results_dst)
        shutil.copytree(results_src, results_dst)
        print(f"   ✓ Copied {len(list(results_src.iterdir()))} files")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Processed {total_processed}/{total_files} files")
    print(f"📂 Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
