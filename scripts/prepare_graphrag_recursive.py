
"""
Prepare GraphRAG Input Data (Recursive Comments)

Creates text input files for GraphRAG indexing from Recursive Reddit Comments.
Grouping: All comments for a single Post are aggregated into one Document.
Filters: Excludes [removed], [deleted], and empty bodies.

Periods:
- P1: 2017-01-01 ~ 2018-06-11
- P2: 2018-06-12 ~ 2019-02-27
- P3: 2019-02-28 ~ 2019-12-31
"""

import pandas as pd
import os
from pathlib import Path
import argparse

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
GRAPHRAG_DIR = PROJECT_ROOT / 'graphrag'

INPUT_FILE = DATA_DIR / 'processed' / 'nk_comments_recursive_roberta_final.csv'

# Period definitions
P1_START, P1_END = '2017-01-01', '2018-06-11'
P2_START, P2_END = '2018-06-12', '2019-02-27'
P3_START, P3_END = '2019-02-28', '2019-12-31'

PERIODS = {
    'period1': {'name': 'P1_Recursive', 'start': P1_START, 'end': P1_END},
    'period2': {'name': 'P2_Recursive', 'start': P2_START, 'end': P2_END},
    'period3': {'name': 'P3_Recursive', 'start': P3_START, 'end': P3_END}
}

def load_data():
    """Load recursive comments data."""
    print(f"Loading data from {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
        
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # Convert created_utc to datetime
    df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
    df.dropna(subset=['created_utc'], inplace=True)
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    
    print(f"  Loaded {len(df):,} comments.")
    return df

def get_period_data(df, period_key):
    """Filter data for specific period."""
    period = PERIODS[period_key]
    mask = (df['date'] >= period['start']) & (df['date'] <= period['end'])
    return df[mask].copy()

def clean_text(text):
    """Clean text content."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.lower() in ['[removed]', '[deleted]', 'nan', '']:
        return ""
    return text

def prepare_input(df, period_key, dry_run=False):
    """Prepare input for a specific period."""
    period_info = PERIODS[period_key]
    print(f"\nProcessing {period_info['name']} ({period_info['start']} ~ {period_info['end']})...")
    
    subset = get_period_data(df, period_key)
    print(f"  Found {len(subset):,} comments in period.")
    
    if subset.empty:
        print("  Warning: No data for this period.")
        return
    
    # Group by Parent Post
    if 'parent_post_id' not in subset.columns:
        # Fallback if parent_post_id is missing (should not happen in processed file)
        print("  Error: 'parent_post_id' column missing.")
        return

    documents = []
    grouped = subset.groupby('parent_post_id')
    
    print(f"  Aggregating into {len(grouped):,} unique Post-based documents...")
    
    for post_id, group in grouped:
        # Get Post Title (from first row, assuming consistent)
        title = clean_text(group['parent_post_title'].iloc[0])
        
        # Collect Valid Comments
        comments = []
        for _, row in group.iterrows():
            body = clean_text(row['body'])
            if body:
                comments.append(body)
        
        if not comments:
            continue
            
        # Construct Document
        doc_parts = []
        if title:
            doc_parts.append(f"TITLE: {title}")
        doc_parts.append("COMMENTS:")
        doc_parts.extend(comments)
        
        full_doc = "\n\n".join(doc_parts)
        documents.append(full_doc)
        
    stats = {
        'period': period_info['name'],
        'comments': len(subset),
        'documents': len(documents)
    }
    
    if dry_run:
        print(f"  [DRY RUN] Would write {len(documents):,} documents.")
        return stats
        
    # Write to file
    output_dir = GRAPHRAG_DIR / period_info['name'] / 'input'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'comments.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n---DOCUMENT_SEPARATOR---\n\n".join(documents))
        
    print(f"  ✓ Wrote {len(documents):,} documents to {output_path}")
    return stats

def main():
    parser = argparse.ArgumentParser(description='Prepare GraphRAG Recursive Input')
    parser.add_argument('--dry-run', action='store_true', help='Show stats without writing files')
    args = parser.parse_args()
    
    df = load_data()
    
    for p_key in ['period1', 'period2', 'period3']:
        prepare_input(df, p_key, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
