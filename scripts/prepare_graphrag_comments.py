"""
Prepare NK Comments for GraphRAG Indexing

Creates text input files for GraphRAG from collected NK comments.
Separates comments by period (P1, P2, P3).
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
COMMENTS_PATH = PROJECT_ROOT / 'data' / 'processed' / 'nk_comments_top3_final.csv'
OUTPUT_DIR = PROJECT_ROOT / 'graphrag_comments'

# Period definitions (timestamps)
PERIODS = {
    'period1': {
        'name': 'P1_PreSingapore',
        'start': datetime(2017, 1, 1).timestamp(),
        'end': datetime(2018, 6, 11).timestamp(),
    },
    'period2': {
        'name': 'P2_SingaporeHanoi', 
        'start': datetime(2018, 6, 13).timestamp(),
        'end': datetime(2019, 2, 27).timestamp(),
    },
    'period3': {
        'name': 'P3_PostHanoi',
        'start': datetime(2019, 3, 1).timestamp(),
        'end': datetime(2019, 12, 31).timestamp(),
    }
}


def get_period(ts):
    """Determine which period a timestamp belongs to."""
    try:
        ts = float(ts)
        for key, period in PERIODS.items():
            if period['start'] <= ts <= period['end']:
                return key
        return None
    except:
        return None


def prepare_comments():
    print("=" * 60)
    print("Preparing NK Comments for GraphRAG")
    print("=" * 60)
    
    # Load comments
    print("\nLoading comments...")
    df = pd.read_csv(COMMENTS_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} comments")
    
    # Filter valid comments
    valid = df[
        (~df['body'].astype(str).str.contains(r'\[removed\]|\[deleted\]', case=False, na=False, regex=True)) &
        (df['body'].astype(str).str.len() > 20)
    ].copy()
    print(f"  Valid comments: {len(valid):,}")
    
    # Assign periods based on created_utc
    valid['period'] = valid['created_utc'].apply(get_period)
    
    # Process each period
    for period_key, period_info in PERIODS.items():
        period_comments = valid[valid['period'] == period_key]
        print(f"\n{period_info['name']}: {len(period_comments):,} comments")
        
        if len(period_comments) == 0:
            continue
            
        # Create output directory
        output_dir = OUTPUT_DIR / period_key / 'input'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create documents from comments
        documents = []
        for _, row in period_comments.iterrows():
            body = str(row.get('body', '')).strip()
            if body and len(body) > 20:
                documents.append(body)
        
        # Write to file
        output_path = output_dir / 'comments.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n\n---DOCUMENT_SEPARATOR---\n\n".join(documents))
        
        print(f"  ✓ Wrote {len(documents):,} documents to {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ Input preparation complete!")
    print("=" * 60)


if __name__ == '__main__':
    prepare_comments()
