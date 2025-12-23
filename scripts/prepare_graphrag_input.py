"""
Prepare GraphRAG Input Data

Creates text input files for GraphRAG indexing from Reddit posts and comments.
Option B: Posts + Top 5 comments per post

Periods:
- P1: Pre-Singapore (2017-01 ~ 2018-05)
- P2: Singapore-Hanoi (2018-06 ~ 2019-02)
- P3: Post-Hanoi (2019-03 ~ 2019-12)
"""

import pandas as pd
import os
from pathlib import Path
import argparse

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
GRAPHRAG_DIR = PROJECT_ROOT / 'graphrag'

# Period definitions
PERIODS = {
    'period1': {
        'name': 'P1_PreSingapore',
        'start': '2017-01',
        'end': '2018-05',
        'description': 'Pre-Singapore Summit (Tension Era)'
    },
    'period2': {
        'name': 'P2_SingaporeHanoi',
        'start': '2018-06',
        'end': '2019-02',
        'description': 'Singapore to Hanoi (Summit Diplomacy)'
    },
    'period3': {
        'name': 'P3_PostHanoi',
        'start': '2019-03',
        'end': '2019-12',
        'description': 'Post-Hanoi (Diplomatic Stall)'
    }
}


def load_data() -> tuple:
    """Load posts and comments data."""
    print("Loading data...")
    
    # Load posts
    posts_path = DATA_DIR / 'final' / 'nk_final.csv'
    posts = pd.read_csv(posts_path)
    print(f"  Loaded {len(posts):,} posts")
    
    # Load comments
    comments_path = DATA_DIR / 'nk' / 'nk_comments_full.csv'
    comments = pd.read_csv(comments_path, usecols=['link_id', 'score', 'body', 'created_utc'], low_memory=False)
    comments['post_id'] = comments['link_id'].str.replace('t3_', '', regex=False)
    comments['score'] = pd.to_numeric(comments['score'], errors='coerce').fillna(0)
    print(f"  Loaded {len(comments):,} comments")
    
    return posts, comments


def filter_period(posts: pd.DataFrame, comments: pd.DataFrame, period_key: str) -> tuple:
    """Filter posts and comments by period."""
    period = PERIODS[period_key]
    
    # Filter posts by month
    period_posts = posts[
        (posts['month'] >= period['start']) & 
        (posts['month'] <= period['end'])
    ].copy()
    
    # Get post IDs for this period
    post_ids = set(period_posts['id'].tolist())
    
    # Filter comments to only those from period posts
    period_comments = comments[comments['post_id'].isin(post_ids)].copy()
    
    print(f"  {period['name']}: {len(period_posts):,} posts, {len(period_comments):,} comments")
    
    return period_posts, period_comments


def get_top_comments(comments: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Get top N comments by score for each post."""
    # Sort by score and get top N per post
    top_comments = (
        comments
        .sort_values('score', ascending=False)
        .groupby('post_id')
        .head(n)
        .reset_index(drop=True)
    )
    return top_comments


def create_document(row: pd.Series, comments_for_post: pd.DataFrame) -> str:
    """Create a single document from a post and its top comments."""
    parts = []
    
    # Title
    title = str(row.get('title', '')).strip()
    if title and title != 'nan':
        parts.append(f"TITLE: {title}")
    
    # Selftext (post body)
    selftext = str(row.get('selftext', '')).strip()
    if selftext and selftext != 'nan' and selftext != '[removed]' and selftext != '[deleted]':
        parts.append(f"POST: {selftext}")
    
    # Top comments
    if not comments_for_post.empty:
        comment_texts = []
        for _, comment in comments_for_post.iterrows():
            body = str(comment.get('body', '')).strip()
            if body and body != 'nan' and body != '[removed]' and body != '[deleted]':
                comment_texts.append(body)
        
        if comment_texts:
            parts.append("COMMENTS:\n" + "\n---\n".join(comment_texts[:5]))
    
    return "\n\n".join(parts)


def prepare_period_input(posts: pd.DataFrame, comments: pd.DataFrame, period_key: str, dry_run: bool = False) -> dict:
    """Prepare input text file for a single period."""
    period = PERIODS[period_key]
    
    # Filter by period
    period_posts, period_comments = filter_period(posts, comments, period_key)
    
    # Get top 5 comments per post
    top_comments = get_top_comments(period_comments, n=5)
    comments_by_post = top_comments.groupby('post_id')
    
    stats = {
        'period': period['name'],
        'posts': len(period_posts),
        'top_comments': len(top_comments),
        'total_documents': len(period_posts)
    }
    
    if dry_run:
        print(f"  [DRY RUN] Would create {len(period_posts):,} documents")
        return stats
    
    # Create output directory
    output_dir = GRAPHRAG_DIR / period_key / 'input'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create documents
    documents = []
    for _, row in period_posts.iterrows():
        post_id = row['id']
        
        # Get comments for this post
        if post_id in comments_by_post.groups:
            post_comments = comments_by_post.get_group(post_id)
        else:
            post_comments = pd.DataFrame()
        
        doc = create_document(row, post_comments)
        if doc.strip():
            documents.append(doc)
    
    # Write to file
    output_path = output_dir / 'posts.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n---DOCUMENT_SEPARATOR---\n\n".join(documents))
    
    stats['documents_written'] = len(documents)
    stats['output_path'] = str(output_path)
    
    print(f"  ✓ Wrote {len(documents):,} documents to {output_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Prepare GraphRAG input data')
    parser.add_argument('--dry-run', action='store_true', help='Show stats without writing files')
    parser.add_argument('--period', type=str, help='Process only specific period (period1, period2, period3)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GraphRAG Input Data Preparation")
    print("Option B: Posts + Top 5 Comments per Post")
    print("=" * 60)
    
    # Load data
    posts, comments = load_data()
    
    # Process periods
    periods_to_process = [args.period] if args.period else list(PERIODS.keys())
    
    all_stats = []
    print("\nProcessing periods...")
    
    for period_key in periods_to_process:
        print(f"\n{PERIODS[period_key]['description']}:")
        stats = prepare_period_input(posts, comments, period_key, dry_run=args.dry_run)
        all_stats.append(stats)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total_docs = 0
    for stats in all_stats:
        docs = stats.get('documents_written', stats.get('total_documents', 0))
        total_docs += docs
        print(f"{stats['period']}: {stats['posts']:,} posts + {stats['top_comments']:,} comments = {docs:,} docs")
    
    print(f"\nTotal documents: {total_docs:,}")
    
    if args.dry_run:
        print("\n[DRY RUN] No files were written. Remove --dry-run to create files.")
    else:
        print("\n✓ Input files ready for GraphRAG indexing")
        print("\nNext steps:")
        for period_key in periods_to_process:
            print(f"  cd graphrag/{period_key} && graphrag index --root .")


if __name__ == '__main__':
    main()
