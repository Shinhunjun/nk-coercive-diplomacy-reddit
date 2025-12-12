"""Collect ALL NK comments only (with pagination)"""
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.collect_all_comments import FullCommentsCollector


def collect_nk_comments():
    """Collect ALL comments for NK posts from merged file."""

    # NK uses merged file in data/nk/ directory
    posts_path = 'data/nk/nk_posts_merged.csv'

    if not os.path.exists(posts_path):
        print(f"Error: No posts file found at {posts_path}!")
        return None

    print("=" * 70)
    print("COLLECTING ALL NK COMMENTS")
    print("=" * 70)

    posts_df = pd.read_csv(posts_path)
    print(f"\nLoaded {len(posts_df):,} NK posts from {posts_path}")

    # Show expected comment count
    total_expected = posts_df['num_comments'].sum() if 'num_comments' in posts_df.columns else 'Unknown'
    if isinstance(total_expected, (int, float)):
        print(f"Expected total comments (from posts): {int(total_expected):,}")
    else:
        print(f"Expected total comments: {total_expected}")

    # Show period distribution
    if 'period' in posts_df.columns:
        print(f"\nPosts by period:")
        print(posts_df['period'].value_counts())

    # Collect ALL comments
    collector = FullCommentsCollector()
    comments = collector.collect_comments_for_posts(
        posts_df,
        max_comments_per_post=5000,  # Safety limit
        max_posts=None  # Process all posts
    )

    if len(comments) > 0:
        # Save to data/nk/ directory
        os.makedirs('data/nk', exist_ok=True)
        output_path = 'data/nk/nk_comments_full.csv'
        comments.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(comments):,} NK comments to {output_path}")

        # Show period distribution
        if 'period' in comments.columns:
            print(f"\nComments by period:")
            print(comments['period'].value_counts())

        # Show parent_id distribution (direct vs reply)
        if 'parent_id' in comments.columns:
            direct = comments['parent_id'].str.startswith('t3_').sum()
            replies = comments['parent_id'].str.startswith('t1_').sum()
            print(f"\nComment types:")
            print(f"  Direct comments (t3_): {direct:,} ({direct/len(comments)*100:.1f}%)")
            print(f"  Replies (t1_): {replies:,} ({replies/len(comments)*100:.1f}%)")

        # Show date range
        if 'created_utc' in comments.columns:
            comments['datetime'] = pd.to_datetime(comments['created_utc'], unit='s')
            print(f"\nDate range: {comments['datetime'].min()} to {comments['datetime'].max()}")

        return comments
    else:
        print(f"\n⚠️ No NK comments collected!")
        return None


if __name__ == '__main__':
    collect_nk_comments()
