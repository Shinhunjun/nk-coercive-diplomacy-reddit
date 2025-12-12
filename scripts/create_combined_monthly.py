"""
Create combined monthly data from posts + comments
Uses RoBERTa sentiment scores for both
"""

import pandas as pd
import os


def create_combined_monthly(
    posts_path: str,
    comments_path: str,
    topic_name: str,
    output_path: str
) -> pd.DataFrame:
    """
    Combine posts and comments into monthly aggregated data.

    Args:
        posts_path: Path to posts CSV with RoBERTa sentiment
        comments_path: Path to comments CSV with RoBERTa sentiment
        topic_name: Topic identifier (e.g., 'nk', 'iran')
        output_path: Where to save monthly data

    Returns:
        Monthly aggregated DataFrame
    """
    print(f"\n{'='*60}")
    print(f"CREATING COMBINED MONTHLY DATA: {topic_name.upper()}")
    print(f"{'='*60}")

    # Load posts
    print(f"\nLoading posts from: {posts_path}")
    posts = pd.read_csv(posts_path)
    print(f"  Total posts: {len(posts)}")

    # Load comments
    print(f"\nLoading comments from: {comments_path}")
    comments = pd.read_csv(comments_path)
    print(f"  Total comments: {len(comments)}")

    # Combine posts and comments
    # Posts: use 'text' or create from title+selftext
    if 'text' not in posts.columns:
        posts['text'] = posts['title'].fillna('') + ' ' + posts['selftext'].fillna('')

    # Comments: use 'body'
    # Rename comment 'body' to 'text' for consistency
    comments_renamed = comments.rename(columns={'body': 'text'})

    # Select common columns
    common_cols = ['created_utc', 'sentiment_score']

    # Ensure both have these columns
    posts_subset = posts[common_cols].copy()
    comments_subset = comments_renamed[common_cols].copy()

    # Add source label
    posts_subset['source'] = 'post'
    comments_subset['source'] = 'comment'

    # Combine
    combined = pd.concat([posts_subset, comments_subset], ignore_index=True)
    print(f"\n✓ Combined total: {len(combined)} (posts: {len(posts)}, comments: {len(comments)})")

    # Convert to datetime
    combined['datetime'] = pd.to_datetime(combined['created_utc'], unit='s')

    # Filter to analysis period (2017-01 to 2019-06)
    combined_filtered = combined[
        (combined['datetime'] >= '2017-01-01') &
        (combined['datetime'] <= '2019-06-30')
    ].copy()

    print(f"\nAnalysis period (2017-01 to 2019-06):")
    print(f"  Total items: {len(combined_filtered)}")
    print(f"  Posts: {len(combined_filtered[combined_filtered['source'] == 'post'])}")
    print(f"  Comments: {len(combined_filtered[combined_filtered['source'] == 'comment'])}")

    # Create month column
    combined_filtered['month'] = combined_filtered['datetime'].dt.to_period('M').astype(str)

    # Aggregate to monthly level
    print(f"\nAggregating to monthly level...")
    monthly = combined_filtered.groupby('month').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()

    monthly.columns = ['month', 'sentiment_mean', 'sentiment_std', 'post_count']

    # Fill missing months in the full time range (2017-01 to 2019-06)
    all_months = pd.period_range('2017-01', '2019-06', freq='M').astype(str)
    monthly_complete = pd.DataFrame({'month': all_months})
    monthly_complete = monthly_complete.merge(monthly, on='month', how='left')

    # Add topic identifier
    monthly_complete['topic'] = topic_name

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    monthly_complete.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")
    print(f"\nMonthly Summary:")
    print(f"  Total months: {len(monthly_complete)}")
    print(f"  Months with data: {monthly_complete['post_count'].notna().sum()}")
    print(f"  Mean sentiment: {monthly_complete['sentiment_mean'].mean():.4f}")
    print(f"  Total items used: {monthly_complete['post_count'].sum():.0f}")

    # Show sample
    print(f"\nSample monthly data (first 10 months with data):")
    sample = monthly_complete[monthly_complete['post_count'].notna()][['month', 'sentiment_mean', 'post_count']].head(10)
    print(sample)

    return monthly_complete


def main():
    """Create combined monthly data for NK and all control groups (Iran, Russia, China)."""

    # 1. NK combined monthly
    print("\n" + "="*60)
    print("STEP 1: NK COMBINED MONTHLY (POSTS + COMMENTS)")
    print("="*60)

    nk_monthly = create_combined_monthly(
        posts_path='data/processed/nk_posts_roberta_sentiment.csv',
        comments_path='data/processed/nk_comments_roberta.csv',
        topic_name='nk',
        output_path='data/processed/nk_monthly_combined_roberta.csv'
    )

    # 2. Iran combined monthly
    print("\n" + "="*60)
    print("STEP 2: IRAN COMBINED MONTHLY (POSTS + COMMENTS)")
    print("="*60)

    iran_monthly = create_combined_monthly(
        posts_path='data/control/iran_posts_roberta.csv',
        comments_path='data/control/iran_comments_roberta.csv',
        topic_name='iran',
        output_path='data/processed/iran_monthly_combined_roberta.csv'
    )

    # 3. Russia combined monthly
    print("\n" + "="*60)
    print("STEP 3: RUSSIA COMBINED MONTHLY (POSTS + COMMENTS)")
    print("="*60)

    russia_monthly = create_combined_monthly(
        posts_path='data/control/russia_posts_roberta.csv',
        comments_path='data/control/russia_comments_roberta.csv',
        topic_name='russia',
        output_path='data/processed/russia_monthly_combined_roberta.csv'
    )

    # 4. China combined monthly
    print("\n" + "="*60)
    print("STEP 4: CHINA COMBINED MONTHLY (POSTS + COMMENTS)")
    print("="*60)

    china_monthly = create_combined_monthly(
        posts_path='data/control/china_posts_roberta.csv',
        comments_path='data/control/china_comments_roberta.csv',
        topic_name='china',
        output_path='data/processed/china_monthly_combined_roberta.csv'
    )

    print("\n" + "="*60)
    print("✓ COMBINED MONTHLY DATA CREATED FOR ALL GROUPS")
    print("="*60)
    print("\nNext steps:")
    print("1. Run parallel trends test with all three control groups")
    print("2. Run DID estimation with all three control groups")
    print("3. Compare results across control groups")


if __name__ == '__main__':
    main()
