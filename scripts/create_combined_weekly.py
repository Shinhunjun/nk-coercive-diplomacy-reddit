"""
Create combined weekly aggregated data from posts and comments
Similar to create_combined_monthly.py but with weekly aggregation
"""

import pandas as pd
import numpy as np


def create_combined_weekly(
    posts_path: str,
    comments_path: str,
    topic_name: str,
    output_path: str
) -> pd.DataFrame:
    """
    Combine posts and comments, then aggregate to weekly level.

    Args:
        posts_path: Path to posts CSV with sentiment scores
        comments_path: Path to comments CSV with sentiment scores
        topic_name: Name of the topic (for logging)
        output_path: Path to save weekly aggregated data

    Returns:
        DataFrame with weekly aggregated sentiment data
    """
    print(f"\nProcessing {topic_name}...")

    # Load posts and comments
    posts = pd.read_csv(posts_path, low_memory=False)
    comments = pd.read_csv(comments_path, low_memory=False)

    print(f"  Posts: {len(posts):,}")
    print(f"  Comments: {len(comments):,}")

    # Select common columns
    common_cols = ['created_utc', 'sentiment_score']
    posts_subset = posts[common_cols].copy()
    comments_subset = comments[common_cols].copy()

    # Add source label
    posts_subset['source'] = 'post'
    comments_subset['source'] = 'comment'

    # Combine
    combined = pd.concat([posts_subset, comments_subset], ignore_index=True)
    print(f"  Combined: {len(combined):,} items")

    # Convert to datetime
    combined['datetime'] = pd.to_datetime(combined['created_utc'], unit='s')

    # Filter to analysis period (2017-01-01 to 2019-06-30)
    combined_filtered = combined[
        (combined['datetime'] >= '2017-01-01') &
        (combined['datetime'] <= '2019-06-30')
    ].copy()

    print(f"  Filtered (2017-2019): {len(combined_filtered):,} items")

    # Create week identifier (ISO week: YYYY-WW)
    # Use Monday as start of week
    combined_filtered['year'] = combined_filtered['datetime'].dt.isocalendar().year
    combined_filtered['week'] = combined_filtered['datetime'].dt.isocalendar().week
    combined_filtered['week_id'] = (
        combined_filtered['year'].astype(str) + '-W' +
        combined_filtered['week'].astype(str).str.zfill(2)
    )

    # Aggregate to weekly level
    # First, get sentiment aggregation
    weekly_sentiment = combined_filtered.groupby('week_id').agg({
        'sentiment_score': ['mean', 'std', 'count'],
        'datetime': 'min'  # First date in the week
    }).reset_index()

    # Flatten column names
    weekly_sentiment.columns = ['week', 'sentiment_mean', 'sentiment_std', 'total_count', 'week_start']

    # Count posts and comments separately
    post_counts = combined_filtered[combined_filtered['source'] == 'post'].groupby('week_id').size()
    comment_counts = combined_filtered[combined_filtered['source'] == 'comment'].groupby('week_id').size()

    # Merge counts
    weekly = weekly_sentiment.copy()
    weekly['post_count'] = weekly['week'].map(post_counts).fillna(0).astype(int)
    weekly['comment_count'] = weekly['week'].map(comment_counts).fillna(0).astype(int)

    # Rename total_count to match
    weekly = weekly.rename(columns={'total_count': 'item_count'})

    # Sort by week
    weekly = weekly.sort_values('week_start')

    # Create complete week range (2017-W01 to 2019-W26)
    start_date = pd.to_datetime('2017-01-02')  # First Monday of 2017
    end_date = pd.to_datetime('2019-06-30')

    # Generate all weeks in range
    all_weeks = []
    current = start_date
    while current <= end_date:
        year = current.isocalendar()[0]
        week = current.isocalendar()[1]
        week_id = f"{year}-W{str(week).zfill(2)}"
        all_weeks.append({
            'week': week_id,
            'week_start': current
        })
        current += pd.Timedelta(days=7)

    all_weeks_df = pd.DataFrame(all_weeks)

    # Merge with actual data (only on 'week' column)
    weekly_complete = all_weeks_df.merge(
        weekly.drop('week_start', axis=1),
        on='week',
        how='left'
    )

    print(f"  Weekly observations: {len(weekly_complete)}")
    print(f"  Weeks with data: {weekly_complete['sentiment_mean'].notna().sum()}")
    print(f"  Weeks without data: {weekly_complete['sentiment_mean'].isna().sum()}")

    # Check data distribution
    if weekly_complete['sentiment_mean'].notna().any():
        print(f"\n  Weekly sentiment distribution:")
        print(f"    Mean: {weekly_complete['sentiment_mean'].mean():.4f}")
        print(f"    Std: {weekly_complete['sentiment_mean'].std():.4f}")
        print(f"    Min: {weekly_complete['sentiment_mean'].min():.4f}")
        print(f"    Max: {weekly_complete['sentiment_mean'].max():.4f}")

        print(f"\n  Weekly item count distribution (posts + comments):")
        print(f"    Mean: {weekly_complete['item_count'].mean():.1f}")
        print(f"    Median: {weekly_complete['item_count'].median():.1f}")
        print(f"    Min: {weekly_complete['item_count'].min():.0f}")
        print(f"    Max: {weekly_complete['item_count'].max():.0f}")
        print(f"    Weeks with <10 items: {(weekly_complete['item_count'] < 10).sum()}")
        print(f"    Weeks with <30 items: {(weekly_complete['item_count'] < 30).sum()}")

    # Save to CSV
    weekly_complete.to_csv(output_path, index=False)
    print(f"  ✓ Saved to: {output_path}")

    print(f"\nSample weekly data (first 10 weeks with data):")
    sample = weekly_complete[weekly_complete['post_count'].notna()][['week', 'sentiment_mean', 'post_count']].head(10)
    print(sample)

    return weekly_complete


def main():
    """Create combined weekly data for NK and all control groups (Iran, Russia, China)."""

    print("="*80)
    print("WEEKLY AGGREGATION: Creating Combined Weekly Data")
    print("="*80)

    # 1. NK combined weekly
    print("\n" + "="*60)
    print("STEP 1: NK COMBINED WEEKLY (POSTS + COMMENTS)")
    print("="*60)

    nk_weekly = create_combined_weekly(
        posts_path='data/processed/nk_posts_roberta_sentiment.csv',
        comments_path='data/processed/nk_comments_roberta.csv',
        topic_name='nk',
        output_path='data/processed/nk_weekly_combined_roberta.csv'
    )

    # 2. Iran combined weekly
    print("\n" + "="*60)
    print("STEP 2: IRAN COMBINED WEEKLY (POSTS + COMMENTS)")
    print("="*60)

    iran_weekly = create_combined_weekly(
        posts_path='data/control/iran_posts_roberta.csv',
        comments_path='data/control/iran_comments_roberta.csv',
        topic_name='iran',
        output_path='data/processed/iran_weekly_combined_roberta.csv'
    )

    # 3. Russia combined weekly
    print("\n" + "="*60)
    print("STEP 3: RUSSIA COMBINED WEEKLY (POSTS + COMMENTS)")
    print("="*60)

    russia_weekly = create_combined_weekly(
        posts_path='data/control/russia_posts_roberta.csv',
        comments_path='data/control/russia_comments_roberta.csv',
        topic_name='russia',
        output_path='data/processed/russia_weekly_combined_roberta.csv'
    )

    # 4. China combined weekly
    print("\n" + "="*60)
    print("STEP 4: CHINA COMBINED WEEKLY (POSTS + COMMENTS)")
    print("="*60)

    china_weekly = create_combined_weekly(
        posts_path='data/control/china_posts_roberta.csv',
        comments_path='data/control/china_comments_roberta.csv',
        topic_name='china',
        output_path='data/processed/china_weekly_combined_roberta.csv'
    )

    print("\n" + "="*60)
    print("✓ COMBINED WEEKLY DATA CREATED FOR ALL GROUPS")
    print("="*60)

    # Summary comparison with monthly
    print("\n" + "="*60)
    print("COMPARISON: WEEKLY vs MONTHLY")
    print("="*60)

    # Load monthly for comparison
    nk_monthly = pd.read_csv('data/processed/nk_monthly_combined_roberta.csv')

    print(f"\n{'Metric':<30} {'Monthly':>12} {'Weekly':>12} {'Ratio':>10}")
    print("-"*60)
    print(f"{'Total observations':<30} {len(nk_monthly):>12} {len(nk_weekly):>12} {len(nk_weekly)/len(nk_monthly):>9.1f}x")
    print(f"{'Observations with data':<30} {nk_monthly['sentiment_mean'].notna().sum():>12} {nk_weekly['sentiment_mean'].notna().sum():>12} {nk_weekly['sentiment_mean'].notna().sum()/nk_monthly['sentiment_mean'].notna().sum():>9.1f}x")
    print(f"{'Avg posts/period':<30} {nk_monthly['post_count'].mean():>12.1f} {nk_weekly['post_count'].mean():>12.1f} {nk_weekly['post_count'].mean()/nk_monthly['post_count'].mean():>9.2f}x")

    print("\nNext steps:")
    print("1. Run parallel trends test with weekly data")
    print("2. Run DID estimation with weekly data")
    print("3. Compare weekly vs monthly results")
    print("4. Check if data quality is sufficient (weeks with <10 items)")


if __name__ == '__main__':
    main()
