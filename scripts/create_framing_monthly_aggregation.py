"""
Create Monthly Aggregated Framing Data from Posts + Comments

This script combines posts and comments and aggregates the diplomacy_scale
to monthly level for DID analysis.

Input: Files with 'diplomacy_scale' column (from create_framing_scale.py)
Output: Monthly aggregated CSV files with mean framing scale per month
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR


def create_framing_monthly(
    posts_path: Path,
    comments_path: Path,
    topic_name: str,
    output_path: Path
) -> pd.DataFrame:
    """
    Combine posts and comments into monthly aggregated framing data.

    Args:
        posts_path: Path to posts CSV with diplomacy_scale
        comments_path: Path to comments CSV with diplomacy_scale
        topic_name: Topic identifier (e.g., 'nk', 'iran')
        output_path: Where to save monthly data

    Returns:
        Monthly aggregated DataFrame
    """
    print(f"\n{'='*80}")
    print(f"CREATING MONTHLY FRAMING DATA: {topic_name.upper()}")
    print(f"{'='*80}")

    # Load posts
    print(f"\nLoading posts from: {posts_path.name}")
    if not posts_path.exists():
        print(f"  WARNING: Posts file not found")
        posts = pd.DataFrame()
    else:
        posts = pd.read_csv(posts_path)
        print(f"  Total posts: {len(posts)}")

    # Load comments
    print(f"\nLoading comments from: {comments_path.name}")
    if not comments_path.exists():
        print(f"  WARNING: Comments file not found")
        comments = pd.DataFrame()
    else:
        comments = pd.read_csv(comments_path)
        print(f"  Total comments: {len(comments)}")

    if len(posts) == 0 and len(comments) == 0:
        print(f"  ERROR: No data found for {topic_name}")
        return pd.DataFrame()

    # Select common columns
    common_cols = ['created_date', 'diplomacy_scale']

    # Prepare posts
    if len(posts) > 0:
        # Rename created_utc to created_date if needed
        if 'created_utc' in posts.columns:
            posts['created_date'] = pd.to_datetime(posts['created_utc'], unit='s')

        posts_subset = posts[common_cols].copy()
        posts_subset['source'] = 'post'
    else:
        posts_subset = pd.DataFrame()

    # Prepare comments
    if len(comments) > 0:
        # Rename created_utc to created_date if needed
        if 'created_utc' in comments.columns:
            comments['created_date'] = pd.to_datetime(comments['created_utc'], unit='s')

        comments_subset = comments[common_cols].copy()
        comments_subset['source'] = 'comment'
    else:
        comments_subset = pd.DataFrame()

    # Combine
    if len(posts_subset) > 0 and len(comments_subset) > 0:
        combined = pd.concat([posts_subset, comments_subset], ignore_index=True)
    elif len(posts_subset) > 0:
        combined = posts_subset
    else:
        combined = comments_subset

    print(f"\n✓ Combined total: {len(combined)} items")
    if len(posts) > 0 and len(comments) > 0:
        print(f"  Posts: {len(posts)}, Comments: {len(comments)}")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(combined['created_date']):
        combined['created_date'] = pd.to_datetime(combined['created_date'])

    # Filter to analysis period (2017-01 to 2019-06)
    combined_filtered = combined[
        (combined['created_date'] >= '2017-01-01') &
        (combined['created_date'] <= '2019-06-30')
    ].copy()

    print(f"\nAnalysis period (2017-01 to 2019-06):")
    print(f"  Total items: {len(combined_filtered)}")

    if len(combined_filtered) == 0:
        print(f"  WARNING: No data in analysis period for {topic_name}")
        return pd.DataFrame()

    # Create month column
    combined_filtered['month'] = combined_filtered['created_date'].dt.to_period('M').astype(str)

    # Show framing scale statistics for filtered data
    print(f"\nDiplomacy scale statistics (2017-2019):")
    print(f"  Mean: {combined_filtered['diplomacy_scale'].mean():.3f}")
    print(f"  Std:  {combined_filtered['diplomacy_scale'].std():.3f}")
    print(f"  Min:  {combined_filtered['diplomacy_scale'].min():.1f}")
    print(f"  Max:  {combined_filtered['diplomacy_scale'].max():.1f}")

    # Aggregate to monthly level
    print(f"\nAggregating to monthly level...")
    monthly = combined_filtered.groupby('month').agg({
        'diplomacy_scale': ['mean', 'std', 'count']
    }).reset_index()

    monthly.columns = ['month', 'framing_mean', 'framing_std', 'item_count']

    # Fill missing months in the full time range (2017-01 to 2019-06)
    all_months = pd.period_range('2017-01', '2019-06', freq='M').astype(str)
    monthly_complete = pd.DataFrame({'month': all_months})
    monthly_complete = monthly_complete.merge(monthly, on='month', how='left')

    # Add topic identifier
    monthly_complete['topic'] = topic_name

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    monthly_complete.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")
    print(f"\nMonthly Summary:")
    print(f"  Total months: {len(monthly_complete)}")
    print(f"  Months with data: {monthly_complete['item_count'].notna().sum()}")
    print(f"  Mean framing: {monthly_complete['framing_mean'].mean():.4f}")
    print(f"  Total items used: {monthly_complete['item_count'].sum():.0f}")

    # Show sample
    print(f"\nSample monthly data (first 10 months with data):")
    sample = monthly_complete[monthly_complete['item_count'].notna()][
        ['month', 'framing_mean', 'item_count']
    ].head(10)
    print(sample.to_string(index=False))

    return monthly_complete


def main():
    """Create monthly framing data for NK and all control groups."""

    print("=" * 80)
    print("Creating Monthly Framing Aggregations")
    print("=" * 80)

    # Define datasets
    datasets = [
        {
            'topic': 'nk',
            'posts': DATA_DIR / 'processed' / 'nk_posts_framing_scaled.csv',
            'comments': DATA_DIR / 'processed' / 'nk_comments_framing_scaled.csv',
            'output': DATA_DIR / 'processed' / 'nk_monthly_framing.csv'
        },
        {
            'topic': 'iran',
            'posts': DATA_DIR / 'processed' / 'iran_posts_framing_scaled.csv',
            'comments': DATA_DIR / 'processed' / 'iran_comments_framing_scaled.csv',
            'output': DATA_DIR / 'processed' / 'iran_monthly_framing.csv'
        },
        {
            'topic': 'russia',
            'posts': DATA_DIR / 'processed' / 'russia_posts_framing_scaled.csv',
            'comments': DATA_DIR / 'processed' / 'russia_comments_framing_scaled.csv',
            'output': DATA_DIR / 'processed' / 'russia_monthly_framing.csv'
        },
        {
            'topic': 'china',
            'posts': DATA_DIR / 'processed' / 'china_posts_framing_scaled.csv',
            'comments': DATA_DIR / 'processed' / 'china_comments_framing_scaled.csv',
            'output': DATA_DIR / 'processed' / 'china_monthly_framing.csv'
        }
    ]

    # Process each dataset
    results = {}
    for dataset in datasets:
        try:
            monthly = create_framing_monthly(
                posts_path=dataset['posts'],
                comments_path=dataset['comments'],
                topic_name=dataset['topic'],
                output_path=dataset['output']
            )
            results[dataset['topic']] = monthly
        except Exception as e:
            print(f"\nERROR processing {dataset['topic']}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("✓ MONTHLY FRAMING DATA CREATED")
    print("=" * 80)

    print("\nSummary by topic:")
    for topic, monthly in results.items():
        if len(monthly) > 0:
            mean_framing = monthly['framing_mean'].mean()
            items_total = monthly['item_count'].sum()
            print(f"  {topic.upper():8s}: Mean framing = {mean_framing:+.3f}, Items = {items_total:.0f}")

    print(f"\nOutput files saved to: {DATA_DIR / 'processed'}")
    print("\nNext steps:")
    print("  1. Run scripts/run_did_analysis_framing.py")
    print("  2. Run scripts/framing_vs_sentiment_comparison.py")


if __name__ == "__main__":
    main()
