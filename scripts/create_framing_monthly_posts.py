"""
Create Monthly Aggregated Framing Data (Posts Only)

Input: data/framing/{topic}_posts_scaled.csv
Output: data/framing/{topic}_monthly_framing.csv
"""

import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def create_monthly_framing(topic: str, input_dir: Path, output_dir: Path) -> pd.DataFrame:
    """Create monthly aggregated framing data for a topic."""
    input_path = input_dir / f"{topic}_posts_scaled.csv"
    output_path = output_dir / f"{topic}_monthly_framing.csv"

    print(f"\n{'='*60}")
    print(f"Processing: {topic.upper()}")
    print(f"{'='*60}")

    if not input_path.exists():
        print(f"  ERROR: File not found: {input_path}")
        return pd.DataFrame()

    # Load
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} posts")

    # Convert created_utc to datetime
    if 'created_utc' in df.columns:
        df['created_date'] = pd.to_datetime(df['created_utc'], unit='s')
    else:
        print(f"  ERROR: 'created_utc' column not found")
        return pd.DataFrame()

    # Filter to analysis period (2017-01 to 2019-06)
    df_filtered = df[
        (df['created_date'] >= '2017-01-01') &
        (df['created_date'] <= '2019-06-30')
    ].copy()
    print(f"  Filtered to analysis period: {len(df_filtered):,} posts")

    if len(df_filtered) == 0:
        print(f"  WARNING: No data in analysis period")
        return pd.DataFrame()

    # Create month column
    df_filtered['month'] = df_filtered['created_date'].dt.to_period('M').astype(str)

    # Aggregate to monthly level
    monthly = df_filtered.groupby('month').agg({
        'diplomacy_scale': ['mean', 'std', 'count']
    }).reset_index()
    monthly.columns = ['month', 'framing_mean', 'framing_std', 'post_count']

    # Fill missing months
    all_months = pd.period_range('2017-01', '2019-06', freq='M').astype(str)
    monthly_complete = pd.DataFrame({'month': all_months})
    monthly_complete = monthly_complete.merge(monthly, on='month', how='left')

    # Add topic identifier
    monthly_complete['topic'] = topic

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_complete.to_csv(output_path, index=False)

    print(f"\n  Monthly Summary:")
    print(f"    Total months: {len(monthly_complete)}")
    print(f"    Months with data: {monthly_complete['post_count'].notna().sum()}")
    print(f"    Mean framing: {monthly_complete['framing_mean'].mean():.4f}")
    print(f"    Total posts: {monthly_complete['post_count'].sum():.0f}")

    # Show pre/post comparison
    pre_months = monthly_complete[monthly_complete['month'] < '2018-03']
    post_months = monthly_complete[monthly_complete['month'] >= '2018-03']

    pre_mean = pre_months['framing_mean'].mean()
    post_mean = post_months['framing_mean'].mean()

    print(f"\n  Pre/Post Intervention Comparison:")
    print(f"    Pre-period mean (2017-01 to 2018-02):  {pre_mean:.4f}")
    print(f"    Post-period mean (2018-03 to 2019-06): {post_mean:.4f}")
    print(f"    Change: {post_mean - pre_mean:+.4f}")

    print(f"\n  ✓ Saved: {output_path}")

    return monthly_complete


def main():
    print("=" * 60)
    print("Creating Monthly Framing Aggregation (Posts Only)")
    print("=" * 60)

    input_dir = project_root / "data" / "framing"
    output_dir = project_root / "data" / "framing"

    topics = ['nk', 'iran', 'russia', 'china']
    results = {}

    for topic in topics:
        monthly = create_monthly_framing(topic, input_dir, output_dir)
        if len(monthly) > 0:
            pre = monthly[monthly['month'] < '2018-03']['framing_mean'].mean()
            post = monthly[monthly['month'] >= '2018-03']['framing_mean'].mean()
            results[topic] = {
                'pre_mean': pre,
                'post_mean': post,
                'change': post - pre,
                'total_posts': monthly['post_count'].sum()
            }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Pre/Post Framing Changes")
    print("=" * 60)
    print(f"{'Topic':<10} {'Pre-Mean':<12} {'Post-Mean':<12} {'Change':<12} {'Posts'}")
    print("-" * 60)
    for topic, data in results.items():
        print(f"{topic.upper():<10} {data['pre_mean']:+.4f}      {data['post_mean']:+.4f}      {data['change']:+.4f}      {data['total_posts']:.0f}")

    print("\n" + "=" * 60)
    print("Interpretation:")
    print("  Positive change = Shift toward DIPLOMACY framing")
    print("  Negative change = Shift toward THREAT framing")
    print("=" * 60)

    print("\n✓ Monthly aggregation complete!")
    print("\nNext: Run scripts/run_did_analysis_framing.py")


if __name__ == "__main__":
    main()
