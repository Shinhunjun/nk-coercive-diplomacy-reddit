"""
Apply OpenAI GPT-3.5-Turbo Framing Classification to Full Dataset

This script applies framing classification to all posts and comments (2017-2019 only)
for NK and all control groups (Iran, Russia, China).

Estimated:
- NK: 12,531 items
- Control groups: ~40,000 items
- Total: ~52,000 items × $0.001 = ~$26

Output files:
- {topic}_posts_framing.csv
- {topic}_comments_framing.csv
"""

import pandas as pd
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from openai_framing_analysis import OpenAIFramingAnalyzer
from config import DATA_DIR

# Date range for filtering (2017-2019 only)
START_DATE = "2017-01-01"
END_DATE = "2019-12-31"


def apply_framing_to_file(
    analyzer: OpenAIFramingAnalyzer,
    input_path: Path,
    output_path: Path,
    data_type: str,  # 'posts' or 'comments'
    topic: str
):
    """
    Apply framing classification to a single file.

    Args:
        analyzer: OpenAIFramingAnalyzer instance
        input_path: Input CSV file path
        output_path: Output CSV file path
        data_type: 'posts' or 'comments'
        topic: Topic name (nk, iran, russia, china)
    """
    print(f"\n{'='*80}")
    print(f"Processing {topic.upper()} {data_type.upper()}")
    print(f"{'='*80}")

    # Check if output file already exists and is complete
    if output_path.exists():
        try:
            df_existing = pd.read_csv(output_path)
            if 'frame' in df_existing.columns and len(df_existing) > 0:
                print(f"  ✓ Already completed: {len(df_existing):,} items")
                print(f"  Skipping {topic} {data_type}")
                return
        except:
            pass

    # Check if file exists
    if not input_path.exists():
        print(f"  WARNING: File not found: {input_path}")
        return

    # Load data
    df = pd.read_csv(input_path)
    print(f"  Total items: {len(df):,}")

    # Determine datetime column
    if 'datetime' in df.columns:
        date_col = 'datetime'
    elif 'created_utc' in df.columns:
        date_col = 'created_utc'
        df['datetime'] = pd.to_datetime(df[date_col], unit='s')
        date_col = 'datetime'
    else:
        print(f"  ERROR: No datetime column found")
        return

    # Convert to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')  # Coerce invalid dates to NaT

    # Filter to 2017-2019
    df_filtered = df[
        (df[date_col] >= START_DATE) &
        (df[date_col] <= END_DATE)
    ].copy()

    print(f"  Filtered (2017-2019): {len(df_filtered):,} items")

    if len(df_filtered) == 0:
        print(f"  WARNING: No data in date range for {topic} {data_type}")
        return

    # Determine title and body columns
    if data_type == 'posts':
        title_col = 'title'
        body_col = 'selftext'
    else:  # comments
        # For comments, use body as title since comments don't have titles
        if 'body' in df_filtered.columns:
            # Filter out [deleted] and [removed] comments
            before_filter = len(df_filtered)
            df_filtered = df_filtered[
                ~df_filtered['body'].isin(['[deleted]', '[removed]'])
            ].copy()
            removed_count = before_filter - len(df_filtered)
            if removed_count > 0:
                print(f"  Removed {removed_count:,} [deleted]/[removed] comments")

            df_filtered['title'] = df_filtered['body'].fillna('').str[:200]  # First 200 chars
            title_col = 'title'
            body_col = 'body'
        else:
            print(f"  ERROR: No body column found for comments")
            return

    if len(df_filtered) == 0:
        print(f"  WARNING: No valid items to classify for {topic} {data_type}")
        return

    # Apply framing classification
    print(f"\n  Classifying {len(df_filtered):,} items...")
    df_framed = analyzer.analyze_dataframe(
        df=df_filtered,
        title_col=title_col,
        body_col=body_col,
        delay=0.1  # 0.1 second delay to avoid rate limits
    )

    # Show results
    print(f"\n  Frame Distribution:")
    frame_counts = df_framed['frame'].value_counts()
    for frame, count in frame_counts.items():
        pct = count / len(df_framed) * 100
        print(f"    {frame:15s}: {count:5d} ({pct:5.1f}%)")

    print(f"\n  Confidence Statistics:")
    print(f"    Mean: {df_framed['frame_confidence'].mean():.3f}")
    print(f"    Std:  {df_framed['frame_confidence'].std():.3f}")
    print(f"    Min:  {df_framed['frame_confidence'].min():.3f}")
    print(f"    Max:  {df_framed['frame_confidence'].max():.3f}")

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_framed.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_path}")

    return df_framed


def main():
    """Main execution function."""
    print("=" * 80)
    print("APPLY OPENAI GPT-3.5-TURBO FRAMING CLASSIFICATION")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("\nERROR: OPENAI_API_KEY environment variable not set!")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        sys.exit(1)

    print(f"\nModel: gpt-4o-mini (best cost/performance)")
    print(f"Date range: 2017-01-01 to 2019-12-31")
    print(f"Estimated total items: ~52,000")
    print(f"Estimated cost: ~$5.50 (70% cheaper than gpt-3.5-turbo)")
    print(f"Estimated time: ~1.5 hours")

    # Initialize analyzer
    print("\nInitializing OpenAI analyzer...")
    analyzer = OpenAIFramingAnalyzer(api_key=api_key, model="gpt-4o-mini")
    print("✓ Analyzer initialized")

    # Define datasets
    datasets = [
        # NK (Treatment group)
        {
            'topic': 'nk',
            'posts_input': DATA_DIR / 'processed' / 'nk_posts_roberta_sentiment.csv',
            'posts_output': DATA_DIR / 'processed' / 'nk_posts_framing.csv',
            'comments_input': DATA_DIR / 'processed' / 'nk_comments_roberta_sentiment.csv',
            'comments_output': DATA_DIR / 'processed' / 'nk_comments_framing.csv'
        },
        # Iran (Control group 1)
        {
            'topic': 'iran',
            'posts_input': DATA_DIR / 'processed' / 'iran_posts_roberta_sentiment.csv',
            'posts_output': DATA_DIR / 'processed' / 'iran_posts_framing.csv',
            'comments_input': DATA_DIR / 'processed' / 'iran_comments_roberta_sentiment.csv',
            'comments_output': DATA_DIR / 'processed' / 'iran_comments_framing.csv'
        },
        # Russia (Control group 2)
        {
            'topic': 'russia',
            'posts_input': DATA_DIR / 'processed' / 'russia_posts_roberta_sentiment.csv',
            'posts_output': DATA_DIR / 'processed' / 'russia_posts_framing.csv',
            'comments_input': DATA_DIR / 'processed' / 'russia_comments_roberta_sentiment.csv',
            'comments_output': DATA_DIR / 'processed' / 'russia_comments_framing.csv'
        },
        # China (Control group 3)
        {
            'topic': 'china',
            'posts_input': DATA_DIR / 'processed' / 'china_posts_roberta_sentiment.csv',
            'posts_output': DATA_DIR / 'processed' / 'china_posts_framing.csv',
            'comments_input': DATA_DIR / 'processed' / 'china_comments_roberta_sentiment.csv',
            'comments_output': DATA_DIR / 'processed' / 'china_comments_framing.csv'
        }
    ]

    # Process each dataset
    total_items = 0
    for dataset in datasets:
        topic = dataset['topic']

        # Process posts
        print(f"\n{'='*80}")
        print(f"TOPIC: {topic.upper()}")
        print(f"{'='*80}")

        try:
            df_posts = apply_framing_to_file(
                analyzer=analyzer,
                input_path=dataset['posts_input'],
                output_path=dataset['posts_output'],
                data_type='posts',
                topic=topic
            )
            if df_posts is not None:
                total_items += len(df_posts)
        except Exception as e:
            print(f"\n  ERROR processing {topic} posts: {str(e)}")
            import traceback
            traceback.print_exc()

        # Process comments
        try:
            df_comments = apply_framing_to_file(
                analyzer=analyzer,
                input_path=dataset['comments_input'],
                output_path=dataset['comments_output'],
                data_type='comments',
                topic=topic
            )
            if df_comments is not None:
                total_items += len(df_comments)
        except Exception as e:
            print(f"\n  ERROR processing {topic} comments: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("✓ FRAMING CLASSIFICATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal items processed: {total_items:,}")
    print(f"Actual cost: ~${total_items * 0.0005:.2f}")
    print(f"\nOutput files saved to: {DATA_DIR / 'processed'}")
    print("\nNext steps:")
    print("  1. Run scripts/create_framing_scale.py")
    print("  2. Run scripts/create_framing_monthly_aggregation.py")
    print("  3. Run scripts/run_did_analysis_framing.py")


if __name__ == "__main__":
    main()
