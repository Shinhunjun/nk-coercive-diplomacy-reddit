"""
Apply Vertex AI Gemini Framing Classification to Reddit Data

This script applies framing analysis to NK and control group data (2017-2019 only).
Uses Vertex AI Gemini 1.5 Flash for classification with BLOCK_NONE safety settings.

Processes 8 datasets:
- NK (posts + comments)
- Iran (posts + comments)
- Russia (posts + comments)
- China (posts + comments)

Output: Files with added 'frame', 'frame_confidence', 'frame_reason' columns
"""

import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vertex_ai_framing_analysis import VertexAIFramingAnalyzer
from config import DATA_DIR, RESULTS_DIR, CONTROL_GROUPS

# Date range for filtering (2017-2019 only)
START_DATE = "2017-01-01"
END_DATE = "2019-12-31"


def load_and_filter_data(file_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Load CSV and filter to date range.

    Args:
        file_path: Path to CSV file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Filtered DataFrame
    """
    print(f"\nLoading: {file_path.name}")
    df = pd.read_csv(file_path)

    # Convert date column
    if 'created_date' in df.columns:
        df['created_date'] = pd.to_datetime(df['created_date'])
    else:
        raise ValueError(f"No 'created_date' column in {file_path}")

    # Filter to date range
    mask = (df['created_date'] >= start_date) & (df['created_date'] <= end_date)
    df_filtered = df[mask].copy()

    print(f"  Original: {len(df):,} rows")
    print(f"  Filtered ({start_date} to {end_date}): {len(df_filtered):,} rows")

    return df_filtered


def apply_framing_to_dataset(
    df: pd.DataFrame,
    analyzer: VertexAIFramingAnalyzer,
    title_col: str = 'title',
    body_col: str = 'selftext',
    delay: float = 0.5
) -> pd.DataFrame:
    """
    Apply framing classification to DataFrame.

    Args:
        df: DataFrame with posts/comments
        analyzer: VertexAIFramingAnalyzer instance
        title_col: Column name for title
        body_col: Column name for body text
        delay: Delay between API calls (seconds)

    Returns:
        DataFrame with framing columns added
    """
    return analyzer.analyze_dataframe(
        df=df,
        title_col=title_col,
        body_col=body_col,
        delay=delay
    )


def process_posts(
    input_path: Path,
    output_path: Path,
    analyzer: VertexAIFramingAnalyzer,
    start_date: str,
    end_date: str
):
    """Process posts dataset with framing classification."""
    # Load and filter
    df = load_and_filter_data(input_path, start_date, end_date)

    if len(df) == 0:
        print("  No data in date range. Skipping.")
        return

    # Apply framing
    print(f"  Applying framing classification to {len(df)} posts...")
    df_framed = apply_framing_to_dataset(
        df=df,
        analyzer=analyzer,
        title_col='title',
        body_col='selftext',
        delay=0.5
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_framed.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Show distribution
    frame_dist = df_framed['frame'].value_counts()
    print(f"  Frame distribution:")
    for frame, count in frame_dist.items():
        pct = count / len(df_framed) * 100
        print(f"    {frame}: {count} ({pct:.1f}%)")


def process_comments(
    input_path: Path,
    output_path: Path,
    analyzer: VertexAIFramingAnalyzer,
    start_date: str,
    end_date: str
):
    """Process comments dataset with framing classification."""
    # Load and filter
    df = load_and_filter_data(input_path, start_date, end_date)

    if len(df) == 0:
        print("  No data in date range. Skipping.")
        return

    # Apply framing (comments use 'body' column)
    print(f"  Applying framing classification to {len(df)} comments...")
    df_framed = apply_framing_to_dataset(
        df=df,
        analyzer=analyzer,
        title_col='body',  # Comments don't have titles, use body as "title"
        body_col='body',    # Same column for body
        delay=0.5
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_framed.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")

    # Show distribution
    frame_dist = df_framed['frame'].value_counts()
    print(f"  Frame distribution:")
    for frame, count in frame_dist.items():
        pct = count / len(df_framed) * 100
        print(f"    {frame}: {count} ({pct:.1f}%)")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Vertex AI Gemini Framing Classification")
    print("=" * 80)

    # Check GCP project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("\nERROR: GOOGLE_CLOUD_PROJECT environment variable not set!")
        print("Please set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        sys.exit(1)

    print(f"\nGCP Project: {project_id}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"Model: gemini-1.5-flash-002")

    # Initialize analyzer
    print("\nInitializing Vertex AI Gemini analyzer...")
    analyzer = VertexAIFramingAnalyzer(project_id=project_id, location="us-central1")
    print("  Analyzer initialized successfully!")

    # Define datasets to process
    datasets = [
        # NK (Treatment group)
        {
            'name': 'NK Posts',
            'input': DATA_DIR / 'processed' / 'nk_posts_roberta_sentiment.csv',
            'output': DATA_DIR / 'processed' / 'nk_posts_framing.csv',
            'type': 'posts'
        },
        {
            'name': 'NK Comments',
            'input': DATA_DIR / 'processed' / 'nk_comments_roberta.csv',
            'output': DATA_DIR / 'processed' / 'nk_comments_framing.csv',
            'type': 'comments'
        },

        # Iran (Control group 1)
        {
            'name': 'Iran Posts',
            'input': DATA_DIR / 'processed' / 'iran_posts_roberta.csv',
            'output': DATA_DIR / 'processed' / 'iran_posts_framing.csv',
            'type': 'posts'
        },
        {
            'name': 'Iran Comments',
            'input': DATA_DIR / 'processed' / 'iran_comments_roberta.csv',
            'output': DATA_DIR / 'processed' / 'iran_comments_framing.csv',
            'type': 'comments'
        },

        # Russia (Control group 2)
        {
            'name': 'Russia Posts',
            'input': DATA_DIR / 'processed' / 'russia_posts_roberta.csv',
            'output': DATA_DIR / 'processed' / 'russia_posts_framing.csv',
            'type': 'posts'
        },
        {
            'name': 'Russia Comments',
            'input': DATA_DIR / 'processed' / 'russia_comments_roberta.csv',
            'output': DATA_DIR / 'processed' / 'russia_comments_framing.csv',
            'type': 'comments'
        },

        # China (Control group 3)
        {
            'name': 'China Posts',
            'input': DATA_DIR / 'processed' / 'china_posts_roberta.csv',
            'output': DATA_DIR / 'processed' / 'china_posts_framing.csv',
            'type': 'posts'
        },
        {
            'name': 'China Comments',
            'input': DATA_DIR / 'processed' / 'china_comments_roberta.csv',
            'output': DATA_DIR / 'processed' / 'china_comments_framing.csv',
            'type': 'comments'
        }
    ]

    # Process each dataset
    print("\n" + "=" * 80)
    print("Processing Datasets")
    print("=" * 80)

    total_items = 0

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Processing: {dataset['name']}")
        print(f"{'='*80}")

        if not dataset['input'].exists():
            print(f"  WARNING: Input file not found: {dataset['input']}")
            continue

        try:
            if dataset['type'] == 'posts':
                process_posts(
                    input_path=dataset['input'],
                    output_path=dataset['output'],
                    analyzer=analyzer,
                    start_date=START_DATE,
                    end_date=END_DATE
                )
            else:  # comments
                process_comments(
                    input_path=dataset['input'],
                    output_path=dataset['output'],
                    analyzer=analyzer,
                    start_date=START_DATE,
                    end_date=END_DATE
                )

            # Count processed items
            if dataset['output'].exists():
                df_result = pd.read_csv(dataset['output'])
                items = len(df_result)
                total_items += items
                print(f"  Total processed: {items:,} items")

        except Exception as e:
            print(f"  ERROR processing {dataset['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 80)
    print("Framing Classification Complete!")
    print("=" * 80)
    print(f"Total items classified: {total_items:,}")
    print(f"Estimated API cost: ${total_items * 0.00003:.2f}")
    print(f"\nOutput files saved to: {DATA_DIR / 'processed'}")
    print("\nNext steps:")
    print("  1. Run scripts/create_framing_scale.py")
    print("  2. Run scripts/create_framing_monthly_aggregation.py")
    print("  3. Run scripts/run_did_analysis_framing.py")


if __name__ == "__main__":
    main()
