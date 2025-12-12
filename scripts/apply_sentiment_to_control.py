"""
Apply BERT Sentiment Analysis to Control Groups
Processes Iran, Russia, and China posts collected in Phase 1
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentiment_analysis import SentimentAnalyzer
from src.config import CONTROL_GROUPS


def load_and_prepare_data(topic: str, data_dir: str = 'data/control') -> pd.DataFrame:
    """Load control group data and prepare text for sentiment analysis."""
    filepath = os.path.join(data_dir, f'{topic}_posts.csv')
    df = pd.read_csv(filepath)

    print(f"\nLoaded {topic.upper()}: {len(df)} posts")

    # Combine title and selftext (same as NK data processing)
    df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')

    # Convert created_utc to datetime if needed
    if 'created_utc' in df.columns:
        # Handle both timestamp and date string formats
        try:
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')
        except:
            df['date'] = pd.to_datetime(df['created_utc'])

    return df


def aggregate_to_monthly(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """
    Aggregate sentiment scores to monthly level.

    Creates 26 monthly observations (2017-01 to 2019-06) matching NK data structure.
    """
    # Extract year-month
    df['month'] = df['date'].dt.to_period('M').astype(str)

    # Group by month and calculate statistics
    monthly = df.groupby('month').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()

    # Flatten column names
    monthly.columns = ['month', 'sentiment_mean', 'sentiment_std', 'post_count']

    # Fill missing months with NaN (if any months have no posts)
    all_months = pd.period_range('2017-01', '2019-06', freq='M').astype(str)
    monthly_complete = pd.DataFrame({'month': all_months})
    monthly_complete = monthly_complete.merge(monthly, on='month', how='left')

    # Add topic identifier
    monthly_complete['topic'] = topic

    print(f"\n{topic.upper()} Monthly Summary:")
    print(f"  Total months: {len(monthly_complete)}")
    print(f"  Months with data: {monthly_complete['post_count'].notna().sum()}")
    print(f"  Mean sentiment: {monthly_complete['sentiment_mean'].mean():.4f}")
    print(f"  Sentiment range: [{monthly_complete['sentiment_mean'].min():.4f}, {monthly_complete['sentiment_mean'].max():.4f}]")

    return monthly_complete


def save_results(df_posts: pd.DataFrame, df_monthly: pd.DataFrame, topic: str):
    """Save sentiment analysis results."""
    # Save post-level data with sentiment scores
    posts_dir = 'data/control'
    os.makedirs(posts_dir, exist_ok=True)
    posts_path = os.path.join(posts_dir, f'{topic}_posts_with_sentiment.csv')
    df_posts.to_csv(posts_path, index=False)
    print(f"✓ Saved post-level data: {posts_path}")

    # Save monthly aggregated data
    monthly_dir = 'data/processed'
    os.makedirs(monthly_dir, exist_ok=True)
    monthly_path = os.path.join(monthly_dir, f'{topic}_monthly.csv')
    df_monthly.to_csv(monthly_path, index=False)
    print(f"✓ Saved monthly data: {monthly_path}")

    return posts_path, monthly_path


def main():
    """Main execution function for Phase 2."""
    print("=" * 60)
    print("PHASE 2: BERT SENTIMENT ANALYSIS FOR CONTROL GROUPS")
    print("=" * 60)

    # Initialize sentiment analyzer (same model as NK data)
    print("\nInitializing BERT sentiment analyzer...")
    analyzer = SentimentAnalyzer()

    topics = ['iran', 'russia', 'china']
    results_summary = {}

    for topic in topics:
        print(f"\n{'=' * 60}")
        print(f"Processing {topic.upper()}")
        print(f"{'=' * 60}")

        try:
            # Load data
            df = load_and_prepare_data(topic)

            # Apply sentiment analysis
            print(f"\nApplying BERT sentiment analysis to {len(df)} posts...")
            df = analyzer.analyze_dataframe(df, text_column='text')

            # Show sample results
            print("\nSample sentiment scores:")
            print(df[['title', 'sentiment_score']].head(3))

            # Aggregate to monthly level
            monthly_df = aggregate_to_monthly(df, topic)

            # Save results
            posts_path, monthly_path = save_results(df, monthly_df, topic)

            results_summary[topic] = {
                'total_posts': len(df),
                'months_with_data': monthly_df['post_count'].notna().sum(),
                'mean_sentiment': float(monthly_df['sentiment_mean'].mean()),
                'posts_file': posts_path,
                'monthly_file': monthly_path
            }

        except Exception as e:
            print(f"\n✗ Error processing {topic}: {e}")
            import traceback
            traceback.print_exc()
            results_summary[topic] = {'error': str(e)}

    # Final summary
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETION SUMMARY")
    print("=" * 60)

    for topic, result in results_summary.items():
        if 'error' in result:
            print(f"\n{topic.upper()}: ✗ {result['error']}")
        else:
            print(f"\n{topic.upper()}:")
            print(f"  Posts analyzed: {result['total_posts']}")
            print(f"  Months with data: {result['months_with_data']}/26")
            print(f"  Mean sentiment: {result['mean_sentiment']:.4f}")
            print(f"  Monthly file: {result['monthly_file']}")

    print("\n✓ Phase 2 complete!")
    print("\nNext: Phase 3 - Parallel Trends Testing")


if __name__ == '__main__':
    main()
