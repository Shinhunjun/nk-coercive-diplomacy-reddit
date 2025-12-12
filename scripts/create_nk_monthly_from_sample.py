"""
Create NK Monthly Data from Sample for DID Analysis
Uses sample data to demonstrate the parallel trends testing workflow
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentiment_analysis import SentimentAnalyzer


def create_nk_monthly_from_sample():
    """
    Create NK monthly sentiment data from sample files.

    Note: This uses sample data for demonstration.
    For full analysis, download complete dataset from Google Drive.
    """
    print("=" * 60)
    print("Creating NK Monthly Data from Sample")
    print("=" * 60)

    # Load sample data
    print("\nLoading sample data...")
    p1 = pd.read_csv('data/sample/posts_period1_sample.csv')
    p2 = pd.read_csv('data/sample/posts_period2_sample.csv')

    print(f"Period 1 (Tension): {len(p1)} posts")
    print(f"Period 2 (Diplomacy): {len(p2)} posts")

    # Combine periods
    df = pd.concat([p1, p2], ignore_index=True)

    # Prepare text
    df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')

    # Check if sentiment scores already exist
    if 'sentiment_score' not in df.columns:
        print("\nApplying BERT sentiment analysis...")
        analyzer = SentimentAnalyzer()
        df = analyzer.analyze_dataframe(df, text_column='text')
    else:
        print("\nUsing existing sentiment scores...")

    # Convert timestamps to datetime
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['date'].dt.to_period('M').astype(str)

    # Aggregate to monthly level
    print("\nAggregating to monthly level...")
    monthly = df.groupby('month').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()

    monthly.columns = ['month', 'sentiment_mean', 'sentiment_std', 'post_count']

    # Fill missing months in the full time range (2017-01 to 2019-06)
    all_months = pd.period_range('2017-01', '2019-06', freq='M').astype(str)
    monthly_complete = pd.DataFrame({'month': all_months})
    monthly_complete = monthly_complete.merge(monthly, on='month', how='left')

    # Add topic identifier
    monthly_complete['topic'] = 'northkorea'

    # Save to processed directory
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/nk_monthly.csv'
    monthly_complete.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")
    print(f"\nMonthly Summary:")
    print(f"  Total months: {len(monthly_complete)}")
    print(f"  Months with data: {monthly_complete['post_count'].notna().sum()}")
    print(f"  Mean sentiment: {monthly_complete['sentiment_mean'].mean():.4f}")

    # Show sample
    print(f"\nSample monthly data:")
    print(monthly_complete[['month', 'sentiment_mean', 'post_count']].head(10))

    return monthly_complete


if __name__ == '__main__':
    create_nk_monthly_from_sample()
