"""
Create NK Monthly Data from Full Dataset
Uses complete NK posts data with pre-computed BERT sentiment scores
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_nk_monthly_from_full():
    """
    Create NK monthly sentiment data from full dataset.
    """
    print("=" * 60)
    print("Creating NK Monthly Data from Full Dataset")
    print("=" * 60)

    # Load full NK posts data with BERT sentiment
    print("\nLoading full NK posts data...")
    nk_full_path = '/Users/hunjunsin/Desktop/Jun/reddit_US_NK/data/processed/posts_final_bert_sentiment.csv'
    df = pd.read_csv(nk_full_path)

    print(f"Total NK posts in dataset: {len(df)}")

    # Convert created_utc to datetime
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['datetime'].dt.to_period('M').astype(str)

    # Filter to analysis period (2017-01 to 2019-06)
    df_filtered = df[(df['month'] >= '2017-01') & (df['month'] <= '2019-06')].copy()

    print(f"Posts in analysis period (2017-01 to 2019-06): {len(df_filtered)}")

    # Use bert_compound as sentiment score (-1 to +1)
    df_filtered['sentiment_score'] = df_filtered['bert_compound']

    # Aggregate to monthly level
    print("\nAggregating to monthly level...")
    monthly = df_filtered.groupby('month').agg({
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
    print(f"  Total posts used: {monthly_complete['post_count'].sum():.0f}")

    # Show sample
    print(f"\nSample monthly data:")
    print(monthly_complete[monthly_complete['post_count'].notna()][['month', 'sentiment_mean', 'post_count']].head(15))

    # Show intervention period specifically
    print(f"\nIntervention period (2018-02 to 2018-04):")
    intervention_data = monthly_complete[monthly_complete['month'].isin(['2018-02', '2018-03', '2018-04'])]
    print(intervention_data[['month', 'sentiment_mean', 'post_count']])

    return monthly_complete


if __name__ == '__main__':
    create_nk_monthly_from_full()
