"""
Create final unified datasets for analysis.
Uses existing sentiment and framing data as primary sources.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAME_SCALE = {"THREAT": -2, "ECONOMIC": -1, "NEUTRAL": 0, "HUMANITARIAN": 1, "DIPLOMACY": 2}


def assign_period(month):
    """Assign period based on month string."""
    if month <= '2018-05':
        return 'P1_PreSingapore'
    elif month <= '2019-02':
        return 'P2_SingaporeHanoi'
    else:
        return 'P3_PostHanoi'


def create_final_dataset_v2(name, sentiment_existing, sentiment_extended, framing_existing, framing_extended, output_path):
    """Create final unified dataset using existing processed data."""
    print(f"\n{'='*60}")
    print(f"CREATING FINAL DATASET: {name.upper()}")
    print(f"{'='*60}")
    
    # Load sentiment data
    df_sent_exist = pd.read_csv(sentiment_existing)
    df_sent_ext = pd.read_csv(sentiment_extended)
    print(f"Sentiment existing: {len(df_sent_exist)}, extended: {len(df_sent_ext)}")
    
    # Load framing data
    df_frame_exist = pd.read_csv(framing_existing)
    df_frame_ext = pd.read_csv(framing_extended)
    print(f"Framing existing: {len(df_frame_exist)}, extended: {len(df_frame_ext)}")
    
    # Use framing data as base (has full coverage 2017-2019.06)
    df_exist = df_frame_exist.copy()
    df_ext = df_frame_ext.copy()
    
    # Merge sentiment from sentiment files (by id)
    sent_lookup = pd.concat([df_sent_exist, df_sent_ext])[['id', 'sentiment_score', 'sentiment_label']].drop_duplicates('id')
    sent_lookup = sent_lookup.set_index('id').to_dict('index')
    
    for df in [df_exist, df_ext]:
        df['sentiment_score'] = df['id'].map(lambda x: sent_lookup.get(x, {}).get('sentiment_score', np.nan))
        df['sentiment_label'] = df['id'].map(lambda x: sent_lookup.get(x, {}).get('sentiment_label', None))
    
    # Combine
    df = pd.concat([df_exist, df_ext], ignore_index=True)
    df = df.drop_duplicates(subset=['id'], keep='first')
    print(f"Combined: {len(df)} posts")
    
    # Add datetime and period
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['datetime'].dt.strftime('%Y-%m')
    df['period'] = df['month'].apply(assign_period)
    df['topic'] = name.lower()
    
    # Filter to analysis period
    df = df[(df['month'] >= '2017-01') & (df['month'] <= '2019-12')].copy()
    print(f"In analysis period: {len(df)} posts")
    
    # Ensure frame_score exists
    if 'frame_score' not in df.columns:
        df['frame_score'] = df['frame'].map(FRAME_SCALE)
    
    # Period stats
    print(f"\nPeriod distribution:")
    for period in ['P1_PreSingapore', 'P2_SingaporeHanoi', 'P3_PostHanoi']:
        subset = df[df['period'] == period]
        sent_cov = subset['sentiment_score'].notna().sum() / len(subset) * 100 if len(subset) > 0 else 0
        frame_cov = subset['frame'].notna().sum() / len(subset) * 100 if len(subset) > 0 else 0
        print(f"  {period}: {len(subset)} posts, sentiment={sent_cov:.0f}%, framing={frame_cov:.0f}%")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return df


def main():
    print("=" * 70)
    print("CREATING FINAL UNIFIED DATASETS V2")
    print("Using existing sentiment and framing data")
    print("=" * 70)
    
    datasets = [
        {
            'name': 'NK',
            'sent_exist': 'data/sentiment/nk_posts_sentiment.csv',
            'sent_ext': 'data/sentiment/nk_posts_hanoi_extended_sentiment.csv',
            'frame_exist': 'data/framing/nk_posts_framed.csv',
            'frame_ext': 'data/framing/nk_posts_hanoi_extended_framed.csv',
            'output': 'data/final/nk_final.csv'
        },
        {
            'name': 'China',
            'sent_exist': 'data/sentiment/china_posts_sentiment.csv',
            'sent_ext': 'data/sentiment/china_posts_hanoi_extended_sentiment.csv',
            'frame_exist': 'data/framing/china_posts_framed.csv',
            'frame_ext': 'data/framing/china_posts_hanoi_extended_framed.csv',
            'output': 'data/final/china_final.csv'
        },
        {
            'name': 'Iran',
            'sent_exist': 'data/sentiment/iran_posts_sentiment.csv',
            'sent_ext': 'data/sentiment/iran_posts_hanoi_extended_sentiment.csv',
            'frame_exist': 'data/framing/iran_posts_framed.csv',
            'frame_ext': 'data/framing/iran_posts_hanoi_extended_framed.csv',
            'output': 'data/final/iran_final.csv'
        },
        {
            'name': 'Russia',
            'sent_exist': 'data/sentiment/russia_posts_sentiment.csv',
            'sent_ext': 'data/sentiment/russia_posts_hanoi_extended_sentiment.csv',
            'frame_exist': 'data/framing/russia_posts_framed.csv',
            'frame_ext': 'data/framing/russia_posts_hanoi_extended_framed.csv',
            'output': 'data/final/russia_final.csv'
        }
    ]
    
    results = {}
    for ds in datasets:
        df = create_final_dataset_v2(
            ds['name'], ds['sent_exist'], ds['sent_ext'],
            ds['frame_exist'], ds['frame_ext'], ds['output']
        )
        results[ds['name']] = len(df)
    
    print("\n" + "=" * 70)
    print("FINAL DATASETS CREATED")
    print("=" * 70)
    for name, count in results.items():
        print(f"  {name}: {count:,} posts")


if __name__ == '__main__':
    main()
