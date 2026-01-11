
import pandas as pd
import numpy as np
import datetime

# Define expected periods
PERIODS = {
    'P1': {'start': '2017-01-01', 'end': '2018-06-11'}, # Pre-Singapore
    'P2': {'start': '2018-06-12', 'end': '2019-02-27'}, # Singapore to Hanoi
    'P3': {'start': '2019-02-28', 'end': '2019-12-31'}  # Post-Hanoi
}

def check_periods_and_quality():
    print("Loading NK Recursive Data...")
    df = pd.read_csv('data/processed/nk_comments_recursive_roberta.csv', low_memory=False)
    
    # Coerce numeric
    df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
    df = df.dropna(subset=['created_utc'])
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    
    print("\n=== 1. PERIOD ASSIGNMENT VERIFICATION ===")
    for p_name, p_range in PERIODS.items():
        mask = (df['date'] >= p_range['start']) & (df['date'] <= p_range['end'])
        subset = df[mask]
        print(f"[{p_name}] ({p_range['start']} ~ {p_range['end']})")
        
        if len(subset) > 0:
            actual_min = subset['date'].min()
            actual_max = subset['date'].max()
            print(f"  - Count: {len(subset):,}")
            print(f"  - Actual Range in Data: {actual_min} to {actual_max}")
        else:
            print("  - Count: 0 (Warning!)")

    print("\n=== 2. SENTIMENT SCORE QUALITY CHECK ===")
    scores = df['roberta_compound']
    print(f"Score Range: {scores.min()} to {scores.max()}")
    print(f"Mean Score: {scores.mean():.4f}")
    print(f"Std Dev: {scores.std():.4f}")
    
    # Distribution
    print("\nDistribution:")
    print(scores.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
    
    print("\n=== 3. QUALITATIVE REASONABILITY TEST ===")
    print("(Checking if high scores actually look positive and low scores look negative)")
    
    # Sample Negatives
    print("\n[Extreme Negative] (Score < -0.9)")
    neg_samples = df[df['roberta_compound'] < -0.9].sample(3)
    for i, row in neg_samples.iterrows():
        print(f"  Score {row['roberta_compound']:.2f}: \"{str(row['body'])[:100].replace(chr(10), ' ')}...\"")
        
    # Sample Positives
    print("\n[Extreme Positive] (Score > 0.9)")
    pos_samples = df[df['roberta_compound'] > 0.9].sample(3)
    for i, row in pos_samples.iterrows():
        print(f"  Score {row['roberta_compound']:.2f}: \"{str(row['body'])[:100].replace(chr(10), ' ')}...\"")
        
    # Sample Neutrals
    print("\n[Neutral] (-0.1 < Score < 0.1)")
    neu_samples = df[(df['roberta_compound'] > -0.1) & (df['roberta_compound'] < 0.1)].sample(3)
    for i, row in neu_samples.iterrows():
        print(f"  Score {row['roberta_compound']:.2f}: \"{str(row['body'])[:100].replace(chr(10), ' ')}...\"")

if __name__ == "__main__":
    check_periods_and_quality()
