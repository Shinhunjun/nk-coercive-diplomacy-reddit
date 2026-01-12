
import pandas as pd
import datetime

file_path = 'data/processed/nk_comments_recursive.csv'

try:
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    print("Columns:", df.columns.tolist())
    
    # Ensure created_utc is numeric
    # If it's not present, look for alternatives
    if 'created_utc' not in df.columns:
        print("Error: created_utc not found")
    else:
        df['date'] = pd.to_datetime(df['created_utc'], unit='s')
        
        # Define Periods
        p1_start = pd.Timestamp('2017-01-01')
        p1_end = pd.Timestamp('2018-06-11')
        p2_end = pd.Timestamp('2019-02-28')
        p3_end = pd.Timestamp('2019-12-31')
        
        def assign_period(date):
            if p1_start <= date <= p1_end:
                return 'P1 (Pre-Summit)'
            elif date <= p2_end:
                return 'P2 (Summit)'
            elif date <= p3_end:
                return 'P3 (Post-Hanoi)'
            else:
                return 'Out of Range'

        df['Period'] = df['date'].apply(assign_period)
        
        # Filter out 'Out of Range'
        df = df[df['Period'] != 'Out of Range']
        
        # Group by Period
        stats = df.groupby('Period').agg(
            Posts=('link_id', 'nunique'),
            Comments=('id', 'count'),
            Avg_Score=('score', 'mean')
        )
        
        print("\nCalculated Stats:")
        print(stats)
        
        total_posts = df['link_id'].nunique()
        total_comments = len(df)
        print(f"\nTotal Unique Posts: {total_posts}")
        print(f"Total Comments: {total_comments}")

except Exception as e:
    print(f"An error occurred: {e}")
