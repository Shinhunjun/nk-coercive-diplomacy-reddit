import pandas as pd
import os
from datetime import datetime

# Define P1, P2, P3
P1_START = datetime(2017, 1, 1).timestamp()
P1_END = datetime(2018, 6, 11).timestamp()
P2_START = datetime(2018, 6, 13).timestamp()
P2_END = datetime(2019, 2, 27).timestamp()
P3_START = datetime(2019, 3, 1).timestamp()
P3_END = datetime(2019, 12, 31).timestamp()

def get_period(timestamp):
    try:
        ts = float(timestamp)
        if P1_START <= ts <= P1_END:
            return 'P1'
        elif P2_START <= ts <= P2_END:
            return 'P2'
        elif P3_START <= ts <= P3_END:
            return 'P3'
        else:
            return 'Out'
    except:
        return 'Error'

files = {
    'North Korea': 'data/processed/nk_comments_roberta.csv',
    'China': 'data/control/china_comments_roberta.csv',
    'Iran': 'data/control/iran_comments_roberta.csv',
    'Russia': 'data/control/russia_comments_roberta.csv'
}

stats = []

for country, path in files.items():
    if os.path.exists(path):
        try:
            # Read only created_utc column to be fast
            df = pd.read_csv(path, usecols=['created_utc'])
            
            df['period'] = df['created_utc'].apply(get_period)
            
            counts = df['period'].value_counts()
            
            stats.append({
                'Country': country,
                'Total': len(df),
                'P1': counts.get('P1', 0),
                'P2': counts.get('P2', 0),
                'P3': counts.get('P3', 0),
                'Out': counts.get('Out', 0)
            })
            print(f"Processed {country}: {len(df)} comments")
        except Exception as e:
            print(f"Error processing {country}: {e}")
    else:
        print(f"File not found: {path}")

print("\n=== Comment Distribution Report ===")
stats_df = pd.DataFrame(stats)
print(stats_df)
print("\nTotal Comments in Analysis Window (P1+P2+P3):", stats_df[['P1', 'P2', 'P3']].sum().sum())
