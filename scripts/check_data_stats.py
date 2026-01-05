
import pandas as pd
import os
import glob
from datetime import datetime

def check_stats():
    print(f"{'File Path':<60} | {'Count':<8} | {'Min Date':<12} | {'Max Date':<12}")
    print("-" * 100)
    
    # helper to parse dates
    def get_date_range(path):
        try:
            df = pd.read_csv(path, low_memory=False)
            if 'created_utc' not in df.columns:
                return "No created_utc", "-", "-"
            
            df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
            df = df.dropna(subset=['created_utc'])
            
            if len(df) == 0:
                return 0, "-", "-"

            dates = pd.to_datetime(df['created_utc'], unit='s')
            return len(df), str(dates.min().date()), str(dates.max().date())
        except Exception as e:
            return "Error", str(e)[:20], ""

    # Scan all csvs in data/nk and data/control
    files = glob.glob("data/nk/*posts*.csv") + glob.glob("data/control/*posts*.csv")
    files.sort()

    for path in files:
        count, min_d, max_d = get_date_range(path)
        print(f"{path:<60} | {str(count):<8} | {min_d:<12} | {max_d:<12}")

if __name__ == "__main__":
    check_stats()
