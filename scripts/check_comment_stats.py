
import pandas as pd
import glob

def check_comment_stats():
    print(f"{'File Path':<50} | {'Count':<10} | {'Min Date':<15} | {'Max Date':<15}")
    print("-" * 100)
    
    files = [
        "data/nk/nk_comments_full.csv",
        "data/control/iran_comments_full.csv",
        "data/control/china_comments_full.csv",
        "data/control/russia_comments_full.csv"
    ]
    
    for path in files:
        try:
            # Read only created_utc column to save memory/speed
            df = pd.read_csv(path, usecols=['created_utc'], low_memory=False)
            
            df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
            df = df.dropna(subset=['created_utc'])
            
            if len(df) == 0:
                print(f"{path:<50} | {'0':<10} | {'-':<15} | {'-':<15}")
                continue

            dates = pd.to_datetime(df['created_utc'], unit='s')
            print(f"{path:<50} | {len(df):<10} | {str(dates.min().date()):<15} | {str(dates.max().date()):<15}")
            
        except ValueError: 
             # Fallback if usecols fails (column name might differ)
            try: 
                df = pd.read_csv(path, nrows=5)
                print(f"{path:<50} | Error: Columns found {list(df.columns)}")
            except:
                print(f"{path:<50} | Error: Read failed")
        except Exception as e:
            print(f"{path:<50} | Error: {str(e)[:30]}")

if __name__ == "__main__":
    check_comment_stats()
