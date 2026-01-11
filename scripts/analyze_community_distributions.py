
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/results")
PERIODS = {
    "P1 (Pre-Summit 2017)": "community_framing_recursive_P1_Recursive.csv",
    "P2 (Summit 2018)": "community_framing_recursive_P2_Recursive.csv",
    "P3 (Ratchet 2019-20)": "community_framing_recursive_P3_Recursive.csv"
}

FRAMES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

print(f"{'Period':<25} | {'Frame':<15} | {'Count':<5} | {'Ratio':<6}")
print("-" * 60)

for p_name, filename in PERIODS.items():
    file_path = DATA_DIR / filename
    if not file_path.exists():
        print(f"{p_name:<25} | {'FILE NOT FOUND'}")
        continue
        
    df = pd.read_csv(file_path)
    total = len(df)
    counts = df['frame'].value_counts()
    
    print(f"[{p_name}] (Total: {total})")
    for frame in FRAMES:
        count = counts.get(frame, 0)
        ratio = (count / total) * 100
        print(f"{'':<25} | {frame:<15} | {count:<5} | {ratio:.1f}%")
    print("-" * 60)
