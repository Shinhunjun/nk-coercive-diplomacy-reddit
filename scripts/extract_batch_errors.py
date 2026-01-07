
import pandas as pd
import glob

def main():
    files = [
        'data/annotations/framing - batch_1.csv',
        'data/annotations/framing - batch_2.csv',
        'data/annotations/framing - batch_pilot.csv'
    ]
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            pass # Skip if not found
            
    if not dfs:
        print("No files found.")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    
    print(f"Total rows: {len(full_df)}")
    
    # Normalize labels
    full_df['final_frame'] = full_df['final_frame'].str.strip().str.upper()
    full_df['v2_annotation'] = full_df['v2_annotation'].str.strip().str.upper()

    print("\n--- TYPE 1: DIPLOMACY -> NEUTRAL Candidates ---")
    type1 = full_df[(full_df['final_frame'] == 'DIPLOMACY') & (full_df['v2_annotation'] == 'NEUTRAL')]
    for idx, row in type1.head(20).iterrows():
        print(f"Title: {row['title']}")
        print(f"Human: {row['final_frame']} -> LLM: {row['v2_annotation']}")
        print("-" * 50)

    print("\n\n--- TYPE 2: KEYWORD SENSITIVITY (NOT HUM -> HUM) ---")
    type2 = full_df[(full_df['final_frame'] != 'HUMANITARIAN') & (full_df['v2_annotation'] == 'HUMANITARIAN')]
    for idx, row in type2.head(20).iterrows():
        print(f"Title: {row['title']}")
        print(f"Human: {row['final_frame']} -> LLM: {row['v2_annotation']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
