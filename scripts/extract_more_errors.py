
import pandas as pd
import sys

def main():
    try:
        df = pd.read_csv('data/annotations/llm_improved_v2_classification.csv')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("--- TYPE 1: DIPLOMACY -> NEUTRAL Candidates ---")
    type1 = df[(df['human_label'].str.upper() == 'DIPLOMACY') & (df['llm_label'].str.upper() == 'NEUTRAL')]
    # Shuffle or just take head
    for idx, row in type1.head(20).iterrows():
        print(f"Index: {idx}")
        print(f"Title: {row['title']}")
        print(f"Reason: {row['reason']}")
        print("-" * 50)

    print("\n\n--- TYPE 2: KEYWORD SENSITIVITY (NOT HUMANITARIAN -> HUMANITARIAN) ---")
    type2 = df[(df['human_label'].str.upper() != 'HUMANITARIAN') & (df['llm_label'].str.upper() == 'HUMANITARIAN')]
    for idx, row in type2.head(20).iterrows():
        print(f"Index: {idx}")
        print(f"Title: {row['title']}")
        print(f"Human: {row['human_label']}")
        print(f"Reason: {row['reason']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
