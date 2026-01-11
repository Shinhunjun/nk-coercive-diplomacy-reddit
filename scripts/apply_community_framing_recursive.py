
"""
Apply Framing Analysis to GraphRAG Communities (Recursive Version)
FULL MODE: P1/P3, All samples.
"""
import pandas as pd
import json
import os
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GRAPHRAG_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/graphrag")
OUTPUT_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/data/results")
FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

def classify_community(title: str, summary: str) -> dict:
    text = f"Title: {title}\nSummary: {summary[:1000]}"
    prompt = f"""Classify Community Report into ONE frame: THREAT, DIPLOMACY, ECONOMIC, HUMANITARIAN, NEUTRAL.
Rules:
1. No Action = NEUTRAL
2. Verbal/Talks = DIPLOMACY
3. Harm to individuals = HUMANITARIAN
4. Mixed = NEUTRAL

Report:
{text}

Response JSON: {{"frame": "CATEGORY", "confidence": 0.0-1.0, "reason": "Explanation"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        if result.get('frame') not in FRAME_CATEGORIES:
            result['frame'] = 'NEUTRAL'
        return result
    except Exception:
        return {"frame": "NEUTRAL", "confidence": 0, "reason": "Error"}

def process_period(period: str):
    print(f"Processing {period}...")
    parquet_path = GRAPHRAG_DIR / period / "output" / "community_reports.parquet"
    output_path = OUTPUT_DIR / f"community_framing_recursive_{period}.csv"
    
    if not parquet_path.exists():
        print(f"  [WARNING] {parquet_path} not found.")
        return

    # FULL DATASET
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} communities")
    
    processed_ids = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            if 'community_id' in existing.columns:
                processed_ids = set(existing['community_id'].astype(str))
            print(f"  Found {len(existing)} existing rows.")
        except:
             pass
    else:
        pd.DataFrame(columns=['community_id', 'level', 'title', 'frame', 'confidence', 'reason', 'period']).to_csv(output_path, index=False)
    
    for idx in tqdm(range(len(df)), desc=f"  {period}"):
        row = df.iloc[idx]
        comm_id = str(row.get('id', idx))
        if comm_id in processed_ids:
            continue

        title = str(row.get('title', ''))
        summary = str(row.get('summary', ''))
        result = classify_community(title, summary)
        
        mini_df = pd.DataFrame([{
            'community_id': comm_id,
            'level': row.get('level', -1),
            'title': title,
            'frame': result.get('frame', 'NEUTRAL'),
            'confidence': result.get('confidence', 0.5),
            'reason': result.get('reason', ''),
            'period': period
        }])
        mini_df.to_csv(output_path, mode='a', header=False, index=False)

    print(f"✓ Saved: {output_path}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # FULL: P1 and P3
    for period in ["P1_Recursive", "P3_Recursive"]: 
        process_period(period)

if __name__ == "__main__":
    main()
