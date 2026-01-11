
"""
Apply Framing Analysis to GraphRAG Edges (Recursive Version)
Uses the "Improved Prompt V2" (Identical to Comment Analysis) for consistency.
FULL MODE: P1/P3, All samples.
"""
import pandas as pd
import json
import os
import time
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

def classify_edge(description: str) -> dict:
    prompt = f"""You are an international relations researcher. Classify the relationship description into ONE of 5 framing categories.
## Rules:
1. No Action = NEUTRAL
2. Verbal vs Physical = DIPLOMACY if only verbal
3. Individual Harm = HUMANITARIAN
4. Conflicting Frames = NEUTRAL

## Frames: THREAT, DIPLOMACY, ECONOMIC, HUMANITARIAN, NEUTRAL

## Description
{description}

## Response Format (JSON only)
{{"frame": "CATEGORY", "confidence": 0.0-1.0, "reason": "Explanation"}}"""

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
    except Exception as e:
        return {"frame": "NEUTRAL", "confidence": 0, "reason": str(e)}

def process_period(period: str):
    print(f"Processing {period}...")
    parquet_path = GRAPHRAG_DIR / period / "output" / "relationships.parquet"
    output_path = OUTPUT_DIR / f"edge_framing_{period}.csv"
    
    if not parquet_path.exists():
        print(f"  [WARNING] {parquet_path} not found. Skipping.")
        return

    # FULL DATASET
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} edges")
    
    # Check existing progress (Resume capability)
    processed = set()
    if output_path.exists():
        try:
            existing = pd.read_csv(output_path)
            # Create simple key map
            # Note: For edges, we just assume append if not careful, but let's try to be smart.
            # Actually, for speed, since we just cleared files, we can skip complex dedup or just rely on index alignment if order preserved.
            # But let's build a source-target set.
            if 'source' in existing.columns:
                 processed = set(zip(existing['source'], existing['target']))
            print(f"  Found {len(existing)} existing rows.")
        except:
            pass
    else:
        pd.DataFrame(columns=['source', 'target', 'description', 'frame', 'confidence', 'reason', 'period']).to_csv(output_path, index=False)
    
    for idx in tqdm(range(len(df)), desc=f"  {period}"):
        row = df.iloc[idx]
        if (row['source'], row['target']) in processed:
            continue
            
        desc = str(row.get('description', ''))
        result = classify_edge(desc)
        
        mini_df = pd.DataFrame([{
            'source': row['source'],
            'target': row['target'],
            'description': desc[:500],
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
