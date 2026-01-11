"""
Apply V2 LLM Framing Classification to GraphRAG Community Reports

Uses the EXACT V2 prompt for methodological consistency.
Processes community_reports from graphrag_comments/ (comment-based indexing).
"""

import pandas as pd
import json
import os
import time
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path

# Configuration
GRAPHRAG_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/graphrag_comments")
OUTPUT_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/data/results")
PERIODS = ["period1", "period2", "period3"]
FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"


def classify_community_v2(title: str, summary: str) -> dict:
    """Classify community using V2 improved prompt."""
    text = f"Title: {title[:200]}\nSummary: {summary[:500]}"
    
    # V2 IMPROVED PROMPT
    prompt = f"""You are an international relations researcher. Classify the following community topic into ONE of 5 framing categories.

## Critical Classification Rules (Apply First!)

### Rule 1: No Action = NEUTRAL
If the community is about general context or factual information without explicit government action, classify as NEUTRAL.

### Rule 2: Verbal vs Physical Actions
If the community focuses on verbal criticism or diplomatic pressure, classify as DIPLOMACY, not THREAT.

### Rule 3: Individual Harm = HUMANITARIAN
If the community involves harm to specific individuals (protesters, defectors, civilians), classify as HUMANITARIAN.

### Rule 4: Conflicting Frames = NEUTRAL
When multiple frames are equally present and competing, classify as NEUTRAL.

## Categories:
- THREAT: Military actions, nuclear weapons, missile tests, military exercises
- DIPLOMACY: Summits, negotiations, diplomatic talks, verbal warnings
- ECONOMIC: Sanctions, trade measures, economic cooperation
- HUMANITARIAN: Human rights, refugees, individual harm
- NEUTRAL: General facts, context, mixed topics

## Community:
{text}

## Response (JSON only):
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "Brief rationale"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Apply the Critical Classification Rules FIRST."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        if result.get('frame') not in FRAME_CATEGORIES:
            result['frame'] = 'NEUTRAL'
            result['confidence'] = 0.5
        
        return result
        
    except Exception as e:
        return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}


def process_period(period: str, sample_n: int = None) -> pd.DataFrame:
    """Process community reports for a single period."""
    parquet_path = GRAPHRAG_DIR / period / "output" / "community_reports.parquet"
    
    if not parquet_path.exists():
        print(f"  [WARNING] {parquet_path} not found")
        return None
    
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} communities from {period}")
    
    # Sample if requested
    if sample_n and sample_n < len(df):
        df = df.sample(sample_n, random_state=42)
        print(f"  Sampled {sample_n} communities for testing")
    
    # Classify each community
    results = []
    for idx in tqdm(range(len(df)), desc=f"  Classifying {period}"):
        row = df.iloc[idx]
        title = str(row.get('title', ''))
        summary = str(row.get('summary', ''))
        
        result = classify_community_v2(title, summary)
        results.append({
            'community_id': row.get('community', idx),
            'title': title[:100],
            'frame': result.get('frame', 'NEUTRAL'),
            'confidence': result.get('confidence', 0.5),
            'reason': result.get('reason', '')
        })
        time.sleep(0.1)
    
    result_df = pd.DataFrame(results)
    result_df['period'] = period
    return result_df


def main(sample_n: int = None):
    print("="*60)
    print("GraphRAG Community Framing Analysis (V2 Prompt)")
    print("="*60)
    
    all_results = []
    
    for period in PERIODS:
        print(f"\nProcessing {period}...")
        result_df = process_period(period, sample_n)
        if result_df is not None:
            all_results.append(result_df)
    
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_sample{sample_n}" if sample_n else ""
    output_path = OUTPUT_DIR / f"community_framing_v2_comments{suffix}.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    
    # Print distribution
    print("\n" + "="*60)
    print("FRAME DISTRIBUTION BY PERIOD")
    print("="*60)
    
    period_labels = {"period1": "P1", "period2": "P2", "period3": "P3"}
    
    for period in PERIODS:
        period_df = combined_df[combined_df['period'] == period]
        print(f"\n{period_labels[period]} (n={len(period_df)})")
        for frame in FRAME_CATEGORIES:
            count = (period_df['frame'] == frame).sum()
            pct = count / len(period_df) * 100 if len(period_df) > 0 else 0
            print(f"  {frame}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=int, default=None, help='Sample N communities per period for testing')
    args = parser.parse_args()
    
    main(sample_n=args.sample)
