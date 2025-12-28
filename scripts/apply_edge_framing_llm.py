"""
Apply LLM Framing Classification to GraphRAG Edge Descriptions

This script classifies edge descriptions from GraphRAG relationships.parquet
using the same framing prompt as posts (GPT-4o-mini).

Replaces the keyword-based approach for methodological consistency with RQ1.
"""

import pandas as pd
import json
import os
import time
from tqdm import tqdm
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configuration
GRAPHRAG_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/graphrag")
OUTPUT_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/data/results")
PERIODS = ["period1", "period2", "period3"]
FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

def classify_edge_description(description: str) -> dict:
    """
    Classify an edge description into a framing category.
    Uses the SAME prompt structure as post classification for consistency.
    """
    # Truncate long descriptions
    text = description[:1000] if description else ""
    
    prompt = f"""이 텍스트를 다음 5가지 프레임 중 하나로 분류하세요:
- THREAT: 군사적 위협, 핵무기, 미사일, 전쟁 위험 강조
- DIPLOMACY: 협상, 대화, 평화, 협력 가능성 강조
- NEUTRAL: 중립적 정보 전달
- ECONOMIC: 경제 제재, 무역 측면 강조
- HUMANITARIAN: 인권, 난민, 북한 주민 문제 강조

텍스트:
{text}

JSON 형식으로 응답:
{{"frame": "카테고리", "confidence": 0.0-1.0, "reason": "간단한 설명"}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a political science researcher analyzing media framing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        # Validate frame
        if result.get('frame') not in FRAME_CATEGORIES:
            result['frame'] = 'NEUTRAL'
            result['confidence'] = 0.5
        
        return result
        
    except Exception as e:
        return {
            "frame": "NEUTRAL",
            "confidence": 0.5,
            "reason": f"Error: {str(e)}"
        }


def process_period(period: str) -> pd.DataFrame:
    """Process all edge descriptions for a single period."""
    parquet_path = GRAPHRAG_DIR / period / "output" / "relationships.parquet"
    
    if not parquet_path.exists():
        print(f"  [WARNING] {parquet_path} not found, skipping...")
        return None
    
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} edges from {period}")
    
    # Classify each edge description
    results = []
    for idx in tqdm(range(len(df)), desc=f"  Classifying {period}"):
        desc = str(df.iloc[idx].get('description', ''))
        result = classify_edge_description(desc)
        results.append({
            'source': df.iloc[idx]['source'],
            'target': df.iloc[idx]['target'],
            'description': desc[:200] + "..." if len(desc) > 200 else desc,
            'frame': result.get('frame', 'NEUTRAL'),
            'confidence': result.get('confidence', 0.5),
            'reason': result.get('reason', '')
        })
        time.sleep(0.1)  # Rate limiting
    
    result_df = pd.DataFrame(results)
    result_df['period'] = period
    return result_df


def main():
    print("="*60)
    print("GraphRAG Edge Description Framing Analysis")
    print("="*60)
    
    all_results = []
    
    for period in PERIODS:
        print(f"\nProcessing {period}...")
        result_df = process_period(period)
        if result_df is not None:
            all_results.append(result_df)
    
    # Combine all results
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "edge_framing_llm_results.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved results to {output_path}")
    
    # Print distribution summary
    print("\n" + "="*60)
    print("FRAME DISTRIBUTION BY PERIOD")
    print("="*60)
    
    period_labels = {"period1": "P1 (Pre-Singapore)", "period2": "P2 (Singapore-Hanoi)", "period3": "P3 (Post-Hanoi)"}
    
    for period in PERIODS:
        period_df = combined_df[combined_df['period'] == period]
        print(f"\n{period_labels[period]} (n={len(period_df)})")
        dist = period_df['frame'].value_counts(normalize=True) * 100
        for frame in FRAME_CATEGORIES:
            pct = dist.get(frame, 0)
            print(f"  {frame}: {pct:.1f}%")


if __name__ == "__main__":
    main()
