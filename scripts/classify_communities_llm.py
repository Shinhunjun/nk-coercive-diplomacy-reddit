"""
LLM-based Community Frame Classification

Classifies community titles and summaries using GPT-4o-mini,
ensuring methodological consistency with edge framing analysis.
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
PERIODS = ["period1", "period2", "period3"]
FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

def classify_community(title: str, summary: str) -> dict:
    """Classify a community based on its title and summary using LLM."""
    text = f"Title: {title}\nSummary: {summary[:800] if summary else 'N/A'}"
    
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
        if result.get('frame') not in FRAME_CATEGORIES:
            result['frame'] = 'NEUTRAL'
        return result
    except Exception as e:
        return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}


def process_period(period: str) -> pd.DataFrame:
    """Process all communities for a single period."""
    comm_path = GRAPHRAG_DIR / period / "output" / "community_reports.parquet"
    if not comm_path.exists():
        print(f"  [WARNING] {comm_path} not found")
        return None
    
    comm_df = pd.read_parquet(comm_path)
    print(f"  Loaded {len(comm_df)} communities from {period}")
    
    results = []
    for idx in tqdm(range(len(comm_df)), desc=f"  {period}"):
        row = comm_df.iloc[idx]
        title = str(row.get('title', ''))
        summary = str(row.get('summary', ''))
        
        result = classify_community(title, summary)
        results.append({
            'community_id': row.get('community', idx),
            'title': title[:100],
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
    print("LLM-based Community Frame Classification")
    print("="*60)
    
    all_results = []
    for period in PERIODS:
        print(f"\nProcessing {period}...")
        result_df = process_period(period)
        if result_df is not None:
            all_results.append(result_df)
            
            # Print distribution
            print(f"\n  {period} Frame Distribution:")
            for frame in FRAME_CATEGORIES:
                pct = (result_df['frame'] == frame).mean() * 100
                print(f"    {frame}: {pct:.1f}%")
    
    # Combine and save
    combined_df = pd.concat(all_results, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "community_framing_llm_results.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    # Summary table
    print("\n" + "="*60)
    print("SUMMARY: Community Frame Distribution by Period")
    print("="*60)
    print(f"{'Frame':<15} {'P1':>8} {'P2':>8} {'P3':>8}")
    print("-"*39)
    for frame in FRAME_CATEGORIES:
        row = ""
        for period in PERIODS:
            period_df = combined_df[combined_df['period'] == period]
            pct = (period_df['frame'] == frame).mean() * 100
            row += f"{pct:>8.1f}"
        print(f"{frame:<15}{row}")


if __name__ == "__main__":
    main()
