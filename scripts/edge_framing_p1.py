"""
Parallel Edge Framing - Period 1 (P1: Pre-Singapore)
Run this script in parallel with period2 and period3 scripts.
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
PERIOD = "period1"
FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o-mini"

def classify_edge(description: str) -> dict:
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
        if result.get('frame') not in FRAME_CATEGORIES:
            result['frame'] = 'NEUTRAL'
        return result
    except Exception as e:
        return {"frame": "ERROR", "confidence": 0, "reason": str(e)}

def main():
    print(f"Processing {PERIOD}...")
    parquet_path = GRAPHRAG_DIR / PERIOD / "output" / "relationships.parquet"
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} edges")
    
    results = []
    for idx in tqdm(range(len(df)), desc=f"  {PERIOD}"):
        row = df.iloc[idx]
        result = classify_edge(str(row.get('description', '')))
        results.append({
            'source': row['source'],
            'target': row['target'],
            'description': str(row.get('description', ''))[:200],
            'frame': result.get('frame', 'NEUTRAL'),
            'confidence': result.get('confidence', 0.5),
            'reason': result.get('reason', '')
        })
        time.sleep(0.05)  # Reduced delay for parallel
    
    result_df = pd.DataFrame(results)
    result_df['period'] = PERIOD
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"edge_framing_{PERIOD}.csv"
    result_df.to_csv(output_path, index=False)
    print(f"✓ Saved: {output_path}")
    
    # Print distribution
    print(f"\n{PERIOD} Frame Distribution:")
    for frame in FRAME_CATEGORIES:
        pct = (result_df['frame'] == frame).mean() * 100
        print(f"  {frame}: {pct:.1f}%")

if __name__ == "__main__":
    main()
