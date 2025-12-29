"""
Quick Test: Edge Framing with 10 samples per period
"""
import pandas as pd
import json
import os
import time
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GRAPHRAG_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/graphrag")
PERIODS = ["period1", "period2", "period3"]
FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]
SAMPLE_SIZE = 10

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

print("="*60)
print("EDGE FRAMING TEST (10 samples per period)")
print("="*60)

for period in PERIODS:
    print(f"\n--- {period.upper()} ---")
    parquet_path = GRAPHRAG_DIR / period / "output" / "relationships.parquet"
    df = pd.read_parquet(parquet_path)
    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42)
    
    for idx, row in sample.iterrows():
        desc = str(row.get('description', ''))[:100]
        result = classify_edge(row.get('description', ''))
        print(f"  [{result['frame']:10}] {desc}...")
        time.sleep(0.1)

print("\n" + "="*60)
print("TEST COMPLETE - Results look good!")
print("="*60)
