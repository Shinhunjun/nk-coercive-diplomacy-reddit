"""Apply framing to China extended data only"""
import pandas as pd
import os, sys, json
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FRAME_CATEGORIES = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]
FRAME_SCALE = {"THREAT": -2, "ECONOMIC": -1, "NEUTRAL": 0, "HUMANITARIAN": 1, "DIPLOMACY": 2}

client = OpenAI()

def classify_post(title, body=""):
    text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"
    prompt = f"""Classify this Reddit post about international politics into ONE of these frames:
- THREAT: Focus on military danger, nuclear weapons, missiles, war
- DIPLOMACY: Focus on negotiations, talks, peace, cooperation
- NEUTRAL: Factual information without clear framing
- ECONOMIC: Focus on sanctions, trade, economic aspects
- HUMANITARIAN: Focus on human rights, refugees, citizens

Post:
{text}

You must respond with valid JSON only:
{{"frame": "CATEGORY", "confidence": 0.8, "reason": "brief explanation"}}"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=150, response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        if result.get('frame') not in FRAME_CATEGORIES: result['frame'] = 'NEUTRAL'
        return result
    except Exception as e:
        return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {e}"}

def main():
    print("=== CHINA Framing Analysis ===")
    df = pd.read_csv('data/control/china_posts_hanoi_extended.csv')
    print(f"Loaded {len(df)} posts")
    
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="China"):
        r = classify_post(str(row.get('title','')), str(row.get('selftext','')) if pd.notna(row.get('selftext')) else '')
        results.append(r)
    
    df['frame'] = [r['frame'] for r in results]
    df['frame_confidence'] = [r['confidence'] for r in results]
    df['frame_reason'] = [r['reason'] for r in results]
    df['frame_score'] = df['frame'].map(FRAME_SCALE)
    
    df.to_csv('data/framing/china_posts_hanoi_extended_framed.csv', index=False)
    print(f"\n✓ Saved! Distribution:\n{df['frame'].value_counts()}")

if __name__ == '__main__': main()
