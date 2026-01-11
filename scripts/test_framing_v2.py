"""Test V2 framing classification on 10 comments."""
import pandas as pd
import json
import time
import os
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print("ERROR: Set OPENAI_API_KEY")
    exit(1)
    
client = OpenAI(api_key=api_key)

def classify_text(text):
    text_content = str(text)[:500] if text else 'N/A'
    
    prompt = f"""You are an international relations researcher. Classify the following Reddit comment into ONE of 5 framing categories.

Critical Classification Rules:
1. No Action = NEUTRAL (questions, speculation, factual reports)
2. Verbal warnings = DIPLOMACY (not THREAT)
3. Individual harm = HUMANITARIAN
4. Conflicting frames = NEUTRAL
5. Domestic politics = NEUTRAL

Categories:
- THREAT: Physical military actions (missiles, nuclear tests, military exercises)
- DIPLOMACY: Summits, negotiations, verbal warnings between states
- ECONOMIC: Sanctions, trade measures
- HUMANITARIAN: Human rights, refugees, individual harm
- NEUTRAL: Questions, factual reports, domestic politics

Comment: {text_content}

Response (JSON only):
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "Brief rationale"}}"""

    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'system', 'content': 'Apply Critical Classification Rules FIRST.'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={'type': 'json_object'}
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        return {'frame': 'ERROR', 'reason': str(e)}

# Load 10 comments
df = pd.read_csv('data/processed/nk_comments_top3_final.csv', low_memory=False)
valid = df[~df['body'].astype(str).str.contains(r'\[removed\]|\[deleted\]', case=False, na=False, regex=True)]
valid = valid[valid['body'].astype(str).str.len() > 50]

print('Testing V2 Prompt on 10 Comments')
print('=' * 70)

for i, (_, row) in enumerate(valid.head(10).iterrows()):
    text = str(row['body'])[:120]
    result = classify_text(row['body'])
    print(f'{i+1}. "{text}..."')
    print(f'   -> {result.get("frame")} | Conf: {result.get("confidence", "?")} | {result.get("reason", "")[:50]}')
    print()
    time.sleep(0.2)

print('Test complete!')
