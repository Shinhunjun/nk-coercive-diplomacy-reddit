
import pandas as pd
import json
import asyncio
import aiohttp
import os
import sys
from datetime import datetime
import time

# Configuration
INPUT_FILE = 'data/processed/china_comments_recursive_roberta_final.csv'
OUTPUT_FILE = 'data/processed/china_framing_results.csv'
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
CONCURRENCY = 25 # Increase for production
TEST_MODE = False
INCLUDE_REASON = True # Set to False to save tokens/time

# V2 Prompt
SYSTEM_PROMPT = """You are a political science researcher analyzing media framing of international relations. Apply the Critical Classification Rules FIRST before classifying."""

def get_prompt(title, body):
    base_prompt = """You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## ⚠️ Critical Classification Rules (Apply First!)

### Rule 1: No Action = NEUTRAL
If the post is a **question, hypothesis, speculation, or factual report** without explicit government action, classify as **NEUTRAL**.
- Example: "What if Ukraine and Russia go to war?" → NEUTRAL (question, no action)

### Rule 2: Verbal vs Physical Actions
If a state is **only verbally criticizing or warning** another state (not taking physical/military action), classify as **DIPLOMACY**, not THREAT.
- Example: "China warns India over military buildup" → DIPLOMACY (verbal warning)

### Rule 3: Individual Harm = HUMANITARIAN
If the harm is to **specific individuals** (protesters, defectors, refugees, civilians), classify as **HUMANITARIAN**, not THREAT.

### Rule 4: Conflicting Frames = NEUTRAL
When **DIPLOMACY and THREAT (or other frames) are equally present** and competing, classify as **NEUTRAL**.

### Rule 5: Domestic Politics = NEUTRAL
**Commentary on domestic political issues**, even if mentioning foreign countries, is NEUTRAL.

---

## Classification Criteria

### THREAT (Military Tension/Conflict)
**Physical military actions that increase conflict possibility**
Include: Missile launches, nuclear tests, military exercises, arms buildup.
Exclude: Verbal warnings (DIPLOMACY), Arms deals (THREAT), but requests to stop (DIPLOMACY).

### DIPLOMACY (Diplomatic Interaction)
**Relationship adjustment through dialogue, negotiation, or verbal pressure**
Include: Summits, negotiations, verbal criticism/warnings, urging to stop actions.

### ECONOMIC (Economic Measures)
**Pressure or cooperation through economic means**
Include: Sanctions, trade measures, aid.

### HUMANITARIAN (Humanitarian Issues)
**Human rights and individual/civilian harm**
Include: Human rights violations, refugee issues, harm to individuals.

### NEUTRAL (Neutral Information)
**Cases not fitting specific frames**
Include: Factual reporting, domestic politics, questions/hypotheticals.

---

## Post Content
Title: {title}
Body: {body}

## Response Format (JSON only)
"""
    if INCLUDE_REASON:
        format_spec = '{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "One sentence explaining classification rationale"}}'
    else:
        format_spec = '{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0}}'
    
    return base_prompt + format_spec

async def classify_comment(session, row, semaphore):
    async with semaphore:
        text = row.get('body', '')
        parent_title = row.get('parent_post_title', '')
        
        # Truncate body
        body_snippet = str(text)[:800] 
        
        prompt = get_prompt(parent_title, body_snippet)
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 100,
            "response_format": {"type": "json_object"}
        }
        
        try:
            async with session.post("https://api.openai.com/v1/chat/completions", json=payload, headers={"Authorization": f"Bearer {API_KEY}"}) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    try:
                        result = json.loads(content)
                        return {
                            'id': row['id'],
                            'frame': result.get('frame', 'NEUTRAL'),
                            'confidence': result.get('confidence', 0.5),
                            'reason': result.get('reason', '') if INCLUDE_REASON else '',
                            'status': 'success'
                        }
                    except:
                        return {'id': row['id'], 'frame': 'NEUTRAL', 'confidence': 0.0, 'status': 'parse_error'}
                elif response.status == 429:
                    return {'id': row['id'], 'status': 'rate_limit'}
                else:
                    return {'id': row['id'], 'status': f'error_{response.status}'}
        except Exception as e:
            return {'id': row['id'], 'status': f'exception_{str(e)}'}

async def process_batch(rows):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = [classify_comment(session, row, semaphore) for row in rows]
        return await asyncio.gather(*tasks)

def main():
    if not API_KEY:
        print("Error: OPENAI_API_KEY not set.")
        return

    print("Loading China Data...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    print(f"Total China Comments: {len(df)}")
    
    # Check for existing progress
    completed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            completed_ids = set(existing['id'].astype(str))
            print(f"Resuming... {len(completed_ids)} already processed.")
        except:
            pass
            
    # Filter out completed
    df['id'] = df['id'].astype(str)
    to_process = df[~df['id'].isin(completed_ids)].to_dict('records')
    print(f"Remaining to process: {len(to_process)}")
    
    if TEST_MODE:
        print("\n>>> TEST MODE: Processing only 10 random samples <<<")
        # Ensure we pick random valid samples
        import random
        random.seed(42)
        if len(to_process) > 10:
            to_process = random.sample(to_process, 10)
        chunk_size = 10
    else:
        chunk_size = CONCURRENCY * 2
    
    if len(to_process) == 0:
        print("All done!")
        return

    timestamp = time.time()
    total_processed = 0
    
    loop = asyncio.get_event_loop()
    
    for i in range(0, len(to_process), chunk_size):
        batch = to_process[i:i+chunk_size]
        results = loop.run_until_complete(process_batch(batch))
        
        # Save results immediately
        results_df = pd.DataFrame([r for r in results if r['status'] == 'success'])
        
        if not results_df.empty:
            if TEST_MODE:
                print("\n--- Test Results (China) ---")
                for _, res in results_df.iterrows():
                    print(f"ID: {res['id']}")
                    print(f"  Frame: {res['frame']} ({res['confidence']})")
                    print(f"  Reason: {res['reason']}")
                    print("-" * 30)
            
            if not TEST_MODE:
                write_header = not os.path.exists(OUTPUT_FILE)
                results_df.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
            
        total_processed += len(batch)
        
    print("Test run complete!")

if __name__ == "__main__":
    main()
