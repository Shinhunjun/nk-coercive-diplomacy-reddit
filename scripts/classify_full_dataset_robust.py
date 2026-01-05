"""
Robust Full Dataset Classification with Rate Limiting
- Retry logic for 429 errors (exponential backoff)
- Reduced thread count (5)
- Per-request delay (100ms)
"""
import pandas as pd
import os
import json
import time
import concurrent.futures
from openai import OpenAI
from dotenv import load_dotenv

# Load env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
OUTPUT_DIR = "data/results/final_framing_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COUNTRIES = {
    "nk": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
    "china": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
    "iran": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
    "russia": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"]
}

MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds (increased)
THREAD_COUNT = 5  # Balanced for speed and rate limit

# ==========================================
# V2 PROMPT
# ==========================================
def get_classification(text, model_id="gpt-4o-mini", retries=0):
    prompt = f"""You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## ⚠️ Critical Classification Rules (Apply First!)

### Rule 1: No Action = NEUTRAL
If the post is a **question, hypothesis, speculation, or factual report** without explicit government action, classify as **NEUTRAL**.
- Example: "What if North Korea attacks?" → NEUTRAL (question, no action)

### Rule 2: Verbal vs Physical Actions
If a state is **only verbally criticizing or warning** another state (not taking physical/military action), classify as **DIPLOMACY**, not THREAT.
- Example: "North Korea warns US over military exercises" → DIPLOMACY (verbal warning)

### Rule 3: Individual Harm = HUMANITARIAN
If the harm is to **specific individuals** (protesters, defectors, refugees, civilians), classify as **HUMANITARIAN**, not THREAT.

### Rule 4: Conflicting Frames = NEUTRAL
When **DIPLOMACY and THREAT (or other frames) are equally present** and competing, classify as **NEUTRAL**.

### Rule 5: Domestic Politics = NEUTRAL
**Commentary on domestic political issues**, even if mentioning foreign countries, is NEUTRAL.
- Example: "Democrats criticize Trump on North Korea policy" → NEUTRAL (domestic politics)

---

## Classification Criteria

### THREAT (Military Tension/Conflict)
**Physical military actions that increase conflict possibility**
- Military actions, arms buildup, ultimatums, cyberattacks on infra.

### DIPLOMACY (Diplomatic Interaction)
**Relationship adjustment through dialogue, negotiation, or verbal pressure**
- Summits, negotiations, treaties, sanctions relief.
- **Verbal criticism, condemnation, or warnings between states.**

### ECONOMIC (Economic Measures)
**Pressure or cooperation through economic means**
- Sanctions, trade measures, economic aid.

### HUMANITARIAN (Humanitarian Issues)
**Human rights and individual/civilian harm**
- Human rights, refugees, aid, civilian harm.

### NEUTRAL (Neutral Information/Opinion)
**Cases not fitting specific frames**
- Factual reporting, domestic politics, questions, pure opinion.

---

## Post
{text}

## Response Format (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "One sentence explaining classification rationale"}}"""

    try:
        # Add delay to avoid rate limit
        time.sleep(0.5)  # 500ms delay
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a political science researcher analyzing media framing of international relations. Apply the Critical Classification Rules FIRST before classifying."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        error_str = str(e)
        
        # Retry on 429 (Rate Limit) with exponential backoff
        if "429" in error_str and retries < MAX_RETRIES:
            wait_time = BASE_DELAY * (2 ** retries)
            print(f"      Rate limit hit, waiting {wait_time:.1f}s (retry {retries+1}/{MAX_RETRIES})...")
            time.sleep(wait_time)
            return get_classification(text, model_id, retries + 1)
        
        return {"frame": "ERROR", "reason": str(e)[:100], "confidence": 0.0}

def process_row(row):
    try:
        title = row.get('title', '')
        body = row.get('selftext', row.get('body', ''))
        text = f"Title: {title}\nBody: {str(body)[:500] if body else 'N/A'}"
        
        result = get_classification(text)
        return {
            "id": row.get('id'),
            "frame": result.get('frame', 'NEUTRAL'),
            "confidence": result.get('confidence', 0.0),
            "reason": result.get('reason', '')
        }
    except Exception:
        return {
            "id": row.get('id'),
            "frame": "ERROR",
            "confidence": 0.0,
            "reason": "Processing Error"
        }

def classify_country(country_key, test_mode=False):
    print(f"\n🚀 Processing {country_key}...")
    
    # Load Data
    dfs = []
    for path in COUNTRIES[country_key]:
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, low_memory=False))
    
    if not dfs:
        print(f"   ❌ No data found for {country_key}")
        return
        
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['id'])
    
    # Test mode: only 10 samples
    if test_mode:
        df = df.head(10)
        print(f"   🧪 TEST MODE: {len(df)} samples")
    else:
        print(f"   Processing {len(df)} posts...")
    
    # Output path
    output_path = f"{OUTPUT_DIR}/{country_key}_framing_v2.csv"
    
    # Check for existing (resume)
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            existing = pd.read_csv(output_path)
            processed_ids = set(existing['id'].astype(str))
            print(f"   🔄 Resuming: {len(processed_ids)} already done.")
        except:
            pass
    
    to_process = df[~df['id'].astype(str).isin(processed_ids)]
    remaining = len(to_process)
    
    if remaining == 0:
        print(f"   ✅ {country_key} already complete!")
        return
    
    print(f"   🔧 {remaining} posts to classify with {THREAD_COUNT} threads...")
    
    # Process with ThreadPoolExecutor
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        future_to_row = {
            executor.submit(process_row, row): row['id'] 
            for _, row in to_process.iterrows()
        }
        
        count = 0
        batch_size = 50
        temp_results = []
        start_time = time.time()
        
        for future in concurrent.futures.as_completed(future_to_row):
            res = future.result()
            temp_results.append(res)
            count += 1
            
            if count % batch_size == 0:
                # Save Batch
                batch_df = pd.DataFrame(temp_results)
                write_header = not os.path.exists(output_path)
                batch_df.to_csv(output_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
                temp_results = []
                
                elapsed = time.time() - start_time
                rate = count / elapsed
                print(f"   {count}/{remaining} done ({rate:.1f} req/s)...", end='\r')
        
        # Save remaining
        if temp_results:
            batch_df = pd.DataFrame(temp_results)
            write_header = not os.path.exists(output_path)
            batch_df.to_csv(output_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')

    print(f"\n   ✅ {country_key} finished! Saved to {output_path}")

def main(test_mode=False):
    print("="*80)
    print("🔧 ROBUST CLASSIFICATION (V2 Prompt)")
    print(f"Threads: {THREAD_COUNT} | Retry: {MAX_RETRIES}x | Delay: 100ms")
    print("="*80)
    
    for country in ['nk', 'china', 'iran', 'russia']:
        classify_country(country, test_mode=test_mode)
    
    print("\n" + "="*80)
    print("🎉 ALL COUNTRIES COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    import sys
    test_mode = "--test" in sys.argv
    main(test_mode=test_mode)
