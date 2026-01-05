
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
INPUT_FILE = "data/comments_to_classify_top3.csv"
OUTPUT_DIR = "data/results/final_framing_v2"
OUTPUT_FILE = f"{OUTPUT_DIR}/comment_framing_v2.csv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# V2 PROMPT (Same as Post Classification)
# ==========================================
def get_classification(text, model_id="gpt-4o-mini"):
    prompt = f"""You are an international relations researcher. Classify the following Reddit comment into ONE of 5 framing categories.

## ⚠️ Critical Classification Rules (Apply First!)

### Rule 1: No Action = NEUTRAL
If the comment is a **question, hypothesis, speculation, or factual report** without explicit government action, classify as **NEUTRAL**.
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
- Example: "Democrats criticize Trump on Russia policy" → NEUTRAL (domestic politics)
- Example: "Trump is an idiot for doing this" → NEUTRAL (domestic/opinion)

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
- Factual reporting, domestic politics, questions, pure opinion without framing the event itself.

---

## Comment
{text}

## Response Format (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "One sentence explaining classification rationale"}}"""

    try:
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
        return {"frame": "ERROR", "reason": str(e), "confidence": 0.0}

def process_row(row):
    try:
        body = row.get('body', '')
        # Truncate very long comments
        text = str(body)[:600]
        
        result = get_classification(text)
        return {
            "id": row.get('id'),
            "parent_post_id": row.get('parent_post_id'),
            "country": row.get('country'),
            "score": row.get('score'),
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

def main():
    print("="*80)
    print("💬 COMMENT CLASSIFICATION (Top 3 per Post)")
    print("Using Model: GPT-4o-mini | Prompt: V2 (Revised)")
    print("="*80)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Input file not found: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE)
    total = len(df)
    print(f"Loaded {total} comments.")
    
    # Resume logic
    processed_ids = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            existing = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(existing['id'].astype(str))
            print(f"🔄 Resuming: {len(processed_ids)} already done.")
        except:
            pass
            
    to_process = df[~df['id'].astype(str).isin(processed_ids)]
    remaining = len(to_process)
    
    if remaining == 0:
        print("✅ All comments already classified!")
        return

    print(f"🚀 Processing {remaining} comments with 30 threads...")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
        future_to_row = {
            executor.submit(process_row, row): row['id'] 
            for _, row in to_process.iterrows()
        }
        
        count = 0
        batch_size = 100
        temp_results = []
        
        start_time = time.time()
        
        for future in concurrent.futures.as_completed(future_to_row):
            res = future.result()
            temp_results.append(res)
            count += 1
            
            if count % batch_size == 0:
                # Save Batch
                batch_df = pd.DataFrame(temp_results)
                write_header = not os.path.exists(OUTPUT_FILE)
                batch_df.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False, encoding='utf-8-sig')
                temp_results = [] # clear buffer
                
                elapsed = time.time() - start_time
                rate = count / elapsed
                print(f"   {count}/{remaining} done ({rate:.1f} req/s)...", end='\r')
        
        # Save remaining
        if temp_results:
            batch_df = pd.DataFrame(temp_results)
            write_header = not os.path.exists(OUTPUT_FILE)
            batch_df.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False, encoding='utf-8-sig')

    print(f"\n✅ Finished! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
