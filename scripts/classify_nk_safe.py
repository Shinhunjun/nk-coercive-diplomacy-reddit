import pandas as pd
import os
import json
import asyncio
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

# Load env
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Configuration
INPUT_FILE = "data/processed/nk_comments_roberta.csv"
OUTPUT_FILE = "data/results/nk_comment_framing_final.csv"
MODEL = "gpt-4o-mini"
CONCURRENCY = 15
MAX_RETRIES = 5

# Periods
P1_START = datetime(2017, 1, 1).timestamp()
P1_END = datetime(2018, 6, 11).timestamp()
P2_START = datetime(2018, 6, 13).timestamp()
P2_END = datetime(2019, 2, 27).timestamp()
P3_START = datetime(2019, 3, 1).timestamp()
P3_END = datetime(2019, 12, 31).timestamp()

def get_period(timestamp):
    try:
        ts = float(timestamp)
        if P1_START <= ts <= P1_END: return 'P1'
        elif P2_START <= ts <= P2_END: return 'P2'
        elif P3_START <= ts <= P3_END: return 'P3'
        else: return 'Out'
    except: return 'Error'

PROMPT_TEMPLATE = """You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## ⚠️ Critical Classification Rules (Apply First!)

### Rule 1: No Action = NEUTRAL
If the post is a **question, hypothesis, speculation, or factual report** without explicit government action, classify as **NEUTRAL**.
- Example: "What if Ukraine and Russia go to war?" → NEUTRAL (question, no action)
- Example: "Russia's submarine activity at highest level" → NEUTRAL (factual report)

### Rule 2: Verbal vs Physical Actions
If a state is **only verbally criticizing or warning** another state (not taking physical/military action), classify as **DIPLOMACY**, not THREAT.
- Example: "China warns India over military buildup" → DIPLOMACY (verbal warning)
- Example: "Russia claims US supporting terrorism" → DIPLOMACY (verbal criticism)

### Rule 3: Individual Harm = HUMANITARIAN
If the harm is to **specific individuals** (protesters, defectors, refugees, civilians), classify as **HUMANITARIAN**, not THREAT.
- Example: "North Korea soldier shot while defecting" → HUMANITARIAN (individual harm)
- Example: "Cyberattack hits app used by Hong Kong protesters" → HUMANITARIAN (targeting civilians)

### Rule 4: Conflicting Frames = NEUTRAL
When **DIPLOMACY and THREAT (or other frames) are equally present** and competing, classify as **NEUTRAL**.
- Example: "Syria shifts to diplomacy while US pushes for war" → NEUTRAL (equal competing frames)

### Rule 5: Domestic Politics = NEUTRAL
**Commentary on domestic political issues**, even if mentioning foreign countries, is NEUTRAL.
- Example: "Democrats criticize Trump on Russia policy" → NEUTRAL (domestic politics)

---

## Classification Criteria

### THREAT (Military Tension/Conflict)
**Physical military actions that increase conflict possibility**

Include:
- Military actions: missile launches, nuclear tests, military exercises, shows of force
- Arms buildup: weapons sales, arms provision, military equipment deployment
- Military threats with NO dialogue possibility (ultimatums)
- Cyberattacks on military/government infrastructure

**Exclude (classify as DIPLOMACY instead):**
- Verbal warnings with possibility of dialogue remaining
- One state verbally criticizing another's actions
- Requests to stop military activities (diplomatic pressure)

---

### DIPLOMACY (Diplomatic Interaction)
**Relationship adjustment through dialogue, negotiation, or verbal pressure**

Include:
- Summit meetings, diplomatic negotiations, bilateral/multilateral talks
- Agreements, treaties, accord signings
- Attempts to improve/normalize relations
- Sanctions relief or easing
- **Verbal criticism, condemnation, or warnings between states**
- **One state urging another to stop certain actions** (diplomatic pressure)
- Even if a summit fails, focus on the summit itself → DIPLOMACY

---

### ECONOMIC (Economic Measures)
**Pressure or cooperation through economic means**

Include:
- Imposition/strengthening of economic sanctions
- Sanctions evasion activities
- Trade measures (tariffs, import/export restrictions)
- Economic cooperation, investment, aid

**Exclude:**
- Arms deals → THREAT
- If the main focus is diplomatic action, not economic → DIPLOMACY

---

### HUMANITARIAN (Humanitarian Issues)
**Human rights and individual/civilian harm**

Include:
- Human rights violations, oppression (targeting civilians within a country)
- Refugee issues
- Humanitarian assistance/aid
- War crimes, genocide
- **Harm to individuals (protesters, defectors, refugees, civilians)**
- **Cyberattacks targeting protesters or civilian groups**

---

### NEUTRAL (Neutral Information)
**Cases not fitting specific frames**

Include:
- Simple factual reporting, analysis, information delivery
- **Domestic politics** (party conflicts, government criticism, etc.)
- **Questions and hypothetical scenarios** ("What if X happens?")
- **Factual descriptions without explicit government action**
- **When multiple frames are equally present and competing**
- Complex cases where priority determination is difficult

---

## Post
{text}

## Response Format (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "One sentence explaining classification rationale"}}"""

async def classify_comment(session, row, semaphore):
    async with semaphore:
        text = str(row.get('body', ''))[:600]
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a political science researcher analyzing media framing of international relations. Apply the Critical Classification Rules FIRST before classifying."},
                {"role": "user", "content": PROMPT_TEMPLATE.format(text=text)}
            ],
            "temperature": 0.0,
            "max_tokens": 200,
            "response_format": {"type": "json_object"}
        }

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = json.loads(data['choices'][0]['message']['content'])
                        return {
                            "id": row['id'],
                            "parent_post_id": row.get('parent_post_id', ''),
                            "created_utc": row['created_utc'],
                            "period": row['period'],
                            "frame": content.get('frame', 'NEUTRAL'),
                            "confidence": content.get('confidence', 0.0),
                            "reason": content.get('reason', '')
                        }
                    elif resp.status == 429: # Rate Limit
                        wait_time = 5 * (2 ** attempt)  # Exponential backoff
                        print(f"⚠️ 429 Rate Limit. Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"⚠️ Error {resp.status}: {await resp.text()}")
                        return {"id": row['id'], "frame": "ERROR", "reason": f"HTTP {resp.status}"}
            except Exception as e:
                print(f"⚠️ Exception: {e}")
                await asyncio.sleep(1)
                
        return {"id": row['id'], "frame": "ERROR", "reason": "Max Retries"}

async def main():
    print(f"🚀 Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    # Filter Periods
    df['period'] = df['created_utc'].apply(get_period)
    df_filtered = df[df['period'].isin(['P1', 'P2', 'P3'])].copy()
    
    # FULL RUN
    total = len(df_filtered)
    print(f"🔍 Analyzable Comments (P1-P3): {total}")
    
    # Check existing
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        done_ids = set(existing['id'].astype(str))
        df_filtered = df_filtered[~df_filtered['id'].astype(str).isin(done_ids)]
        print(f"🔄 Resuming... {len(done_ids)} done, {len(df_filtered)} remaining.")
    
    if len(df_filtered) == 0:
        print("✅ Nothing to do.")
        return

    # Semaphore for concurrency
    sem = asyncio.Semaphore(CONCURRENCY)
    
    results = []
    batch_size = 50
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in df_filtered.iterrows():
            tasks.append(classify_comment(session, row, sem))
            
        # Process in batches to save progress
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = await asyncio.gather(*batch)
            
            # Save batch
            res_df = pd.DataFrame(batch_results)
            write_header = not os.path.exists(OUTPUT_FILE)
            res_df.to_csv(OUTPUT_FILE, mode='a', header=write_header, index=False)
            
            print(f"✅ Processed {i + len(batch)}/{total}...", end='\r')
            await asyncio.sleep(0.5) # Slight delay between batches

    print(f"\n🎉 Done! Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
