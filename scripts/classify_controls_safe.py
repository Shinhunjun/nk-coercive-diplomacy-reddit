import pandas as pd
import os
import json
import asyncio
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "gpt-4o-mini"
CONCURRENCY = 15
MAX_RETRIES = 5

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
            "max_tokens": 200, # Reduced tokens for speed
            "response_format": {"type": "json_object"}
        }

        headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
        
        for attempt in range(MAX_RETRIES):
            try:
                # Add slight delay to spread requests
                if attempt > 0: await asyncio.sleep(2 ** attempt)
                
                async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = json.loads(data['choices'][0]['message']['content'])
                        return {
                            "id": row['id'],
                            "parent_post_id": row.get('parent_post_id', ''),
                            "created_utc": row['created_utc'],
                            "period": row['period'],
                            "country": row['country'],
                            "frame": content.get('frame', 'NEUTRAL')
                        }
                    elif resp.status == 429:
                        wait_time = 5 * (2 ** attempt)
                        print(f"⚠️ 429 Rate Limit ({row['country']}). Waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"⚠️ Error {resp.status}: {await resp.text()}")
                        return {"id": row['id'], "frame": "ERROR"}
            except Exception as e:
                print(f"⚠️ Exception: {e}")
                await asyncio.sleep(1)
        return {"id": row['id'], "frame": "ERROR"}

async def process_file(country, input_path, output_path, session, sem):
    print(f"🚀 Loading {country} from {input_path}...")
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    df['period'] = df['created_utc'].apply(get_period)
    df['country'] = country
    
    df_filtered = df[df['period'].isin(['P1', 'P2', 'P3'])].copy()
    print(f"🔍 {country}: {len(df_filtered)} comments.")

    if os.path.exists(output_path):
        done = pd.read_csv(output_path)
        done_ids = set(done['id'].astype(str))
        df_filtered = df_filtered[~df_filtered['id'].astype(str).isin(done_ids)]
        print(f"🔄 Resuming {country}: {len(df_filtered)} remaining.")
    
    if len(df_filtered) == 0: return

    tasks = [classify_comment(session, row, sem) for _, row in df_filtered.iterrows()]
    
    batch_size = 50
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i+batch_size]
        results = await asyncio.gather(*batch)
        
        # Save
        res_df = pd.DataFrame(results)
        write_header = not os.path.exists(output_path)
        res_df.to_csv(output_path, mode='a', header=write_header, index=False)
        print(f"✅ {country}: {i + len(batch)} processed...", end='\r')

async def main():
    configs = [
        ('China', 'data/control/china_comments_roberta.csv', 'data/results/china_comment_framing_final.csv'),
        ('Iran', 'data/control/iran_comments_roberta.csv', 'data/results/iran_comment_framing_final.csv'),
        ('Russia', 'data/control/russia_comments_roberta.csv', 'data/results/russia_comment_framing_final.csv')
    ]
    
    sem = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for country, inp, out in configs:
            tasks.append(process_file(country, inp, out, session, sem))
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
