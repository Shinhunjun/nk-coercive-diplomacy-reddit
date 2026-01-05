
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
    "North Korea": {
        "files": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
        "output": f"{OUTPUT_DIR}/nk_framing_v2.csv"
    },
    "China": {
        "files": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
        "output": f"{OUTPUT_DIR}/china_framing_v2.csv"
    },
    "Iran": {
        "files": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
        "output": f"{OUTPUT_DIR}/iran_framing_v2.csv"
    },
    "Russia": {
        "files": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"],
        "output": f"{OUTPUT_DIR}/russia_framing_v2.csv"
    }
}

# ==========================================
# V2 PROMPT
# ==========================================
def get_classification(text, model_id="gpt-4o-mini"):
    prompt = f"""You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

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

# Worker function for threading
def process_row(row, model_id):
    try:
        title = row.get('title', '')
        body = row.get('selftext', '')
        text = f"Title: {title}\nBody: {str(body)[:500]}"
        
        result = get_classification(text, model_id)
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


# Wrapper for multiprocessing
def process_country_wrapper(args):
    country_name, config_data = args
    # Re-initialize client inside process to avoid pickle issues
    global client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    classify_country(country_name, config_data)

def classify_country(country_name, config):
    print(f"\n🚀 Starting {country_name}...")
    
    # Load Data
    dfs = []
    for f in config['files']:
        if os.path.exists(f):
            dfs.append(pd.read_csv(f, low_memory=False))
            
    if not dfs:
        print(f"❌ No data for {country_name}")
        return

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.drop_duplicates(subset=['id'])
    
    # Check for existing results to resume
    output_path = config['output']
    processed_ids = set()
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        processed_ids = set(existing_df['id'].astype(str))
        print(f"   🔄 Resuming {country_name}: {len(processed_ids)} already done.")
    
    # Filter for unprocessed
    to_process = full_df[~full_df['id'].astype(str).isin(processed_ids)]
    total = len(to_process)
    
    if total == 0:
        print(f"✅ {country_name} already complete!")
        return

    print(f"   Processing {total} posts for {country_name}...")
    
    results = []
    # Using ThreadPoolExecutor INSIDE each Process
    # Reduced max_workers per process to 10 to avoid hitting global limits (4 countries * 10 = 40 threads total)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_row = {
            executor.submit(process_row, row, "gpt-4o-mini"): row['id'] 
            for _, row in to_process.iterrows()
        }
        
        count = 0
        batch_size = 50
        temp_results = []
        
        for future in concurrent.futures.as_completed(future_to_row):
            res = future.result()
            temp_results.append(res)
            count += 1
            
            if count % batch_size == 0:
                # Save Batch
                batch_df = pd.DataFrame(temp_results)
                write_header = not os.path.exists(output_path)
                batch_df.to_csv(output_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
                temp_results = [] # clear buffer
                # print(f"   [{country_name}] {count}/{total} done...", end='\r') # Avoid clutter in parallel output
        
        # Save remaining
        if temp_results:
            batch_df = pd.DataFrame(temp_results)
            write_header = not os.path.exists(output_path)
            batch_df.to_csv(output_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
            
    print(f"\n✅ {country_name} Finished! Saved to {output_path}")

def main():
    print("="*80)
    print("🌍 UNIVERSE CLASSIFICATION: NK, China, Iran, Russia (PARALLEL EXECUTION)")
    print("Using Model: GPT-4o-mini | Prompt: V2 (Revised)")
    print("="*80)
    
    # Run Countries in Parallel Processes
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_country_wrapper, COUNTRIES.items())

if __name__ == "__main__":
    main()
