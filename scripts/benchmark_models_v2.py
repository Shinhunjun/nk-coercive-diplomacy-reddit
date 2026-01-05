"""
Benchmark Models: GPT-4o-mini vs GPT-4o using the V2 (Best) Prompt.
This script:
1. Loads existing GPT-4o-mini results (from llm_improved_v2_classification.csv)
2. Runs GPT-4o on the same dataset using the V2 Prompt
3. Compares performance metrics side-by-side.
"""

import pandas as pd
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

# ==========================================
# EXACT V2 PROMPT (Best Performer)
# ==========================================
def classify_post_v2(title: str, body: str = "", model: str = "gpt-4o") -> dict:
    text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"
    
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
            model=model,
            messages=[
                {"role": "system", "content": "You are a political science researcher analyzing media framing of international relations. Apply the Critical Classification Rules FIRST before classifying."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Zero temperature for benchmark consistency
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        valid_frames = ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN']
        if result.get('frame') not in valid_frames:
            result['frame'] = 'NEUTRAL'
            result['confidence'] = 0.5
            
        return result
        
    except Exception as e:
        return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}


def main():
    print("=" * 70)
    print("🧪 MODEL BENCHMARK: GPT-4o-mini vs GPT-4o (Using V2 Prompt)")
    print("=" * 70)
    
    # 1. Load Data
    batch_pilot = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_pilot.csv')
    batch_1 = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_1.csv')
    batch_2 = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_2.csv')
    combined = pd.concat([batch_pilot, batch_1, batch_2], ignore_index=True)
    combined['final_frame_clean'] = combined['final_frame'].astype(str).str.strip().str.upper()
    
    valid_frames = ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN']
    valid = combined[combined['final_frame_clean'].isin(valid_frames)]
    valid = valid.dropna(subset=['title'])
    valid = valid[valid['title'].str.len() > 0]
    total = len(valid)
    
    # 2. Check for Baseline (GPT-4o-mini) Results
    baseline_path = 'data/annotations/llm_improved_v2_classification.csv'
    if os.path.exists(baseline_path):
        print(f"✅ Loaded Baseline (GPT-4o-mini) results from: {baseline_path}")
        baseline_df = pd.read_csv(baseline_path)
    else:
        print("⚠️ Baseline file not found! Please run v2 script first.")
        return

    # 3. Run Benchmark (GPT-4o)
    print(f"\n🚀 Running GPT-4o on {total} samples...")
    gpt4o_results = []
    
    for idx in range(total):
        row = valid.iloc[idx]
        title = str(row['title'])
        body = str(row.get('text', '')) if pd.notna(row.get('text')) else ''
        
        result = classify_post_v2(title, body, model="gpt-4o")
        
        gpt4o_results.append({
            'post_id': row.get('post_id', idx),
            'title': title[:100],
            'human_frame': row['final_frame_clean'],
            'gpt4o_frame': result.get('frame', 'NEUTRAL'),
            'gpt4o_reason': result.get('reason', '')
        })
        
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  Progress: {idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%)", flush=True)
        
        time.sleep(0.1) # Faster rate limit for Tier 4+ (assuming user has it, or just cautious)

    gpt4o_df = pd.DataFrame(gpt4o_results)
    
    # 4. Compare Results
    # Merge Mini and 4o results
    # Assuming order is preserved (which it is if we iterate same valid df)
    # But safer to merge on post_id if available, or just index if consistent
    
    # Let's align by index since we just ran on 'valid' df
    gpt4o_df['gpt4o_mini_frame'] = baseline_df['llm_frame'].values # Assuming 1:1 match
    
    print("\n" + "=" * 70)
    print("📊 BENCHMARK RESULTS")
    print("=" * 70)
    
    # Accuracy
    acc_mini = (gpt4o_df['human_frame'] == gpt4o_df['gpt4o_mini_frame']).mean()
    acc_4o = (gpt4o_df['human_frame'] == gpt4o_df['gpt4o_frame']).mean()
    
    print(f"\n🏆 Accuracy Comparison:")
    print(f"  GPT-4o-mini : {acc_mini:.1%} (Legacy)")
    print(f"  GPT-4o      : {acc_4o:.1%} (New)")
    delta = acc_4o - acc_mini
    print(f"  Difference  : {delta:+.1%}")
    
    # Kappa
    kappa_mini = cohen_kappa_score(gpt4o_df['human_frame'], gpt4o_df['gpt4o_mini_frame'])
    kappa_4o = cohen_kappa_score(gpt4o_df['human_frame'], gpt4o_df['gpt4o_frame'])
    print(f"\n📈 Kappa Comparison:")
    print(f"  GPT-4o-mini : {kappa_mini:.3f}")
    print(f"  GPT-4o      : {kappa_4o:.3f}")
    
    # Per Frame F1
    print("\n🔍 Per-Frame F1 Score Delta:")
    report_mini = classification_report(gpt4o_df['human_frame'], gpt4o_df['gpt4o_mini_frame'], output_dict=True, zero_division=0)
    report_4o = classification_report(gpt4o_df['human_frame'], gpt4o_df['gpt4o_frame'], output_dict=True, zero_division=0)
    
    print(f"  {'Frame':12s} | {'Mini F1':8s} | {'4o F1':8s} | {'Delta':6s}")
    print("-" * 50)
    for frame in valid_frames:
        f1_mini = report_mini[frame]['f1-score']
        f1_4o = report_4o[frame]['f1-score']
        print(f"  {frame:12s} | {f1_mini:.3f}    | {f1_4o:.3f}    | {f1_4o - f1_mini:+.3f}")

    # Save
    gpt4o_df.to_csv('data/annotations/benchmark_gpt4o_results.csv', index=False, encoding='utf-8-sig')
    print("\n💾 Saved: data/annotations/benchmark_gpt4o_results.csv")

if __name__ == "__main__":
    main()
