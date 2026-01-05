"""
Benchmark Models: Open Source Models via Groq API vs GPT-4o-mini (Baseline).
Models to test:
1. Llama 3.3 70B (llama-3.3-70b-versatile)
2. Mixtral 8x7B (mixtral-8x7b-32768)
3. DeepSeek R1 Distill (deepseek-r1-distill-llama-70b)

Using the V2 (Best) Prompt.
"""

import pandas as pd
import json
import time
import os
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, classification_report

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Use OpenAI client compatible with Groq
client = OpenAI(
    api_key=groq_api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODELS = {
    "Llama-3.3-70B": "llama-3.3-70b-versatile",
    "Llama-3.1-8B": "llama-3.1-8b-instant"
}

# ==========================================
# EXACT V2 PROMPT (Best Performer)
# ==========================================
def classify_post_v2(title: str, body: str = "", model_id: str = "") -> dict:
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
            model=model_id,
            messages=[
                {"role": "system", "content": "You are a political science researcher analyzing media framing of international relations. Apply the Critical Classification Rules FIRST before classifying."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
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
        print(f"Error with {model_id}: {e}")
        return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}


def main():
    print("=" * 80)
    print("🧪 GROQ MODEL BENCHMARK: Llama 3.3, Mixtral, DeepSeek vs Baseline")
    print("=" * 80)
    
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
    
    # ---------------------------------------------------------
    # TEST MODE DISABLED - RUNNING FULL BENCHMARK
    # ---------------------------------------------------------
    # total = 10 
    # print(f"⚠️ TEST MODE ACTIVE: Processing only {total} samples per model.")
    # ---------------------------------------------------------
    
    # 2. Check for Baseline (GPT-4o-mini)
    baseline_path = 'data/annotations/llm_improved_v2_classification.csv'
    if os.path.exists(baseline_path):
        print(f"✅ Loaded Baseline (GPT-4o-mini) results")
        baseline_df = pd.read_csv(baseline_path)
    else:
        print("⚠️ Baseline file not found! Continuing without comparison.")
        baseline_df = None

    all_results = valid[['sample_id', 'post_id', 'title', 'final_frame_clean']].copy()
    all_results.rename(columns={'final_frame_clean': 'human_frame'}, inplace=True)
    
    if baseline_df is not None:
        all_results['gpt4o_mini_frame'] = baseline_df['llm_frame'].values

    for model_name, model_id in MODELS.items():
        print(f"\n🚀 Running {model_name} ({model_id})...")
        model_frames = []
        
        for idx in range(total):
            row = valid.iloc[idx]
            result = classify_post_v2(row['title'], str(row.get('text', '')), model_id)
            model_frames.append(result.get('frame', 'NEUTRAL'))
            
            if (idx + 1) % 50 == 0:
                print(f"   {idx+1}/{total}...", end="\r")
            
            # Groq rate limits can be strict on free tier
            time.sleep(0.5) 
            
        all_results[f'{model_name}_frame'] = model_frames
        
        # Immediate metrics for this model
        acc = (all_results['human_frame'] == all_results[f'{model_name}_frame']).mean()
        kappa = cohen_kappa_score(all_results['human_frame'], all_results[f'{model_name}_frame'])
        print(f"\n✅ {model_name} Finished | Accuracy: {acc:.1%} | Kappa: {kappa:.3f}")
        
        # Save incrementally
        all_results.to_csv('data/annotations/benchmark_groq_results.csv', index=False, encoding='utf-8-sig')
        print(f"📦 Progress saved for {model_name}")

    # 4. Final Comparison Table
    print("\n" + "=" * 80)
    print("📊 FINAL BENCHMARK LEADERBOARD")
    print("=" * 80)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Kappa':<8} | {'Delta (Acc)'}")
    print("-" * 65)
    
    # Baseline
    if baseline_df is not None:
        base_acc = (all_results['human_frame'] == all_results['gpt4o_mini_frame']).mean()
        base_kappa = cohen_kappa_score(all_results['human_frame'], all_results['gpt4o_mini_frame'])
        print(f"{'GPT-4o-mini (Base)':<20} | {base_acc:.1%}    | {base_kappa:.3f}    | -")
    else:
        base_acc = 0

    # New Models
    for model_name in MODELS.keys():
        acc = (all_results['human_frame'] == all_results[f'{model_name}_frame']).mean()
        kappa = cohen_kappa_score(all_results['human_frame'], all_results[f'{model_name}_frame'])
        delta = acc - base_acc
        print(f"{model_name:<20} | {acc:.1%}    | {kappa:.3f}    | {delta:+.1%}")

    # Save
    all_results.to_csv('data/annotations/benchmark_groq_results.csv', index=False, encoding='utf-8-sig')
    print("\n💾 Saved: data/annotations/benchmark_groq_results.csv")

if __name__ == "__main__":
    main()
