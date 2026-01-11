"""
Apply framing classification to collected comments using GPT-4o V2 improved prompt.
EXACT COPY of the prompt from reclassify_with_improved_prompt_v2.py
"""

import pandas as pd
import json
import time
import os
import sys
from openai import OpenAI
from tqdm import tqdm

# Configuration
COMMENT_FILES = {
    'nk': 'data/processed/nk_comments_top3_final.csv',
    'china': 'data/control/china_comments_top3_final.csv',
    'iran': 'data/control/iran_comments_top3_final.csv',
    'russia': 'data/control/russia_comments_top3_final.csv'
}

OUTPUT_SUFFIX = '_framing.csv'

# API Key
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    client = OpenAI(api_key=api_key)
else:
    print("ERROR: Please set OPENAI_API_KEY environment variable")
    sys.exit(1)


def classify_text(text: str) -> dict:
    """Classify a single text using the EXACT V2 IMPROVED prompt."""
    
    # Truncate if too long
    text_content = str(text)[:500] if text else 'N/A'
    
    # EXACT COPY OF V2 PROMPT (lines 30-136 from reclassify_with_improved_prompt_v2.py)
    prompt = f"""You are an international relations researcher. Classify the following Reddit comment into ONE of 5 framing categories.

## ⚠️ Critical Classification Rules (Apply First!)

### Rule 1: No Action = NEUTRAL
If the comment is a **question, hypothesis, speculation, or factual report** without explicit government action, classify as **NEUTRAL**.
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

## Comment
{text_content}

## Response Format (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "One sentence explaining classification rationale"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political science researcher analyzing media framing of international relations. Apply the Critical Classification Rules FIRST before classifying."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=200,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        
        # Validate frame
        valid_frames = ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN']
        if result.get('frame') not in valid_frames:
            result['frame'] = 'NEUTRAL'
            result['confidence'] = 0.5
            
        return result
        
    except Exception as e:
        return {
            "frame": "NEUTRAL",
            "confidence": 0.5,
            "reason": f"Error: {str(e)}"
        }


def process_topic(topic: str, path: str):
    """Process all comments for a single topic."""
    print(f"\nProcessing {topic.upper()}...")
    
    if not os.path.exists(path):
        print(f"  File not found: {path}")
        return
    
    df = pd.read_csv(path, low_memory=False)
    print(f"  Loaded {len(df):,} comments")
    
    # Filter out removed/deleted
    valid = df[~df['body'].astype(str).str.contains(r'\[removed\]|\[deleted\]', case=False, na=False, regex=True)]
    valid = valid[valid['body'].astype(str).str.len() > 20]
    print(f"  Valid comments: {len(valid):,}")
    
    # Check for existing output (for resumption)
    output_path = path.replace('.csv', OUTPUT_SUFFIX)
    start_idx = 0
    results = []
    
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        start_idx = len(existing)
        results = existing.to_dict('records')
        print(f"  Resuming from index {start_idx}")
    
    # Process remaining
    for idx in tqdm(range(start_idx, len(valid)), desc=f"Classifying {topic}"):
        row = valid.iloc[idx]
        text = str(row.get('body', ''))
        
        result = classify_text(text)
        
        results.append({
            'comment_id': row.get('id', ''),
            'parent_post_id': row.get('parent_post_id', ''),
            'body': text[:200],
            'frame': result.get('frame', 'NEUTRAL'),
            'confidence': result.get('confidence', 0.5),
            'reason': result.get('reason', '')
        })
        
        # Save every 100 comments
        if (idx + 1) % 100 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
        
        # Rate limiting (0.15s = ~6.6 requests/sec, safe for tier 1)
        time.sleep(0.15)
    
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"  ✅ Saved to: {output_path}")
    
    # Summary
    print(f"\n  Frame distribution:")
    print(results_df['frame'].value_counts())


def main():
    print("=" * 60)
    print("FRAMING CLASSIFICATION FOR COLLECTED COMMENTS (V2 EXACT PROMPT)")
    print("=" * 60)
    
    for topic, path in COMMENT_FILES.items():
        process_topic(topic, path)
    
    print("\n" + "=" * 60)
    print("✅ Framing classification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
