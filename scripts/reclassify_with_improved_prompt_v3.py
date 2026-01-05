"""
Re-classify human annotation samples with IMPROVED framing prompt.
Version 3 - Additional rules based on v2 error analysis.
January 2025
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


def classify_post(title: str, body: str = "") -> dict:
    """Classify a single post using the IMPROVED v3 prompt."""
    text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"
    
    prompt = f"""You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## ⚠️ Critical Classification Rules (Apply in Order!)

### Rule 1: Missile/Nuclear Actions = THREAT (Highest Priority)
If **missile launches, nuclear tests, or weapons deployment** are mentioned, classify as **THREAT** regardless of other context.
- Example: "North Korea fires 2 projectiles after offering talks" → THREAT (missile launch overrides dialogue mention)
- Example: "Iran activates advanced centrifuges" → THREAT

### Rule 2: Strategy/Military Planning = THREAT
If the post discusses **military strategy, war planning, or meetings of military/intelligence advisers**, classify as **THREAT**.
- Example: "Iran's grand strategy tests U.S. and its allies" → THREAT (strategy = threat)
- Example: "Trump's military advisers held unusual meeting at CIA on Iran" → THREAT

### Rule 3: Summit/Diplomatic Event = DIPLOMACY
If the post **analyzes or discusses a summit/diplomatic event**, classify as **DIPLOMACY**, even if it's commentary.
- Example: "4 winners and 4 losers from Trump-Kim summit" → DIPLOMACY (summit analysis)
- Example: "Trump will remove sanctions because he likes Kim" → DIPLOMACY (sanction relief)

### Rule 4: Corporate Actions = NEUTRAL
Actions by **corporations** (Apple, Siemens, etc.) are NOT state actions → **NEUTRAL**.
- Example: "Apple under fire for labelling Crimea as Russia" → NEUTRAL (corporate action)
- Example: "Siemens to press charges" → NEUTRAL (corporate action)

### Rule 5: No Explicit Government Action = NEUTRAL
If the post is a **question, hypothesis, or factual report** without explicit government action, classify as **NEUTRAL**.
- Exception: If military strategy or planning is discussed → THREAT (Rule 2)
- Example: "What if Ukraine and Russia go to war?" → NEUTRAL (question)

### Rule 6: Verbal Criticism/Warning = DIPLOMACY
If a state is **only verbally criticizing or warning** another state (not taking physical/military action), classify as **DIPLOMACY**.
- Example: "China warns India over military buildup" → DIPLOMACY (verbal warning)
- Exception: If the warning explicitly states "no dialogue possible" → THREAT

### Rule 7: Individual/Civilian Harm = HUMANITARIAN
If the harm is to **specific individuals** (protesters, defectors, refugees), classify as **HUMANITARIAN**.
- Example: "North Korea soldier shot while defecting" → HUMANITARIAN

### Rule 8: Conflicting Frames = NEUTRAL
When **DIPLOMACY and THREAT are equally present** and competing, classify as **NEUTRAL**.
- Exception: If missile/nuclear actions are mentioned → THREAT (Rule 1 overrides)

### Rule 9: Domestic Politics = NEUTRAL
**Commentary on domestic political issues**, even if mentioning foreign countries, is **NEUTRAL**.

---

## Classification Categories

### THREAT
- Physical military actions: missile launches, nuclear tests, military exercises
- Arms deals, weapons deployment
- Military strategy discussion, war planning
- Military/intelligence adviser meetings on adversaries

### DIPLOMACY
- Summit meetings, negotiations, talks
- Agreements, treaties
- Sanctions relief or easing
- Verbal criticism/warnings between states
- Summit analysis/commentary

### ECONOMIC
- Economic sanctions (imposition/strengthening)
- Trade measures (tariffs, restrictions)
- Sanctions evasion

### HUMANITARIAN
- Human rights violations
- Harm to individuals (protesters, defectors, civilians)
- Refugee issues

### NEUTRAL
- Factual reporting without government action
- Domestic politics
- Corporate actions
- Questions/hypotheticals
- Conflicting frames of equal weight

---

## Post
{text}

## Response (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "Brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political science researcher. Apply the Critical Classification Rules IN ORDER."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
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
    print("📊 LLM vs Human Framing Analysis (IMPROVED PROMPT v3)")
    print("=" * 70)
    
    # Load files
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
    print(f"\n✅ Valid samples: {total}")
    print(valid['final_frame_clean'].value_counts())
    print()
    
    results = []
    
    for idx in range(total):
        row = valid.iloc[idx]
        title = str(row['title'])
        body = str(row.get('text', '')) if pd.notna(row.get('text')) else ''
        
        result = classify_post(title, body)
        
        results.append({
            'sample_id': row.get('sample_id', idx),
            'post_id': row.get('post_id', ''),
            'title': title[:100],
            'human_frame': row['final_frame_clean'],
            'llm_frame': result.get('frame', 'NEUTRAL'),
            'llm_confidence': result.get('confidence', 0.5),
            'llm_reason': result.get('reason', '')
        })
        
        if (idx + 1) % 10 == 0 or idx == total - 1:
            print(f"  Progress: {idx + 1}/{total} ({(idx + 1) / total * 100:.1f}%)", flush=True)
        
        time.sleep(0.15)
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 70)
    print("📊 RESULTS (v3)")
    print("=" * 70)
    
    matches = results_df['human_frame'] == results_df['llm_frame']
    accuracy = matches.sum() / len(results_df)
    print(f"\n✅ Overall Accuracy: {accuracy:.1%} ({matches.sum()}/{len(results_df)})")
    
    kappa = cohen_kappa_score(results_df['human_frame'], results_df['llm_frame'])
    print(f"📈 Cohen's Kappa: {kappa:.3f}")
    
    print("\n📊 Per-Frame:")
    print("-" * 50)
    for frame in valid_frames:
        human_count = (results_df['human_frame'] == frame).sum()
        llm_count = (results_df['llm_frame'] == frame).sum()
        tp = ((results_df['human_frame'] == frame) & (results_df['llm_frame'] == frame)).sum()
        recall = tp / human_count if human_count > 0 else 0
        precision = tp / llm_count if llm_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"  {frame:12s}: Human={human_count:3d}, LLM={llm_count:3d} | P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(results_df['human_frame'], results_df['llm_frame'], labels=valid_frames)
    print("         " + "  ".join(f"{f[:6]:>6s}" for f in valid_frames))
    for i, frame in enumerate(valid_frames):
        row_str = " ".join(f"{cm[i, j]:6d}" for j in range(len(valid_frames)))
        print(f"{frame[:8]:8s} {row_str}")
    
    # Save
    output_path = 'data/annotations/llm_improved_v3_classification.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print("✅ Complete!")


if __name__ == "__main__":
    main()
