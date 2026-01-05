"""
Re-classify human annotation samples with refined framing prompt.
Uses GPT-4o-mini with comprehensive classification criteria.
Final Version - January 2025
"""

import pandas as pd
import json
import time
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix
import numpy as np

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)


def classify_post(title: str, body: str = "") -> dict:
    """Classify a single post using the refined prompt."""
    text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"
    
    prompt = f"""You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## Classification Principles

### 1. Actor Identification
- Identify whether the actor is a **state, government, or state representative**
- Actions by corporations or private individuals should NOT be interpreted as inter-state relations

### 2. Action Type Assessment
- Determine if the action is military, diplomatic, or economic
- Consider both the purpose and outcome of the action

---

## Classification Criteria (In Priority Order)

### THREAT (Military Tension/Conflict)
**Actions that increase the possibility of inter-state armed conflict**

Include:
- Military actions: missile launches, nuclear tests, military exercises, shows of force, cyberattacks
- Arms buildup: weapons sales, arms provision, military equipment deployment
- Military threats: war threats, attack warnings, military retaliation statements
- Military cooperation: joint military exercises, strengthening military alliances (increases regional tension)
- Military actions related to territorial/maritime disputes
- Designation as terrorist state/organization
- Ultimatum-style warnings with **no possibility of dialogue**

⚠️ **Critical Exception:**
- Arms deals appear economic but are classified as **THREAT**
  - Example: "Trump signs $110B arms deal with Saudi Arabia" → THREAT

---

### DIPLOMACY (Diplomatic Interaction)
**Relationship adjustment through dialogue and negotiation**

Include:
- Summit meetings, diplomatic negotiations, bilateral/multilateral talks
- Agreements, treaties, accord signings
- Attempts to improve/normalize relations
- Sanctions relief or easing
- **Non-military** criticism/condemnation of other countries by states
- Warnings or demands where dialogue possibility remains
- Even if a summit fails, focus on **the summit itself** → DIPLOMACY

**Decision Criteria:**
- Pressure through verbal means without direct use of force → DIPLOMACY
- Possibility of future dialogue exists → DIPLOMACY

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
- If text contains "sanctions" but **another topic is central**, classify accordingly
  - Example: Diplomatic statement criticizing sanctions → Context-dependent, may be DIPLOMACY

---

### HUMANITARIAN (Humanitarian Issues)
**Human rights and humanitarian matters**

Include:
- Human rights violations, oppression (targeting civilians within a country)
- Refugee issues
- Humanitarian assistance/aid
- War crimes, genocide

---

### NEUTRAL (Neutral Information)
**Cases not fitting specific frames**

Include:
- Simple factual reporting, analysis, information delivery
- **Domestic politics** of a single country (party conflicts, government criticism, etc.)
- Complex cases where priority determination is difficult

---

## Classification Decision Flow

```
1. Military action/weapons-related? → THREAT
2. Diplomatic dialogue/negotiation-related? → DIPLOMACY  
3. Economic sanctions/trade-related? → ECONOMIC
4. Human rights/refugees-related? → HUMANITARIAN
5. None of the above or undeterminable → NEUTRAL
```

---

## Post
{text}

## Response Format (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "One sentence explaining classification rationale"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political science researcher analyzing media framing of international relations."},
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


def main():
    print("=" * 70)
    print("📊 LLM vs Human Framing Agreement Analysis")
    print("=" * 70)
    
    # Load all three annotation files
    batch_pilot = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_pilot.csv')
    batch_1 = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_1.csv')
    batch_2 = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_2.csv')
    
    print(f"\n📁 Loaded files:")
    print(f"  - batch_pilot: {len(batch_pilot)} rows")
    print(f"  - batch_1: {len(batch_1)} rows")
    print(f"  - batch_2: {len(batch_2)} rows")
    
    # Combine all batches
    combined = pd.concat([batch_pilot, batch_1, batch_2], ignore_index=True)
    
    # Clean and filter
    combined['final_frame_clean'] = combined['final_frame'].astype(str).str.strip().str.upper()
    
    # Filter: only valid frames (exclude empty, nan, unmatched)
    valid_frames = ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN']
    valid = combined[combined['final_frame_clean'].isin(valid_frames)]
    valid = valid.dropna(subset=['title'])
    valid = valid[valid['title'].str.len() > 0]
    
    total = len(valid)
    print(f"\n✅ Valid samples for analysis: {total}")
    print(f"📊 Human annotation distribution:")
    print(valid['final_frame_clean'].value_counts())
    print()
    
    results = []
    
    # Manual progress display (instead of tqdm)
    for idx in range(total):
        row = valid.iloc[idx]
        title = str(row['title'])
        body = str(row.get('text', '')) if pd.notna(row.get('text')) else ''
        
        result = classify_post(title, body)
        
        results.append({
            'sample_id': row.get('sample_id', idx),
            'post_id': row.get('post_id', ''),
            'title': title[:100],  # Truncate for display
            'human_frame': row['final_frame_clean'],
            'llm_frame': result.get('frame', 'NEUTRAL'),
            'llm_confidence': result.get('confidence', 0.5),
            'llm_reason': result.get('reason', '')
        })
        
        # Progress display every 10 samples
        if (idx + 1) % 10 == 0 or idx == total - 1:
            pct = (idx + 1) / total * 100
            print(f"  Progress: {idx + 1}/{total} ({pct:.1f}%)", flush=True)
        
        # Rate limiting
        time.sleep(0.15)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # ========== Calculate Metrics ==========
    print("\n" + "=" * 70)
    print("📊 ANALYSIS RESULTS")
    print("=" * 70)
    
    # Overall accuracy
    matches = results_df['human_frame'] == results_df['llm_frame']
    accuracy = matches.sum() / len(results_df)
    print(f"\n✅ Overall Accuracy: {accuracy:.1%} ({matches.sum()}/{len(results_df)})")
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(results_df['human_frame'], results_df['llm_frame'])
    print(f"📈 Cohen's Kappa: {kappa:.3f}")
    
    if kappa < 0.2:
        kappa_interp = "Poor agreement"
    elif kappa < 0.4:
        kappa_interp = "Fair agreement"
    elif kappa < 0.6:
        kappa_interp = "Moderate agreement"
    elif kappa < 0.8:
        kappa_interp = "Substantial agreement"
    else:
        kappa_interp = "Almost perfect agreement"
    print(f"   Interpretation: {kappa_interp}")
    
    # Per-class metrics
    print("\n📊 Per-Frame Results:")
    print("-" * 60)
    for frame in valid_frames:
        human_count = (results_df['human_frame'] == frame).sum()
        llm_count = (results_df['llm_frame'] == frame).sum()
        
        # True positives (both agree on this frame)
        tp = ((results_df['human_frame'] == frame) & (results_df['llm_frame'] == frame)).sum()
        
        recall = tp / human_count if human_count > 0 else 0
        precision = tp / llm_count if llm_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  {frame:12s}: Human={human_count:3d}, LLM={llm_count:3d} | "
              f"Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")
    
    # Confusion Matrix
    print("\n📊 Confusion Matrix:")
    print("-" * 60)
    cm = confusion_matrix(results_df['human_frame'], results_df['llm_frame'], labels=valid_frames)
    cm_df = pd.DataFrame(cm, index=valid_frames, columns=valid_frames)
    print("         " + "  ".join(f"{f[:6]:>6s}" for f in valid_frames))
    for i, frame in enumerate(valid_frames):
        row_str = " ".join(f"{cm[i, j]:6d}" for j in range(len(valid_frames)))
        print(f"{frame[:8]:8s} {row_str}")
    
    # Classification report
    print("\n📊 Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(results_df['human_frame'], results_df['llm_frame'], 
                                labels=valid_frames, zero_division=0))
    
    # Save results
    output_path = 'data/annotations/llm_revised_classification_final.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Results saved to: {output_path}")
    
    # Show some disagreement examples
    disagreements = results_df[~matches]
    if len(disagreements) > 0:
        print(f"\n⚠️ Disagreement Examples (showing first 10):")
        print("-" * 60)
        for _, row in disagreements.head(10).iterrows():
            print(f"  Title: {row['title'][:60]}...")
            print(f"    Human: {row['human_frame']} | LLM: {row['llm_frame']}")
            print(f"    Reason: {row['llm_reason'][:80]}...")
            print()
    
    print("=" * 70)
    print("✅ Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
