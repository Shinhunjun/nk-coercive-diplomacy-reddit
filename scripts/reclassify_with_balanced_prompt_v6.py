"""
Re-classify human annotation samples with BALANCED framing prompt.
Version 6 - Balanced between V2 (conservative) and V5 (aggressive).
Goal: Improve THREAT/DIPLOMACY recall while maintaining NEUTRAL precision.
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
    """Classify a single post using the BALANCED v6 prompt."""
    text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"
    
    prompt = f"""You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## Classification Priority Rules

### 1. THREAT - Military Actions & Strategy
Classify as THREAT if the post involves:
- **Military actions**: missile launches, nuclear tests, military exercises, submarine activity
- **Military strategy discussions**: grand strategy, military alliances, intelligence meetings
- **Arms deals or weapons deployment**
- **Nuclear agreement breaches or escalation**

Examples:
- "North Korea fires 2 projectiles" → THREAT
- "Iran's grand strategy tests U.S." → THREAT
- "Russian Submarine Activity Highest" → THREAT

### 2. DIPLOMACY - Diplomatic Events & Sanctions
Classify as DIPLOMACY if the post involves:
- **Summits, talks, negotiations** (including analysis/commentary about them)
- **Sanctions** (relief, imposition, or statements about them)
- **Treaties and agreements**
- **Verbal criticism or accusations between states** (without military action)

Examples:
- "4 winners and losers from Trump-Kim summit" → DIPLOMACY (summit analysis)
- "Trump will remove North Korea sanctions" → DIPLOMACY
- "NATO accuses Russia of blocking observation" → DIPLOMACY

### 3. NEUTRAL - Default Category
Classify as NEUTRAL if:
- **Corporate actions** (Apple, Siemens, etc.)
- **Domestic politics** without international implications
- **Pure factual reporting** without clear framing toward THREAT/DIPLOMACY
- **Questions or speculation** without military/diplomatic substance

Examples:
- "Apple under fire for labelling Crimea" → NEUTRAL (corporate)
- "Democrats criticize Trump" → NEUTRAL (domestic)

### 4. ECONOMIC - Pure Economic Issues
Classify as ECONOMIC only for:
- **Trade tariffs and restrictions** (NOT sanctions)
- **Economic cooperation without political context**

### 5. HUMANITARIAN - Individual Harm
Classify as HUMANITARIAN for:
- **Human rights violations**
- **Harm to specific individuals** (refugees, protesters, defectors)

---

## Decision Guide
1. Does it involve military action/strategy/weapons? → THREAT
2. Does it involve diplomatic events/sanctions/interstate criticism? → DIPLOMACY
3. Is it corporate/domestic/pure factual? → NEUTRAL
4. Is it purely economic (not sanctions)? → ECONOMIC
5. Does it involve individual human rights harm? → HUMANITARIAN

---

## Post
{text}

## Response (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "Brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political science researcher. Follow the decision guide. Be balanced: don't over-classify as THREAT/DIPLOMACY, but also don't miss obvious cases."},
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
    print("📊 LLM Framing Analysis (BALANCED PROMPT v6)")
    print("=" * 70)
    
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
    print("📊 RESULTS (v6 - BALANCED)")
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
    
    output_path = 'data/annotations/llm_balanced_v6_classification.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: {output_path}")


if __name__ == "__main__":
    main()
