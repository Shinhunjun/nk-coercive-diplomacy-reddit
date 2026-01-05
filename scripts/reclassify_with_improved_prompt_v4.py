"""
Re-classify human annotation samples with IMPROVED framing prompt.
Version 4 - Refined based on V2 failure analysis (Over-neutralization).
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
    """Classify a single post using the IMPROVED v4 prompt."""
    text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"
    
    prompt = f"""You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## ⚠️ Critical Priority Rules (Apply in Order!)

### 1. Kinetic Military Action & Capabilities = THREAT (Highest Priority)
If **missile launches, nuclear tests, weapon deployments, or troop movements** are mentioned, classify as **THREAT**.
- Also include: Reports of "increasing military capabilities," "reactivating nuclear sites," or "submarine activity."

### 2. Military Strategy, Posture & Intelligence = THREAT
If the post discusses **Grand Strategy, War Planning, Intelligence Reports/Meetings, or Military Alliances**, classify as **THREAT**.
- Even if it's just a report or analysis (e.g., "Iran's grand strategy tests US," "Intel report claims X").
- *Correction from previous rules:* Do NOT classify these as NEUTRAL just because there is no "kinetic action." The discussion of military strategy itself is a THREAT frame.

### 3. Diplomatic Events & Analysis = DIPLOMACY
If the post discusses **summits, negotiations, treaties, or sanctions (relief/imposition)**, classify as **DIPLOMACY**.
- Include **analysis/commentary** on summits (e.g., "Winners of Trump-Kim summit").
- *Correction:* Do NOT classify summit analysis as NEUTRAL.

### 4. Military Threats vs Diplomatic Warnings
- **THREAT**: Explicit threats of destruction, war, or physical retaliation (e.g., "We will destroy X," "Mobilizing army," "Ultimatum").
- **DIPLOMACY**: Verbal warnings, condemnations, or urged restraint WITHOUT physical action (e.g., "China warns India," "Russia criticizes US").

### 5. Corporate & Domestic = NEUTRAL
- **Corporate Actions**: Actions by companies (Apple, Siemens) are **NEUTRAL** (unless state-directed cyberwarfare).
- **Domestic Politics**: Party politics, elections, or domestic scandals are **NEUTRAL**.

### 6. Default Fallback
- **NEUTRAL**: Simple factual reporting on non-strategic/non-military topics, questions without strategic implication, or general info.
- **HUMANITARIAN**: Specific harm to individuals (refugees, defectors, protesters).
- **ECONOMIC**: Trade deals, economic indicators (EXCLUDING sanctions, which are usually DIPLOMACY/ECONOMIC hybrid, but prioritize DIPLOMACY if political).

---

## Classification Categories

### THREAT
- Physical military actions (missiles, nukes, drills)
- Military strategy/posture discussions
- Intelligence reports on adversaries
- Grand strategy, military alliances
- Explicit threats of war/destruction

### DIPLOMACY
- Summits, talks, negotiations (and analysis of them)
- Sanctions (relief, imposition, discussion)
- Treaties, agreements
- Verbal warnings/condemnations between states

### ECONOMIC
- Trade tariffs, export/import restrictions
- General economic news (stock markets, trade deficits)
- *Note:* Sanctions often fit DIPLOMACY better if the context is political pressure.

### HUMANITARIAN
- Human rights abuses
- Harm to civilians/refugees/protesters

### NEUTRAL
- Corporate actions
- Domestic politics
- General factual reporting (non-strategic)

---

## Post
{text}

## Response (JSON only)
{{"frame": "THREAT|DIPLOMACY|ECONOMIC|HUMANITARIAN|NEUTRAL", "confidence": 0.0-1.0, "reason": "Brief explanation"}}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a political science researcher. Apply the Priority Rules strictly. Don't over-use NEUTRAL for strategic/diplomatic analysis."},
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
    print("📊 LLM Framing Analysis (IMPROVED PROMPT v4)")
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
    print("📊 RESULTS (v4)")
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
    
    # Save
    output_path = 'data/annotations/llm_improved_v4_classification.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: {output_path}")


if __name__ == "__main__":
    main()
