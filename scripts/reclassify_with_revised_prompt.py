"""
Re-classify human annotation samples with revised framing prompt.
Uses GPT-4o-mini with conflict-potential-first criteria.
"""

import pandas as pd
import json
import time
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)

def classify_post(title: str, body: str = "") -> dict:
    """Classify a single post using the revised prompt."""
    text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"
    
    prompt = f"""You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

## Classification Criteria (Priority Order)

### Step 1: THREAT First
First, determine if this post describes actions that **increase or have potential to increase inter-state conflict**.

→ **THREAT** if:
- Military actions: missile launches, nuclear tests, military exercises, shows of force
- Arms buildup: weapons sales, arms deals, military deployments
- Military threats: war threats, attack warnings, military retaliation

⚠️ Note: Arms deals may look economic but **increase conflict potential → THREAT**
Example: "Trump signs $110B arms deal with Saudi Arabia" → THREAT (arms buildup)

### Step 2: If NOT THREAT, classify as below
Only if conflict potential is LOW or NONE:

- **DIPLOMACY**: Negotiations, summits, dialogue, agreements, peace efforts, improving relations
- **ECONOMIC**: Economic sanctions, trade measures, financial restrictions (NOT arms deals)
- **HUMANITARIAN**: Human rights issues, refugees, humanitarian aid, **civilian suppression/crackdowns within a country**
- **NEUTRAL**: Factual reporting, analysis, **domestic politics (party criticism, internal government affairs)**

## Post
{text}

## Response Format (JSON)
{{"frame": "CATEGORY", "confidence": 0.0-1.0, "reason": "Brief explanation including conflict potential assessment"}}"""

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
    # Load human annotation files
    batch_pilot = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_pilot.csv')
    batch_1 = pd.read_csv('data/annotations/framing_human_annotation_Moon - batch_1.csv')
    
    # Combine
    combined = pd.concat([batch_pilot, batch_1], ignore_index=True)
    
    # Filter valid entries (exclude unmatched/empty)
    combined['final_frame_clean'] = combined['final_frame'].str.strip().str.upper()
    valid = combined[~combined['final_frame_clean'].str.contains('UNMATCHED', case=False, na=True)]
    valid = valid.dropna(subset=['title'])
    
    print(f"📊 Total samples to classify: {len(valid)}")
    print("=" * 60)
    
    results = []
    
    for idx in tqdm(range(len(valid)), desc="Classifying with revised prompt"):
        row = valid.iloc[idx]
        title = str(row['title'])
        body = str(row.get('text', '')) if pd.notna(row.get('text')) else ''
        
        result = classify_post(title, body)
        
        results.append({
            'post_id': row['post_id'],
            'title': title,
            'human_frame': row['final_frame_clean'],
            'llm_frame_revised': result.get('frame', 'NEUTRAL'),
            'llm_confidence_revised': result.get('confidence', 0.5),
            'llm_reason_revised': result.get('reason', '')
        })
        
        # Rate limiting
        time.sleep(0.2)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate agreement
    matches = results_df['human_frame'] == results_df['llm_frame_revised']
    accuracy = matches.sum() / len(results_df)
    
    print("\n" + "=" * 60)
    print(f"✅ Classification complete!")
    print(f"📊 Overall Accuracy: {accuracy:.1%} ({matches.sum()}/{len(results_df)})")
    
    # Save results
    output_path = 'data/annotations/llm_revised_classification.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 Saved to: {output_path}")
    
    # Show per-class metrics
    print("\n📊 Per-Frame Results:")
    print("-" * 40)
    for frame in ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN']:
        human_count = (results_df['human_frame'] == frame).sum()
        llm_count = (results_df['llm_frame_revised'] == frame).sum()
        frame_matches = ((results_df['human_frame'] == frame) & (results_df['llm_frame_revised'] == frame)).sum()
        if human_count > 0:
            recall = frame_matches / human_count
            print(f"  {frame:12s}: Human={human_count:3d}, LLM={llm_count:3d}, Recall={recall:.1%}")

if __name__ == "__main__":
    main()
