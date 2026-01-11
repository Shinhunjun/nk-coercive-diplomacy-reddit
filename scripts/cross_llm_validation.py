"""
Cross-LLM Validation for Framing Classification
Validates GPT-4o-mini classifications against alternative LLMs
"""

import pandas as pd
from groq import Groq
import json
import time
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
import numpy as np

# Configuration
GROQ_API_KEY = "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"
SAMPLE_SIZE = 500
RANDOM_SEED = 42

# V2 Framing Prompt
V2_PROMPT = """You are an expert political scientist analyzing framing in international relations discourse.

**FRAMING CODEBOOK**

**THREAT**: Emphasizes military danger, security risk, or adversarial posture
- Keywords: nuclear weapons, missile tests, military threat, regime collapse, attack, sanctions
- Example: "North Korea's nuclear program poses an existential threat to regional stability"

**DIPLOMACY**: Emphasizes negotiation, dialogue, or peaceful resolution
- Keywords: talks, summit, negotiation, peace process, dialogue, agreement, denuclearization
- Example: "The Singapore Summit represents a historic opportunity for diplomatic engagement"

**NEUTRAL**: Factual reporting without clear threat or diplomacy emphasis
- Example: "North Korea's economy contracted 3.5% last year"

**HUMANITARIAN**: Emphasizes human rights, suffering, or humanitarian conditions
- Keywords: human rights violations, refugees, famine, prison camps
- Example: "International organizations report widespread malnutrition in North Korea"

**ECONOMIC**: Emphasizes trade, sanctions, or economic relations
- Keywords: sanctions relief, trade agreements, economic cooperation, investment
- Example: "New sanctions target North Korea's coal exports"

**INSTRUCTIONS**
1. Classify the post into ONE category
2. Return ONLY valid JSON: {"frame": "CATEGORY", "confidence": 0.0-1.0, "reason": "brief explanation"}
3. Use exact category names: THREAT, DIPLOMACY, NEUTRAL, HUMANITARIAN, or ECONOMIC"""


import re

def clean_json_output(text):
    """Extract JSON object from text using regex"""
    try:
        # Find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            return text[start:end+1]
        return text
    except:
        return text

def classify_with_groq(text, model, client, max_retries=3):
    """Classify text using Groq model with retries"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": V2_PROMPT},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            cleaned_content = clean_json_output(content)
            
            try:
                result = json.loads(cleaned_content)
                return result.get('frame'), result.get('confidence')
            except json.JSONDecodeError:
                # If strict JSON parsing fails, try a looser regex approach for specific keys
                frame_match = re.search(r'"frame"\s*:\s*"([^"]+)"', content)
                conf_match = re.search(r'"confidence"\s*:\s*([\d\.]+)', content)
                
                if frame_match:
                    frame = frame_match.group(1)
                    conf = float(conf_match.group(1)) if conf_match else 0.0
                    return frame, conf
                
                if attempt == max_retries - 1:
                    print(f"\nFailed to parse JSON for: {text[:30]}... Output: {content[:50]}...")
            
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\nAPI Error: {e}")
            time.sleep(1 * (attempt + 1))  # Exponential backoff
            
    return None, None


def main():
    print("🔬 Cross-LLM Validation for Framing Classification")
    print("="*70)
    
    # Load NK posts with existing GPT-4o-mini classifications
    print("\n📂 Loading data...")
    nk_posts = pd.read_csv('../data/processed/nk_posts_framing.csv')
    print(f"   Total posts: {len(nk_posts)}")
    
    # Random sample
    print(f"\n🎲 Sampling {SAMPLE_SIZE} posts...")
    sample = nk_posts.sample(min(SAMPLE_SIZE, len(nk_posts)), random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"   Sample size: {len(sample)}")
    
    # Setup Groq client
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    # Models to test
    models = [
        ("Llama 3.3 70B", "llama-3.3-70b-versatile"),
        ("Llama 3.1 8B", "llama-3.1-8b-instant")
    ]
    
    results = {
        'post_id': sample['id'].tolist(),
        'title': sample['title'].tolist(),
        'gpt4o_mini': sample['frame'].tolist()
    }
    
    # Setup OpenAI client
    from openai import OpenAI
    import os
    from dotenv import load_dotenv
    
    # Load .env explicitly
    load_dotenv('../.env')
    openai_api_key = os.getenv('GRAPHRAG_API_KEY') or os.getenv('OPENAI_API_KEY')
    
    if not openai_api_key:
        print("❌ Error: OPENAI_API_KEY not found in .env")
        return

    openai_client = OpenAI(api_key=openai_api_key)

    # Models to test
    models = [
        ("GPT-4o", "gpt-4o"),
        ("Llama 3.3 70B", "llama-3.3-70b-versatile"),
        ("Llama 3.1 8B", "llama-3.1-8b-instant")
    ] 

    # Classify with each model
    for model_name, model_id in models:
        print(f"\n🤖 Classifying with {model_name}...")
        print("-"*70)
        
        frames = []
        confidences = []
        
        # Check partial
        start_idx = 0
        partial_path = f'../data/results/partial_{model_id}.csv'
        
        try:
            if os.path.exists(partial_path):
                partial_df = pd.read_csv(partial_path)
                print(f"   Resuming from {len(partial_df)} processed posts...")
                frames = partial_df['frame'].tolist()
                confidences = partial_df['confidence'].tolist()
                start_idx = len(frames)
            else:
                print("   Starting fresh...")
        except Exception as e:
            print(f"   Error checking partial: {e}")
            start_idx = 0
            
        print(f"   Loop range: {start_idx} to {len(sample)}")
            
        for idx in tqdm(range(start_idx, len(sample)), initial=start_idx, total=len(sample)):
            row = sample.iloc[idx]
            
            # Determine client (Groq vs OpenAI)
            if "gpt" in model_id:
                try:
                    response = openai_client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": V2_PROMPT},
                            {"role": "user", "content": row['title']}
                        ],
                        temperature=0.0,
                        max_tokens=200
                    )
                    content = response.choices[0].message.content
                    cleaned = clean_json_output(content)
                    result = json.loads(cleaned)
                    frame, conf = result.get('frame'), result.get('confidence')
                except Exception as e:
                    print(f"Error: {e}")
                    frame, conf = None, None
            else:
                frame, conf = classify_with_groq(row['title'], model_id, groq_client)
                
            frames.append(frame)
            confidences.append(conf)
            
            # Save partial checkpoint
            if (idx + 1) % 50 == 0:
                temp_df = pd.DataFrame({
                    'frame': frames,
                    'confidence': confidences
                })
                temp_df.to_csv(partial_path, index=False)
            
            # Rate limiting
            time.sleep(0.1)
        
    
    # Save results
    results_df = pd.DataFrame(results)
    output_path = '../data/results/cross_llm_validation.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    
    # Calculate agreement metrics
    print("\n" + "="*70)
    print("📊 AGREEMENT METRICS")
    print("="*70)
    
    gpt4o_frames = results_df['gpt4o_mini'].tolist()
    
    for model_name, _ in models:
        model_frames = results_df[f'{model_name}_frame'].tolist()
        
        # Remove None values
        valid_pairs = [(g, m) for g, m in zip(gpt4o_frames, model_frames) if m is not None]
        if not valid_pairs:
            continue
            
        gpt_valid = [g for g, m in valid_pairs]
        model_valid = [m for g, m in valid_pairs]
        
        # Calculate metrics
        accuracy = accuracy_score(gpt_valid, model_valid)
        kappa = cohen_kappa_score(gpt_valid, model_valid)
        
        print(f"\n{model_name} vs GPT-4o-mini:")
        print(f"  Agreement: {accuracy:.1%}")
        print(f"  Cohen's κ: {kappa:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(gpt_valid, model_valid, 
                             labels=['THREAT', 'DIPLOMACY', 'NEUTRAL', 'HUMANITARIAN', 'ECONOMIC'])
        print(f"  Confusion Matrix:")
        print(cm)
    
    print("\n✅ Cross-LLM validation complete!")


if __name__ == "__main__":
    main()
