import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score
import os
import json
import time
from tqdm import tqdm
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import re

# Load API keys
load_dotenv('../.env')
groq_api_key = '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'
openai_api_key = os.getenv('GRAPHRAG_API_KEY') or os.getenv('OPENAI_API_KEY')

groq_client = Groq(api_key=groq_api_key)
openai_client = OpenAI(api_key=openai_api_key)

# V2 Prompt
V2_PROMPT = """You are an international relations researcher. Classify the following Reddit post into ONE of 5 framing categories.

### Critical Classification Rules (Apply First!)
1. **No Action = NEUTRAL**: If the post is a question, hypothesis, speculation, or factual report without explicit government action, classify as NEUTRAL.
2. **Verbal vs. Physical Actions**: If a state is only verbally criticizing or warning another (not taking physical/military action), classify as DIPLOMACY, not THREAT.
3. **Individual Harm = HUMANITARIAN**: If the harm is to specific individuals (protesters, defectors, refugees, civilians), classify as HUMANITARIAN, not THREAT.
4. **Conflicting Frames = NEUTRAL**: When DIPLOMACY and THREAT (or other frames) are equally present and competing, classify as NEUTRAL.
5. **Domestic Politics = NEUTRAL**: Commentary on domestic political issues, even if mentioning foreign countries, is NEUTRAL.

### Categories:
1. THREAT: Physical military actions (missile launches, nuclear tests, military exercises). Exclude: verbal warnings/criticism.
2. DIPLOMACY: Dialogue, negotiation, verbal pressure, summits, agreements. Includes verbal criticism between states.
3. ECONOMIC: Sanctions, trade measures, economic cooperation.
4. HUMANITARIAN: Human rights, refugees, aid, harm to individuals.
5. NEUTRAL: Factual reporting, analysis, questions, domestic politics, conflicting frames.

Return ONLY a JSON object: {"frame": "CATEGORY", "confidence": 0.0-1.0}"""

def clean_json_output(text):
    """Extract JSON from potentially messy LLM output"""
    match = re.search(r'\{[^{}]*\}', text)
    if match:
        return match.group(0)
    return text

def classify_with_groq(title, model_id, client):
    """Classify with Groq API"""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": V2_PROMPT},
                    {"role": "user", "content": title}
                ],
                temperature=0.0,
                max_tokens=200
            )
            content = response.choices[0].message.content
            cleaned = clean_json_output(content)
            result = json.loads(cleaned)
            return result.get('frame'), result.get('confidence')
        except Exception as e:
            if attempt < 2:
                time.sleep(1 * (attempt + 1))
            else:
                return None, None

def classify_with_openai(title, model_id, client):
    """Classify with OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": V2_PROMPT},
                {"role": "user", "content": title}
            ],
            temperature=0.0,
            max_tokens=200
        )
        content = response.choices[0].message.content
        cleaned = clean_json_output(content)
        result = json.loads(cleaned)
        return result.get('frame'), result.get('confidence')
    except Exception as e:
        return None, None

def main():
    print("="*70)
    print("📊 Human-vs-Model Benchmark")
    print("="*70)
    
    # Load all batch files
    batch_files = [
        '../data/annotations/framing - batch_1.csv',
        '../data/annotations/framing - batch_2.csv',
        '../data/annotations/framing - batch_pilot.csv'
    ]
    
    dfs = []
    for f in batch_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"Loaded {f}: {len(df)} rows")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📂 Combined: {len(combined)} rows")
    
    # Keep only needed columns
    combined = combined[['post_id', 'title', 'final_frame', 'v2_annotation']].copy()
    combined = combined.rename(columns={'final_frame': 'human', 'v2_annotation': 'gpt4o_mini'})
    
    # Models to run
    models = [
        ("GPT-4o", "gpt-4o", classify_with_openai, openai_client),
        ("Llama 3.3 70B", "llama-3.3-70b-versatile", classify_with_groq, groq_client),
        ("Llama 3.1 8B", "llama-3.1-8b-instant", classify_with_groq, groq_client)
    ]
    
    # Check for partial results
    output_path = '../data/results/human_benchmark_full.csv'
    if os.path.exists(output_path):
        print(f"\n📂 Loading existing partial results...")
        combined = pd.read_csv(output_path)
    
    for model_name, model_id, classify_fn, client in models:
        col_name = f'{model_name}'
        
        # Skip if already done
        if col_name in combined.columns and combined[col_name].notna().all():
            print(f"\n✅ {model_name} already completed. Skipping...")
            continue
        
        print(f"\n🤖 Classifying with {model_name}...")
        
        frames = []
        for idx, row in tqdm(combined.iterrows(), total=len(combined)):
            # Skip if already classified
            if col_name in combined.columns and pd.notna(combined.at[idx, col_name]):
                frames.append(combined.at[idx, col_name])
                continue
                
            frame, conf = classify_fn(row['title'], model_id, client)
            frames.append(frame)
            
            # Checkpoint every 50
            if (idx + 1) % 50 == 0:
                combined[col_name] = frames + [None] * (len(combined) - len(frames))
                combined.to_csv(output_path, index=False)
            
            time.sleep(0.1)
        
        combined[col_name] = frames
        combined.to_csv(output_path, index=False)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("📊 FINAL RESULTS: Model vs Human Agreement")
    print("="*70)
    
    results = []
    for model_name in ['gpt4o_mini', 'GPT-4o', 'Llama 3.3 70B', 'Llama 3.1 8B']:
        if model_name not in combined.columns:
            continue
            
        valid = combined.dropna(subset=['human', model_name])
        if len(valid) == 0:
            continue
            
        acc = accuracy_score(valid['human'], valid[model_name])
        kappa = cohen_kappa_score(valid['human'], valid[model_name])
        
        results.append({
            'model': model_name,
            'n': len(valid),
            'accuracy': f"{acc:.1%}",
            'kappa': f"{kappa:.3f}"
        })
        
        print(f"\n{model_name} (N={len(valid)}):")
        print(f"  Accuracy vs Human: {acc:.1%}")
        print(f"  Cohen's κ vs Human: {kappa:.3f}")
    
    # Save final
    combined.to_csv(output_path, index=False)
    print(f"\n💾 Saved to {output_path}")
    
    # Save metrics
    pd.DataFrame(results).to_csv('../data/results/human_benchmark_metrics.csv', index=False)
    print(f"📊 Metrics saved to human_benchmark_metrics.csv")

if __name__ == "__main__":
    main()
