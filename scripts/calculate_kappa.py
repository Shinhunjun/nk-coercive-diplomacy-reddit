"""
Calculate Cohen's Kappa: Human vs LLM (Original Prompt and V2 Prompt)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
import os

# Load human annotation data
print("="*80)
print("📊 COHEN'S KAPPA: HUMAN vs LLM")
print("="*80)

# Load combined data (already has human and original LLM)
combined = pd.read_csv('data/annotations/framing_combined_human_llm.csv')
print(f"\nLoaded {len(combined)} annotated samples")

# Clean frames - handle whitespace and case
combined['human_frame'] = combined['human_final_frame'].str.strip().str.upper()
combined['llm_original'] = combined['llm_annotation'].str.strip().str.upper()

# Filter valid frames
valid_frames = ['THREAT', 'DIPLOMACY', 'ECONOMIC', 'HUMANITARIAN', 'NEUTRAL']
mask = (combined['human_frame'].isin(valid_frames)) & (combined['llm_original'].isin(valid_frames))
combined = combined[mask]
print(f"Valid samples: {len(combined)}")

# ===== ORIGINAL PROMPT ANALYSIS =====
print("\n" + "="*80)
print("🔵 ORIGINAL PROMPT - Cohen's Kappa")
print("="*80)

kappa_original = cohen_kappa_score(combined['human_frame'], combined['llm_original'])
accuracy_original = accuracy_score(combined['human_frame'], combined['llm_original'])

print(f"\nCohen's Kappa: {kappa_original:.4f}")
print(f"Accuracy:      {accuracy_original:.4f}")

print("\nClassification Report:")
print(classification_report(combined['human_frame'], combined['llm_original'], 
                            labels=valid_frames, zero_division=0))

# ===== V2 PROMPT ANALYSIS =====
# Need to check if V2 classifications exist for these samples
print("\n" + "="*80)
print("🟠 V2 PROMPT - Cohen's Kappa")
print("="*80)

# Load V2 results for NK (most samples are NK)
v2_files = {
    'nk': 'data/results/final_framing_v2/nk_framing_v2.csv',
    'china': 'data/results/final_framing_v2/china_framing_v2.csv',
    'iran': 'data/results/final_framing_v2/iran_framing_v2.csv',
    'russia': 'data/results/final_framing_v2/russia_framing_v2.csv'
}

v2_all = []
for country, path in v2_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['country'] = country.upper()
        v2_all.append(df)

v2_df = pd.concat(v2_all, ignore_index=True)
v2_df = v2_df.rename(columns={'frame': 'llm_v2'})
v2_df['llm_v2'] = v2_df['llm_v2'].str.strip().str.upper()

# Merge with human annotations
merged = pd.merge(combined, v2_df[['id', 'llm_v2']], 
                  left_on='post_id', right_on='id', how='inner')

print(f"\nMatched samples with V2 classifications: {len(merged)}")

if len(merged) > 0:
    # Filter valid V2 frames
    mask_v2 = merged['llm_v2'].isin(valid_frames)
    merged = merged[mask_v2]
    print(f"Valid V2 samples: {len(merged)}")
    
    kappa_v2 = cohen_kappa_score(merged['human_frame'], merged['llm_v2'])
    accuracy_v2 = accuracy_score(merged['human_frame'], merged['llm_v2'])
    
    print(f"\nCohen's Kappa: {kappa_v2:.4f}")
    print(f"Accuracy:      {accuracy_v2:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(merged['human_frame'], merged['llm_v2'], 
                                labels=valid_frames, zero_division=0))
    
    # ===== COMPARISON SUMMARY =====
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<20} {'Original':<15} {'V2':<15}")
    print("-"*50)
    print(f"{'Cohen Kappa':<20} {kappa_original:.4f}         {kappa_v2:.4f}")
    print(f"{'Accuracy':<20} {accuracy_original:.4f}         {accuracy_v2:.4f}")
    print(f"{'N Samples':<20} {len(combined):<15} {len(merged):<15}")
else:
    print("\n⚠️ No matching samples found between human annotations and V2 results")
    print("This may happen if post_ids don't match between datasets")
