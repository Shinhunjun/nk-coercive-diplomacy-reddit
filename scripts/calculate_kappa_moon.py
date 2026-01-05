"""
Calculate Cohen's Kappa: Human Final Annotation vs LLM (Original and V2 Prompt)
Using Moon's batch files directly
"""
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, accuracy_score, classification_report
import os

print("="*80)
print("📊 COHEN'S KAPPA: HUMAN (Moon) vs LLM")
print("="*80)

# Load all Moon's annotation files
batch_files = [
    'data/annotations/framing_human_annotation_Moon - batch_1.csv',
    'data/annotations/framing_human_annotation_Moon - batch_2.csv',
    'data/annotations/framing_human_annotation_Moon - batch_pilot.csv'
]

dfs = []
for f in batch_files:
    df = pd.read_csv(f)
    dfs.append(df)
    print(f"Loaded {f}: {len(df)} samples")

human_df = pd.concat(dfs, ignore_index=True)
print(f"\nTotal human annotations: {len(human_df)}")

# Clean up
human_df['human_frame'] = human_df['final_frame'].str.strip().str.upper()
human_df['post_id'] = human_df['post_id'].astype(str)

# Filter valid frames
valid_frames = ['THREAT', 'DIPLOMACY', 'ECONOMIC', 'HUMANITARIAN', 'NEUTRAL']
human_df = human_df[human_df['human_frame'].isin(valid_frames)]
print(f"Valid human annotations: {len(human_df)}")

# ===== LOAD ORIGINAL PROMPT RESULTS =====
# These are in data/final/*_framing_final.csv
print("\n" + "-"*80)
print("Loading Original Prompt classifications...")

original_files = {
    'china': 'data/final/china_framing_final.csv',
    'iran': 'data/final/iran_framing_final.csv',
    'nk': 'data/final/nk_framing_final.csv',
    'russia': 'data/final/russia_framing_final.csv'
}

original_dfs = []
for country, path in original_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path, usecols=['id', 'frame'], low_memory=False)
        df['id'] = df['id'].astype(str)
        original_dfs.append(df)

original_df = pd.concat(original_dfs, ignore_index=True)
original_df['original_frame'] = original_df['frame'].str.strip().str.upper()
print(f"Loaded Original prompt classifications: {len(original_df)}")

# ===== LOAD V2 PROMPT RESULTS =====
print("\n" + "-"*80)
print("Loading V2 Prompt classifications...")

v2_files = {
    'china': 'data/results/final_framing_v2/china_framing_v2.csv',
    'iran': 'data/results/final_framing_v2/iran_framing_v2.csv',
    'nk': 'data/results/final_framing_v2/nk_framing_v2.csv',
    'russia': 'data/results/final_framing_v2/russia_framing_v2.csv'
}

v2_dfs = []
for country, path in v2_files.items():
    if os.path.exists(path):
        df = pd.read_csv(path, usecols=['id', 'frame'])
        df['id'] = df['id'].astype(str)
        v2_dfs.append(df)

v2_df = pd.concat(v2_dfs, ignore_index=True)
v2_df['v2_frame'] = v2_df['frame'].str.strip().str.upper()
print(f"Loaded V2 prompt classifications: {len(v2_df)}")

# ===== MERGE AND CALCULATE =====
print("\n" + "="*80)
print("🔵 ORIGINAL PROMPT - Cohen's Kappa")
print("="*80)

merged_orig = pd.merge(human_df, original_df[['id', 'original_frame']], 
                       left_on='post_id', right_on='id', how='inner')
merged_orig = merged_orig[merged_orig['original_frame'].isin(valid_frames)]
print(f"\nMatched samples: {len(merged_orig)}")

if len(merged_orig) > 0:
    kappa_orig = cohen_kappa_score(merged_orig['human_frame'], merged_orig['original_frame'])
    accuracy_orig = accuracy_score(merged_orig['human_frame'], merged_orig['original_frame'])
    
    print(f"Cohen's Kappa: {kappa_orig:.4f}")
    print(f"Accuracy:      {accuracy_orig:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(merged_orig['human_frame'], merged_orig['original_frame'], 
                                labels=valid_frames, zero_division=0))

print("\n" + "="*80)
print("🟠 V2 PROMPT - Cohen's Kappa")
print("="*80)

merged_v2 = pd.merge(human_df, v2_df[['id', 'v2_frame']], 
                     left_on='post_id', right_on='id', how='inner')
merged_v2 = merged_v2[merged_v2['v2_frame'].isin(valid_frames)]
print(f"\nMatched samples: {len(merged_v2)}")

if len(merged_v2) > 0:
    kappa_v2 = cohen_kappa_score(merged_v2['human_frame'], merged_v2['v2_frame'])
    accuracy_v2 = accuracy_score(merged_v2['human_frame'], merged_v2['v2_frame'])
    
    print(f"Cohen's Kappa: {kappa_v2:.4f}")
    print(f"Accuracy:      {accuracy_v2:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(merged_v2['human_frame'], merged_v2['v2_frame'], 
                                labels=valid_frames, zero_division=0))

# ===== COMPARISON =====
if len(merged_orig) > 0 and len(merged_v2) > 0:
    print("\n" + "="*80)
    print("📊 COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Metric':<20} {'Original':<15} {'V2':<15} {'Diff':<15}")
    print("-"*60)
    print(f"{'Cohen Kappa':<20} {kappa_orig:.4f}         {kappa_v2:.4f}         {kappa_v2 - kappa_orig:+.4f}")
    print(f"{'Accuracy':<20} {accuracy_orig:.4f}         {accuracy_v2:.4f}         {accuracy_v2 - accuracy_orig:+.4f}")
    print(f"{'N Samples':<20} {len(merged_orig):<15} {len(merged_v2):<15}")
