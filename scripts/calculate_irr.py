"""
Calculate inter-rater reliability (Cohen's Kappa) for human annotation batches.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import numpy as np

# Paths
ANNOTATIONS_DIR = "data/annotations"
pilot_path = f"{ANNOTATIONS_DIR}/batch_pilot_annotated.csv"
batch1_path = f"{ANNOTATIONS_DIR}/batch_1_annotated.csv"

# Valid frame labels
VALID_LABELS = ["THREAT", "DIPLOMACY", "NEUTRAL", "ECONOMIC", "HUMANITARIAN"]

def clean_label(label):
    """Standardize label: uppercase, strip whitespace."""
    if pd.isna(label):
        return None
    label_clean = str(label).strip().upper()
    # Handle known variations
    if label_clean in VALID_LABELS:
        return label_clean
    # If it contains Korean or is ambiguous, mark as None
    return None

def calculate_kappa(df, batch_name):
    """Calculate Cohen's Kappa for a batch."""
    # Clean labels
    df['a1_clean'] = df['annotator_1_frame'].apply(clean_label)
    df['a2_clean'] = df['annotator_2_frame'].apply(clean_label)
    
    # Filter rows where both annotators provided valid labels
    valid_mask = df['a1_clean'].notna() & df['a2_clean'].notna()
    valid_df = df[valid_mask].copy()
    
    total_rows = len(df)
    valid_rows = len(valid_df)
    excluded_rows = total_rows - valid_rows
    
    print(f"\n{'='*60}")
    print(f"Batch: {batch_name}")
    print(f"{'='*60}")
    print(f"Total samples: {total_rows}")
    print(f"Valid for Kappa calculation: {valid_rows}")
    print(f"Excluded (missing/ambiguous): {excluded_rows}")
    
    if valid_rows == 0:
        print("ERROR: No valid rows for Kappa calculation!")
        return None, None
    
    # Calculate agreement
    agreement = (valid_df['a1_clean'] == valid_df['a2_clean']).sum()
    agreement_rate = agreement / valid_rows
    print(f"\nRaw Agreement: {agreement}/{valid_rows} ({agreement_rate*100:.1f}%)")
    
    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(valid_df['a1_clean'], valid_df['a2_clean'])
    print(f"Cohen's Kappa: {kappa:.3f}")
    
    # Interpretation
    if kappa < 0.20:
        interpretation = "Poor"
    elif kappa < 0.40:
        interpretation = "Fair"
    elif kappa < 0.60:
        interpretation = "Moderate"
    elif kappa < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost Perfect"
    print(f"Interpretation: {interpretation}")
    
    # Confusion matrix like breakdown
    print(f"\n--- Label Distribution ---")
    print("Annotator 1:")
    print(valid_df['a1_clean'].value_counts())
    print("\nAnnotator 2:")
    print(valid_df['a2_clean'].value_counts())
    
    return kappa, agreement_rate

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("INTER-RATER RELIABILITY ANALYSIS")
    print("="*60)
    
    # Load and process pilot batch
    pilot_df = pd.read_csv(pilot_path)
    kappa_pilot, agree_pilot = calculate_kappa(pilot_df, "Pilot")
    
    # Load and process batch 1
    batch1_df = pd.read_csv(batch1_path)
    kappa_batch1, agree_batch1 = calculate_kappa(batch1_df, "Batch 1")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"| Batch   | Kappa | Agreement | Interpretation |")
    print(f"|---------|-------|-----------|----------------|")
    if kappa_pilot is not None:
        interp_pilot = "Substantial" if kappa_pilot >= 0.61 else ("Moderate" if kappa_pilot >= 0.41 else "Fair/Poor")
        print(f"| Pilot   | {kappa_pilot:.3f} | {agree_pilot*100:.1f}%      | {interp_pilot} |")
    if kappa_batch1 is not None:
        interp_batch1 = "Substantial" if kappa_batch1 >= 0.61 else ("Moderate" if kappa_batch1 >= 0.41 else "Fair/Poor")
        print(f"| Batch 1 | {kappa_batch1:.3f} | {agree_batch1*100:.1f}%      | {interp_batch1} |")
