"""
Calculate Inter-Rater Reliability (Cohen's Kappa) for annotation batches.
Use this after each annotation batch to track agreement.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
import sys

def calculate_irr(csv_path, batch_name="Full Dataset"):
    """
    Calculate IRR metrics from annotation spreadsheet.
    
    Args:
        csv_path: Path to annotation CSV
        batch_name: Name of the batch for reporting
    """
    
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter to rows where both annotators have labeled
    df_labeled = df[
        (df['annotator_1_frame'].notna()) & 
        (df['annotator_1_frame'] != '') &
        (df['annotator_2_frame'].notna()) & 
        (df['annotator_2_frame'] != '')
    ].copy()
    
    if len(df_labeled) == 0:
        print(f"No completed annotations found in {csv_path}")
        return
    
    print("=" * 70)
    print(f"INTER-RATER RELIABILITY: {batch_name}")
    print("=" * 70)
    print(f"Total annotated: {len(df_labeled)} / {len(df)} posts")
    
    # Get labels
    labels_1 = df_labeled['annotator_1_frame'].str.upper()
    labels_2 = df_labeled['annotator_2_frame'].str.upper()
    
    # Calculate agreement metrics
    exact_agreement = (labels_1 == labels_2).sum()
    agreement_pct = exact_agreement / len(df_labeled) * 100
    
    # Cohen's Kappa
    kappa = cohen_kappa_score(labels_1, labels_2)
    
    print(f"\n{'='*70}")
    print("AGREEMENT METRICS")
    print("=" * 70)
    print(f"Exact Agreement: {exact_agreement} / {len(df_labeled)} ({agreement_pct:.1f}%)")
    print(f"Cohen's Kappa: {kappa:.3f}")
    
    # Interpret Kappa
    if kappa < 0.0:
        interpretation = "Poor (worse than chance)"
    elif kappa < 0.20:
        interpretation = "Slight"
    elif kappa < 0.40:
        interpretation = "Fair"
    elif kappa < 0.60:
        interpretation = "Moderate"
    elif kappa < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost Perfect"
    
    print(f"Interpretation: {interpretation}")
    
    # Confusion matrix
    print(f"\n{'='*70}")
    print("CONFUSION MATRIX (Annotator 1 vs Annotator 2)")
    print("=" * 70)
    
    categories = sorted(set(labels_1) | set(labels_2))
    cm = confusion_matrix(labels_1, labels_2, labels=categories)
    
    # Print as DataFrame for readability
    cm_df = pd.DataFrame(cm, index=categories, columns=categories)
    print("\nAnnotator 1 (rows) vs Annotator 2 (columns):")
    print(cm_df.to_string())
    
    # Per-category agreement
    print(f"\n{'='*70}")
    print("PER-CATEGORY AGREEMENT")
    print("=" * 70)
    
    for cat in categories:
        cat_mask = (labels_1 == cat) | (labels_2 == cat)
        cat_agree = ((labels_1 == cat) & (labels_2 == cat)).sum()
        cat_total = cat_mask.sum()
        
        if cat_total > 0:
            cat_pct = cat_agree / cat_total * 100
            print(f"{cat:12}: {cat_agree:3} / {cat_total:3} agreed ({cat_pct:.1f}%)")
    
    # Disagreements
    print(f"\n{'='*70}")
    print("DISAGREEMENTS TO REVIEW")
    print("=" * 70)
    
    disagreements = df_labeled[labels_1 != labels_2].copy()
    print(f"Total disagreements: {len(disagreements)}")
    
    if len(disagreements) > 0:
        print("\nTop 10 disagreements:")
        for idx, row in disagreements.head(10).iterrows():
            print(f"\nPost ID: {row['post_id']}")
            print(f"  Title: {row['title'][:80]}...")
            print(f"  Annotator 1: {row['annotator_1_frame']}")
            print(f"  Annotator 2: {row['annotator_2_frame']}")
        
        # Save full disagreements to file
        disagreement_path = csv_path.replace('.csv', '_disagreements.csv')
        disagreements[['sample_id', 'post_id', 'country', 'title', 
                      'annotator_1_frame', 'annotator_2_frame', 'notes']].to_csv(
            disagreement_path, index=False
        )
        print(f"\n✓ Full disagreements saved to: {disagreement_path}")
    
    # Summary for codebook
    print(f"\n{'='*70}")
    print("FOR CODEBOOK DOCUMENTATION")
    print("=" * 70)
    print(f"Batch: {batch_name}")
    print(f"N Posts: {len(df_labeled)}")
    print(f"Agreement: {agreement_pct:.1f}%")
    print(f"Cohen's Kappa: {kappa:.3f} ({interpretation})")
    
    return {
        'n_posts': len(df_labeled),
        'agreement_pct': agreement_pct,
        'kappa': kappa,
        'interpretation': interpretation,
        'disagreements': len(disagreements)
    }


def main():
    """Run IRR calculation."""
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        batch_name = sys.argv[2] if len(sys.argv) > 2 else "Batch"
    else:
        csv_path = 'data/sample/human_benchmark_sample_annotation.csv'
        batch_name = "Full Dataset"
    
    calculate_irr(csv_path, batch_name)


if __name__ == '__main__':
    main()
