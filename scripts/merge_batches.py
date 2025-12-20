"""
Merge all annotated batch files into final dataset.
Run this after all batches are annotated.
"""

import pandas as pd
import os

def merge_batches():
    """Merge all batch files into final annotated dataset."""
    
    batch_files = [
        'data/sample/batch_pilot.csv',
        'data/sample/batch_1.csv',
        'data/sample/batch_2.csv',
        'data/sample/batch_3.csv',
        'data/sample/batch_4.csv',
        'data/sample/batch_5.csv',
        'data/sample/batch_6.csv',
    ]
    
    dfs = []
    for f in batch_files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            print(f"Loaded {f}: {len(df)} posts")
            dfs.append(df)
        else:
            print(f"Warning: {f} not found")
    
    # Merge all
    merged = pd.concat(dfs, ignore_index=True)
    
    # Sort by sample_id
    merged = merged.sort_values('sample_id').reset_index(drop=True)
    
    # Save
    output_path = 'data/sample/human_benchmark_sample_annotation_final.csv'
    merged.to_csv(output_path, index=False)
    
    print(f"\n✅ Merged {len(merged)} posts to {output_path}")
    
    # Check completeness
    annotated_1 = merged['annotator_1_frame'].notna().sum()
    annotated_2 = merged['annotator_2_frame'].notna().sum()
    final = merged['final_frame'].notna().sum()
    
    print(f"\nAnnotation status:")
    print(f"  Annotator 1: {annotated_1}/{len(merged)} ({annotated_1/len(merged)*100:.1f}%)")
    print(f"  Annotator 2: {annotated_2}/{len(merged)} ({annotated_2/len(merged)*100:.1f}%)")
    print(f"  Final frame: {final}/{len(merged)} ({final/len(merged)*100:.1f}%)")

if __name__ == '__main__':
    merge_batches()
