import pandas as pd
import os
import shutil

BASE_DIR = 'data/anonymized/annotations'

def consolidate_human_annotations():
    print("Consolidating human annotations...")
    frames = []
    
    # Process batch files
    batches = [
        ('framing - batch_pilot.csv', 'pilot'),
        ('framing - batch_1.csv', 'batch_1'),
        ('framing - batch_2.csv', 'batch_2')
    ]
    
    for filename, batch_name in batches:
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Rename columns to standard names
            df = df.rename(columns={
                'annotator_1_frame': 'annotator_1',
                'annotator_2_frame': 'annotator_2',
                'final_frame': 'gold_label'
            })
            
            # Select columns
            cols = ['post_id', 'annotator_1', 'annotator_2', 'gold_label']
            clean_df = df[cols].copy()
            clean_df['batch'] = batch_name
            frames.append(clean_df)
    
    if frames:
        combined_df = pd.concat(frames, ignore_index=True)
        # Remove duplicates if any
        combined_df = combined_df.drop_duplicates(subset=['post_id'])
        
        output_path = os.path.join(BASE_DIR, 'human_ground_truth.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Created {output_path} with {len(combined_df)} rows")
        return True
    return False

def setup_validation_results():
    print("Setting up validation results...")
    # Use final_analysis_v2.csv as the main validation result
    src = os.path.join(BASE_DIR, 'final_analysis_v2.csv')
    dst = os.path.join(BASE_DIR, 'validation_results.csv')
    
    if os.path.exists(src):
        df = pd.read_csv(src)
        # Select relevant columns
        cols = ['post_id', 'human_frame', 'llm_frame', 'llm_confidence']
        # Handle case where column names might strictly match
        available_cols = [c for c in cols if c in df.columns]
        
        clean_df = df[available_cols]
        clean_df = clean_df.rename(columns={
            'human_frame': 'gold_label',
            'llm_frame': 'model_prediction',
            'llm_confidence': 'confidence_score'
        })
        
        clean_df.to_csv(dst, index=False)
        print(f"Created {dst} with {len(clean_df)} rows")
        return True
    return False

def cleanup_files():
    print("Cleaning up old files...")
    keep_files = {'human_ground_truth.csv', 'validation_results.csv', 'CODEBOOK.md'}
    
    for f in os.listdir(BASE_DIR):
        if f not in keep_files and not f.startswith('.'):
            path = os.path.join(BASE_DIR, f)
            os.remove(path)
            print(f"Deleted {f}")

if __name__ == "__main__":
    if consolidate_human_annotations() and setup_validation_results():
        cleanup_files()
        print("\nCleanup complete!")
        print(f"Remaining files in {BASE_DIR}:")
        for f in os.listdir(BASE_DIR):
            if not f.startswith('.'):
                print(f" - {f}")
