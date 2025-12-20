"""
Create stratified proportional sample for human annotation benchmark.
Generates ~1,330 samples stratified by Country × Period × Frame.
"""

import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configuration
TARGET_PER_COUNTRY = 300
MIN_PER_STRATUM = 10
COUNTRIES = ['nk', 'china', 'iran', 'russia']

def create_stratified_sample():
    """Create stratified proportional sample across all countries."""
    
    print("=" * 70)
    print("CREATING HUMAN BENCHMARK SAMPLE")
    print(f"Target: ~{TARGET_PER_COUNTRY} samples per country")
    print(f"Minimum per stratum: {MIN_PER_STRATUM}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)
    
    all_samples = []
    
    for country in COUNTRIES:
        print(f"\n{'='*50}")
        print(f"Processing {country.upper()}")
        print("=" * 50)
        
        # Load data
        df = pd.read_csv(f'data/final/{country}_final.csv')
        df['country'] = country.upper()
        print(f"Total posts: {len(df):,}")
        
        # Calculate sample sizes per stratum
        strata_counts = df.groupby(['period', 'frame']).size().reset_index(name='population')
        strata_counts['proportion'] = strata_counts['population'] / len(df)
        strata_counts['target_n'] = (strata_counts['proportion'] * TARGET_PER_COUNTRY).round().astype(int)
        strata_counts['sample_n'] = strata_counts['target_n'].clip(lower=MIN_PER_STRATUM)
        
        # Sample from each stratum
        country_samples = []
        for _, row in strata_counts.iterrows():
            stratum_df = df[(df['period'] == row['period']) & (df['frame'] == row['frame'])]
            n_sample = min(row['sample_n'], len(stratum_df))  # Can't sample more than available
            
            sampled = stratum_df.sample(n=n_sample, random_state=RANDOM_SEED)
            country_samples.append(sampled)
            print(f"  {row['period']:20} × {row['frame']:12}: {n_sample:3} samples (from {len(stratum_df):,})")
        
        country_df = pd.concat(country_samples, ignore_index=True)
        print(f"\nCountry total: {len(country_df)} samples")
        all_samples.append(country_df)
    
    # Combine all samples
    final_sample = pd.concat(all_samples, ignore_index=True)
    
    # Shuffle the final sample
    final_sample = final_sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print("\n" + "=" * 70)
    print("FINAL SAMPLE SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(final_sample):,}")
    print(f"\nBy country:")
    print(final_sample['country'].value_counts().to_string())
    print(f"\nBy period:")
    print(final_sample['period'].value_counts().to_string())
    print(f"\nBy frame:")
    print(final_sample['frame'].value_counts().to_string())
    
    return final_sample


def create_annotation_spreadsheet(df, output_path):
    """Create annotation-ready spreadsheet with only necessary columns."""
    
    # Select columns for annotation (Title + Text + Country only, as discussed)
    annotation_df = pd.DataFrame({
        'sample_id': range(1, len(df) + 1),
        'post_id': df['id'],
        'country': df['country'],
        'title': df['title'],
        'text': df['selftext'].fillna(''),  # Some posts may not have body text
        'annotator_1_frame': '',  # To be filled by annotator
        'annotator_2_frame': '',  # To be filled by annotator
        'final_frame': '',        # Consensus after discussion
        'notes': ''               # Optional notes
    })
    
    # Also save a version with LLM labels (hidden from annotators, for later validation)
    validation_df = df[['id', 'country', 'period', 'frame', 'frame_confidence', 'title', 'selftext']].copy()
    validation_df.columns = ['post_id', 'country', 'period', 'llm_frame', 'llm_confidence', 'title', 'text']
    
    # Save files
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    annotation_path = output_path.replace('.csv', '_annotation.csv')
    validation_path = output_path.replace('.csv', '_validation.csv')
    
    annotation_df.to_csv(annotation_path, index=False)
    validation_df.to_csv(validation_path, index=False)
    
    print(f"\n✓ Annotation spreadsheet: {annotation_path}")
    print(f"✓ Validation data (with LLM labels): {validation_path}")
    
    return annotation_path, validation_path


def main():
    # Create sample
    sample_df = create_stratified_sample()
    
    # Create annotation spreadsheet
    output_path = 'data/sample/human_benchmark_sample.csv'
    sample_df.to_csv(output_path, index=False)
    print(f"\n✓ Full sample saved: {output_path}")
    
    # Create annotation-ready version
    create_annotation_spreadsheet(sample_df, output_path)
    
    print("\n" + "=" * 70)
    print("DONE! Files created:")
    print("=" * 70)
    print("1. data/sample/human_benchmark_sample.csv - Full sample with all metadata")
    print("2. data/sample/human_benchmark_sample_annotation.csv - For annotators (Title + Text + Country)")
    print("3. data/sample/human_benchmark_sample_validation.csv - For validation (includes LLM labels)")


if __name__ == '__main__':
    main()
