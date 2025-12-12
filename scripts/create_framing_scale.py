"""
Convert Categorical Framing to Continuous Scale

This script converts categorical framing classifications (THREAT, DIPLOMACY, etc.)
into a continuous diplomacy-threat scale for DID regression analysis.

Scale Formula:
  diplomacy_scale = (DIPLOMACY × +2) + (THREAT × -2) + (NEUTRAL/ECONOMIC/HUMANITARIAN × 0)

Range: -2 (strong threat framing) to +2 (strong diplomacy framing)

Input: Files with 'frame' column (from apply_vertex_ai_framing.py)
Output: Same files with added 'diplomacy_scale' column
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR


def create_framing_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert categorical frame to continuous scale.

    Args:
        df: DataFrame with 'frame' column

    Returns:
        DataFrame with added 'diplomacy_scale' column
    """
    # Create scale based on frame
    df = df.copy()

    # Scale formula: DIPLOMACY=+2, THREAT=-2, others=0
    df['diplomacy_scale'] = 0.0  # Default for NEUTRAL/ECONOMIC/HUMANITARIAN

    # Apply scale values
    df.loc[df['frame'] == 'DIPLOMACY', 'diplomacy_scale'] = 2.0
    df.loc[df['frame'] == 'THREAT', 'diplomacy_scale'] = -2.0

    # Note: NEUTRAL, ECONOMIC, HUMANITARIAN all remain 0.0

    return df


def process_file(input_path: Path, output_path: Path):
    """
    Process a single file to add framing scale.

    Args:
        input_path: Path to input CSV with 'frame' column
        output_path: Path to output CSV with scale column
    """
    print(f"\nProcessing: {input_path.name}")

    # Check if file exists
    if not input_path.exists():
        print(f"  WARNING: File not found: {input_path}")
        return

    # Load data
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} rows")

    # Check for frame column
    if 'frame' not in df.columns:
        print(f"  ERROR: 'frame' column not found in {input_path}")
        return

    # Create scale
    df_scaled = create_framing_scale(df)

    # Show scale distribution
    print(f"\n  Frame distribution:")
    frame_counts = df_scaled['frame'].value_counts()
    for frame, count in frame_counts.items():
        pct = count / len(df_scaled) * 100
        print(f"    {frame}: {count} ({pct:.1f}%)")

    print(f"\n  Diplomacy Scale statistics:")
    print(f"    Mean: {df_scaled['diplomacy_scale'].mean():.3f}")
    print(f"    Std:  {df_scaled['diplomacy_scale'].std():.3f}")
    print(f"    Min:  {df_scaled['diplomacy_scale'].min():.1f}")
    print(f"    Max:  {df_scaled['diplomacy_scale'].max():.1f}")

    # Show value counts
    scale_counts = df_scaled['diplomacy_scale'].value_counts().sort_index()
    print(f"\n  Scale value distribution:")
    for value, count in scale_counts.items():
        pct = count / len(df_scaled) * 100
        if value == 2.0:
            label = "DIPLOMACY"
        elif value == -2.0:
            label = "THREAT"
        else:
            label = "NEUTRAL/ECONOMIC/HUMANITARIAN"
        print(f"    {value:+.1f} ({label}): {count} ({pct:.1f}%)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path}")


def main():
    """Main execution function."""
    print("=" * 80)
    print("Creating Framing Scale (Categorical → Continuous)")
    print("=" * 80)

    print("\nScale Formula:")
    print("  diplomacy_scale = (DIPLOMACY × +2) + (THREAT × -2) + (others × 0)")
    print("\nScale Range:")
    print("  -2.0 = Strong THREAT framing")
    print("   0.0 = NEUTRAL/ECONOMIC/HUMANITARIAN framing")
    print("  +2.0 = Strong DIPLOMACY framing")

    # Define input/output files
    datasets = [
        # NK (Treatment group)
        {
            'name': 'NK Posts',
            'input': DATA_DIR / 'processed' / 'nk_posts_framing.csv',
            'output': DATA_DIR / 'processed' / 'nk_posts_framing_scaled.csv'
        },
        {
            'name': 'NK Comments',
            'input': DATA_DIR / 'processed' / 'nk_comments_framing.csv',
            'output': DATA_DIR / 'processed' / 'nk_comments_framing_scaled.csv'
        },

        # Iran (Control group 1)
        {
            'name': 'Iran Posts',
            'input': DATA_DIR / 'processed' / 'iran_posts_framing.csv',
            'output': DATA_DIR / 'processed' / 'iran_posts_framing_scaled.csv'
        },
        {
            'name': 'Iran Comments',
            'input': DATA_DIR / 'processed' / 'iran_comments_framing.csv',
            'output': DATA_DIR / 'processed' / 'iran_comments_framing_scaled.csv'
        },

        # Russia (Control group 2)
        {
            'name': 'Russia Posts',
            'input': DATA_DIR / 'processed' / 'russia_posts_framing.csv',
            'output': DATA_DIR / 'processed' / 'russia_posts_framing_scaled.csv'
        },
        {
            'name': 'Russia Comments',
            'input': DATA_DIR / 'processed' / 'russia_comments_framing.csv',
            'output': DATA_DIR / 'processed' / 'russia_comments_framing_scaled.csv'
        },

        # China (Control group 3)
        {
            'name': 'China Posts',
            'input': DATA_DIR / 'processed' / 'china_posts_framing.csv',
            'output': DATA_DIR / 'processed' / 'china_posts_framing_scaled.csv'
        },
        {
            'name': 'China Comments',
            'input': DATA_DIR / 'processed' / 'china_comments_framing.csv',
            'output': DATA_DIR / 'processed' / 'china_comments_framing_scaled.csv'
        }
    ]

    # Process each dataset
    print("\n" + "=" * 80)
    print("Processing Datasets")
    print("=" * 80)

    total_processed = 0

    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset['name']}")
        print(f"{'='*80}")

        try:
            process_file(
                input_path=dataset['input'],
                output_path=dataset['output']
            )

            # Count processed
            if dataset['output'].exists():
                df = pd.read_csv(dataset['output'])
                total_processed += len(df)

        except Exception as e:
            print(f"  ERROR processing {dataset['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print("\n" + "=" * 80)
    print("Framing Scale Creation Complete!")
    print("=" * 80)
    print(f"Total items processed: {total_processed:,}")
    print(f"\nOutput files saved to: {DATA_DIR / 'processed'}")
    print("\nNext steps:")
    print("  1. Run scripts/create_framing_monthly_aggregation.py")
    print("  2. Run scripts/run_did_analysis_framing.py")


if __name__ == "__main__":
    main()
