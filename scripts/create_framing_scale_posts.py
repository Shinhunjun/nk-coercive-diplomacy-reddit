"""
Convert Categorical Framing to Continuous Scale (Posts Only)

Scale Formula:
  diplomacy_scale = DIPLOMACY(+2) + THREAT(-2) + Others(0)
  Range: -2 (strong threat) to +2 (strong diplomacy)

Input: data/framing/{topic}_posts_framed.csv
Output: data/framing/{topic}_posts_scaled.csv
"""

import pandas as pd
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def create_framing_scale(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical frame to continuous scale."""
    df = df.copy()

    # Scale: DIPLOMACY=+2, THREAT=-2, others=0
    df['diplomacy_scale'] = 0.0
    df.loc[df['frame'] == 'DIPLOMACY', 'diplomacy_scale'] = 2.0
    df.loc[df['frame'] == 'THREAT', 'diplomacy_scale'] = -2.0

    return df


def process_topic(topic: str, input_dir: Path, output_dir: Path):
    """Process a single topic."""
    input_path = input_dir / f"{topic}_posts_framed.csv"
    output_path = output_dir / f"{topic}_posts_scaled.csv"

    print(f"\n{'='*60}")
    print(f"Processing: {topic.upper()}")
    print(f"{'='*60}")

    if not input_path.exists():
        print(f"  WARNING: File not found: {input_path}")
        return None

    # Load
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df):,} posts")

    # Check for frame column
    if 'frame' not in df.columns:
        print(f"  ERROR: 'frame' column not found")
        return None

    # Create scale
    df_scaled = create_framing_scale(df)

    # Show distribution
    print(f"\n  Frame distribution:")
    for frame, count in df_scaled['frame'].value_counts().items():
        pct = count / len(df_scaled) * 100
        print(f"    {frame}: {count:,} ({pct:.1f}%)")

    print(f"\n  Diplomacy Scale statistics:")
    print(f"    Mean: {df_scaled['diplomacy_scale'].mean():.3f}")
    print(f"    Std:  {df_scaled['diplomacy_scale'].std():.3f}")

    # Scale value distribution
    print(f"\n  Scale value distribution:")
    for value in [-2.0, 0.0, 2.0]:
        count = (df_scaled['diplomacy_scale'] == value).sum()
        pct = count / len(df_scaled) * 100
        label = {-2.0: "THREAT", 0.0: "NEUTRAL/ECON/HUMAN", 2.0: "DIPLOMACY"}[value]
        print(f"    {value:+.1f} ({label}): {count:,} ({pct:.1f}%)")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    df_scaled.to_csv(output_path, index=False)
    print(f"\n  ✓ Saved: {output_path}")

    return df_scaled


def main():
    print("=" * 60)
    print("Creating Framing Scale (Posts Only)")
    print("=" * 60)
    print("\nScale: DIPLOMACY=+2, THREAT=-2, Others=0")

    input_dir = project_root / "data" / "framing"
    output_dir = project_root / "data" / "framing"

    topics = ['nk', 'iran', 'russia', 'china']
    results = {}

    for topic in topics:
        df = process_topic(topic, input_dir, output_dir)
        if df is not None:
            results[topic] = {
                'count': len(df),
                'mean': df['diplomacy_scale'].mean(),
                'threat_pct': (df['frame'] == 'THREAT').mean() * 100,
                'diplomacy_pct': (df['frame'] == 'DIPLOMACY').mean() * 100
            }

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Topic':<10} {'Count':<10} {'Mean':<10} {'THREAT%':<10} {'DIPLOMACY%'}")
    print("-" * 50)
    for topic, data in results.items():
        print(f"{topic.upper():<10} {data['count']:<10,} {data['mean']:+.3f}     {data['threat_pct']:.1f}%       {data['diplomacy_pct']:.1f}%")

    print("\n✓ Framing scale creation complete!")
    print("\nNext: Run scripts/create_framing_monthly_posts.py")


if __name__ == "__main__":
    main()
