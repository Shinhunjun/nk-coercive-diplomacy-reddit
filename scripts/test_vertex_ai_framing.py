"""
Test Vertex AI Gemini Framing Classification (100 samples)

This script tests the framing classifier on a small sample to verify it works correctly.
"""

import pandas as pd
import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from vertex_ai_framing_analysis import VertexAIFramingAnalyzer
from config import DATA_DIR

# Date range for filtering (2017-2019 only)
START_DATE = "2017-01-01"
END_DATE = "2019-12-31"


def test_framing_classification():
    """Test framing classification on 100 NK posts."""
    print("=" * 80)
    print("Testing Vertex AI Gemini Framing Classification (100 samples)")
    print("=" * 80)

    # Check GCP project ID
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        print("\nERROR: GOOGLE_CLOUD_PROJECT environment variable not set!")
        print("Set it with: export GOOGLE_CLOUD_PROJECT='mlops-compute-lab'")
        sys.exit(1)

    print(f"\nGCP Project: {project_id}")
    print(f"Model: gemini-1.5-flash-002")
    print(f"Sample size: 100 posts")

    # Initialize analyzer
    print("\nInitializing Vertex AI Gemini analyzer...")
    try:
        analyzer = VertexAIFramingAnalyzer(project_id=project_id, location="us-central1")
        print("✓ Analyzer initialized successfully!")
    except Exception as e:
        print(f"✗ ERROR initializing analyzer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Load NK posts
    posts_path = DATA_DIR / 'processed' / 'nk_posts_roberta_sentiment.csv'
    print(f"\nLoading NK posts from: {posts_path.name}")

    if not posts_path.exists():
        print(f"✗ ERROR: File not found: {posts_path}")
        sys.exit(1)

    df = pd.read_csv(posts_path)
    print(f"  Total posts: {len(df)}")

    # Convert date and filter
    df['datetime'] = pd.to_datetime(df['datetime'])
    df_filtered = df[
        (df['datetime'] >= START_DATE) &
        (df['datetime'] <= END_DATE)
    ].copy()

    print(f"  Filtered (2017-2019): {len(df_filtered)} posts")

    # Take first 100 samples
    df_sample = df_filtered.head(100).copy()
    print(f"  Sample size: {len(df_sample)} posts")

    # Apply framing classification
    print("\n" + "="*80)
    print("Classifying 100 posts...")
    print("="*80)

    df_framed = analyzer.analyze_dataframe(
        df=df_sample,
        title_col='title',
        body_col='selftext',
        delay=0.5  # 0.5 second delay between API calls
    )

    # Show results
    print("\n" + "="*80)
    print("Classification Results")
    print("="*80)

    # Frame distribution
    print("\nFrame Distribution:")
    frame_counts = df_framed['frame'].value_counts()
    for frame, count in frame_counts.items():
        pct = count / len(df_framed) * 100
        print(f"  {frame:15s}: {count:3d} ({pct:5.1f}%)")

    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {df_framed['frame_confidence'].mean():.3f}")
    print(f"  Std:  {df_framed['frame_confidence'].std():.3f}")
    print(f"  Min:  {df_framed['frame_confidence'].min():.3f}")
    print(f"  Max:  {df_framed['frame_confidence'].max():.3f}")

    # Show some examples
    print("\n" + "="*80)
    print("Sample Classifications (first 10)")
    print("="*80)

    for idx in range(min(10, len(df_framed))):
        row = df_framed.iloc[idx]
        title = row['title'][:80] + "..." if len(row['title']) > 80 else row['title']
        frame = row['frame']
        confidence = row['frame_confidence']
        reason = row['frame_reason'][:100] + "..." if len(str(row['frame_reason'])) > 100 else row['frame_reason']

        print(f"\n{idx+1}. {frame} (confidence: {confidence:.2f})")
        print(f"   Title: {title}")
        print(f"   Reason: {reason}")

    # Save results
    output_path = DATA_DIR / 'sample' / 'test_framing_100.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_framed.to_csv(output_path, index=False)

    print("\n" + "="*80)
    print("✓ Test Complete!")
    print("="*80)
    print(f"Results saved to: {output_path}")
    print(f"\nCost: ${len(df_framed) * 0.00003:.4f}")
    print(f"Time: ~{len(df_framed) * 0.5:.0f} seconds ({len(df_framed) * 0.5 / 60:.1f} minutes)")

    # Summary recommendation
    print("\n" + "="*80)
    print("Next Steps")
    print("="*80)

    if frame_counts.get('THREAT', 0) + frame_counts.get('DIPLOMACY', 0) > 50:
        print("✓ Classification looks good! Proceed with full dataset:")
        print("  python scripts/apply_vertex_ai_framing.py")
    else:
        print("⚠ Warning: Low THREAT/DIPLOMACY classification rate.")
        print("  Review sample results before proceeding with full dataset.")


if __name__ == "__main__":
    test_framing_classification()
