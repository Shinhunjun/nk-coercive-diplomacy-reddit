"""
Analyze Iran Framing Distribution and Pre/Post Event Changes

This script analyzes the framing distribution for Iran (posts + comments combined)
and compares the distribution before and after the NK-US summit announcement.

Event date: 2018-03-08 (NK-US summit announcement)
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR

# Event date
EVENT_DATE = "2018-03-08"


def load_iran_data():
    """Load Iran posts and comments."""
    posts_path = DATA_DIR / 'processed' / 'iran_posts_framing.csv'
    comments_path = DATA_DIR / 'processed' / 'iran_comments_framing.csv'

    # Load posts
    posts = pd.read_csv(posts_path)
    posts['type'] = 'post'

    # Load comments
    comments = pd.read_csv(comments_path)
    comments['type'] = 'comment'

    # Combine
    combined = pd.concat([posts, comments], ignore_index=True)

    # Convert datetime
    combined['datetime'] = pd.to_datetime(combined['datetime'])

    return combined


def analyze_framing_distribution(df, title="Overall"):
    """Analyze framing distribution."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    total = len(df)
    print(f"Total items: {total:,}\n")

    # Frame distribution
    frame_counts = df['frame'].value_counts().sort_index()

    print("Frame Distribution:")
    print("-" * 60)
    for frame, count in frame_counts.items():
        pct = count / total * 100
        print(f"  {frame:15s}: {count:5,} ({pct:5.1f}%)")

    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {df['frame_confidence'].mean():.3f}")
    print(f"  Std:  {df['frame_confidence'].std():.3f}")

    return frame_counts


def compare_pre_post(df, event_date):
    """Compare framing distribution before and after event."""
    # Split into pre and post
    pre = df[df['datetime'] < event_date].copy()
    post = df[df['datetime'] >= event_date].copy()

    print(f"\n{'='*60}")
    print(f"PRE-EVENT vs POST-EVENT COMPARISON")
    print(f"Event: NK-US Summit Announcement ({event_date})")
    print(f"{'='*60}")

    print(f"\nPre-event:  {len(pre):,} items")
    print(f"Post-event: {len(post):,} items")

    # Get frame distributions
    pre_dist = pre['frame'].value_counts(normalize=True) * 100
    post_dist = post['frame'].value_counts(normalize=True) * 100

    # Combine into comparison table
    all_frames = sorted(set(list(pre_dist.index) + list(post_dist.index)))

    print(f"\n{'Frame':<15} {'Pre %':>10} {'Post %':>10} {'Change':>10}")
    print("-" * 60)

    for frame in all_frames:
        pre_pct = pre_dist.get(frame, 0)
        post_pct = post_dist.get(frame, 0)
        change = post_pct - pre_pct

        # Add arrow indicator
        if abs(change) > 1.0:
            if change > 0:
                arrow = "↑"
            else:
                arrow = "↓"
        else:
            arrow = "→"

        print(f"{frame:<15} {pre_pct:>9.1f}% {post_pct:>9.1f}% {change:>+9.1f}% {arrow}")

    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    # Threat frame change
    threat_change = post_dist.get('THREAT', 0) - pre_dist.get('THREAT', 0)
    diplomacy_change = post_dist.get('DIPLOMACY', 0) - pre_dist.get('DIPLOMACY', 0)

    print(f"\nTHREAT frame:    {pre_dist.get('THREAT', 0):.1f}% → {post_dist.get('THREAT', 0):.1f}% ({threat_change:+.1f}%)")
    print(f"DIPLOMACY frame: {pre_dist.get('DIPLOMACY', 0):.1f}% → {post_dist.get('DIPLOMACY', 0):.1f}% ({diplomacy_change:+.1f}%)")

    # Interpretation
    print("\nInterpretation:")
    if threat_change < -5:
        print("  ✓ THREAT framing significantly decreased after event")
    elif threat_change > 5:
        print("  ✗ THREAT framing significantly increased after event")
    else:
        print("  → THREAT framing remained relatively stable")

    if diplomacy_change > 5:
        print("  ✓ DIPLOMACY framing significantly increased after event")
    elif diplomacy_change < -5:
        print("  ✗ DIPLOMACY framing significantly decreased after event")
    else:
        print("  → DIPLOMACY framing remained relatively stable")


def main():
    """Main execution."""
    print("="*60)
    print("IRAN FRAMING ANALYSIS")
    print("Posts + Comments Combined (2017-2019)")
    print("="*60)

    # Load data
    print("\nLoading data...")
    df = load_iran_data()

    # Overall distribution
    analyze_framing_distribution(df, title="OVERALL DISTRIBUTION (2017-2019)")

    # By type
    print(f"\n{'='*60}")
    print("BY TYPE")
    print(f"{'='*60}")

    posts = df[df['type'] == 'post']
    comments = df[df['type'] == 'comment']

    print(f"\nPosts:    {len(posts):,}")
    print(f"Comments: {len(comments):,}")

    # Pre/Post comparison
    compare_pre_post(df, EVENT_DATE)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
