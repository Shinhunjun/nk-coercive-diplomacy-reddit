"""
Analyze Russia Posts Framing Distribution
"""

import pandas as pd
from pathlib import Path

# Load Russia posts
posts_path = Path('data/processed/russia_posts_framing.csv')
df = pd.read_csv(posts_path)

# Convert datetime
df['datetime'] = pd.to_datetime(df['datetime'])

# Event date
event_date = '2018-03-08'

# Overall distribution
print('='*60)
print('RUSSIA POSTS FRAMING DISTRIBUTION')
print('='*60)
print(f'\nTotal items: {len(df):,}\n')

print('Frame Distribution:')
print('-'*60)
frame_counts = df['frame'].value_counts().sort_index()
for frame, count in frame_counts.items():
    pct = count / len(df) * 100
    print(f'  {frame:15s}: {count:5,} ({pct:5.1f}%)')

# Pre/Post comparison
pre = df[df['datetime'] < event_date]
post = df[df['datetime'] >= event_date]

print('\n' + '='*60)
print('PRE-EVENT vs POST-EVENT COMPARISON')
print(f'Event: NK-US Summit Announcement ({event_date})')
print('='*60)

print(f'\nPre-event:  {len(pre):,} items')
print(f'Post-event: {len(post):,} items')

# Get frame distributions
pre_dist = pre['frame'].value_counts(normalize=True) * 100
post_dist = post['frame'].value_counts(normalize=True) * 100

# Combine into comparison table
all_frames = sorted(set(list(pre_dist.index) + list(post_dist.index)))

print(f'\n{"Frame":<15} {"Pre %":>10} {"Post %":>10} {"Change":>10}')
print('-'*60)

for frame in all_frames:
    pre_pct = pre_dist.get(frame, 0)
    post_pct = post_dist.get(frame, 0)
    change = post_pct - pre_pct

    # Add arrow indicator
    if abs(change) > 1.0:
        arrow = '↑' if change > 0 else '↓'
    else:
        arrow = '→'

    print(f'{frame:<15} {pre_pct:>9.1f}% {post_pct:>9.1f}% {change:>+9.1f}% {arrow}')

# Key findings
print('\n' + '='*60)
print('KEY FINDINGS')
print('='*60)

threat_change = post_dist.get('THREAT', 0) - pre_dist.get('THREAT', 0)
diplomacy_change = post_dist.get('DIPLOMACY', 0) - pre_dist.get('DIPLOMACY', 0)

print(f'\nTHREAT frame:    {pre_dist.get("THREAT", 0):.1f}% → {post_dist.get("THREAT", 0):.1f}% ({threat_change:+.1f}%)')
print(f'DIPLOMACY frame: {pre_dist.get("DIPLOMACY", 0):.1f}% → {post_dist.get("DIPLOMACY", 0):.1f}% ({diplomacy_change:+.1f}%)')

print('\nInterpretation:')
if abs(threat_change) < 3:
    print('  ✓ THREAT framing remained stable (good control group)')
else:
    print(f'  ✗ THREAT framing changed by {threat_change:+.1f}% (potential issue)')

if abs(diplomacy_change) < 3:
    print('  ✓ DIPLOMACY framing remained stable (good control group)')
else:
    print(f'  ✗ DIPLOMACY framing changed by {diplomacy_change:+.1f}% (potential issue)')
