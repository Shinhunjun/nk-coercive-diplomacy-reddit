#!/usr/bin/env python3
"""
Generate improved ratio comparison figure with modern minimalist style.
v2: Mohit's requested improvements - larger ticks, no grid, more colors, no box
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# Modern style setup - NO grid, NO box
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.left'] = True
plt.rcParams['axes.spines.bottom'] = True
plt.rcParams['axes.grid'] = False  # No grid lines

# Output directory
output_dir = "paper/figures"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv('data/final/nk_final.csv')

# Helper to get ratios
def get_ratios(subset):
    # Sentiment (using 0.05 threshold)
    pos = len(subset[subset['sentiment_score'] > 0.05])
    neg = len(subset[subset['sentiment_score'] < -0.05])
    sent_ratio = pos / neg if neg > 0 else 0
    
    # Framing
    dip = len(subset[subset['frame'] == 'DIPLOMACY'])
    threat = len(subset[subset['frame'] == 'THREAT'])
    frame_ratio = dip / threat if threat > 0 else 0
    
    return sent_ratio, frame_ratio

# Define periods
p1 = df[df['period'] == 'P1_PreSingapore']
p2 = df[df['period'] == 'P2_SingaporeHanoi']
p3 = df[df['period'] == 'P3_PostHanoi']

# Calculate
s1, f1 = get_ratios(p1)
s2, f2 = get_ratios(p2)
s3, f3 = get_ratios(p3)

# Prepare data
periods = ['Pre-Singapore\n(P1)', 'Singapore-Hanoi\n(P2)', 'Post-Hanoi\n(P3)']
sentiment_ratios = [s1, s2, s3]
framing_ratios = [f1, f2, f3]

# Colors - more vibrant
color_sentiment = '#FF6B6B'  # Coral red
color_framing = '#4ECDC4'    # Teal/turquoise

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot lines with markers
x = range(3)
ax.plot(x, sentiment_ratios, marker='o', markersize=14, linewidth=3, 
        color=color_sentiment, label='Sentiment Ratio\n(Positive / Negative)')
ax.plot(x, framing_ratios, marker='s', markersize=14, linewidth=3, 
        color=color_framing, label='Framing Ratio\n(Diplomacy / Threat)')

# Annotate values with larger font
for i, (s, f) in enumerate(zip(sentiment_ratios, framing_ratios)):
    # Sentiment annotation
    offset_s = 0.08 if i == 1 else -0.08
    ax.annotate(f"{s:.2f}", (i, s + offset_s), ha='center', va='bottom' if i == 1 else 'top',
                fontsize=14, fontweight='bold', color=color_sentiment)
    # Framing annotation
    offset_f = 0.08 if i == 1 else -0.08
    ax.annotate(f"{f:.2f}", (i, f + offset_f), ha='center', va='bottom' if i == 1 else 'top',
                fontsize=14, fontweight='bold', color=color_framing)

# Styling
ax.set_xticks(x)
ax.set_xticklabels(periods, fontsize=14, fontweight='bold')
ax.set_ylabel('Ratio Value', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1.8)

# Larger tick labels
ax.tick_params(axis='both', which='major', labelsize=14, length=8, width=2)

# Thicker spines
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# Legend - outside the plot
ax.legend(loc='upper left', fontsize=12, frameon=False)

# Title
ax.set_title('Evolution of Discourse Ratios', fontsize=16, fontweight='bold', pad=20)

# Save
output_path = os.path.join(output_dir, "fig_ratio_comparison_3p.pdf")
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
output_path_png = os.path.join(output_dir, "fig_ratio_comparison_3p.png")
plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {output_path}")
print(f"Saved: {output_path_png}")
plt.close()
