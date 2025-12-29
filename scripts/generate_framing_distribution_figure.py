"""
Generate Figure: Relationship Framing Distribution (LLM-classified)
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

# Data from LLM classification results
data = {
    'Period': ['P1\n(Pre-Singapore)', 'P2\n(Singapore-Hanoi)', 'P3\n(Post-Hanoi)'],
    'THREAT': [48.4, 28.0, 25.6],
    'DIPLOMACY': [20.6, 38.9, 32.2],
    'NEUTRAL': [22.0, 24.6, 26.8],
    'ECONOMIC': [5.7, 6.2, 8.3],
    'HUMANITARIAN': [3.3, 2.3, 7.1]
}

df = pd.DataFrame(data)

# Create stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

colors = {
    'THREAT': '#E63946',      # Red
    'DIPLOMACY': '#2A9D8F',   # Teal
    'NEUTRAL': '#A8DADC',     # Light blue
    'ECONOMIC': '#F4A261',    # Orange
    'HUMANITARIAN': '#9B59B6' # Purple
}

frames = ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN']
bottom = [0, 0, 0]

for frame in frames:
    ax.bar(df['Period'], df[frame], bottom=bottom, label=frame, color=colors[frame], width=0.6)
    bottom = [b + v for b, v in zip(bottom, df[frame])]

# Formatting
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_xlabel('')
ax.set_title('Relationship Framing Distribution Across Periods (LLM-classified)', fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 105)

# Add percentage labels for main frames
for i, period in enumerate(df['Period']):
    # THREAT label
    threat_val = df.loc[i, 'THREAT']
    ax.text(i, threat_val/2, f'{threat_val:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    
    # DIPLOMACY label
    dip_val = df.loc[i, 'DIPLOMACY']
    dip_pos = threat_val + dip_val/2
    ax.text(i, dip_pos, f'{dip_val:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

plt.tight_layout()

# Save
output_path = Path('/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/paper/figures/fig_framing_distribution.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved: {output_path}")
