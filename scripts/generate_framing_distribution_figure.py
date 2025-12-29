"""
Generate improved Framing Distribution visualization.
Grouped bar chart: X-axis = Frame types, Bars = Periods (P1, P2, P3)
Shows all 5 frames with percentages labeled.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'

# Edge framing data (from LLM classification)
edge_data = {
    'Frame': ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN'],
    'P1': [48.4, 20.6, 22.0, 5.7, 3.3],
    'P2': [28.0, 38.9, 24.6, 6.2, 2.3],
    'P3': [25.6, 32.2, 26.8, 8.3, 7.1]
}

# Community framing data (from LLM classification)
community_data = {
    'Frame': ['THREAT', 'DIPLOMACY', 'NEUTRAL', 'ECONOMIC', 'HUMANITARIAN'],
    'P1': [52.0, 24.6, 13.0, 3.1, 7.3],
    'P2': [32.7, 42.5, 17.0, 3.3, 4.6],
    'P3': [26.2, 34.6, 16.8, 8.4, 14.0]
}

# Colors for periods
period_colors = ['#2C3E50', '#3498DB', '#95A5A6']  # Dark blue, Light blue, Gray
period_labels = ['P1 (Pre-Singapore)', 'P2 (Singapore-Hanoi)', 'P3 (Post-Hanoi)']

def create_grouped_bar_chart(data, title, output_path):
    """Create grouped bar chart for framing distribution."""
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df['Frame']))
    width = 0.25
    
    bars1 = ax.bar(x - width, df['P1'], width, label=period_labels[0], color=period_colors[0])
    bars2 = ax.bar(x, df['P2'], width, label=period_labels[1], color=period_colors[1])
    bars3 = ax.bar(x + width, df['P3'], width, label=period_labels[2], color=period_colors[2])
    
    # Add percentage labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Frame Category', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Frame'], fontsize=11)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 60)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()

# Generate figures
output_dir = Path('/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/paper/figures')

# Edge framing visualization
create_grouped_bar_chart(
    edge_data, 
    'Edge Framing Distribution Across Periods',
    output_dir / 'fig_framing_distribution.png'
)

# Community framing visualization  
create_grouped_bar_chart(
    community_data,
    'Community Framing Distribution Across Periods',
    output_dir / 'fig_community_framing_distribution.png'
)

print("\n✓ All figures generated!")
