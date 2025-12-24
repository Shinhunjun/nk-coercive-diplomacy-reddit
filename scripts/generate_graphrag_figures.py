#!/usr/bin/env python3
"""
Generate visualizations for GraphRAG results in paper.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13

OUTPUT_DIR = Path("paper/latex/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    'p1': '#4a7c59',  # Green
    'p2': '#f4a259',  # Orange
    'p3': '#bc4749',  # Red
    'threat': '#d62828',
    'peace': '#2a9d8f',
    'neutral': '#6c757d'
}


def create_network_density_figure():
    """Create network density trend visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    periods = ['P1\n(Pre-Singapore)', 'P2\n(Singapore-Hanoi)', 'P3\n(Post-Hanoi)']
    colors = [COLORS['p1'], COLORS['p2'], COLORS['p3']]
    
    # Panel A: Density
    density = [0.0013, 0.0032, 0.0037]
    axes[0].bar(periods, density, color=colors, edgecolor='black', linewidth=1)
    axes[0].set_ylabel('Network Density')
    axes[0].set_title('(A) Network Density')
    axes[0].set_ylim(0, 0.005)
    for i, v in enumerate(density):
        axes[0].text(i, v + 0.0002, f'{v:.4f}', ha='center', fontsize=10)
    
    # Panel B: Node Count
    nodes = [2656, 1043, 879]
    axes[1].bar(periods, nodes, color=colors, edgecolor='black', linewidth=1)
    axes[1].set_ylabel('Number of Nodes')
    axes[1].set_title('(B) Entity Count')
    for i, v in enumerate(nodes):
        axes[1].text(i, v + 50, f'{v:,}', ha='center', fontsize=10)
    
    # Panel C: Communities
    communities = [354, 153, 107]
    axes[2].bar(periods, communities, color=colors, edgecolor='black', linewidth=1)
    axes[2].set_ylabel('Number of Communities')
    axes[2].set_title('(C) Community Count')
    for i, v in enumerate(communities):
        axes[2].text(i, v + 10, f'{v}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_network_topology.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_network_topology.pdf', bbox_inches='tight')
    plt.close()
    print("Created: fig_network_topology.png/pdf")


def create_framing_distribution_figure():
    """Create relationship framing distribution visualization."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    periods = ['P1\n(Pre-Singapore)', 'P2\n(Singapore-Hanoi)', 'P3\n(Post-Hanoi)']
    
    threat = [61.5, 42.4, 40.9]
    peace = [7.8, 24.7, 18.6]
    neutral = [30.8, 32.9, 40.5]
    
    x = np.arange(len(periods))
    width = 0.6
    
    # Stacked bar
    bars1 = ax.bar(x, threat, width, label='Threat', color=COLORS['threat'])
    bars2 = ax.bar(x, peace, width, bottom=threat, label='Peace', color=COLORS['peace'])
    bars3 = ax.bar(x, neutral, width, bottom=np.array(threat)+np.array(peace), 
                   label='Neutral', color=COLORS['neutral'])
    
    # Add annotations
    for i, (t, p, n) in enumerate(zip(threat, peace, neutral)):
        ax.text(i, t/2, f'{t:.1f}%', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, t + p/2, f'{p:.1f}%', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, t + p + n/2, f'{n:.1f}%', ha='center', va='center', color='white', fontweight='bold')
    
    ax.set_ylabel('Percentage of Relationships')
    ax.set_title('Relationship Framing Distribution Across Diplomatic Periods')
    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.set_ylim(0, 105)
    
    # Add arrows for key changes
    ax.annotate('', xy=(1, 45), xytext=(0, 63),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(0.5, 58, '-19pp', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_framing_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_framing_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("Created: fig_framing_distribution.png/pdf")


def create_keyword_comparison_figure():
    """Create keyword comparison visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    data = {
        'P1': [('military', 252), ('nuclear', 203), ('international', 152), 
               ('security', 117), ('diplomatic', 96), ('geopolitical', 92)],
        'P2': [('nuclear', 91), ('military', 84), ('international', 67),
               ('diplomatic', 62), ('relations', 50), ('efforts', 41)],
        'P3': [('international', 58), ('diplomatic', 44), ('military', 38),
               ('kim', 37), ('trump', 33), ('security', 30)]
    }
    
    period_colors = [COLORS['p1'], COLORS['p2'], COLORS['p3']]
    titles = ['(A) P1: Pre-Singapore', '(B) P2: Singapore-Hanoi', '(C) P3: Post-Hanoi']
    
    for ax, (period, keywords), color, title in zip(axes, data.items(), period_colors, titles):
        words = [k[0] for k in keywords]
        counts = [k[1] for k in keywords]
        
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, color=color, edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_title(title)
        
        for i, v in enumerate(counts):
            ax.text(v + 2, i, str(v), va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig_keyword_evolution.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'fig_keyword_evolution.pdf', bbox_inches='tight')
    plt.close()
    print("Created: fig_keyword_evolution.png/pdf")


if __name__ == "__main__":
    print("Generating GraphRAG visualizations...")
    create_network_density_figure()
    create_framing_distribution_figure()
    create_keyword_comparison_figure()
    print("\nAll figures saved to:", OUTPUT_DIR)
