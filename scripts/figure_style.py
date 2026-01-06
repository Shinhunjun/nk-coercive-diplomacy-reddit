#!/usr/bin/env python3
"""
Master style configuration for all paper figures.
Apply this at the start of any figure generation script.
Mohit's style: larger ticks, no grid, vibrant colors, no box frame.
"""

import matplotlib.pyplot as plt

def apply_modern_style():
    """Apply modern minimalist style to matplotlib figures."""
    plt.rcParams.update({
        # Font settings
        'font.family': 'sans-serif',
        'font.size': 12,
        
        # Axes settings - NO grid, minimal spines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.linewidth': 1.5,
        'axes.grid': False,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'axes.labelweight': 'bold',
        
        # Tick settings - larger
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        
        # Legend settings
        'legend.frameon': False,
        'legend.fontsize': 11,
        
        # Figure settings
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })

# Vibrant color palette for figures
COLORS = {
    'primary': '#4ECDC4',      # Teal
    'secondary': '#FF6B6B',    # Coral
    'accent1': '#45B7D1',      # Sky blue
    'accent2': '#96CEB4',      # Sage green
    'accent3': '#FFEAA7',      # Light yellow
    'accent4': '#DDA0DD',      # Plum
    'nk': '#FF6B6B',           # North Korea - Coral
    'china': '#4ECDC4',        # China - Teal
    'iran': '#45B7D1',         # Iran - Sky blue
    'russia': '#96CEB4',       # Russia - Sage green
    'diplomacy': '#2ECC71',    # Diplomacy frame - Green (Singapore Summit success)
    'threat': '#E74C3C',       # Threat frame - Red (Hanoi Summit failure)
    'neutral': '#95A5A6',      # Neutral - Gray
    'economic': '#F39C12',     # Economic - Orange
    'humanitarian': '#9B59B6', # Humanitarian - Purple
}

# Period colors
PERIOD_COLORS = {
    'P1': '#757575',  # Gray
    'P2': '#4CAF50',  # Green
    'P3': '#F44336',  # Red
    'Transition': '#FFC107',  # Yellow
}

if __name__ == "__main__":
    apply_modern_style()
    print("Modern style applied!")
