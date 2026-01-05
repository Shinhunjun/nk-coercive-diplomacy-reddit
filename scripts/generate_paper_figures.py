"""
Generate Paper Figures: Timeline, Framing Trends, Frame Heatmap
Updated with modern minimalist style (Mohit's recommendations)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "scripts"))

from config import DATA_DIR
from figure_style import apply_modern_style, COLORS

OUTPUT_DIR = project_root / 'paper' / 'figures'

# Apply modern style
apply_modern_style()


def create_framing_trends_figure():
    """Create framing trends figure with modern style."""
    # Load data
    countries = ['nk', 'china', 'iran']
    data = {}
    
    for name in countries:
        filepath = DATA_DIR / 'framing' / f'{name}_monthly_framing.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['month'] = pd.to_datetime(df['month'].astype(str) + '-01')
            data[name] = df
    
    colors = {'nk': COLORS['nk'], 'china': COLORS['china'], 'iran': COLORS['iran']}
    labels = {'nk': 'North Korea', 'china': 'China', 'iran': 'Iran'}
    markers = {'nk': 'o', 'china': 's', 'iran': '^'}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each country
    for name, df in data.items():
        ax.plot(df['month'], df['framing_mean'], markers[name] + '-', 
                color=colors[name], label=labels[name], linewidth=2.5, markersize=8, alpha=0.9)
    
    # Event lines (vertical dashed)
    events = [
        ('2018-06-12', 'Singapore Summit', COLORS['diplomacy']),
        ('2019-02-27', 'Hanoi Summit', COLORS['threat']),
    ]
    for date, label, color in events:
        ax.axvline(pd.to_datetime(date), color=color, linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Framing Score', fontsize=12, fontweight='bold')
    ax.set_title('Monthly Framing Score Trends: NK vs. Control Groups', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Thicker spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save as both PNG and PDF
    save_path_pdf = OUTPUT_DIR / 'fig2_framing_trends.pdf'
    save_path_png = OUTPUT_DIR / 'fig2_framing_trends.png'
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path_pdf}")
    print(f"✓ Saved: {save_path_png}")
    plt.close()


def create_sentiment_trends_figure():
    """Create sentiment trends figure with modern style."""
    # Load data
    countries = ['nk', 'china', 'iran', 'russia']
    data = {}
    
    for name in countries:
        filepath = DATA_DIR / 'sentiment' / f'{name}_monthly_sentiment.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['month'] = pd.to_datetime(df['month'].astype(str) + '-01')
            data[name] = df
    
    colors = {
        'nk': COLORS['nk'], 
        'china': COLORS['china'], 
        'iran': COLORS['iran'], 
        'russia': COLORS['russia']
    }
    labels = {'nk': 'North Korea', 'china': 'China', 'iran': 'Iran', 'russia': 'Russia'}
    markers = {'nk': 'o', 'china': 's', 'iran': '^', 'russia': 'd'}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each country
    for name, df in data.items():
        ax.plot(df['month'], df['sentiment_mean'], markers[name] + '-', 
                color=colors[name], label=labels[name], linewidth=2.5, markersize=8, alpha=0.9)
    
    # Event lines
    events = [
        ('2018-06-12', 'Singapore Summit', COLORS['diplomacy']),
        ('2019-02-27', 'Hanoi Summit', COLORS['threat']),
    ]
    for date, label, color in events:
        ax.axvline(pd.to_datetime(date), color=color, linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
    ax.set_title('Monthly Sentiment Score Trends: NK vs. Control Groups', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Thicker spines
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.tight_layout()
    
    # Save as both PNG and PDF
    save_path_pdf = OUTPUT_DIR / 'fig6_sentiment_trends.pdf'
    save_path_png = OUTPUT_DIR / 'fig6_sentiment_trends.png'
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path_pdf}")
    print(f"✓ Saved: {save_path_png}")
    plt.close()


def create_frame_heatmap():
    """Create frame distribution heatmap with modern style."""
    # Load framing data
    filepath = DATA_DIR / 'final' / 'nk_framing_final.csv'
    if not filepath.exists():
        print(f"Error: {filepath} not found")
        return
    
    df = pd.read_csv(filepath)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    
    # Count frames by month
    frame_counts = df.groupby(['month', 'frame']).size().unstack(fill_value=0)
    
    # Normalize to percentages
    frame_pct = frame_counts.div(frame_counts.sum(axis=1), axis=0) * 100
    
    # Reorder frames
    frame_order = ['THREAT', 'ECONOMIC', 'NEUTRAL', 'HUMANITARIAN', 'DIPLOMACY']
    frame_pct = frame_pct.reindex(columns=[f for f in frame_order if f in frame_pct.columns])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Use a more vibrant colormap
    im = ax.imshow(frame_pct.T.values, aspect='auto', cmap='viridis', vmin=0, vmax=60)
    
    ax.set_yticks(range(len(frame_pct.columns)))
    ax.set_yticklabels(frame_pct.columns, fontsize=12, fontweight='bold')
    
    # Show every 3rd month
    months = frame_pct.index.tolist()
    ax.set_xticks(range(0, len(months), 3))
    ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha='right', fontsize=11)
    
    ax.set_xlabel('Month', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frame Category', fontsize=12, fontweight='bold')
    ax.set_title('Frame Category Distribution Over Time (North Korea)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=11, fontweight='bold')
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Save as both PNG and PDF
    save_path_pdf = OUTPUT_DIR / 'fig5_frame_heatmap.pdf'
    save_path_png = OUTPUT_DIR / 'fig5_frame_heatmap.png'
    plt.savefig(save_path_pdf, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path_pdf}")
    print(f"✓ Saved: {save_path_png}")
    plt.close()


def main():
    print("=" * 70)
    print("Generating Paper Figures (Modern Style)")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Creating Framing Trends figure...")
    create_framing_trends_figure()
    
    print("\n2. Creating Sentiment Trends figure...")
    create_sentiment_trends_figure()
    
    print("\n3. Creating Frame Heatmap...")
    create_frame_heatmap()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
