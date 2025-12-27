"""
Generate Paper Figures: Timeline, Framing Trends, Frame Heatmap
- Remove "Figure N" labels from titles
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

from config import DATA_DIR

OUTPUT_DIR = project_root / 'paper' / 'figures'


def create_timeline_figure():
    """Create research timeline figure without Figure N label."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Define periods
    periods = [
        ('2017-01', '2018-02', 'P1: Pre-Announcement', '#808080', 0.3),
        ('2018-03', '2018-05', 'Transition', '#FFD700', 0.3),
        ('2018-06', '2019-01', 'P2: Singapore-Hanoi', 'green', 0.1),
        ('2019-02', '2019-02', 'Transition', '#FFD700', 0.3),
        ('2019-03', '2019-12', 'P3: Post-Hanoi', '#E63946', 0.3),
    ]
    
    # Create timeline
    for start, end, label, color, alpha in periods:
        start_date = pd.to_datetime(start + '-01')
        end_date = pd.to_datetime(end + '-28')
        ax.axvspan(start_date, end_date, alpha=alpha, color=color, label=label if label not in [p[2] for p in periods[:periods.index((start, end, label, color, alpha))]] else '')
    
    # Key events
    events = [
        ('2018-03-08', 'Summit\nAnnouncement', -0.3),
        ('2018-06-12', 'Singapore\nSummit', 0.3),
        ('2019-02-27', 'Hanoi\nSummit', -0.3),
    ]
    
    for date, label, y_offset in events:
        event_date = pd.to_datetime(date)
        ax.axvline(event_date, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.annotate(label, xy=(event_date, 0.5 + y_offset), 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))
    
    # Formatting
    ax.set_xlim(pd.to_datetime('2017-01-01'), pd.to_datetime('2019-12-31'))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Title without Figure N
    ax.set_title('Research Timeline and Key Events', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=11)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'fig1_timeline.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_framing_trends_figure():
    """Create framing trends figure without Figure N label."""
    # Load data
    countries = ['nk', 'china', 'iran']
    data = {}
    
    for name in countries:
        filepath = DATA_DIR / 'framing' / f'{name}_monthly_framing.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['month'] = pd.to_datetime(df['month'].astype(str) + '-01')
            data[name] = df
    
    colors = {'nk': '#E63946', 'china': '#457B9D', 'iran': '#2A9D8F'}
    labels = {'nk': 'North Korea', 'china': 'China', 'iran': 'Iran'}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each country
    for name, df in data.items():
        ax.plot(df['month'], df['framing_mean'], 'o-', 
                color=colors[name], label=labels[name], linewidth=2, markersize=4, alpha=0.8)
    
    # Event lines
    events = [
        ('2018-06-12', 'Singapore Summit'),
        ('2019-02-27', 'Hanoi Summit'),
    ]
    for date, label in events:
        ax.axvline(pd.to_datetime(date), color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Period shading
    ax.axvspan(pd.to_datetime('2017-01-01'), pd.to_datetime('2018-02-28'), alpha=0.1, color='gray', label='P1')
    ax.axvspan(pd.to_datetime('2018-06-01'), pd.to_datetime('2019-01-31'), alpha=0.1, color='green', label='P2')
    ax.axvspan(pd.to_datetime('2019-03-01'), pd.to_datetime('2019-12-31'), alpha=0.1, color='red', label='P3')
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Framing Score', fontsize=11)
    ax.set_title('Monthly Framing Score Trends: NK vs. Control Groups', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'fig2_framing_trends.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_sentiment_trends_figure():
    """Create sentiment trends figure without Figure N label."""
    # Load data
    countries = ['nk', 'china', 'iran', 'russia']
    data = {}
    
    for name in countries:
        filepath = DATA_DIR / 'sentiment' / f'{name}_monthly_sentiment.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['month'] = pd.to_datetime(df['month'].astype(str) + '-01')
            data[name] = df
    
    colors = {'nk': '#E63946', 'china': '#457B9D', 'iran': '#2A9D8F', 'russia': '#9B59B6'}
    labels = {'nk': 'North Korea', 'china': 'China', 'iran': 'Iran', 'russia': 'Russia'}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot each country
    for name, df in data.items():
        ax.plot(df['month'], df['sentiment_mean'], 'o-', 
                color=colors[name], label=labels[name], linewidth=2, markersize=4, alpha=0.8)
    
    # Event lines
    events = [
        ('2018-06-12', 'Singapore Summit'),
        ('2019-02-27', 'Hanoi Summit'),
    ]
    for date, label in events:
        ax.axvline(pd.to_datetime(date), color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Period shading
    ax.axvspan(pd.to_datetime('2017-01-01'), pd.to_datetime('2018-02-28'), alpha=0.1, color='gray', label='P1')
    ax.axvspan(pd.to_datetime('2018-06-01'), pd.to_datetime('2019-01-31'), alpha=0.1, color='green', label='P2')
    ax.axvspan(pd.to_datetime('2019-03-01'), pd.to_datetime('2019-12-31'), alpha=0.1, color='red', label='P3')
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Sentiment Score', fontsize=11)
    ax.set_title('Monthly Sentiment Score Trends: NK vs. Control Groups', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'fig6_sentiment_trends.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_frame_heatmap():
    """Create frame distribution heatmap without Figure N label."""
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
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im = ax.imshow(frame_pct.T.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=60)
    
    ax.set_yticks(range(len(frame_pct.columns)))
    ax.set_yticklabels(frame_pct.columns)
    
    # Show every 3rd month
    months = frame_pct.index.tolist()
    ax.set_xticks(range(0, len(months), 3))
    ax.set_xticklabels([months[i] for i in range(0, len(months), 3)], rotation=45, ha='right')
    
    ax.set_xlabel('Month', fontsize=11)
    ax.set_ylabel('Frame Category', fontsize=11)
    ax.set_title('Frame Category Distribution Over Time (North Korea)', fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Percentage (%)', fontsize=10)
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'fig5_frame_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Generating Paper Figures (without Figure N labels)")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Creating Timeline figure...")
    create_timeline_figure()
    
    print("\n2. Creating Framing Trends figure...")
    create_framing_trends_figure()
    
    print("\n3. Creating Sentiment Trends figure...")
    create_sentiment_trends_figure()
    
    print("\n4. Creating Frame Heatmap...")
    create_frame_heatmap()
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
