"""
Generate Paper Figures: DID Visualizations
- Figure 4: Framing DID (A: Singapore Effect, B: Hanoi Effect)
- Figure 4b: Sentiment DID (A: Singapore Effect, B: Hanoi Effect)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR

# DID values from ICWSM_draft.md
FRAMING_DID = {
    'singapore': {'China': 1.28, 'Iran': 0.85},
    'hanoi': {'China': -0.88, 'Iran': -0.30}
}

SENTIMENT_DID = {
    'singapore': {'China': 0.21, 'Iran': 0.10, 'Russia': 0.14},
    'hanoi': {'China': -0.11, 'Iran': -0.06, 'Russia': -0.12}
}

# Period definitions (full data range: 2017-01 to 2019-12)
PERIODS = {
    'P1': ('2017-01', '2018-02'),  # Pre-Announcement
    'P2': ('2018-06', '2019-01'),  # Singapore-Hanoi
    'P3': ('2019-03', '2019-12')   # Post-Hanoi
}


def load_monthly_data(outcome='framing'):
    """Load monthly aggregated data for all countries."""
    data = {}
    
    if outcome == 'framing':
        files = {
            'NK': DATA_DIR / 'framing' / 'nk_monthly_framing.csv',
            'China': DATA_DIR / 'framing' / 'china_monthly_framing.csv',
            'Iran': DATA_DIR / 'framing' / 'iran_monthly_framing.csv',
        }
        score_col = 'framing_mean'
    else:  # sentiment
        files = {
            'NK': DATA_DIR / 'sentiment' / 'nk_monthly_sentiment.csv',
            'China': DATA_DIR / 'sentiment' / 'china_monthly_sentiment.csv',
            'Iran': DATA_DIR / 'sentiment' / 'iran_monthly_sentiment.csv',
            'Russia': DATA_DIR / 'sentiment' / 'russia_monthly_sentiment.csv',
        }
        score_col = 'sentiment_mean'
    
    for name, filepath in files.items():
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['month'] = pd.to_datetime(df['month'].astype(str) + '-01')
            data[name] = df
            print(f"  Loaded {name}: {len(df)} months, {score_col} range: {df[score_col].min():.2f} to {df[score_col].max():.2f}")
        else:
            print(f"  Warning: {filepath} not found")
    
    return data, score_col


def get_period_means(df, score_col, period):
    """Calculate mean score for a given period."""
    start, end = PERIODS[period]
    start_date = pd.to_datetime(start + '-01')
    end_date = pd.to_datetime(end + '-01')
    
    mask = (df['month'] >= start_date) & (df['month'] <= end_date)
    result = df.loc[mask, score_col].mean()
    return result


def create_did_figure(outcome='framing', save_path=None):
    """
    Create 2-panel DID figure: (A) Singapore Effect, (B) Hanoi Effect
    
    Args:
        outcome: 'framing' or 'sentiment'
        save_path: Path to save figure
    """
    data, score_col = load_monthly_data(outcome)
    
    if 'NK' not in data:
        print("Error: NK data not found")
        return
    
    # Get DID values
    did_values = FRAMING_DID if outcome == 'framing' else SENTIMENT_DID
    
    # Colors
    colors = {
        'NK': '#E63946',
        'China': '#457B9D',
        'Iran': '#2A9D8F',
        'Russia': '#9B59B6'
    }
    
    # Create figure with extra space for legend
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Main title with outcome type
    outcome_title = "Framing" if outcome == 'framing' else "Sentiment"
    fig.suptitle(f'{outcome_title} Difference-in-Differences Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    controls = ['China', 'Iran'] if outcome == 'framing' else ['China', 'Iran', 'Russia']
    x_pos = [0, 1]
    
    # Panel A: Singapore Effect (P1 → P2)
    ax = axes[0]
    ax.set_title('(A) Singapore Summit Effect (P1 → P2)', fontsize=12, fontweight='bold', pad=10)
    
    # Plot NK
    nk_p1 = get_period_means(data['NK'], score_col, 'P1')
    nk_p2 = get_period_means(data['NK'], score_col, 'P2')
    print(f"  NK P1→P2: {nk_p1:.3f} → {nk_p2:.3f}")
    ax.plot(x_pos, [nk_p1, nk_p2], 'o-', color=colors['NK'], linewidth=2.5, 
            markersize=10, label='North Korea', zorder=3)
    
    # Plot controls
    for ctrl in controls:
        if ctrl in data:
            ctrl_p1 = get_period_means(data[ctrl], score_col, 'P1')
            ctrl_p2 = get_period_means(data[ctrl], score_col, 'P2')
            print(f"  {ctrl} P1→P2: {ctrl_p1:.3f} → {ctrl_p2:.3f}")
            ax.plot(x_pos, [ctrl_p1, ctrl_p2], 's--', color=colors[ctrl], 
                    linewidth=2, markersize=8, label=ctrl, alpha=0.8, zorder=2)
    
    # Add DID annotations at left-center area (avoid covering data lines in Panel A)
    for i, ctrl in enumerate(controls):
        if ctrl in did_values['singapore']:
            val = did_values['singapore'][ctrl]
            sign = '+' if val > 0 else ''
            ax.text(0.55, 0.98 - i*0.05, f'DiD ({ctrl}): {sign}{val:.2f}', 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   color=colors[ctrl], verticalalignment='top', horizontalalignment='center',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['P1\n(Pre-Singapore)', 'P2\n(Singapore-Hanoi)'])
    ax.set_ylabel(f'{outcome_title} Score', fontsize=11)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    # ax.grid removed for modern style
    
    # Panel B: Hanoi Effect (P2 → P3)
    ax = axes[1]
    ax.set_title('(B) Hanoi Summit Effect (P2 → P3)', fontsize=12, fontweight='bold', pad=10)
    
    # Plot NK
    nk_p2 = get_period_means(data['NK'], score_col, 'P2')
    nk_p3 = get_period_means(data['NK'], score_col, 'P3')
    print(f"  NK P2→P3: {nk_p2:.3f} → {nk_p3:.3f}")
    ax.plot(x_pos, [nk_p2, nk_p3], 'o-', color=colors['NK'], linewidth=2.5, 
            markersize=10, label='North Korea', zorder=3)
    
    # Plot controls
    for ctrl in controls:
        if ctrl in data:
            ctrl_p2 = get_period_means(data[ctrl], score_col, 'P2')
            ctrl_p3 = get_period_means(data[ctrl], score_col, 'P3')
            print(f"  {ctrl} P2→P3: {ctrl_p2:.3f} → {ctrl_p3:.3f}")
            ax.plot(x_pos, [ctrl_p2, ctrl_p3], 's--', color=colors[ctrl], 
                    linewidth=2, markersize=8, label=ctrl, alpha=0.8, zorder=2)
    
    # Add DID annotations at top-right corner (avoid covering data)
    for i, ctrl in enumerate(controls):
        if ctrl in did_values['hanoi']:
            val = did_values['hanoi'][ctrl]
            sign = '+' if val > 0 else ''
            ax.text(0.98, 0.98 - i*0.05, f'DiD ({ctrl}): {sign}{val:.2f}', 
                   transform=ax.transAxes, fontsize=9, fontweight='bold',
                   color=colors[ctrl], verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['P2\n(Singapore-Hanoi)', 'P3\n(Post-Hanoi)'])
    ax.set_ylabel(f'{outcome_title} Score', fontsize=11)
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
    # ax.grid removed for modern style
    
    # Single legend below the figure (outside graphs)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(controls)+1, 
               fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved: {save_path}")
    
    plt.close()


def main():
    print("=" * 70)
    print("Generating Paper DID Figures")
    print("=" * 70)
    
    output_dir = project_root / 'paper' / 'figures'
    
    # Figure 4: Framing DID
    print("\n1. Generating Figure 4: Framing DID...")
    create_did_figure(
        outcome='framing',
        save_path=output_dir / 'fig4_did_visualization.pdf'
    )
    
    # Figure 4b: Sentiment DID
    print("\n2. Generating Figure 4b: Sentiment DID...")
    create_did_figure(
        outcome='sentiment',
        save_path=output_dir / 'fig4b_sentiment_did_visualization.pdf'
    )
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
