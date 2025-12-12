"""
Compare Framing DID vs Sentiment DID Results

This script compares DID results between framing-based and sentiment-based analyses
to understand whether discourse framing changes align with sentiment changes.

Key Questions:
1. Do framing and sentiment DID show similar patterns?
2. Is discourse framing a driver of sentiment, or vice versa?
3. Which approach provides stronger evidence of coercive diplomacy effects?
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR, RESULTS_DIR, FIGURES_DIR


def load_results():
    """Load all DID results (framing and sentiment)."""
    print("=" * 80)
    print("Loading DID Results")
    print("=" * 80)

    results = {}

    # Load framing results
    print("\n1. Framing-based DID:")
    try:
        with open(RESULTS_DIR / 'framing_parallel_trends_results.json', 'r') as f:
            results['framing_pt'] = json.load(f)
        print("   ✓ Parallel trends loaded")

        with open(RESULTS_DIR / 'framing_did_slope_results.json', 'r') as f:
            results['framing_slope'] = json.load(f)
        print("   ✓ Slope DID loaded")

        with open(RESULTS_DIR / 'framing_did_level_results.json', 'r') as f:
            results['framing_level'] = json.load(f)
        print("   ✓ Level DID loaded")
    except FileNotFoundError as e:
        print(f"   ✗ ERROR: Framing results not found. Run run_did_analysis_framing.py first.")
        print(f"     {e}")
        return None

    # Load sentiment results
    print("\n2. Sentiment-based DID:")
    try:
        with open(RESULTS_DIR / 'parallel_trends_results.json', 'r') as f:
            results['sentiment_pt'] = json.load(f)
        print("   ✓ Parallel trends loaded")

        with open(RESULTS_DIR / 'did_all_controls_results.json', 'r') as f:
            results['sentiment_slope'] = json.load(f)
        print("   ✓ Slope DID loaded")

        with open(RESULTS_DIR / 'did_level_change_results.json', 'r') as f:
            results['sentiment_level'] = json.load(f)
        print("   ✓ Level DID loaded")
    except FileNotFoundError as e:
        print(f"   ✗ ERROR: Sentiment results not found.")
        print(f"     {e}")
        return None

    print("\n✓ All results loaded successfully")
    return results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create comparison table of framing vs sentiment DID."""
    print("\n" + "="*80)
    print("Creating Comparison Table")
    print("="*80)

    comparison_data = []

    control_groups = ['Iran', 'Russia', 'China']

    for control in control_groups:
        # Parallel trends
        framing_pt = results['framing_pt'][control]
        sentiment_pt = results['sentiment_pt'][control]

        # Slope DID
        framing_slope = results['framing_slope'][control]
        sentiment_slope = results['sentiment_slope'][control]

        # Level DID
        framing_level = results['framing_level'][control]
        sentiment_level = results['sentiment_level'][control]

        comparison_data.append({
            'Control': control,
            'Method': 'Framing',
            'PT_beta': framing_pt['beta4_treat_time'],
            'PT_p': framing_pt['p_value'],
            'PT_pass': framing_pt['pass'],
            'Slope_beta': framing_slope['did_estimate_monthly'],
            'Slope_p': framing_slope['p_value'],
            'Slope_cum15': framing_slope['cumulative_15mo'],
            'Level_beta': framing_level['did_estimate'],
            'Level_p': framing_level['p_value']
        })

        comparison_data.append({
            'Control': control,
            'Method': 'Sentiment',
            'PT_beta': sentiment_pt['beta4_treat_time'],
            'PT_p': sentiment_pt['p_value'],
            'PT_pass': sentiment_pt['pass'],
            'Slope_beta': sentiment_slope['did_estimate_monthly'],
            'Slope_p': sentiment_slope['p_value'],
            'Slope_cum15': sentiment_slope['cumulative_15mo'],
            'Level_beta': sentiment_level['did_estimate'],
            'Level_p': sentiment_level['p_value']
        })

    df = pd.DataFrame(comparison_data)

    print("\nComparison Table:")
    print(df.to_string(index=False))

    return df


def analyze_consistency(comparison_df: pd.DataFrame) -> dict:
    """Analyze consistency between framing and sentiment results."""
    print("\n" + "="*80)
    print("Consistency Analysis")
    print("="*80)

    analysis = {}

    control_groups = ['Iran', 'Russia', 'China']

    for control in control_groups:
        print(f"\n{control} Control:")

        framing = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Framing')].iloc[0]
        sentiment = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Sentiment')].iloc[0]

        # Parallel trends
        pt_both_pass = framing['PT_pass'] and sentiment['PT_pass']
        print(f"  Parallel Trends: Framing {'PASS' if framing['PT_pass'] else 'FAIL'}, "
              f"Sentiment {'PASS' if sentiment['PT_pass'] else 'FAIL'}")

        # Slope DID - Check direction consistency
        slope_same_direction = (framing['Slope_beta'] * sentiment['Slope_beta']) > 0
        print(f"  Slope DID: Framing β={framing['Slope_beta']:+.4f} (p={framing['Slope_p']:.3f}), "
              f"Sentiment β={sentiment['Slope_beta']:+.4f} (p={sentiment['Slope_p']:.3f})")
        print(f"    Direction: {'✓ Same' if slope_same_direction else '✗ Different'}")

        # Level DID - Check direction consistency
        level_same_direction = (framing['Level_beta'] * sentiment['Level_beta']) > 0
        print(f"  Level DID: Framing β={framing['Level_beta']:+.4f} (p={framing['Level_p']:.3f}), "
              f"Sentiment β={sentiment['Level_beta']:+.4f} (p={sentiment['Level_p']:.3f})")
        print(f"    Direction: {'✓ Same' if level_same_direction else '✗ Different'}")

        # Statistical significance
        framing_slope_sig = framing['Slope_p'] < 0.10
        sentiment_slope_sig = sentiment['Slope_p'] < 0.10
        framing_level_sig = framing['Level_p'] < 0.10
        sentiment_level_sig = sentiment['Level_p'] < 0.10

        print(f"  Significance:")
        print(f"    Slope:  Framing {'✓' if framing_slope_sig else '✗'}, "
              f"Sentiment {'✓' if sentiment_slope_sig else '✗'}")
        print(f"    Level:  Framing {'✓' if framing_level_sig else '✗'}, "
              f"Sentiment {'✓' if sentiment_level_sig else '✗'}")

        analysis[control] = {
            'parallel_trends_both_pass': pt_both_pass,
            'slope_same_direction': slope_same_direction,
            'level_same_direction': level_same_direction,
            'framing_slope_sig': framing_slope_sig,
            'sentiment_slope_sig': sentiment_slope_sig,
            'framing_level_sig': framing_level_sig,
            'sentiment_level_sig': sentiment_level_sig
        }

    return analysis


def create_visualization(comparison_df: pd.DataFrame, output_path: Path):
    """Create visualization comparing framing vs sentiment DID."""
    print("\n" + "="*80)
    print("Creating Visualization")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Framing vs Sentiment DID Comparison', fontsize=16, fontweight='bold')

    control_groups = ['Iran', 'Russia', 'China']

    for idx, control in enumerate(control_groups):
        framing = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Framing')].iloc[0]
        sentiment = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Sentiment')].iloc[0]

        # Row 1: Slope DID
        ax1 = axes[0, idx]
        x = ['Framing', 'Sentiment']
        y = [framing['Slope_beta'], sentiment['Slope_beta']]
        colors = ['#2ecc71' if val > 0 else '#e74c3c' for val in y]

        ax1.bar(x, y, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1)
        ax1.set_title(f'{control} Control: Slope DID', fontweight='bold')
        ax1.set_ylabel('β₆ (Monthly Slope Change)')
        ax1.grid(axis='y', alpha=0.3)

        # Add p-values
        for i, (val, p) in enumerate(zip(y, [framing['Slope_p'], sentiment['Slope_p']])):
            sig = '**' if p < 0.05 else '*' if p < 0.10 else 'ns'
            ax1.text(i, val + (0.001 if val > 0 else -0.001), sig,
                    ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

        # Row 2: Level DID
        ax2 = axes[1, idx]
        y2 = [framing['Level_beta'], sentiment['Level_beta']]
        colors2 = ['#2ecc71' if val > 0 else '#e74c3c' for val in y2]

        ax2.bar(x, y2, color=colors2, alpha=0.7, edgecolor='black')
        ax2.axhline(0, color='black', linestyle='--', linewidth=1)
        ax2.set_title(f'{control} Control: Level DID', fontweight='bold')
        ax2.set_ylabel('β₃ (Level Change)')
        ax2.grid(axis='y', alpha=0.3)

        # Add p-values
        for i, (val, p) in enumerate(zip(y2, [framing['Level_p'], sentiment['Level_p']])):
            sig = '**' if p < 0.05 else '*' if p < 0.10 else 'ns'
            ax2.text(i, val + (0.005 if val > 0 else -0.005), sig,
                    ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")

    plt.close()


def generate_summary_report(comparison_df: pd.DataFrame, consistency: dict) -> str:
    """Generate text summary report."""
    print("\n" + "="*80)
    print("Generating Summary Report")
    print("="*80)

    report = []
    report.append("=" * 80)
    report.append("FRAMING VS SENTIMENT DID COMPARISON REPORT")
    report.append("=" * 80)

    report.append("\n1. PARALLEL TRENDS TEST")
    report.append("-" * 80)
    report.append(f"{'Control':<10} {'Framing PT':<15} {'Sentiment PT':<15} {'Both Pass'}")
    report.append("-" * 80)

    for control in ['Iran', 'Russia', 'China']:
        framing = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Framing')].iloc[0]
        sentiment = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Sentiment')].iloc[0]
        both_pass = consistency[control]['parallel_trends_both_pass']

        report.append(f"{control:<10} "
                     f"{'PASS' if framing['PT_pass'] else 'FAIL':<15} "
                     f"{'PASS' if sentiment['PT_pass'] else 'FAIL':<15} "
                     f"{'✓' if both_pass else '✗'}")

    report.append("\n2. SLOPE CHANGE DID (Monthly)")
    report.append("-" * 80)
    report.append(f"{'Control':<10} {'Method':<12} {'β₆':<12} {'P-value':<10} {'Sig'}")
    report.append("-" * 80)

    for control in ['Iran', 'Russia', 'China']:
        framing = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Framing')].iloc[0]
        sentiment = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Sentiment')].iloc[0]

        for method, data in [('Framing', framing), ('Sentiment', sentiment)]:
            sig = '**' if data['Slope_p'] < 0.05 else '*' if data['Slope_p'] < 0.10 else ''
            report.append(f"{control if method=='Framing' else '':<10} "
                         f"{method:<12} "
                         f"{data['Slope_beta']:+.6f}   "
                         f"{data['Slope_p']:.4f}    "
                         f"{sig}")

        # Direction consistency
        same_dir = consistency[control]['slope_same_direction']
        report.append(f"{'':10} Direction: {'✓ Same' if same_dir else '✗ Different'}")
        report.append("")

    report.append("3. LEVEL CHANGE DID")
    report.append("-" * 80)
    report.append(f"{'Control':<10} {'Method':<12} {'β₃':<12} {'P-value':<10} {'Sig'}")
    report.append("-" * 80)

    for control in ['Iran', 'Russia', 'China']:
        framing = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Framing')].iloc[0]
        sentiment = comparison_df[(comparison_df['Control']==control) & (comparison_df['Method']=='Sentiment')].iloc[0]

        for method, data in [('Framing', framing), ('Sentiment', sentiment)]:
            sig = '**' if data['Level_p'] < 0.05 else '*' if data['Level_p'] < 0.10 else ''
            report.append(f"{control if method=='Framing' else '':<10} "
                         f"{method:<12} "
                         f"{data['Level_beta']:+.6f}   "
                         f"{data['Level_p']:.4f}    "
                         f"{sig}")

        # Direction consistency
        same_dir = consistency[control]['level_same_direction']
        report.append(f"{'':10} Direction: {'✓ Same' if same_dir else '✗ Different'}")
        report.append("")

    report.append("4. KEY FINDINGS")
    report.append("-" * 80)

    # Count consistencies
    all_same_slope = all(consistency[c]['slope_same_direction'] for c in ['Iran', 'Russia', 'China'])
    all_same_level = all(consistency[c]['level_same_direction'] for c in ['Iran', 'Russia', 'China'])

    if all_same_slope and all_same_level:
        report.append("✓ CONSISTENT: Framing and sentiment show same direction across all controls")
        report.append("  → Discourse framing changes align with sentiment changes")
        report.append("  → Framing may be a driver of sentiment (or vice versa)")
    else:
        report.append("✗ INCONSISTENT: Framing and sentiment show different patterns")
        report.append("  → Discourse framing changes do NOT align with sentiment")
        report.append("  → Different mechanisms at work")

    # Significance comparison
    report.append("\n5. STATISTICAL STRENGTH")
    report.append("-" * 80)

    for control in ['Iran', 'Russia', 'China']:
        c = consistency[control]
        report.append(f"\n{control} Control:")
        report.append(f"  Slope DID:  Framing {'sig' if c['framing_slope_sig'] else 'ns'}, "
                     f"Sentiment {'sig' if c['sentiment_slope_sig'] else 'ns'}")
        report.append(f"  Level DID:  Framing {'sig' if c['framing_level_sig'] else 'ns'}, "
                     f"Sentiment {'sig' if c['sentiment_level_sig'] else 'ns'}")

    report.append("\n" + "=" * 80)

    return "\n".join(report)


def main():
    """Main execution function."""
    print("=" * 80)
    print("FRAMING vs SENTIMENT DID COMPARISON")
    print("=" * 80)

    # Load results
    results = load_results()
    if results is None:
        print("\nERROR: Could not load all required results.")
        return

    # Create comparison table
    comparison_df = create_comparison_table(results)

    # Analyze consistency
    consistency = analyze_consistency(comparison_df)

    # Create visualization
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    viz_path = FIGURES_DIR / 'framing_vs_sentiment_comparison.png'
    create_visualization(comparison_df, viz_path)

    # Generate summary report
    report_text = generate_summary_report(comparison_df, consistency)
    print("\n" + report_text)

    # Save report
    report_path = RESULTS_DIR / 'framing_vs_sentiment_comparison.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\n✓ Saved report: {report_path}")

    # Save comparison data
    comparison_path = RESULTS_DIR / 'framing_vs_sentiment_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✓ Saved comparison data: {comparison_path}")

    # Save consistency analysis
    consistency_path = RESULTS_DIR / 'framing_vs_sentiment_consistency.json'
    with open(consistency_path, 'w') as f:
        json.dump(consistency, f, indent=2)
    print(f"✓ Saved consistency analysis: {consistency_path}")

    print("\n" + "="*80)
    print("✓ COMPARISON COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
