"""
Run DID analysis with all three control groups (Iran, Russia, China)
Generates comprehensive comparison report
"""

import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.did_analysis import DIDAnalyzer


def main():
    """Execute DID analysis with all control groups - WEEKLY VERSION."""

    print("=" * 80)
    print("DID ANALYSIS: NK COERCIVE DIPLOMACY (WEEKLY)")
    print("All Control Groups (Iran, Russia, China)")
    print("=" * 80)

    # Load NK data (treatment group)
    print("\nLoading NK weekly data (treatment group)...")
    nk_weekly = pd.read_csv('data/processed/nk_weekly_combined_roberta.csv')
    # Add 'month' column as alias for 'week' (for compatibility with DIDAnalyzer)
    nk_weekly['month'] = nk_weekly['week']
    # Add month_date column using week_start (for DIDAnalyzer)
    nk_weekly['month_date'] = pd.to_datetime(nk_weekly['week_start'])
    print(f"  NK: {len(nk_weekly)} weeks")
    print(f"  Weeks with data: {nk_weekly['sentiment_mean'].notna().sum()}")
    print(f"  Mean sentiment: {nk_weekly['sentiment_mean'].mean():.4f}")

    # Control groups
    control_groups = {
        'Iran': 'data/processed/iran_weekly_combined_roberta.csv',
        'Russia': 'data/processed/russia_weekly_combined_roberta.csv',
        'China': 'data/processed/china_weekly_combined_roberta.csv'
    }

    results = {}

    # Run DID for each control group
    for control_name, control_path in control_groups.items():
        print("\n" + "=" * 80)
        print(f"CONTROL GROUP: {control_name.upper()}")
        print("=" * 80)

        # Load control data
        control_weekly = pd.read_csv(control_path)
        # Add 'month' column as alias for 'week' (for compatibility with DIDAnalyzer)
        control_weekly['month'] = control_weekly['week']
        # Add month_date column using week_start (for DIDAnalyzer)
        control_weekly['month_date'] = pd.to_datetime(control_weekly['week_start'])
        print(f"\nLoaded {control_name} data: {len(control_weekly)} weeks")
        print(f"  Weeks with data: {control_weekly['sentiment_mean'].notna().sum()}")
        print(f"  Mean sentiment: {control_weekly['sentiment_mean'].mean():.4f}")

        # Initialize DID analyzer
        analyzer = DIDAnalyzer(
            treatment_data=nk_weekly,
            control_data=control_weekly,
            control_name=control_name
        )

        # Run full analysis
        control_results = analyzer.run_full_analysis()
        results[control_name] = control_results

        # Print results summary (main output is already printed by DIDAnalyzer)
        print("\n" + "-" * 80)
        print(f"RESULTS SUMMARY: {control_name}")
        print("-" * 80)

        # Parallel trends test
        ols = control_results['ols']
        print(f"\nParallel Trends Test (β₄):")
        print(f"  Time×Treat coefficient: {ols['beta4_pre_trend_diff']:.4f}")
        print(f"  P-value: {ols['beta4_pvalue']:.4f}")
        print(f"  Verdict: {'PASS' if ols['beta4_pvalue'] > 0.10 else 'FAIL'}")

        # DID estimate
        use_clustered = 'clustered' in control_results
        did = control_results['clustered'] if use_clustered else control_results['ols']
        se_type = "Clustered SE" if use_clustered else "OLS SE"

        print(f"\nDID Estimate (Slope Change β₆) with {se_type}:")
        print(f"  Coefficient: {did['did_estimate']:.4f}")
        print(f"  Std Error: {did['se']:.4f}")
        print(f"  P-value: {did['p_value']:.4f}")
        print(f"  95% CI: [{did['ci_lower']:.4f}, {did['ci_upper']:.4f}]")

        if did['p_value'] < 0.05:
            direction = "positive" if did['did_estimate'] > 0 else "negative"
            print(f"  ✓ Statistically significant {direction} effect (p < 0.05)")
        elif did['p_value'] < 0.10:
            print(f"  ~ Marginally significant (p < 0.10)")
        else:
            print(f"  ✗ Not statistically significant (p > 0.10)")

    # Comparison table
    print("\n" + "=" * 80)
    print("COMPARISON ACROSS ALL CONTROL GROUPS")
    print("=" * 80)

    print("\nParallel Trends Test Results:")
    print("-" * 80)
    print(f"{'Control Group':<15} {'Coefficient':>12} {'P-value':>10} {'Verdict':>10}")
    print("-" * 80)
    for control_name in ['Iran', 'Russia', 'China']:
        ols = results[control_name]['ols']
        verdict = 'PASS' if ols['beta4_pvalue'] > 0.10 else 'FAIL'
        print(f"{control_name:<15} {ols['beta4_pre_trend_diff']:>12.4f} {ols['beta4_pvalue']:>10.4f} {verdict:>10}")

    print("\n\nDID Estimates (Slope Change β₆) with Clustered SE:")
    print("-" * 80)
    print(f"{'Control Group':<15} {'Coefficient':>12} {'Std Error':>12} {'P-value':>10} {'95% CI':>25}")
    print("-" * 80)
    for control_name in ['Iran', 'Russia', 'China']:
        use_clustered = 'clustered' in results[control_name]
        did = results[control_name]['clustered'] if use_clustered else results[control_name]['ols']
        ci_str = f"[{did['ci_lower']:.4f}, {did['ci_upper']:.4f}]"
        print(f"{control_name:<15} {did['did_estimate']:>12.4f} {did['se']:>12.4f} {did['p_value']:>10.4f} {ci_str:>25}")

    # Save results
    os.makedirs('data/results', exist_ok=True)

    # Save detailed results to JSON
    import json
    with open('data/results/did_weekly_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE (WEEKLY)")
    print("=" * 80)
    print("\nResults saved to:")
    print("  - data/results/did_weekly_results.json")
    print("  - visualizations/parallel_trends_{iran,russia,china}.png")
    print("  - visualizations/did_event_study_{iran,russia,china}.png")

    print("\nNext steps:")
    print("1. Compare WEEKLY vs MONTHLY results")
    print("2. Check statistical power improvement")
    print("3. Assess data quality (weeks with sparse data)")
    print("4. Decide which aggregation level to use for final analysis")

    return results


if __name__ == '__main__':
    main()
