"""
Compare Slope Change DID vs Level Change DID
Shows difference between trend-based and mean-based approaches
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def estimate_level_change_did(nk_data, control_data, control_name):
    """
    Standard 2x2 DID: Compares PRE vs POST means

    Model: Y_it = β₀ + β₁*Treat + β₂*Post + β₃*(Treat×Post)

    β₃ = DID estimate (difference in mean changes)
    """
    # Prepare data
    nk_data = nk_data.copy()
    control_data = control_data.copy()

    nk_data['treat'] = 1
    nk_data['group'] = 'NK'
    control_data['treat'] = 0
    control_data['group'] = control_name

    combined = pd.concat([nk_data, control_data], ignore_index=True)
    combined['month_date'] = pd.to_datetime(combined['month'] + '-01')
    combined['post'] = (combined['month_date'] >= '2018-03-01').astype(int)

    # Remove missing values
    data = combined.dropna(subset=['sentiment_mean'])

    print(f"\n{'='*70}")
    print(f"LEVEL CHANGE DID: NK vs {control_name}")
    print(f"{'='*70}")

    # Calculate means for 2x2 table
    nk_pre = data[(data['treat'] == 1) & (data['post'] == 0)]['sentiment_mean'].mean()
    nk_post = data[(data['treat'] == 1) & (data['post'] == 1)]['sentiment_mean'].mean()
    control_pre = data[(data['treat'] == 0) & (data['post'] == 0)]['sentiment_mean'].mean()
    control_post = data[(data['treat'] == 0) & (data['post'] == 1)]['sentiment_mean'].mean()

    nk_change = nk_post - nk_pre
    control_change = control_post - control_pre
    did_estimate = nk_change - control_change

    print("\n2×2 DID Table (Mean Sentiment Scores):")
    print("-" * 70)
    print(f"{'Group':<15} {'Pre (2017-2018.02)':>20} {'Post (2018.03-2019.06)':>20} {'Change (Δ)':>12}")
    print("-" * 70)
    print(f"{'NK':<15} {nk_pre:>20.4f} {nk_post:>20.4f} {nk_change:>12.4f}")
    print(f"{control_name:<15} {control_pre:>20.4f} {control_post:>20.4f} {control_change:>12.4f}")
    print("-" * 70)
    print(f"{'DID Estimate':<15} {'':>20} {'':>20} {did_estimate:>12.4f}")
    print("-" * 70)

    # Regression estimation
    model_ols = smf.ols('sentiment_mean ~ treat + post + treat:post', data=data).fit()

    print(f"\nRegression Results:")
    print(f"  β₃ (Treat×Post): {model_ols.params['treat:post']:+.4f}")
    print(f"  SE: {model_ols.bse['treat:post']:.4f}")
    print(f"  P-value: {model_ols.pvalues['treat:post']:.4f}")
    print(f"  95% CI: [{model_ols.conf_int().loc['treat:post', 0]:.4f}, {model_ols.conf_int().loc['treat:post', 1]:.4f}]")
    print(f"  R²: {model_ols.rsquared:.4f}")

    # Clustered SE
    try:
        model_cluster = smf.ols('sentiment_mean ~ treat + post + treat:post', data=data).fit(
            cov_type='cluster',
            cov_kwds={'groups': data['month']}
        )
        print(f"\nWith Clustered SE:")
        print(f"  β₃ (Treat×Post): {model_cluster.params['treat:post']:+.4f}")
        print(f"  SE: {model_cluster.bse['treat:post']:.4f}")
        print(f"  P-value: {model_cluster.pvalues['treat:post']:.4f}")
        print(f"  95% CI: [{model_cluster.conf_int().loc['treat:post', 0]:.4f}, {model_cluster.conf_int().loc['treat:post', 1]:.4f}]")

        return {
            'did_estimate': float(model_cluster.params['treat:post']),
            'se': float(model_cluster.bse['treat:post']),
            'p_value': float(model_cluster.pvalues['treat:post']),
            'ci_lower': float(model_cluster.conf_int().loc['treat:post', 0]),
            'ci_upper': float(model_cluster.conf_int().loc['treat:post', 1]),
            'means': {
                'nk_pre': nk_pre,
                'nk_post': nk_post,
                'control_pre': control_pre,
                'control_post': control_post
            }
        }
    except Exception as e:
        print(f"\n⚠️ Clustered SE failed: {e}")
        return {
            'did_estimate': float(model_ols.params['treat:post']),
            'se': float(model_ols.bse['treat:post']),
            'p_value': float(model_ols.pvalues['treat:post']),
            'ci_lower': float(model_ols.conf_int().loc['treat:post', 0]),
            'ci_upper': float(model_ols.conf_int().loc['treat:post', 1]),
            'means': {
                'nk_pre': nk_pre,
                'nk_post': nk_post,
                'control_pre': control_pre,
                'control_post': control_post
            }
        }


def main():
    """Compare Level Change DID across all control groups."""

    print("="*70)
    print("LEVEL CHANGE DID COMPARISON")
    print("Standard 2×2 DID (Pre vs Post Means)")
    print("="*70)

    # Load data
    nk_monthly = pd.read_csv('data/processed/nk_monthly_combined_roberta.csv')

    control_groups = {
        'Iran': 'data/processed/iran_monthly_combined_roberta.csv',
        'Russia': 'data/processed/russia_monthly_combined_roberta.csv',
        'China': 'data/processed/china_monthly_combined_roberta.csv'
    }

    results = {}

    for control_name, control_path in control_groups.items():
        control_monthly = pd.read_csv(control_path)
        results[control_name] = estimate_level_change_did(nk_monthly, control_monthly, control_name)

    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY: LEVEL CHANGE DID ACROSS ALL CONTROL GROUPS")
    print("="*70)

    print("\nDID Estimates (Level Change - Post vs Pre Means):")
    print("-" * 70)
    print(f"{'Control':<12} {'DID Est':>10} {'SE':>8} {'P-value':>10} {'95% CI':>20}")
    print("-" * 70)
    for control_name in ['Iran', 'Russia', 'China']:
        r = results[control_name]
        ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
        print(f"{control_name:<12} {r['did_estimate']:>10.4f} {r['se']:>8.4f} {r['p_value']:>10.4f} {ci_str:>20}")

    print("\n" + "="*70)
    print("COMPARISON: SLOPE CHANGE vs LEVEL CHANGE")
    print("="*70)

    print("\nSlope Change DID (Current Method):")
    print("  - Estimates: β₆ (Treat×Time×Post) = +0.0067")
    print("  - P-value: ~0.14 (all controls)")
    print("  - Interpretation: Monthly trend change")

    print("\nLevel Change DID (Alternative):")
    print("  - Estimates: β₃ (Treat×Post) = [see above]")
    print("  - Interpretation: One-time mean shift")

    # Save results
    import json
    os.makedirs('data/results', exist_ok=True)
    with open('data/results/did_level_change_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to: data/results/did_level_change_results.json")


if __name__ == '__main__':
    main()
