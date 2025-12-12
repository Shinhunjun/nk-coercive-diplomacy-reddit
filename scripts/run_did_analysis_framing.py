"""
Difference-in-Differences Analysis Using Framing Scale

This script runs DID analysis using the diplomacy_scale instead of sentiment.
Tests parallel trends and estimates treatment effects with all three control groups.

Model: framing_scale ~ treat + time + post + treat:time + treat:post + treat:time:post
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR, RESULTS_DIR


def load_monthly_framing(topic: str) -> pd.DataFrame:
    """Load monthly framing data for a topic."""
    file_path = DATA_DIR / 'framing' / f'{topic}_monthly_framing.csv'
    print(f"Loading {topic}: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_csv(file_path)
    count_col = 'post_count' if 'post_count' in df.columns else 'item_count'
    print(f"  Loaded {len(df)} months, {df[count_col].sum():.0f} total items")

    return df


def prepare_did_data(nk_data: pd.DataFrame, control_data: pd.DataFrame, control_name: str) -> pd.DataFrame:
    """
    Prepare combined data for DID analysis.

    Args:
        nk_data: NK monthly framing data
        control_data: Control group monthly framing data
        control_name: Name of control group

    Returns:
        Combined DataFrame ready for DID regression
    """
    # Add treatment indicator
    nk_data = nk_data.copy()
    control_data = control_data.copy()

    nk_data['treat'] = 1
    control_data['treat'] = 0

    # Combine
    combined = pd.concat([nk_data, control_data], ignore_index=True)

    # Create time variable (0-29 for 30 months)
    combined['time'] = combined.groupby('topic').cumcount()

    # Create post-intervention indicator (intervention at 2018-03, which is month 14)
    # Pre: 2017-01 to 2018-02 (months 0-13)
    # Post: 2018-03 to 2019-06 (months 14-29)
    combined['post'] = (combined['time'] >= 14).astype(int)

    # Create interaction terms
    combined['treat_time'] = combined['treat'] * combined['time']
    combined['treat_post'] = combined['treat'] * combined['post']
    combined['treat_time_post'] = combined['treat'] * combined['time'] * combined['post']

    print(f"\nDID Data Prepared: NK vs {control_name}")
    print(f"  Total observations: {len(combined)}")
    print(f"  NK: {len(combined[combined['treat']==1])}, {control_name}: {len(combined[combined['treat']==0])}")
    print(f"  Pre-period: {len(combined[combined['post']==0])}, Post-period: {len(combined[combined['post']==1])}")

    return combined


def test_parallel_trends(combined_data: pd.DataFrame, control_name: str) -> dict:
    """
    Test parallel trends assumption in pre-intervention period.

    Model: framing_mean ~ treat + time + treat:time (pre-period only)
    H0: β₄ (treat:time) = 0 (parallel trends)
    """
    print(f"\n{'='*80}")
    print(f"PARALLEL TRENDS TEST: NK vs {control_name}")
    print(f"{'='*80}")

    # Pre-intervention period only
    pre_data = combined_data[combined_data['post'] == 0].copy()

    print(f"Pre-intervention period: {len(pre_data)} observations")
    print(f"  Time range: {pre_data['time'].min()} to {pre_data['time'].max()}")

    # Run regression
    model = smf.ols(
        'framing_mean ~ treat + time + treat:time',
        data=pre_data
    ).fit(cov_type='cluster', cov_kwds={'groups': pre_data['month']})

    print(f"\nRegression Results:")
    print(model.summary())

    # Extract treat:time coefficient (parallel trends test)
    treat_time_coef = model.params['treat:time']
    treat_time_se = model.bse['treat:time']
    treat_time_p = model.pvalues['treat:time']

    print(f"\nParallel Trends Test:")
    print(f"  β₄ (treat:time): {treat_time_coef:.6f}")
    print(f"  Standard Error:  {treat_time_se:.6f}")
    print(f"  P-value:         {treat_time_p:.6f}")

    if treat_time_p > 0.10:
        print(f"  ✓ PASS: Parallel trends assumption satisfied (p > 0.10)")
    else:
        print(f"  ✗ FAIL: Parallel trends assumption violated (p ≤ 0.10)")

    return {
        'control_group': control_name,
        'beta4_treat_time': float(treat_time_coef),
        'se': float(treat_time_se),
        'p_value': float(treat_time_p),
        'pass': bool(treat_time_p > 0.10),
        'n_obs': len(pre_data)
    }


def run_slope_change_did(combined_data: pd.DataFrame, control_name: str) -> dict:
    """
    Run slope change DID (ITS-style).

    Model: framing_mean ~ treat + time + post + treat:time + treat:post + treat:time:post
    β₆ (treat:time:post) = Monthly slope change after intervention
    """
    print(f"\n{'='*80}")
    print(f"SLOPE CHANGE DID: NK vs {control_name}")
    print(f"{'='*80}")

    # Run regression with clustered standard errors
    model = smf.ols(
        'framing_mean ~ treat + time + post + treat:time + treat:post + treat:time:post',
        data=combined_data
    ).fit(cov_type='cluster', cov_kwds={'groups': combined_data['month']})

    print(f"\nRegression Results:")
    print(model.summary())

    # Extract β₆ (treat:time:post) - slope change
    beta6 = model.params['treat:time:post']
    se6 = model.bse['treat:time:post']
    p6 = model.pvalues['treat:time:post']
    ci_lower = beta6 - 1.96 * se6
    ci_upper = beta6 + 1.96 * se6

    # Calculate cumulative effect over 15 months post-intervention
    cumulative_15mo = beta6 * 15

    print(f"\nSlope Change DID Results:")
    print(f"  β₆ (treat:time:post):     {beta6:.6f} per month")
    print(f"  Standard Error:           {se6:.6f}")
    print(f"  P-value:                  {p6:.6f}")
    print(f"  95% CI:                   [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  Cumulative (15 months):   {cumulative_15mo:.6f}")

    if p6 < 0.05:
        print(f"  ✓ SIGNIFICANT at p < 0.05")
    elif p6 < 0.10:
        print(f"  ○ Marginally significant at p < 0.10")
    else:
        print(f"  ✗ Not significant")

    return {
        'control_group': control_name,
        'did_estimate_monthly': float(beta6),
        'se': float(se6),
        'p_value': float(p6),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'cumulative_15mo': float(cumulative_15mo),
        'n_obs': len(combined_data),
        'r_squared': float(model.rsquared)
    }


def run_level_change_did(combined_data: pd.DataFrame, control_name: str) -> dict:
    """
    Run level change DID (mean shift).

    Model: framing_mean ~ treat + time + post + treat:post
    β₃ (treat:post) = Immediate level change after intervention
    """
    print(f"\n{'='*80}")
    print(f"LEVEL CHANGE DID: NK vs {control_name}")
    print(f"{'='*80}")

    # Run regression with clustered standard errors
    model = smf.ols(
        'framing_mean ~ treat + time + post + treat:post',
        data=combined_data
    ).fit(cov_type='cluster', cov_kwds={'groups': combined_data['month']})

    print(f"\nRegression Results:")
    print(model.summary())

    # Extract β₃ (treat:post) - level change
    beta3 = model.params['treat:post']
    se3 = model.bse['treat:post']
    p3 = model.pvalues['treat:post']
    ci_lower = beta3 - 1.96 * se3
    ci_upper = beta3 + 1.96 * se3

    # Calculate means for interpretation
    pre_nk = combined_data[(combined_data['treat']==1) & (combined_data['post']==0)]['framing_mean'].mean()
    post_nk = combined_data[(combined_data['treat']==1) & (combined_data['post']==1)]['framing_mean'].mean()
    pre_control = combined_data[(combined_data['treat']==0) & (combined_data['post']==0)]['framing_mean'].mean()
    post_control = combined_data[(combined_data['treat']==0) & (combined_data['post']==1)]['framing_mean'].mean()

    print(f"\nLevel Change DID Results:")
    print(f"  β₃ (treat:post):          {beta3:.6f}")
    print(f"  Standard Error:           {se3:.6f}")
    print(f"  P-value:                  {p3:.6f}")
    print(f"  95% CI:                   [{ci_lower:.6f}, {ci_upper:.6f}]")

    print(f"\nMean Values:")
    print(f"  NK Pre:       {pre_nk:.4f}")
    print(f"  NK Post:      {post_nk:.4f}")
    print(f"  Control Pre:  {pre_control:.4f}")
    print(f"  Control Post: {post_control:.4f}")

    if p3 < 0.05:
        print(f"  ✓ SIGNIFICANT at p < 0.05")
    elif p3 < 0.10:
        print(f"  ○ Marginally significant at p < 0.10")
    else:
        print(f"  ✗ Not significant")

    return {
        'control_group': control_name,
        'did_estimate': float(beta3),
        'se': float(se3),
        'p_value': float(p3),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'means': {
            'nk_pre': float(pre_nk),
            'nk_post': float(post_nk),
            'control_pre': float(pre_control),
            'control_post': float(post_control)
        },
        'n_obs': len(combined_data),
        'r_squared': float(model.rsquared)
    }


def main():
    """Main execution function."""
    print("=" * 80)
    print("DIFFERENCE-IN-DIFFERENCES ANALYSIS: FRAMING SCALE")
    print("=" * 80)

    print("\nOutcome Variable: diplomacy_scale")
    print("  Range: -2 (THREAT) to +2 (DIPLOMACY)")
    print("\nIntervention: 2018-03-08 (NK-US summit announcement)")
    print("  Pre-period:  2017-01 to 2018-02 (14 months)")
    print("  Post-period: 2018-03 to 2019-06 (16 months)")

    # Load data
    print("\n" + "="*80)
    print("Loading Monthly Framing Data")
    print("="*80)

    nk_monthly = load_monthly_framing('nk')
    iran_monthly = load_monthly_framing('iran')
    russia_monthly = load_monthly_framing('russia')
    china_monthly = load_monthly_framing('china')

    # Control groups
    control_groups = {
        'Iran': iran_monthly,
        'Russia': russia_monthly,
        'China': china_monthly
    }

    # Results storage
    parallel_trends_results = {}
    slope_did_results = {}
    level_did_results = {}

    # Run analysis for each control group
    for control_name, control_data in control_groups.items():
        print(f"\n{'='*80}")
        print(f"ANALYSIS: NK vs {control_name}")
        print(f"{'='*80}")

        # Prepare data
        combined = prepare_did_data(nk_monthly, control_data, control_name)

        # Test parallel trends
        pt_result = test_parallel_trends(combined, control_name)
        parallel_trends_results[control_name] = pt_result

        # Run slope change DID
        slope_result = run_slope_change_did(combined, control_name)
        slope_did_results[control_name] = slope_result

        # Run level change DID
        level_result = run_level_change_did(combined, control_name)
        level_did_results[control_name] = level_result

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save parallel trends results
    pt_output = RESULTS_DIR / 'framing_parallel_trends_results.json'
    with open(pt_output, 'w') as f:
        json.dump(parallel_trends_results, f, indent=2)
    print(f"\n✓ Saved parallel trends results: {pt_output}")

    # Save slope DID results
    slope_output = RESULTS_DIR / 'framing_did_slope_results.json'
    with open(slope_output, 'w') as f:
        json.dump(slope_did_results, f, indent=2)
    print(f"✓ Saved slope DID results: {slope_output}")

    # Save level DID results
    level_output = RESULTS_DIR / 'framing_did_level_results.json'
    with open(level_output, 'w') as f:
        json.dump(level_did_results, f, indent=2)
    print(f"✓ Saved level DID results: {level_output}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Framing-Based DID Results")
    print("="*80)

    print("\n1. Parallel Trends Test:")
    for control, result in parallel_trends_results.items():
        status = "✓ PASS" if result['pass'] else "✗ FAIL"
        print(f"   {control:8s}: β₄={result['beta4_treat_time']:+.6f}, p={result['p_value']:.4f} {status}")

    print("\n2. Slope Change DID (Monthly):")
    for control, result in slope_did_results.items():
        sig = "**" if result['p_value'] < 0.05 else "*" if result['p_value'] < 0.10 else ""
        print(f"   {control:8s}: β₆={result['did_estimate_monthly']:+.6f}, p={result['p_value']:.4f} {sig}")

    print("\n3. Level Change DID:")
    for control, result in level_did_results.items():
        sig = "**" if result['p_value'] < 0.05 else "*" if result['p_value'] < 0.10 else ""
        print(f"   {control:8s}: β₃={result['did_estimate']:+.6f}, p={result['p_value']:.4f} {sig}")

    print("\nInterpretation:")
    print("  Positive β = Shift toward DIPLOMACY framing (coercive diplomacy SUCCEEDED)")
    print("  Negative β = Shift toward THREAT framing (coercive diplomacy FAILED)")

    print("\nNext steps:")
    print("  Run scripts/framing_vs_sentiment_comparison.py to compare with sentiment DID")


if __name__ == "__main__":
    main()
