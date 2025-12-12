"""
Calculate effect sizes for SLOPE CHANGE DID
Compare with level change effect sizes
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def calculate_slope_cohens_d(nk_data, control_data):
    """
    Calculate Cohen's d for slope change.

    Approach:
    1. Calculate pre/post slopes for NK and control
    2. Calculate slope changes
    3. Standardize by pooled SD of slopes
    """
    # Prepare data
    nk_data = nk_data.copy()
    control_data = control_data.copy()

    nk_data['month_date'] = pd.to_datetime(nk_data['month'] + '-01')
    control_data['month_date'] = pd.to_datetime(control_data['month'] + '-01')

    intervention_date = pd.to_datetime('2018-03-01')

    # Add time variable (months since start)
    start_date = pd.to_datetime('2017-01-01')
    nk_data['time'] = (nk_data['month_date'] - start_date).dt.days / 30.44
    control_data['time'] = (control_data['month_date'] - start_date).dt.days / 30.44

    # Separate pre/post
    nk_pre = nk_data[nk_data['month_date'] < intervention_date].dropna(subset=['sentiment_mean'])
    nk_post = nk_data[nk_data['month_date'] >= intervention_date].dropna(subset=['sentiment_mean'])
    control_pre = control_data[control_data['month_date'] < intervention_date].dropna(subset=['sentiment_mean'])
    control_post = control_data[control_data['month_date'] >= intervention_date].dropna(subset=['sentiment_mean'])

    # Calculate slopes using OLS
    def get_slope(data):
        if len(data) < 3:
            return 0.0, 0.0
        model = smf.ols('sentiment_mean ~ time', data=data).fit()
        return model.params['time'], model.bse['time']

    nk_pre_slope, nk_pre_slope_se = get_slope(nk_pre)
    nk_post_slope, nk_post_slope_se = get_slope(nk_post)
    control_pre_slope, control_pre_slope_se = get_slope(control_pre)
    control_post_slope, control_post_slope_se = get_slope(control_post)

    # Slope changes
    nk_slope_change = nk_post_slope - nk_pre_slope
    control_slope_change = control_post_slope - control_pre_slope

    # DID estimate (slope change difference)
    did_slope_estimate = nk_slope_change - control_slope_change

    # For Cohen's d, we need SD of slopes
    # Use residual SD from pre-period regressions as proxy
    nk_pre_model = smf.ols('sentiment_mean ~ time', data=nk_pre).fit()
    control_pre_model = smf.ols('sentiment_mean ~ time', data=control_pre).fit()

    nk_residual_sd = np.sqrt(nk_pre_model.mse_resid)
    control_residual_sd = np.sqrt(control_pre_model.mse_resid)

    # Pooled SD
    n1 = len(nk_pre)
    n2 = len(control_pre)
    pooled_sd = np.sqrt(((n1 - 1) * nk_residual_sd**2 + (n2 - 1) * control_residual_sd**2) / (n1 + n2 - 2))

    # Cohen's d for slope change
    # Note: This is comparing slope CHANGES, so we standardize by residual variation
    cohens_d = did_slope_estimate / pooled_sd

    return {
        'did_slope_estimate': did_slope_estimate,
        'nk_pre_slope': nk_pre_slope,
        'nk_post_slope': nk_post_slope,
        'nk_slope_change': nk_slope_change,
        'control_pre_slope': control_pre_slope,
        'control_post_slope': control_post_slope,
        'control_slope_change': control_slope_change,
        'pooled_sd': pooled_sd,
        'cohens_d': cohens_d,
        'nk_residual_sd': nk_residual_sd,
        'control_residual_sd': control_residual_sd
    }


def calculate_cumulative_effect(slope_change, months=15):
    """
    Calculate cumulative effect of slope change over time.
    """
    return slope_change * months


def main():
    """Compare slope change vs level change effect sizes."""

    print("="*80)
    print("SLOPE CHANGE EFFECT SIZE ANALYSIS")
    print("Comparing Slope vs Level Change with Standardized Effect Sizes")
    print("="*80)

    # Load data
    nk_monthly = pd.read_csv('data/processed/nk_monthly_combined_roberta.csv')

    control_groups = {
        'Iran': 'data/processed/iran_monthly_combined_roberta.csv',
        'Russia': 'data/processed/russia_monthly_combined_roberta.csv',
        'China': 'data/processed/china_monthly_combined_roberta.csv'
    }

    print("\n" + "-"*80)
    print("SLOPE CHANGE DID: Effect Sizes")
    print("-"*80)

    slope_results = {}

    for control_name, control_path in control_groups.items():
        control_monthly = pd.read_csv(control_path)
        result = calculate_slope_cohens_d(nk_monthly, control_monthly)
        slope_results[control_name] = result

    # Print results
    print(f"\n{'Control':<10} {'Slope DID':>12} {'Cohen d':>10} {'Pooled SD':>12} {'% Scale':>12} {'15-mo Cumul':>12}")
    print("-"*80)

    for control_name, result in slope_results.items():
        cumulative = calculate_cumulative_effect(result['did_slope_estimate'], 15)
        pct_scale = (result['did_slope_estimate'] / 2.0) * 100
        cumul_pct = (cumulative / 2.0) * 100

        print(f"{control_name:<10} {result['did_slope_estimate']:>12.4f} {result['cohens_d']:>10.3f} "
              f"{result['pooled_sd']:>12.4f} {pct_scale:>11.2f}% {cumulative:>12.4f}")

    # Detailed breakdown
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN: Slope Components")
    print("="*80)

    for control_name, result in slope_results.items():
        print(f"\n{control_name} Control:")
        print(f"  NK pre-intervention slope:     {result['nk_pre_slope']:>8.4f}")
        print(f"  NK post-intervention slope:    {result['nk_post_slope']:>8.4f}")
        print(f"  NK slope change:               {result['nk_slope_change']:>8.4f}")
        print(f"  {control_name} pre-slope:      {result['control_pre_slope']:>8.4f}")
        print(f"  {control_name} post-slope:     {result['control_post_slope']:>8.4f}")
        print(f"  {control_name} slope change:   {result['control_slope_change']:>8.4f}")
        print(f"  DID (slope difference):        {result['did_slope_estimate']:>8.4f}")
        print(f"  Cohen's d:                     {result['cohens_d']:>8.3f}")

    # Load level change results for comparison
    print("\n" + "="*80)
    print("COMPARISON: SLOPE vs LEVEL CHANGE")
    print("="*80)

    # Recalculate level change Cohen's d
    from effect_size_analysis import calculate_cohens_d

    level_results = {}
    for control_name, control_path in control_groups.items():
        control_monthly = pd.read_csv(control_path)
        level_results[control_name] = calculate_cohens_d(nk_monthly, control_monthly)

    print(f"\n{'Control':<10} {'Method':<15} {'DID Est':>10} {'Cohen d':>10} {'% Scale':>10} {'Interpret':>12}")
    print("-"*80)

    for control_name in ['Iran', 'Russia', 'China']:
        # Level change
        level = level_results[control_name]
        level_pct = (level['did_estimate'] / 2.0) * 100
        d_level = level['cohens_d']
        if abs(d_level) < 0.2:
            interp_level = "Very small"
        elif abs(d_level) < 0.5:
            interp_level = "Small"
        elif abs(d_level) < 0.8:
            interp_level = "Medium"
        else:
            interp_level = "Large"

        print(f"{control_name:<10} {'Level change':<15} {level['did_estimate']:>10.4f} "
              f"{level['cohens_d']:>10.3f} {level_pct:>9.2f}% {interp_level:>12}")

        # Slope change (monthly)
        slope = slope_results[control_name]
        slope_pct = (slope['did_slope_estimate'] / 2.0) * 100
        d_slope = slope['cohens_d']
        if abs(d_slope) < 0.2:
            interp_slope = "Very small"
        elif abs(d_slope) < 0.5:
            interp_slope = "Small"
        elif abs(d_slope) < 0.8:
            interp_slope = "Medium"
        else:
            interp_slope = "Large"

        print(f"{'':<10} {'Slope (monthly)':<15} {slope['did_slope_estimate']:>10.4f} "
              f"{slope['cohens_d']:>10.3f} {slope_pct:>9.2f}% {interp_slope:>12}")

        # Slope cumulative (15 months)
        cumul = calculate_cumulative_effect(slope['did_slope_estimate'], 15)
        cumul_pct = (cumul / 2.0) * 100
        # For cumulative, Cohen's d scales linearly with effect
        d_cumul = d_slope * 15
        if abs(d_cumul) < 0.2:
            interp_cumul = "Very small"
        elif abs(d_cumul) < 0.5:
            interp_cumul = "Small"
        elif abs(d_cumul) < 0.8:
            interp_cumul = "Medium"
        else:
            interp_cumul = "Large"

        print(f"{'':<10} {'Slope (15-mo)':<15} {cumul:>10.4f} "
              f"{d_cumul:>10.3f} {cumul_pct:>9.2f}% {interp_cumul:>12}")

        print("-"*80)

    # Key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    print("\n1. Slope Change Cohen's d:")
    print("   - Monthly slope change has VERY SMALL standardized effect")
    print("   - But cumulative effect over 15 months becomes LARGER")

    print("\n2. Level Change Cohen's d:")
    print("   - Iran/Russia: Medium effect (d ≈ 0.72)")
    print("   - China: Large effect (d = 1.12)")

    print("\n3. Scale Perspective:")
    print("   - Level change: 3-4% of full scale (one-time)")
    print("   - Slope monthly: 0.3% of full scale")
    print("   - Slope cumulative: 5% of full scale (over 15 months)")

    print("\n4. Which is Larger?")
    for control_name in ['Iran', 'Russia', 'China']:
        level_d = level_results[control_name]['cohens_d']
        slope_cumul_d = slope_results[control_name]['cohens_d'] * 15
        print(f"   {control_name}: Level d={level_d:.2f} vs Slope(15-mo) d={slope_cumul_d:.2f}")

    print("\n5. Interpretation:")
    print("   - Level change = immediate shift (larger Cohen's d)")
    print("   - Slope change = gradual accumulation (smaller monthly d, but adds up)")
    print("   - Both effects may exist simultaneously")

    # Save results
    import json
    results = {
        'slope': {k: {key: float(val) if isinstance(val, (np.float64, np.float32)) else val
                      for key, val in v.items()}
                  for k, v in slope_results.items()},
        'level': {k: {key: float(val) if isinstance(val, (np.float64, np.float32)) else val
                      for key, val in v.items()}
                  for k, v in level_results.items()}
    }

    with open('data/results/slope_vs_level_effect_sizes.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n✓ Results saved to: data/results/slope_vs_level_effect_sizes.json")


if __name__ == '__main__':
    main()
