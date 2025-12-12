"""
Calculate standardized effect sizes for DID estimates
Addresses scale-dependence issue
"""

import pandas as pd
import numpy as np


def calculate_cohens_d(nk_data, control_data):
    """
    Calculate Cohen's d for DID estimate.

    Cohen's d = (DID estimate) / (Pooled SD)

    Interpretation:
    - d = 0.2: Small effect
    - d = 0.5: Medium effect
    - d = 0.8: Large effect
    """
    # Separate pre/post
    nk_data = nk_data.copy()
    control_data = control_data.copy()

    nk_data['month_date'] = pd.to_datetime(nk_data['month'] + '-01')
    control_data['month_date'] = pd.to_datetime(control_data['month'] + '-01')

    intervention_date = pd.to_datetime('2018-03-01')

    nk_pre = nk_data[nk_data['month_date'] < intervention_date]['sentiment_mean'].dropna()
    nk_post = nk_data[nk_data['month_date'] >= intervention_date]['sentiment_mean'].dropna()
    control_pre = control_data[control_data['month_date'] < intervention_date]['sentiment_mean'].dropna()
    control_post = control_data[control_data['month_date'] >= intervention_date]['sentiment_mean'].dropna()

    # Calculate means
    nk_pre_mean = nk_pre.mean()
    nk_post_mean = nk_post.mean()
    control_pre_mean = control_pre.mean()
    control_post_mean = control_post.mean()

    # DID estimate
    did_estimate = (nk_post_mean - nk_pre_mean) - (control_post_mean - control_pre_mean)

    # Pooled standard deviation (pre-period)
    n1 = len(nk_pre)
    n2 = len(control_pre)
    sd1 = nk_pre.std()
    sd2 = control_pre.std()

    pooled_sd = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))

    # Cohen's d
    cohens_d = did_estimate / pooled_sd

    return {
        'did_estimate': did_estimate,
        'pooled_sd': pooled_sd,
        'cohens_d': cohens_d,
        'nk_pre_sd': sd1,
        'control_pre_sd': sd2
    }


def calculate_percentage_change(nk_data, control_data):
    """
    Calculate percentage change relative to pre-intervention mean.
    """
    nk_data = nk_data.copy()
    control_data = control_data.copy()

    nk_data['month_date'] = pd.to_datetime(nk_data['month'] + '-01')
    control_data['month_date'] = pd.to_datetime(control_data['month'] + '-01')

    intervention_date = pd.to_datetime('2018-03-01')

    nk_pre = nk_data[nk_data['month_date'] < intervention_date]['sentiment_mean'].dropna()
    nk_post = nk_data[nk_data['month_date'] >= intervention_date]['sentiment_mean'].dropna()
    control_pre = control_data[control_data['month_date'] < intervention_date]['sentiment_mean'].dropna()
    control_post = control_data[control_data['month_date'] >= intervention_date]['sentiment_mean'].dropna()

    nk_pre_mean = nk_pre.mean()
    nk_post_mean = nk_post.mean()
    control_pre_mean = control_pre.mean()
    control_post_mean = control_post.mean()

    # Percentage changes
    nk_pct_change = ((nk_post_mean - nk_pre_mean) / abs(nk_pre_mean)) * 100
    control_pct_change = ((control_post_mean - control_pre_mean) / abs(control_pre_mean)) * 100

    # DID as percentage points
    did_pct = nk_pct_change - control_pct_change

    return {
        'nk_pct_change': nk_pct_change,
        'control_pct_change': control_pct_change,
        'did_pct': did_pct
    }


def calculate_scale_normalized_effect(did_estimate, scale_range=2.0):
    """
    Normalize DID estimate by scale range.

    For sentiment scale [-1, +1], range = 2
    """
    normalized = (did_estimate / scale_range) * 100
    return normalized


def main():
    """Calculate standardized effect sizes for all control groups."""

    print("="*80)
    print("EFFECT SIZE ANALYSIS: Scale-Normalized Comparisons")
    print("="*80)

    # Load data
    nk_monthly = pd.read_csv('data/processed/nk_monthly_combined_roberta.csv')

    control_groups = {
        'Iran': 'data/processed/iran_monthly_combined_roberta.csv',
        'Russia': 'data/processed/russia_monthly_combined_roberta.csv',
        'China': 'data/processed/china_monthly_combined_roberta.csv'
    }

    print("\n" + "-"*80)
    print("LEVEL CHANGE DID EFFECT SIZES")
    print("-"*80)

    print(f"\n{'Control':<10} {'DID Est':>10} {'Cohen d':>10} {'Pooled SD':>12} {'% Change':>12} {'Interpret':>15}")
    print("-"*80)

    for control_name, control_path in control_groups.items():
        control_monthly = pd.read_csv(control_path)

        # Cohen's d
        cohens = calculate_cohens_d(nk_monthly, control_monthly)

        # Percentage change
        pct = calculate_percentage_change(nk_monthly, control_monthly)

        # Scale normalization
        scale_norm = calculate_scale_normalized_effect(cohens['did_estimate'], scale_range=2.0)

        # Interpretation
        d = cohens['cohens_d']
        if abs(d) < 0.2:
            interp = "Very small"
        elif abs(d) < 0.5:
            interp = "Small"
        elif abs(d) < 0.8:
            interp = "Medium"
        else:
            interp = "Large"

        print(f"{control_name:<10} {cohens['did_estimate']:>10.4f} {cohens['cohens_d']:>10.3f} "
              f"{cohens['pooled_sd']:>12.4f} {scale_norm:>11.2f}% {interp:>15}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    print("\n1. Cohen's d (Standardized Effect Size):")
    print("   - Accounts for variability in the data")
    print("   - Rule of thumb: 0.2 = small, 0.5 = medium, 0.8 = large")

    print("\n2. Scale-Normalized (% of total range):")
    print("   - DID estimate / scale range [-1, +1]")
    print("   - Shows % change relative to full scale")

    print("\n3. Percentage Change:")
    print("   - Change relative to pre-intervention mean")
    print("   - Contextualizes absolute effect size")

    # Detailed breakdown for China (strongest effect)
    print("\n" + "="*80)
    print("DETAILED BREAKDOWN: CHINA (Strongest Effect)")
    print("="*80)

    china_monthly = pd.read_csv('data/processed/china_monthly_combined_roberta.csv')
    cohens_china = calculate_cohens_d(nk_monthly, china_monthly)
    pct_china = calculate_percentage_change(nk_monthly, china_monthly)

    print(f"\nRaw DID Estimate: {cohens_china['did_estimate']:.4f}")
    print(f"  → Scale range: -1 to +1 (total = 2)")
    print(f"  → Normalized: {cohens_china['did_estimate']/2*100:.2f}% of full scale")

    print(f"\nCohen's d: {cohens_china['cohens_d']:.3f}")
    print(f"  → NK pre-period SD: {cohens_china['nk_pre_sd']:.4f}")
    print(f"  → China pre-period SD: {cohens_china['control_pre_sd']:.4f}")
    print(f"  → Pooled SD: {cohens_china['pooled_sd']:.4f}")
    print(f"  → Effect size: {cohens_china['did_estimate']:.4f} / {cohens_china['pooled_sd']:.4f} = {cohens_china['cohens_d']:.3f}")

    print(f"\nPercentage Changes:")
    print(f"  → NK: {pct_china['nk_pct_change']:+.2f}%")
    print(f"  → China: {pct_china['control_pct_change']:+.2f}%")
    print(f"  → DID: {pct_china['did_pct']:+.2f} percentage points")

    # Compare with slope change
    print("\n" + "="*80)
    print("COMPARISON: SLOPE vs LEVEL CHANGE")
    print("="*80)

    slope_change_per_month = 0.0067
    slope_15_months = slope_change_per_month * 15

    print(f"\nSlope Change DID:")
    print(f"  → Monthly: {slope_change_per_month:.4f} ({slope_change_per_month/2*100:.2f}% of scale)")
    print(f"  → 15-month cumulative: {slope_15_months:.4f} ({slope_15_months/2*100:.2f}% of scale)")

    print(f"\nLevel Change DID (China):")
    print(f"  → One-time: {cohens_china['did_estimate']:.4f} ({cohens_china['did_estimate']/2*100:.2f}% of scale)")

    print(f"\nTotal Effect (if both exist):")
    print(f"  → Level + Cumulative Slope: {cohens_china['did_estimate'] + slope_15_months:.4f}")
    print(f"  → Percentage of scale: {(cohens_china['did_estimate'] + slope_15_months)/2*100:.2f}%")


if __name__ == '__main__':
    main()
