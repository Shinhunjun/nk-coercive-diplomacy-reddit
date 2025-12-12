"""
3-Period DID Analysis: Hanoi Summit Impact

Analyzes framing changes across 3 periods:
- Period 1: Pre-Singapore (2017.01 ~ 2018.05)
- Period 2: Singapore-Hanoi (2018.06 ~ 2019.02)
- Period 3: Post-Hanoi (2019.03 ~ 2019.06)

Uses China as control group for DID comparison.
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import json
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Period definitions
PERIODS = {
    'pre_singapore': {
        'name': 'Pre-Singapore Summit',
        'start': '2017-01',
        'end': '2018-05',
        'period_id': 1
    },
    'singapore_hanoi': {
        'name': 'Singapore to Hanoi',
        'start': '2018-06',
        'end': '2019-02',
        'period_id': 2
    },
    'post_hanoi': {
        'name': 'Post-Hanoi Collapse',
        'start': '2019-03',
        'end': '2019-06',
        'period_id': 3
    }
}


def load_framing_data():
    """Load framing data for NK and China."""
    data_dir = Path('data/framing')
    
    nk_df = pd.read_csv(data_dir / 'nk_monthly_framing.csv')
    china_df = pd.read_csv(data_dir / 'china_monthly_framing.csv')
    
    return nk_df, china_df


def assign_period(month: str) -> tuple:
    """Assign period ID and name based on month."""
    for period_key, period_info in PERIODS.items():
        if period_info['start'] <= month <= period_info['end']:
            return period_info['period_id'], period_info['name']
    return 0, 'Unknown'


def prepare_did_data(nk_df: pd.DataFrame, china_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare combined data for DID analysis."""
    
    # Add topic if not present
    if 'topic' not in nk_df.columns:
        nk_df['topic'] = 'nk'
    if 'topic' not in china_df.columns:
        china_df['topic'] = 'china'
    
    # Combine
    combined = pd.concat([nk_df, china_df], ignore_index=True)
    
    # Assign periods
    combined['period_id'], combined['period_name'] = zip(
        *combined['month'].apply(assign_period)
    )
    
    # Filter only periods 1, 2, 3
    combined = combined[combined['period_id'] > 0]
    
    # Create treatment indicator (NK = 1, China = 0)
    combined['treatment'] = (combined['topic'] == 'nk').astype(int)
    
    return combined


def run_3period_did(df: pd.DataFrame) -> dict:
    """
    Run DID analysis comparing 3 periods.
    
    Comparisons:
    1. Period 1 vs Period 2: Singapore Summit Effect
    2. Period 2 vs Period 3: Hanoi Collapse Effect
    3. Period 1 vs Period 3: Overall Change
    """
    results = {}
    
    comparisons = [
        ('singapore_effect', 1, 2, 'Singapore Summit Effect'),
        ('hanoi_effect', 2, 3, 'Hanoi Collapse Effect'),
        ('overall_change', 1, 3, 'Overall Change')
    ]
    
    for comp_key, period_pre, period_post, comp_name in comparisons:
        print(f"\n{'='*60}")
        print(f"DID: {comp_name} (Period {period_pre} vs {period_post})")
        print('='*60)
        
        # Filter data for these two periods
        subset = df[df['period_id'].isin([period_pre, period_post])].copy()
        
        # Create post indicator
        subset['post'] = (subset['period_id'] == period_post).astype(int)
        
        # DID interaction
        subset['did'] = subset['treatment'] * subset['post']
        
        # Summary statistics
        print("\n--- Mean Framing by Group & Period ---")
        summary = subset.groupby(['topic', 'period_id'])['framing_mean'].agg(['mean', 'std', 'count'])
        print(summary)
        
        # Calculate DID manually
        nk_pre = subset[(subset['treatment']==1) & (subset['post']==0)]['framing_mean'].mean()
        nk_post = subset[(subset['treatment']==1) & (subset['post']==1)]['framing_mean'].mean()
        china_pre = subset[(subset['treatment']==0) & (subset['post']==0)]['framing_mean'].mean()
        china_post = subset[(subset['treatment']==0) & (subset['post']==1)]['framing_mean'].mean()
        
        did_estimate = (nk_post - nk_pre) - (china_post - china_pre)
        
        print(f"\n--- DID Calculation ---")
        print(f"NK Pre:    {nk_pre:.4f}")
        print(f"NK Post:   {nk_post:.4f}")
        print(f"NK Change: {nk_post - nk_pre:.4f}")
        print(f"China Pre:    {china_pre:.4f}")
        print(f"China Post:   {china_post:.4f}")
        print(f"China Change: {china_post - china_pre:.4f}")
        print(f"\nDID Estimate: {did_estimate:.4f}")
        
        # OLS regression for significance
        try:
            model = smf.ols('framing_mean ~ treatment + post + did', data=subset)
            result = model.fit(cov_type='HC3')
            
            print(f"\n--- OLS Regression ---")
            print(result.summary().tables[1])
            
            did_coef = result.params.get('did', np.nan)
            did_pvalue = result.pvalues.get('did', np.nan)
            did_ci = result.conf_int().loc['did'].values if 'did' in result.params else [np.nan, np.nan]
            
            # Effect size (Cohen's d)
            pooled_std = subset['framing_mean'].std()
            cohens_d = did_estimate / pooled_std if pooled_std > 0 else np.nan
            
        except Exception as e:
            print(f"Regression error: {e}")
            did_coef = did_estimate
            did_pvalue = np.nan
            did_ci = [np.nan, np.nan]
            cohens_d = np.nan
        
        results[comp_key] = {
            'comparison': comp_name,
            'period_pre': period_pre,
            'period_post': period_post,
            'nk_pre': float(nk_pre),
            'nk_post': float(nk_post),
            'nk_change': float(nk_post - nk_pre),
            'china_pre': float(china_pre),
            'china_post': float(china_post),
            'china_change': float(china_post - china_pre),
            'did_estimate': float(did_estimate),
            'did_coefficient': float(did_coef),
            'did_pvalue': float(did_pvalue) if not np.isnan(did_pvalue) else None,
            'did_ci_lower': float(did_ci[0]) if not np.isnan(did_ci[0]) else None,
            'did_ci_upper': float(did_ci[1]) if not np.isnan(did_ci[1]) else None,
            'cohens_d': float(cohens_d) if not np.isnan(cohens_d) else None,
            'significant': bool(did_pvalue < 0.05) if not np.isnan(did_pvalue) else None
        }
        
        # Interpretation
        sig_str = "✅ SIGNIFICANT" if results[comp_key]['significant'] else "❌ Not significant"
        print(f"\n{sig_str} (p = {did_pvalue:.4f})" if not np.isnan(did_pvalue) else "")
        print(f"Cohen's d = {cohens_d:.3f}" if not np.isnan(cohens_d) else "")
    
    return results


def analyze_framing_distribution(df: pd.DataFrame) -> dict:
    """Analyze framing distribution changes across periods."""
    
    print("\n" + "="*60)
    print("FRAMING DISTRIBUTION BY PERIOD")
    print("="*60)
    
    period_stats = {}
    
    for period_id in [1, 2, 3]:
        period_name = [p['name'] for p in PERIODS.values() if p['period_id'] == period_id][0]
        
        nk_data = df[(df['topic'] == 'nk') & (df['period_id'] == period_id)]
        
        mean_framing = nk_data['framing_mean'].mean()
        std_framing = nk_data['framing_mean'].std()
        total_posts = nk_data['post_count'].sum()
        n_months = len(nk_data)
        
        period_stats[f'period_{period_id}'] = {
            'name': period_name,
            'mean_framing': float(mean_framing),
            'std_framing': float(std_framing),
            'total_posts': int(total_posts),
            'n_months': int(n_months)
        }
        
        print(f"\n{period_name} (Period {period_id}):")
        print(f"  Mean Framing: {mean_framing:.4f}")
        print(f"  Std: {std_framing:.4f}")
        print(f"  Total Posts: {total_posts:,}")
        print(f"  Months: {n_months}")
    
    # T-tests between periods
    print("\n--- T-tests Between Periods (NK only) ---")
    
    nk_p1 = df[(df['topic'] == 'nk') & (df['period_id'] == 1)]['framing_mean']
    nk_p2 = df[(df['topic'] == 'nk') & (df['period_id'] == 2)]['framing_mean']
    nk_p3 = df[(df['topic'] == 'nk') & (df['period_id'] == 3)]['framing_mean']
    
    # Period 1 vs 2
    t_stat_12, p_val_12 = stats.ttest_ind(nk_p1, nk_p2)
    print(f"Period 1 vs 2: t={t_stat_12:.3f}, p={p_val_12:.4f}")
    
    # Period 2 vs 3
    t_stat_23, p_val_23 = stats.ttest_ind(nk_p2, nk_p3)
    print(f"Period 2 vs 3: t={t_stat_23:.3f}, p={p_val_23:.4f}")
    
    # Period 1 vs 3
    t_stat_13, p_val_13 = stats.ttest_ind(nk_p1, nk_p3)
    print(f"Period 1 vs 3: t={t_stat_13:.3f}, p={p_val_13:.4f}")
    
    period_stats['ttests'] = {
        'p1_vs_p2': {'t_stat': float(t_stat_12), 'p_value': float(p_val_12)},
        'p2_vs_p3': {'t_stat': float(t_stat_23), 'p_value': float(p_val_23)},
        'p1_vs_p3': {'t_stat': float(t_stat_13), 'p_value': float(p_val_13)}
    }
    
    return period_stats


def main():
    print("=" * 70)
    print("3-PERIOD DID ANALYSIS: HANOI SUMMIT IMPACT")
    print("=" * 70)
    print("\nPeriods:")
    for key, info in PERIODS.items():
        print(f"  Period {info['period_id']}: {info['name']} ({info['start']} ~ {info['end']})")
    
    # Load data
    print("\n--- Loading Data ---")
    nk_df, china_df = load_framing_data()
    print(f"NK months: {len(nk_df)}, China months: {len(china_df)}")
    
    # Prepare DID data
    combined_df = prepare_did_data(nk_df, china_df)
    print(f"Combined observations: {len(combined_df)}")
    
    # Run 3-period DID
    did_results = run_3period_did(combined_df)
    
    # Framing distribution analysis
    period_stats = analyze_framing_distribution(combined_df)
    
    # Combine results
    final_results = {
        'analysis': '3-Period DID Analysis: Hanoi Summit Impact',
        'periods': PERIODS,
        'did_results': did_results,
        'period_statistics': period_stats,
        'control_group': 'China',
        'treatment_group': 'North Korea'
    }
    
    # Save results
    output_path = Path('data/results/hanoi_3period_did_results.json')
    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for key, result in did_results.items():
        sig = "✅" if result['significant'] else "❌"
        print(f"\n{result['comparison']}:")
        print(f"  DID Estimate: {result['did_estimate']:+.4f}")
        print(f"  P-value: {result['did_pvalue']:.4f}" if result['did_pvalue'] else "  P-value: N/A")
        print(f"  Cohen's d: {result['cohens_d']:.3f}" if result['cohens_d'] else "  Cohen's d: N/A")
        print(f"  Significance: {sig}")
    
    return final_results


if __name__ == '__main__':
    main()
