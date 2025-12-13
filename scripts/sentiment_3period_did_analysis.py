"""
3-Period DID Analysis for Sentiment Data
Periods:
- Period 1: Pre-Singapore (2017.01 - 2018.05)
- Period 2: Singapore-Hanoi (2018.06 - 2019.02)
- Period 3: Post-Hanoi (2019.03 - 2019.12)
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_and_prepare_data():
    """Load all sentiment data and prepare for analysis."""
    print("=" * 70)
    print("LOADING SENTIMENT DATA")
    print("=" * 70)
    
    # NK existing data (2017.01 - 2019.06)
    nk_existing = pd.read_csv('data/processed/nk_posts_roberta_sentiment.csv')
    nk_existing['datetime'] = pd.to_datetime(nk_existing['created_utc'], unit='s')
    print(f"NK existing: {len(nk_existing)} posts")
    
    # NK extended data (2019.07 - 2019.12)
    nk_extended = pd.read_csv('data/sentiment/nk_posts_hanoi_extended_sentiment.csv')
    nk_extended['datetime'] = pd.to_datetime(nk_extended['created_utc'], unit='s')
    print(f"NK extended: {len(nk_extended)} posts")
    
    # Combine NK
    nk_all = pd.concat([nk_existing, nk_extended], ignore_index=True)
    nk_all['topic'] = 'nk'
    print(f"NK total: {len(nk_all)} posts")
    
    # Control groups - existing
    china_existing = pd.read_csv('data/processed/china_posts_roberta_sentiment.csv')
    iran_existing = pd.read_csv('data/processed/iran_posts_roberta_sentiment.csv')
    russia_existing = pd.read_csv('data/processed/russia_posts_roberta_sentiment.csv')
    
    # Control groups - extended
    china_extended = pd.read_csv('data/sentiment/china_posts_hanoi_extended_sentiment.csv')
    iran_extended = pd.read_csv('data/sentiment/iran_posts_hanoi_extended_sentiment.csv')
    russia_extended = pd.read_csv('data/sentiment/russia_posts_hanoi_extended_sentiment.csv')
    
    # Combine control groups
    for df in [china_existing, china_extended]: 
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    for df in [iran_existing, iran_extended]: 
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    for df in [russia_existing, russia_extended]: 
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    
    china_all = pd.concat([china_existing, china_extended], ignore_index=True)
    china_all['topic'] = 'china'
    print(f"China total: {len(china_all)} posts")
    
    iran_all = pd.concat([iran_existing, iran_extended], ignore_index=True)
    iran_all['topic'] = 'iran'
    print(f"Iran total: {len(iran_all)} posts")
    
    russia_all = pd.concat([russia_existing, russia_extended], ignore_index=True)
    russia_all['topic'] = 'russia'
    print(f"Russia total: {len(russia_all)} posts")
    
    return nk_all, china_all, iran_all, russia_all


def assign_period(df):
    """Assign period based on date."""
    df = df.copy()
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    
    def get_period(month):
        if month <= '2018-05':
            return 'P1_PreSingapore'
        elif month <= '2019-02':
            return 'P2_SingaporeHanoi'
        else:
            return 'P3_PostHanoi'
    
    df['period'] = df['month'].apply(get_period)
    return df


def calculate_period_stats(df, topic_name):
    """Calculate sentiment statistics by period."""
    period_stats = df.groupby('period').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).round(4)
    period_stats.columns = ['mean', 'std', 'count']
    period_stats['topic'] = topic_name
    return period_stats.reset_index()


def run_did_analysis(treatment_df, control_df, treatment_name, control_name, period1, period2):
    """Run DID between two periods."""
    # Filter data
    t_p1 = treatment_df[treatment_df['period'] == period1]['sentiment_score']
    t_p2 = treatment_df[treatment_df['period'] == period2]['sentiment_score']
    c_p1 = control_df[control_df['period'] == period1]['sentiment_score']
    c_p2 = control_df[control_df['period'] == period2]['sentiment_score']
    
    # Calculate means
    t_p1_mean, t_p2_mean = t_p1.mean(), t_p2.mean()
    c_p1_mean, c_p2_mean = c_p1.mean(), c_p2.mean()
    
    # DID estimate
    did = (t_p2_mean - t_p1_mean) - (c_p2_mean - c_p1_mean)
    
    # Statistical test (t-test on difference)
    treatment_diff = t_p2.mean() - t_p1.mean()
    control_diff = c_p2.mean() - c_p1.mean()
    
    # Pooled standard error approximation
    n_t1, n_t2, n_c1, n_c2 = len(t_p1), len(t_p2), len(c_p1), len(c_p2)
    se_t = np.sqrt(t_p1.var()/n_t1 + t_p2.var()/n_t2)
    se_c = np.sqrt(c_p1.var()/n_c1 + c_p2.var()/n_c2)
    se_did = np.sqrt(se_t**2 + se_c**2)
    
    # t-statistic and p-value
    t_stat = did / se_did if se_did > 0 else 0
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=min(n_t1, n_t2, n_c1, n_c2) - 1))
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((t_p2.var() + c_p2.var()) / 2)
    cohens_d = did / pooled_std if pooled_std > 0 else 0
    
    return {
        'comparison': f'{period1} → {period2}',
        'treatment': treatment_name,
        'control': control_name,
        'treatment_p1_mean': round(t_p1_mean, 4),
        'treatment_p2_mean': round(t_p2_mean, 4),
        'control_p1_mean': round(c_p1_mean, 4),
        'control_p2_mean': round(c_p2_mean, 4),
        'did_estimate': round(did, 4),
        'se': round(se_did, 4),
        't_stat': round(t_stat, 4),
        'p_value': round(p_value, 4),
        'cohens_d': round(cohens_d, 4),
        'significant': '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.1 else ''))
    }


def main():
    print("=" * 70)
    print("3-PERIOD SENTIMENT DID ANALYSIS")
    print("=" * 70)
    print("\nPeriods:")
    print("  P1: Pre-Singapore (2017.01 - 2018.05)")
    print("  P2: Singapore-Hanoi (2018.06 - 2019.02)")
    print("  P3: Post-Hanoi (2019.03 - 2019.12)")
    print()
    
    # Load data
    nk, china, iran, russia = load_and_prepare_data()
    
    # Assign periods
    nk = assign_period(nk)
    china = assign_period(china)
    iran = assign_period(iran)
    russia = assign_period(russia)
    
    # Calculate period statistics
    print("\n" + "=" * 70)
    print("PERIOD STATISTICS")
    print("=" * 70)
    
    for df, name in [(nk, 'NK'), (china, 'China'), (iran, 'Iran'), (russia, 'Russia')]:
        stats_df = calculate_period_stats(df, name)
        print(f"\n{name}:")
        for _, row in stats_df.iterrows():
            print(f"  {row['period']}: mean={row['mean']:.4f}, std={row['std']:.4f}, n={int(row['count'])}")
    
    # Run DID analyses
    print("\n" + "=" * 70)
    print("DID ANALYSIS RESULTS")
    print("=" * 70)
    
    results = []
    
    # Use Iran as primary control (has data for all periods: 2017.01 - 2019.06)
    # Note: China/Russia only have data from 2018.11
    control = iran
    control_name = 'Iran'
    
    print(f"\n*** Primary Control: {control_name} (full period coverage) ***")
    
    # Analysis 1: Singapore Summit Effect (P1 → P2)
    print("\n--- Singapore Summit Effect (P1 → P2) ---")
    result1 = run_did_analysis(nk, control, 'NK', control_name, 'P1_PreSingapore', 'P2_SingaporeHanoi')
    results.append(result1)
    print(f"DID Estimate: {result1['did_estimate']:.4f} {result1['significant']}")
    print(f"P-value: {result1['p_value']:.4f}")
    print(f"Cohen's d: {result1['cohens_d']:.4f}")
    
    # Analysis 2: Hanoi Collapse Effect (P2 → P3)
    print("\n--- Hanoi Collapse Effect (P2 → P3) ---")
    result2 = run_did_analysis(nk, control, 'NK', control_name, 'P2_SingaporeHanoi', 'P3_PostHanoi')
    results.append(result2)
    print(f"DID Estimate: {result2['did_estimate']:.4f} {result2['significant']}")
    print(f"P-value: {result2['p_value']:.4f}")
    print(f"Cohen's d: {result2['cohens_d']:.4f}")
    
    # Analysis 3: Full Effect (P1 → P3)
    print("\n--- Full Effect (P1 → P3) ---")
    result3 = run_did_analysis(nk, control, 'NK', control_name, 'P1_PreSingapore', 'P3_PostHanoi')
    results.append(result3)
    print(f"DID Estimate: {result3['did_estimate']:.4f} {result3['significant']}")
    print(f"P-value: {result3['p_value']:.4f}")
    print(f"Cohen's d: {result3['cohens_d']:.4f}")
    
    # Robustness: China and Russia for P2 → P3 only (they lack P1 data)
    print("\n" + "=" * 70)
    print("ROBUSTNESS: P2→P3 with China/Russia (no P1 data for these)")
    print("=" * 70)
    
    for ctrl, ctrl_name in [(china, 'China'), (russia, 'Russia')]:
        print(f"\n--- Control: {ctrl_name} (P2→P3 only) ---")
        r2 = run_did_analysis(nk, ctrl, 'NK', ctrl_name, 'P2_SingaporeHanoi', 'P3_PostHanoi')
        print(f"P2→P3: DID={r2['did_estimate']:.4f} {r2['significant']} (p={r2['p_value']:.4f})")
        results.append(r2)
    
    # Save results
    results_df = pd.DataFrame(results)
    os.makedirs('data/results', exist_ok=True)
    results_df.to_csv('data/results/sentiment_3period_did_results.csv', index=False)
    print(f"\n✓ Results saved to: data/results/sentiment_3period_did_results.csv")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Findings (China as Control):")
    print(f"  1. Singapore Summit Effect (P1→P2): DID = {result1['did_estimate']:.4f} {result1['significant']}")
    print(f"  2. Hanoi Collapse Effect (P2→P3): DID = {result2['did_estimate']:.4f} {result2['significant']}")
    print(f"  3. Net Effect (P1→P3): DID = {result3['did_estimate']:.4f} {result3['significant']}")
    
    return results_df


if __name__ == '__main__':
    main()
