"""
High-Confidence DiD Robustness Check
- Only use predictions with confidence >= 0.9
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Settings
CONFIDENCE_THRESHOLD = 0.9

# Load V2 data with metadata
V2_POSTS = "data/results/final_framing_v2"
META_FILES = {
    "nk": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
    "china": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
    "iran": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
    "russia": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"]
}

SCALE = {'THREAT': -2, 'ECONOMIC': -1, 'NEUTRAL': 0, 'HUMANITARIAN': 1, 'DIPLOMACY': 2, 'ERROR': 0}

def assign_period(month):
    if month <= '2018-02':
        return 'P1'
    elif '2018-03' <= month <= '2018-05':
        return None
    elif '2018-06' <= month <= '2019-01':
        return 'P2'
    elif month == '2019-02':
        return None
    elif '2019-03' <= month <= '2019-12':
        return 'P3'
    return None

def create_monthly_highconf(country, threshold):
    res_df = pd.read_csv(f"{V2_POSTS}/{country}_framing_v2.csv")
    
    # Filter by confidence
    res_df = res_df[res_df['confidence'] >= threshold]
    
    meta_dfs = []
    for p in META_FILES[country]:
        if os.path.exists(p):
            meta_dfs.append(pd.read_csv(p, usecols=['id', 'created_utc'], low_memory=False))
    meta_df = pd.concat(meta_dfs).drop_duplicates(subset=['id'])
    
    merged = pd.merge(res_df, meta_df, on='id', how='inner')
    merged['date'] = pd.to_datetime(merged['created_utc'], unit='s')
    merged['month'] = merged['date'].dt.to_period('M').astype(str)
    merged['framing_score'] = merged['frame'].map(SCALE).fillna(0)
    
    monthly = merged.groupby('month').agg({
        'framing_score': ['mean', 'std', 'count']
    }).reset_index()
    monthly.columns = ['month', 'framing_mean', 'framing_std', 'post_count']
    monthly['topic'] = country
    monthly['period'] = monthly['month'].apply(assign_period)
    monthly = monthly.dropna(subset=['period'])
    
    return monthly

def test_parallel_trends(nk_data, ctrl_data, period):
    nk = nk_data[nk_data['period'] == period].copy()
    ctrl = ctrl_data[ctrl_data['period'] == period].copy()
    nk['treat'] = 1
    ctrl['treat'] = 0
    combined = pd.concat([nk, ctrl], ignore_index=True)
    months = sorted(combined['month'].unique())
    month_map = {m: i for i, m in enumerate(months)}
    combined['time'] = combined['month'].map(month_map)
    
    try:
        model = smf.ols('framing_mean ~ treat + time + treat:time', data=combined).fit(
            cov_type='cluster', cov_kwds={'groups': combined['month']}
        )
        return model.pvalues['treat:time']
    except:
        return None

def run_did(nk_data, ctrl_data, p_start, p_end):
    nk = nk_data[nk_data['period'].isin([p_start, p_end])].copy()
    ctrl = ctrl_data[ctrl_data['period'].isin([p_start, p_end])].copy()
    nk['treat'] = 1
    ctrl['treat'] = 0
    combined = pd.concat([nk, ctrl], ignore_index=True)
    combined['post'] = (combined['period'] == p_end).astype(int)
    combined['did'] = combined['treat'] * combined['post']
    months = sorted(combined['month'].unique())
    month_map = {m: i for i, m in enumerate(months)}
    combined['time'] = combined['month'].map(month_map)
    
    try:
        model = smf.ols('framing_mean ~ treat + time + post + did', data=combined).fit(
            cov_type='cluster', cov_kwds={'groups': combined['month']}
        )
        coef = model.params['did']
        pval = model.pvalues['did']
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        return coef, pval, sig
    except:
        return None, None, "ERROR"

def main():
    print("="*80)
    print(f"📊 HIGH-CONFIDENCE DiD (Confidence >= {CONFIDENCE_THRESHOLD})")
    print("="*80)
    
    # Create monthly aggregates
    data = {}
    for country in ['nk', 'china', 'iran', 'russia']:
        df = create_monthly_highconf(country, CONFIDENCE_THRESHOLD)
        data[country] = df
        total = df['post_count'].sum()
        print(f"{country}: {len(df)} months, {total:.0f} high-conf posts")
    
    nk = data['nk']
    
    print("\n" + "="*80)
    print("🏛️ SINGAPORE (P1 → P2)")
    print("="*80)
    
    print("\n📉 Parallel Trends (P1)")
    for ctrl in ['china', 'russia']:
        pval = test_parallel_trends(nk, data[ctrl], 'P1')
        result = "✅ PASS" if pval and pval > 0.10 else "❌ FAIL"
        print(f"   {ctrl.title():8} | p = {pval:.4f} {result}" if pval else f"   {ctrl.title():8} | ERROR")
    
    print("\n📊 DiD (P1 → P2)")
    for ctrl in ['china', 'russia']:
        coef, pval, sig = run_did(nk, data[ctrl], 'P1', 'P2')
        if coef:
            print(f"   {ctrl.title():8} | DiD: {coef:+.4f} | p: {pval:.4f} {sig}")
    
    print("\n" + "="*80)
    print("🏛️ HANOI (P2 → P3)")
    print("="*80)
    
    print("\n📉 Parallel Trends (P2)")
    for ctrl in ['china', 'russia']:
        pval = test_parallel_trends(nk, data[ctrl], 'P2')
        result = "✅ PASS" if pval and pval > 0.10 else "❌ FAIL"
        print(f"   {ctrl.title():8} | p = {pval:.4f} {result}" if pval else f"   {ctrl.title():8} | ERROR")
    
    print("\n📊 DiD (P2 → P3)")
    for ctrl in ['china', 'russia']:
        coef, pval, sig = run_did(nk, data[ctrl], 'P2', 'P3')
        if coef:
            print(f"   {ctrl.title():8} | DiD: {coef:+.4f} | p: {pval:.4f} {sig}")

if __name__ == "__main__":
    main()
