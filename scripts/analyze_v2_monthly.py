"""
Create monthly aggregates from V2 framing data and run same DiD analysis
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Load V2 post-level data and create monthly aggregates
POST_RESULTS_DIR = "data/results/final_framing_v2"
META_FILES = {
    "nk": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
    "china": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
    "iran": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
    "russia": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"]
}

SCALE = {'THREAT': -2, 'ECONOMIC': -1, 'NEUTRAL': 0, 'HUMANITARIAN': 1, 'DIPLOMACY': 2, 'ERROR': 0}

def create_monthly_v2(country):
    res_df = pd.read_csv(f"{POST_RESULTS_DIR}/{country}_framing_v2.csv")
    
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
    
    return monthly

def prepare_did_data(nk_data, control_data, control_name):
    nk_data = nk_data.copy()
    control_data = control_data.copy()
    nk_data['treat'] = 1
    control_data['treat'] = 0
    combined = pd.concat([nk_data, control_data], ignore_index=True)
    combined['time'] = combined.groupby('topic').cumcount()
    combined['post'] = (combined['time'] >= 14).astype(int)  # Pre: 0-13, Post: 14+
    return combined

def test_parallel_trends(combined, control_name):
    pre_data = combined[combined['post'] == 0].copy()
    try:
        model = smf.ols('framing_mean ~ treat + time + treat:time', data=pre_data).fit(
            cov_type='cluster', cov_kwds={'groups': pre_data['month']}
        )
        pval = model.pvalues['treat:time']
        result = "✓ PASS" if pval > 0.10 else "✗ FAIL"
        return pval, result
    except:
        return None, "ERROR"

def run_level_did(combined, control_name):
    try:
        model = smf.ols('framing_mean ~ treat + time + post + treat:post', data=combined).fit(
            cov_type='cluster', cov_kwds={'groups': combined['month']}
        )
        coef = model.params['treat:post']
        pval = model.pvalues['treat:post']
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        return coef, pval, sig
    except:
        return None, None, "ERROR"

def main():
    print("="*90)
    print("📊 V2 FRAMING - MONTHLY AGGREGATED DiD ANALYSIS")
    print("="*90)
    
    # Create monthly aggregates from V2
    monthly = {}
    for country in ['nk', 'china', 'iran', 'russia']:
        df = create_monthly_v2(country)
        monthly[country] = df
        print(f"Created {country}: {len(df)} months, {df['post_count'].sum():.0f} posts")
    
    nk = monthly['nk']
    
    print("\n" + "="*90)
    print("📉 PARALLEL TRENDS TEST (p > 0.10 = PASS)")
    print("="*90)
    
    for ctrl in ['china', 'iran', 'russia']:
        combined = prepare_did_data(nk, monthly[ctrl], ctrl)
        pval, result = test_parallel_trends(combined, ctrl)
        if pval:
            print(f"NK vs {ctrl.title():8} | β₄ p-value: {pval:.4f} | {result}")
    
    print("\n" + "="*90)
    print("📊 LEVEL CHANGE DiD (treat:post coefficient)")
    print("="*90)
    
    for ctrl in ['china', 'iran', 'russia']:
        combined = prepare_did_data(nk, monthly[ctrl], ctrl)
        coef, pval, sig = run_level_did(combined, ctrl)
        if coef:
            print(f"NK vs {ctrl.title():8} | DiD: {coef:+.4f} | p: {pval:.4f} {sig}")

if __name__ == "__main__":
    main()
