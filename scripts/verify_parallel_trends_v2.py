import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Configuration
POST_RESULTS_DIR = "data/results/final_framing_v2"

META_FILES = {
    "nk": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
    "china": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
    "iran": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
    "russia": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"]
}

SCALE = {
    'THREAT': -2, 'ECONOMIC': -1, 'NEUTRAL': 0, 'HUMANITARIAN': 1, 'DIPLOMACY': 2, 'ERROR': 0
}

def load_post_data(country_key):
    res_path = f"{POST_RESULTS_DIR}/{country_key}_framing_v2.csv"
    if not os.path.exists(res_path): return None
    res_df = pd.read_csv(res_path)
    
    meta_dfs = []
    for p in META_FILES[country_key]:
        if os.path.exists(p):
            meta_dfs.append(pd.read_csv(p, usecols=['id', 'created_utc'], low_memory=False))
            
    if not meta_dfs: return None
    meta_df = pd.concat(meta_dfs, ignore_index=True).drop_duplicates(subset=['id'])
    
    merged = pd.merge(res_df, meta_df, on='id', how='inner')
    merged['date'] = pd.to_datetime(merged['created_utc'], unit='s')
    return merged[['id', 'frame', 'date']]

def assign_period(month_str):
    if month_str <= '2018-02': return 'P1'
    elif '2018-03' <= month_str <= '2018-05': return None
    elif '2018-06' <= month_str <= '2019-01': return 'P2'
    elif month_str == '2019-02': return None
    elif '2019-03' <= month_str <= '2019-12': return 'P3'
    return None

def test_parallel_trends(df_treat, df_control, treat_name, control_name, period_code, label):
    t = df_treat[df_treat['period'] == period_code].copy()
    c = df_control[df_control['period'] == period_code].copy()
    
    t['is_treat'] = 1
    c['is_treat'] = 0
    
    combined = pd.concat([t, c], ignore_index=True)
    
    months = sorted(combined['month'].unique())
    month_map = {m: i for i, m in enumerate(months)}
    combined['time_trend'] = combined['month'].map(month_map)
    combined['treat_time'] = combined['is_treat'] * combined['time_trend']
    
    try:
        model = smf.ols('framing_score ~ is_treat + time_trend + treat_time', data=combined).fit(
            cov_type='cluster', cov_kwds={'groups': combined['month']}
        )
        pval = model.pvalues['treat_time']
        result = "✅ PASS" if pval > 0.05 else "❌ FAIL"
        
        print(f"{label:20} | {control_name:8} | Period: {period_code} | p={pval:.4f} {result}")
        return pval > 0.05
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("="*80)
    print("📉 PARALLEL TRENDS TEST (Post-Only V2 Data)")
    print("="*80)
    
    datasets = {}
    for country in ['nk', 'china', 'iran', 'russia']:
        df = load_post_data(country)
        if df is not None:
            df['month'] = df['date'].dt.to_period('M').astype(str)
            df['framing_score'] = df['frame'].map(SCALE).fillna(0)
            df['period'] = df['month'].apply(assign_period)
            datasets[country] = df.dropna(subset=['period'])
            
    nk = datasets['nk']
    
    print("\n1. Pre-Singapore (P1)")
    print("-" * 80)
    for ctrl in ['china', 'iran', 'russia']:
        test_parallel_trends(nk, datasets[ctrl], 'NK', ctrl.title(), 'P1', 'Pre-Summit')
        
    print("\n2. Pre-Hanoi (P2)")
    print("-" * 80)
    for ctrl in ['china', 'iran', 'russia']:
        test_parallel_trends(nk, datasets[ctrl], 'NK', ctrl.title(), 'P2', 'Between-Summit')

if __name__ == "__main__":
    main()
