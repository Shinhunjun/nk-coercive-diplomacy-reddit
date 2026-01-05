"""
Complete DiD Analysis for BOTH Periods (Singapore & Hanoi)
- Original Prompt vs V2 Prompt
- Monthly Aggregated Data
- Parallel Trends + Level Change DiD
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# ==========================================
# DATA LOADING
# ==========================================

# Original monthly data paths
ORIGINAL_MONTHLY = {
    "nk": "data/framing/nk_monthly_framing.csv",
    "china": "data/framing/china_monthly_framing.csv",
    "iran": "data/framing/iran_monthly_framing.csv",
    "russia": "data/framing/russia_monthly_framing.csv"
}

# V2 post-level data paths  
V2_POSTS = "data/results/final_framing_v2"
META_FILES = {
    "nk": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
    "china": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
    "iran": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
    "russia": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"]
}

SCALE = {'THREAT': -2, 'ECONOMIC': -1, 'NEUTRAL': 0, 'HUMANITARIAN': 1, 'DIPLOMACY': 2, 'ERROR': 0}

def load_original_monthly():
    data = {}
    for country, path in ORIGINAL_MONTHLY.items():
        df = pd.read_csv(path)
        df['topic'] = country
        data[country] = df
    return data

def create_v2_monthly():
    data = {}
    for country in ['nk', 'china', 'iran', 'russia']:
        res_df = pd.read_csv(f"{V2_POSTS}/{country}_framing_v2.csv")
        
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
        data[country] = monthly
    return data

# ==========================================
# PERIOD DEFINITIONS
# ==========================================

def assign_period(month):
    """Assign period with transition exclusion"""
    if month <= '2018-02':  # Pre-Singapore
        return 'P1'
    elif '2018-03' <= month <= '2018-05':  # Transition
        return None
    elif '2018-06' <= month <= '2019-01':  # Singapore-Hanoi
        return 'P2'
    elif month == '2019-02':  # Hanoi month
        return None
    elif '2019-03' <= month <= '2019-12':  # Post-Hanoi
        return 'P3'
    return None

# ==========================================
# ANALYSIS FUNCTIONS
# ==========================================

def test_parallel_trends(nk_data, ctrl_data, period):
    """Test parallel trends in specified period"""
    nk = nk_data[nk_data['period'] == period].copy()
    ctrl = ctrl_data[ctrl_data['period'] == period].copy()
    
    nk['treat'] = 1
    ctrl['treat'] = 0
    
    combined = pd.concat([nk, ctrl], ignore_index=True)
    months = sorted(combined['month'].unique())
    month_map = {m: i for i, m in enumerate(months)}
    combined['time'] = combined['month'].map(month_map)
    combined['treat_time'] = combined['treat'] * combined['time']
    
    try:
        model = smf.ols('framing_mean ~ treat + time + treat:time', data=combined).fit(
            cov_type='cluster', cov_kwds={'groups': combined['month']}
        )
        pval = model.pvalues['treat:time']
        return pval
    except:
        return None

def run_did(nk_data, ctrl_data, p_start, p_end):
    """Run Level Change DiD"""
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

def analyze_dataset(data, label):
    """Run full analysis on a dataset"""
    print(f"\n{'='*90}")
    print(f"📊 {label}")
    print(f"{'='*90}")
    
    # Add period to all
    for country in data:
        data[country]['period'] = data[country]['month'].apply(assign_period)
        data[country] = data[country].dropna(subset=['period'])
    
    nk = data['nk']
    
    # ===== SINGAPORE (P1 -> P2) =====
    print(f"\n{'─'*90}")
    print("🏛️ SINGAPORE SUMMIT (P1 → P2)")
    print(f"{'─'*90}")
    
    print("\n📉 Parallel Trends (Pre-Singapore, P1) - p > 0.10 = PASS")
    for ctrl in ['china', 'iran', 'russia']:
        pval = test_parallel_trends(nk, data[ctrl], 'P1')
        result = "✅ PASS" if pval and pval > 0.10 else "❌ FAIL"
        print(f"   {ctrl.title():8} | p = {pval:.4f} {result}" if pval else f"   {ctrl.title():8} | ERROR")
    
    print("\n📊 DiD Level Change (P1 → P2)")
    for ctrl in ['china', 'iran', 'russia']:
        coef, pval, sig = run_did(nk, data[ctrl], 'P1', 'P2')
        if coef:
            print(f"   {ctrl.title():8} | DiD: {coef:+.4f} | p: {pval:.4f} {sig}")
    
    # ===== HANOI (P2 -> P3) =====
    print(f"\n{'─'*90}")
    print("🏛️ HANOI SUMMIT (P2 → P3)")
    print(f"{'─'*90}")
    
    print("\n📉 Parallel Trends (Pre-Hanoi, P2) - p > 0.10 = PASS")
    for ctrl in ['china', 'iran', 'russia']:
        pval = test_parallel_trends(nk, data[ctrl], 'P2')
        result = "✅ PASS" if pval and pval > 0.10 else "❌ FAIL"
        print(f"   {ctrl.title():8} | p = {pval:.4f} {result}" if pval else f"   {ctrl.title():8} | ERROR")
    
    print("\n📊 DiD Level Change (P2 → P3)")
    for ctrl in ['china', 'iran', 'russia']:
        coef, pval, sig = run_did(nk, data[ctrl], 'P2', 'P3')
        if coef:
            print(f"   {ctrl.title():8} | DiD: {coef:+.4f} | p: {pval:.4f} {sig}")

def main():
    print("="*90)
    print("📊 COMPLETE TWO-PERIOD DiD ANALYSIS (Monthly Aggregated)")
    print("="*90)
    
    # Load Original
    print("\nLoading Original Monthly Data...")
    original = load_original_monthly()
    
    # Load V2
    print("Creating V2 Monthly Aggregates...")
    v2 = create_v2_monthly()
    
    # Analyze both
    analyze_dataset(original, "ORIGINAL PROMPT DATA")
    analyze_dataset(v2, "V2 PROMPT DATA")

if __name__ == "__main__":
    main()
