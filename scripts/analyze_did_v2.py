
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os
import glob

# Configuration
RESULTS_DIR = "data/results/final_framing_v2"
DATA_DIR = "data"

META_FILES = {
    "nk": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
    "china": ["data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv"],
    "iran": ["data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv"],
    "russia": ["data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"]
}

# Diplomacy Scale
SCALE = {
    'THREAT': -2,
    'ECONOMIC': -1,
    'NEUTRAL': 0,
    'HUMANITARIAN': 1,
    'DIPLOMACY': 2,
    'ERROR': 0  # Should be filtered out, but 0 is safety
}

def load_and_merge(country_key):
    # 1. Load V2 Results
    res_path = f"{RESULTS_DIR}/{country_key}_framing_v2.csv"
    if not os.path.exists(res_path):
        print(f"❌ Missing result file: {res_path}")
        return None
    
    res_df = pd.read_csv(res_path)
    # Check if 'id' column exists, sometimes it might be index if not saved properly
    if 'id' not in res_df.columns:
         print(f"❌ 'id' column missing in {res_path}")
         return None

    # 2. Load Original Metadata (for Dates)
    meta_dfs = []
    for p in META_FILES[country_key]:
        if os.path.exists(p):
            meta_dfs.append(pd.read_csv(p, usecols=['id', 'created_utc'], low_memory=False))
            
    if not meta_dfs:
        print(f"❌ Missing metadata files for {country_key}")
        return None
        
    meta_df = pd.concat(meta_dfs, ignore_index=True)
    meta_df = meta_df.drop_duplicates(subset=['id'])
    
    # 3. Merge
    merged = pd.merge(res_df, meta_df, on='id', how='inner')
    
    # 4. Parse Dates
    merged['date'] = pd.to_datetime(merged['created_utc'], unit='s')
    merged['month'] = merged['date'].dt.to_period('M').astype(str)
    
    # 5. Apply Scale
    merged['framing_score'] = merged['frame'].map(SCALE).fillna(0)
    
    return merged

def assign_period(month_str):
    # P1: Pre-Summit (Until Feb 2018)
    if month_str <= '2018-02':
        return 'P1'
    # Transition: Mar-May 2018 (Excluded)
    elif '2018-03' <= month_str <= '2018-05':
        return None
    # P2: Singapore Era (June 2018 - Jan 2019)
    elif '2018-06' <= month_str <= '2019-01':
        return 'P2'
    # Hanoi Summit: Feb 2019 (Excluded)
    elif month_str == '2019-02':
        return None
    # P3: Post-Hanoi (Mar 2019 - Dec 2019)
    elif '2019-03' <= month_str <= '2019-12':
        return 'P3'
    return None

def run_did_ols(df_treat, df_control, treat_name, control_name, p_start, p_end, label):
    # Filter Periods
    t = df_treat[df_treat['period'].isin([p_start, p_end])].copy()
    c = df_control[df_control['period'].isin([p_start, p_end])].copy()
    
    t['is_treat'] = 1
    c['is_treat'] = 0
    
    combined = pd.concat([t, c], ignore_index=True)
    combined['is_post'] = (combined['period'] == p_end).astype(int)
    combined['did'] = combined['is_treat'] * combined['is_post']
    
    # OLS
    # Cluster by Month to account for temporal correlation
    model = smf.ols('framing_score ~ is_treat + is_post + did', data=combined).fit(
        cov_type='cluster', cov_kwds={'groups': combined['month']}
    )
    
    coef = model.params['did']
    pval = model.pvalues['did']
    
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    
    print(f"{label:25} | {control_name:8} | DiD: {coef:+.3f} | p: {pval:.4f} {sig}")
    return coef, pval

def main():
    print("="*80)
    print("📊 DiD ANALYSIS: V2 FRAMING (Diplomacy Scale: -2 THREAT to +2 DIPLOMACY)")
    print("="*80)
    
    # Load Data
    data = {}
    for country in ['nk', 'china', 'iran', 'russia']:
        print(f"Loading {country}...", end='\r')
        df = load_and_merge(country)
        if df is not None:
            df['period'] = df['month'].apply(assign_period)
            df = df.dropna(subset=['period']) # Drop transition/out-of-range
            data[country] = df
            print(f"Loaded {country}: {len(df)} posts ({df['month'].min()} ~ {df['month'].max()})")
    
    print("-" * 80)
    print(f"{'Analysis':25} | {'Control':8} | {'Result':20} | {'Significance'}")
    print("-" * 80)
    
    nk = data['nk']
    
    # Singapore Effect (P1 -> P2)
    # Expectation: Shift towards Diplomacy (+ve DiD)
    for control in ['china', 'iran', 'russia']:
        run_did_ols(nk, data[control], 'NK', control.title(), 'P1', 'P2', 'Singapore (P1->P2)')

    print("-" * 80)
    
    # Hanoi Failure Effect (P2 -> P3)
    # Expectation: Shift towards Threat (-ve DiD)
    for control in ['china', 'iran', 'russia']:
        run_did_ols(nk, data[control], 'NK', control.title(), 'P2', 'P3', 'Hanoi Fail (P2->P3)')

if __name__ == "__main__":
    main()
