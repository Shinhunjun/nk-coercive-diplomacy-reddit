"""
DiD Analysis using FULL ORIGINAL FRAMING DATA (data/final/*_framing_final.csv)
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Full original framing files
ORIGINAL_FILES = {
    "nk": "data/final/nk_framing_final.csv",
    "china": "data/final/china_framing_final.csv",
    "iran": "data/final/iran_framing_final.csv", 
    "russia": "data/final/russia_framing_final.csv"
}

SCALE = {
    'THREAT': -2, 'ECONOMIC': -1, 'NEUTRAL': 0, 'HUMANITARIAN': 1, 'DIPLOMACY': 2
}

def load_data(country_key):
    path = ORIGINAL_FILES[country_key]
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None
    
    df = pd.read_csv(path, low_memory=False)
    
    # Parse date
    if 'created_utc' in df.columns:
        df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    elif 'datetime' in df.columns:
        df['date'] = pd.to_datetime(df['datetime'])
    
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['framing_score'] = df['frame'].map(SCALE).fillna(0)
    
    return df

def assign_period(month_str):
    if month_str <= '2018-02': return 'P1'
    elif '2018-03' <= month_str <= '2018-05': return None
    elif '2018-06' <= month_str <= '2019-01': return 'P2'
    elif month_str == '2019-02': return None
    elif '2019-03' <= month_str <= '2019-12': return 'P3'
    return None

def test_parallel_trends(df_treat, df_control, control_name, period_code):
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
        return pval, result
    except:
        return None, "ERROR"

def run_did(df_treat, df_control, p_start, p_end):
    t = df_treat[df_treat['period'].isin([p_start, p_end])].copy()
    c = df_control[df_control['period'].isin([p_start, p_end])].copy()
    
    t['is_treat'] = 1
    c['is_treat'] = 0
    
    combined = pd.concat([t, c], ignore_index=True)
    combined['is_post'] = (combined['period'] == p_end).astype(int)
    combined['did'] = combined['is_treat'] * combined['is_post']
    
    model = smf.ols('framing_score ~ is_treat + is_post + did', data=combined).fit(
        cov_type='cluster', cov_kwds={'groups': combined['month']}
    )
    
    coef = model.params['did']
    pval = model.pvalues['did']
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    
    return coef, pval, sig

def main():
    print("="*90)
    print("📊 FULL ORIGINAL FRAMING DATA ANALYSIS (data/final/*_framing_final.csv)")
    print("="*90)
    
    # Load data
    datasets = {}
    for country in ['nk', 'china', 'iran', 'russia']:
        df = load_data(country)
        if df is not None:
            df['period'] = df['month'].apply(assign_period)
            df = df.dropna(subset=['period'])
            datasets[country] = df
            print(f"Loaded {country}: {len(df)} posts ({df['month'].min()} ~ {df['month'].max()})")
    
    nk = datasets['nk']
    
    # Parallel Trends Test
    print("\n" + "="*90)
    print("📉 PARALLEL TRENDS TEST")
    print("="*90)
    
    print("\n1. Pre-Singapore (P1)")
    print("-" * 90)
    for ctrl in ['china', 'iran', 'russia']:
        pval, result = test_parallel_trends(nk, datasets[ctrl], ctrl.title(), 'P1')
        if pval:
            print(f"Pre-Summit           | {ctrl.title():8} | Period: P1 | p={pval:.4f} {result}")
        
    print("\n2. Pre-Hanoi (P2)")
    print("-" * 90)
    for ctrl in ['china', 'iran', 'russia']:
        pval, result = test_parallel_trends(nk, datasets[ctrl], ctrl.title(), 'P2')
        if pval:
            print(f"Between-Summit       | {ctrl.title():8} | Period: P2 | p={pval:.4f} {result}")
    
    # DiD Analysis
    print("\n" + "="*90)
    print("📊 DiD ANALYSIS")
    print("="*90)
    
    print(f"\n{'Analysis':25} | {'Control':8} | {'DiD Coef':10} | {'P-value':10} | {'Sig'}")
    print("-" * 90)
    
    # Singapore (P1 -> P2)
    for ctrl in ['china', 'iran', 'russia']:
        coef, pval, sig = run_did(nk, datasets[ctrl], 'P1', 'P2')
        print(f"{'Singapore (P1->P2)':25} | {ctrl.title():8} | {coef:+.4f}     | {pval:.4f}     | {sig}")
    
    print("-" * 90)
    
    # Hanoi (P2 -> P3)
    for ctrl in ['china', 'iran', 'russia']:
        coef, pval, sig = run_did(nk, datasets[ctrl], 'P2', 'P3')
        print(f"{'Hanoi Fail (P2->P3)':25} | {ctrl.title():8} | {coef:+.4f}     | {pval:.4f}     | {sig}")

if __name__ == "__main__":
    main()
