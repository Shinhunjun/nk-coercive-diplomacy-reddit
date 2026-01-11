
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
import sys

warnings.filterwarnings("ignore")

# SCORING CONFIG
# THREAT = -2
# DIPLOMACY = +2
# REST = 0
SCALE_MAP = {
    'THREAT': -2,
    'DIPLOMACY': 2,
    'ECONOMIC': 0,
    'NEUTRAL': 0,
    'HUMANITARIAN': 0
}

def analyze_framing_did():
    print("Loading Data for DiD Analysis...")
    
    # 1. Load Data
    nk_p1 = pd.read_csv('data/processed/nk_p1_framing_results.csv')
    nk_p2p3 = pd.read_csv('data/processed/nk_p2_p3_framing_results.csv')
    china_partial = pd.read_csv('data/processed/china_framing_results.csv')
    
    nk_frames = pd.concat([nk_p1, nk_p2p3])
    
    # Load Dates
    nk_dates = pd.read_csv('data/processed/nk_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    china_dates = pd.read_csv('data/processed/china_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    
    # Merge
    nk_frames['id'] = nk_frames['id'].astype(str)
    china_partial['id'] = china_partial['id'].astype(str)
    nk_dates['id'] = nk_dates['id'].astype(str)
    china_dates['id'] = china_dates['id'].astype(str)
    
    nk = nk_frames.merge(nk_dates, on='id')
    china = china_partial.merge(china_dates, on='id')
    
    # 2. Apply Scale & Treatment
    nk['score'] = nk['frame'].map(SCALE_MAP).fillna(0)
    china['score'] = china['frame'].map(SCALE_MAP).fillna(0)
    
    nk['treated'] = 1
    china['treated'] = 0
    
    # 3. Combine
    df = pd.concat([nk, china])
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['date'].dt.to_period('M')
    
    # 4. Define Periods
    # Summit Date: 2018-06-12 (Start of P2)
    # Hanoi Date: 2019-02-28 (Start of P3)
    
    summit_date = pd.Timestamp('2018-06-12')
    hanoi_date = pd.Timestamp('2019-02-28')
    
    print("\n--- DATA COUNTS ---")
    print(f"Total Observations: {len(df)}")
    print(f"NK: {len(nk)} / China: {len(china)}")
    
    # Aggregation for Robust Estimator (Monthly Means)
    monthly = df.groupby(['month', 'treated'])['score'].mean().reset_index()
    monthly['month_num'] = monthly['month'].astype(int)
    
    # Filter Periods based on dates
    # (Since monthly data is period object, we need to map back to timestamps for filtering or use month strings)
    
    # Convert month to timestamp (start of month)
    monthly['ts'] = monthly['month'].dt.to_timestamp()
    
    print("\n" + "="*60)
    print("DiD RESULTS: Framing Score (-2 to +2)")
    print("="*60)
    
    # --- MODEL 1: Singapore Summit Effect (P1 vs P2) ---
    # P1: < 2018-06
    # P2: 2018-06 to 2019-02
    
    data_p1p2 = monthly[(monthly['ts'] < '2019-03-01')].copy() # Up to Feb 2019
    data_p1p2['post'] = (data_p1p2['ts'] >= '2018-06-01').astype(int) # June 2018 onwards is Post
    
    print("\n[1] Singapore Summit Effect (P1 -> P2)")
    model_p1p2 = smf.ols('score ~ treated * post', data=data_p1p2).fit()
    
    coef_1 = model_p1p2.params['treated:post']
    pval_1 = model_p1p2.pvalues['treated:post']
    
    print(model_p1p2.summary().tables[1])
    print(f">> DiD Coefficient: {coef_1:.4f} (p={pval_1:.4f})")
    if pval_1 < 0.1:
        print(">> RESULT: ✅ SIGNIFICANT Positive Shift (Conflict -> Peace)")
    else:
        print(">> RESULT: ❌ No Significant Shift")
        
    # --- MODEL 2: Ratchet Effect (Hanoi Collapse) (P2 vs P3) ---
    # P2: 2018-06 to 2019-02 (Pre-Hanoi)
    # P3: >= 2019-03 (Post-Hanoi)
    
    # Note: We only have partial P3 for China (up to June 2019 based on log), but that's enough for immediate effect
    data_p2p3 = monthly[(monthly['ts'] >= '2018-06-01')].copy()
    data_p2p3['post_hanoi'] = (data_p2p3['ts'] >= '2019-03-01').astype(int)
    
    print("\n[2] Hanoi Collapse Effect (P2 -> P3)")
    # Hypothesis: If Ratchet Effect holds, Coefficient should be INSIGNIFICANT or SMALL Negative
    # If Diplomacy failed completely, Coefficient should be LARGE Negative (returning to P1 levels)
    
    model_p2p3 = smf.ols('score ~ treated * post_hanoi', data=data_p2p3).fit()
    
    coef_2 = model_p2p3.params['treated:post_hanoi']
    pval_2 = model_p2p3.pvalues['treated:post_hanoi']
    
    print(model_p2p3.summary().tables[1])
    print(f">> DiD Coefficient: {coef_2:.4f} (p={pval_2:.4f})")
    
    if pval_2 > 0.1:
        print(">> RESULT: ✅ Ratchet Effect CONFIRMED (No significant reversal)")
    elif coef_2 < 0:
        print(">> RESULT: ⚠️ Reversal Detected (Negative Shift)")
    else:
        print(">> RESULT: ❓ Positive Shift Continued")

if __name__ == "__main__":
    analyze_framing_did()
