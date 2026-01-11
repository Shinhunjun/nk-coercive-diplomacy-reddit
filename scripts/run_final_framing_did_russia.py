
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from datetime import datetime

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

def analyze_russia_did():
    print("Loading FULL Data (NK & Russia 81k)...")
    
    # 1. Load Framing Results
    nk_p1 = pd.read_csv('data/processed/nk_p1_framing_results.csv')
    nk_p2p3 = pd.read_csv('data/processed/nk_p2_p3_framing_results.csv')
    russia_full = pd.read_csv('data/processed/russia_framing_results.csv')
    
    nk_frames = pd.concat([nk_p1, nk_p2p3])
    
    # Load Dates
    nk_dates = pd.read_csv('data/processed/nk_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    russia_dates = pd.read_csv('data/processed/russia_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'], low_memory=False)
    
    # Clean Russia Dates
    russia_dates['created_utc'] = pd.to_numeric(russia_dates['created_utc'], errors='coerce')
    russia_dates = russia_dates.dropna(subset=['created_utc'])
    
    # Merge
    nk_frames['id'] = nk_frames['id'].astype(str)
    russia_full['id'] = russia_full['id'].astype(str)
    nk_dates['id'] = nk_dates['id'].astype(str)
    russia_dates['id'] = russia_dates['id'].astype(str)
    
    nk = nk_frames.merge(nk_dates, on='id')
    russia = russia_full.merge(russia_dates, on='id')
    
    # 2. Apply Scale & Treatment
    nk['score'] = nk['frame'].map(SCALE_MAP).fillna(0)
    russia['score'] = russia['frame'].map(SCALE_MAP).fillna(0)
    
    nk['treated'] = 1
    russia['treated'] = 0
    
    # 3. Combine & Monthly Aggregation
    df = pd.concat([nk, russia])
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['date'].dt.to_period('M')
    
    monthly = df.groupby(['month', 'treated'])['score'].mean().reset_index()
    monthly['month_num'] = (monthly['month'].dt.year - 2017) * 12 + (monthly['month'].dt.month)
    monthly['ts'] = monthly['month'].dt.to_timestamp()
    
    print("\n--- DATA COUNTS ---")
    print(f"NK Total: {len(nk)}")
    print(f"Russia Total: {len(russia)}")
    
    # --- [STEP 1] PARALLEL TRENDS VERIFICATION (P1 for Singapore Effect) ---
    print("\n" + "="*65)
    print("1. PARALLEL TRENDS [P1] (Verify Baseline for Summit)")
    print("="*65)
    
    data_p1 = monthly[monthly['ts'] < '2018-03-01'].copy() # Buffer Excluded
    model_pt1 = smf.ols('score ~ month_num * treated', data=data_p1).fit()
    pt1_pval = model_pt1.pvalues['month_num:treated']
    print(model_pt1.summary().tables[1])
    print(f"\n>> P1 PT Interaction P-Value: {pt1_pval:.4f}")
    if pt1_pval > 0.05:
        print(">> VERDICT: ✅ PASS (Trends parallel before Summit)")
    else:
        print(">> VERDICT: ❌ FAIL (Trends divergent before Summit)")

    # --- [STEP 2] SINGAPORE SUMMIT DiD (P1 -> P2) ---
    print("\n" + "="*65)
    print("2. SINGAPORE SUMMIT DiD (P1 vs P2)")
    print("="*65)
    
    data_p1p2 = monthly[(monthly['ts'] < '2019-02-01')].copy() # Exclude Feb 2019
    data_p1p2['post'] = (data_p1p2['ts'] >= '2018-06-12').astype(int)
    
    model_s = smf.ols('score ~ treated * post', data=data_p1p2).fit()
    s_coef = model_s.params['treated:post']
    s_pval = model_s.pvalues['treated:post']
    print(model_s.summary().tables[1])
    print(f"\n>> DiD Coefficient: {s_coef:.4f} (p={s_pval:.4f})")

    # --- [STEP 3] PARALLEL TRENDS VERIFICATION (P2 for Ratchet Effect) ---
    print("\n" + "="*65)
    print("3. PARALLEL TRENDS [P2] (Verify Baseline for Ratchet Effect)")
    print("="*65)
    
    data_p2_solo = monthly[(monthly['ts'] >= '2018-06-12') & (monthly['ts'] < '2019-02-01')].copy()
    if len(data_p2_solo) > 4:
        model_pt2 = smf.ols('score ~ month_num * treated', data=data_p2_solo).fit()
        pt2_pval = model_pt2.pvalues['month_num:treated']
        print(model_pt2.summary().tables[1])
        print(f"\n>> P2 PT Interaction P-Value: {pt2_pval:.4f}")
        if pt2_pval > 0.05:
            print(">> VERDICT: ✅ PASS (Trends parallel in P2 Summit era)")
        else:
            print(">> VERDICT: ❌ FAIL (Trends divergent in P2)")
    else:
        print(">> SKIP: Not enough monthly data points in P2 for trend test.")

    # --- [STEP 4] HANOI / RATCHET DiD (P2 -> P3) ---
    print("\n" + "="*65)
    print("4. HANOI COLLAPSE & RATCHET DiD (P2 vs P3)")
    print("="*65)
    
    data_p2p3 = monthly[monthly['ts'] >= '2018-06-12'].copy()
    data_p2p3 = data_p2p3[data_p2p3['ts'] != '2019-02-01'] # Exclude Feb 2019
    data_p2p3['post_hanoi'] = (data_p2p3['ts'] >= '2019-03-01').astype(int)
    
    model_r = smf.ols('score ~ treated * post_hanoi', data=data_p2p3).fit()
    r_coef = model_r.params['treated:post_hanoi']
    r_pval = model_r.pvalues['treated:post_hanoi']
    print(model_r.summary().tables[1])
    print(f"\n>> DiD Coefficient: {r_coef:.4f} (p={r_pval:.4f})")

if __name__ == "__main__":
    analyze_russia_did()
