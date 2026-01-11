
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

def final_analysis():
    print("Loading FULL Data (NK & China 62k)...")
    
    # 1. Load Framing Results
    nk_p1 = pd.read_csv('data/processed/nk_p1_framing_results.csv')
    nk_p2p3 = pd.read_csv('data/processed/nk_p2_p3_framing_results.csv')
    china_full = pd.read_csv('data/processed/china_framing_results.csv')
    
    nk_frames = pd.concat([nk_p1, nk_p2p3])
    
    # Load Dates
    nk_dates = pd.read_csv('data/processed/nk_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    china_dates = pd.read_csv('data/processed/china_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    
    # Merge
    nk_frames['id'] = nk_frames['id'].astype(str)
    china_full['id'] = china_full['id'].astype(str)
    nk_dates['id'] = nk_dates['id'].astype(str)
    china_dates['id'] = china_dates['id'].astype(str)
    
    nk = nk_frames.merge(nk_dates, on='id')
    china = china_full.merge(china_dates, on='id')
    
    # 2. Apply Scale & Treatment
    nk['score'] = nk['frame'].map(SCALE_MAP).fillna(0)
    china['score'] = china['frame'].map(SCALE_MAP).fillna(0)
    
    nk['treated'] = 1
    china['treated'] = 0
    
    # 3. Combine & Monthly Aggregation
    df = pd.concat([nk, china])
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['date'].dt.to_period('M')
    
    monthly = df.groupby(['month', 'treated'])['score'].mean().reset_index()
    monthly['month_num'] = (monthly['month'].dt.year - 2017) * 12 + (monthly['month'].dt.month)
    monthly['ts'] = monthly['month'].dt.to_timestamp()
    
    print("\n--- DATA COUNTS ---")
    print(f"NK Total: {len(nk)}")
    print(f"China Total (Final): {len(china)}")
    
    # --- [STEP 1] PARALLEL TRENDS VERIFICATION (P1) ---
    print("\n" + "="*60)
    print("1. PARALLEL TRENDS VERIFICATION (Pre-Summit)")
    print("="*60)
    
    # Exclude buffer period (March - May 2018) to match Post-level analysis
    data_p1 = monthly[monthly['ts'] < '2018-03-01'].copy()
    print("\n--- P1 Monthly Means (NK vs China) [Buffer Excluded] ---")
    print(data_p1[['month', 'treated', 'score']].pivot(index='month', columns='treated', values='score'))
    model_pt = smf.ols('score ~ month_num * treated', data=data_p1).fit()
    
    pt_pval = model_pt.pvalues['month_num:treated']
    print(model_pt.summary().tables[1])
    print(f"\n>> PT Interation P-Value: {pt_pval:.4f}")
    if pt_pval > 0.1:
        print(">> VERDICT: ✅ PASS (Trends are parallel)")
    else:
        print(">> VERDICT: ❌ FAIL (Trends are divergent)")

    # --- [STEP 2] SINGAPORE SUMMIT DiD (P1 vs P2) ---
    print("\n" + "="*60)
    print("2. SINGAPORE SUMMIT EFFECT (P1 -> P2)")
    print("="*60)
    
    data_p1p2 = monthly[(monthly['ts'] < '2019-02-01')].copy() # Exclude Feb 2019
    data_p1p2['post'] = (data_p1p2['ts'] >= '2018-06-01').astype(int)
    
    model_summit = smf.ols('score ~ treated * post', data=data_p1p2).fit()
    
    s_coef = model_summit.params['treated:post']
    s_pval = model_summit.pvalues['treated:post']
    
    print(model_summit.summary().tables[1])
    print(f"\n>> DiD Coefficient: {s_coef:.4f} (p={s_pval:.4f})")
    
    # --- [STEP 2.5] PARALLEL TRENDS VERIFICATION (P2 for Ratchet Effect) ---
    print("\n" + "="*60)
    print("2.5. PARALLEL TRENDS [P2] (Verify Baseline for Ratchet Effect)")
    print("="*60)
    
    # P2 period: 2018-06-12 to 2019-02-01 (excluding Feb 2019 buffer)
    data_p2_solo = monthly[(monthly['ts'] >= '2018-06-12') & (monthly['ts'] < '2019-02-01')].copy()
    
    if len(data_p2_solo) > 3:
        model_pt2 = smf.ols('score ~ month_num * treated', data=data_p2_solo).fit()
        pt2_pval = model_pt2.pvalues['month_num:treated']
        print(model_pt2.summary().tables[1])
        print(f"\n>> P2 PT Interation P-Value: {pt2_pval:.4f}")
        if pt2_pval > 0.05:
            print(">> VERDICT: ✅ PASS (Trends parallel in P2)")
        else:
            print(">> VERDICT: ❌ FAIL (Trends divergent in P2)")
    else:
        print(">> SKIP: Not enough monthly data points in P2 for trend test.")

    # --- [STEP 3] RATCHET EFFECT DiD (P2 vs P3) ---
    print("\n" + "="*60)
    print("3. HANOI COLLAPSE & RATCHET EFFECT (P2 -> P3)")

    print("="*60)
    
    data_p2p3 = monthly[(monthly['ts'] >= '2018-06-01') & (monthly['ts'] != '2019-02-01')].copy() # Exclude Feb 2019
    data_p2p3['post_hanoi'] = (data_p2p3['ts'] >= '2019-03-01').astype(int)
    
    model_ratchet = smf.ols('score ~ treated * post_hanoi', data=data_p2p3).fit()
    
    r_coef = model_ratchet.params['treated:post_hanoi']
    r_pval = model_ratchet.pvalues['treated:post_hanoi']
    
    print(model_ratchet.summary().tables[1])
    print(f"\n>> DiD Coefficient: {r_coef:.4f} (p={r_pval:.4f})")
    
    print("\n" + "="*60)
    print("SUMMARY OF FINDINGS")
    print("="*60)
    if s_pval < 0.05 and s_coef > 0:
        print(f"- Summit Effect: Significant Positive Shift ({s_coef:.3f})")
    if r_pval > 0.05:
        print(f"- Ratchet Effect: CONFIRMED (No significant drop after Hanoi, p={r_pval:.3f})")
    elif r_coef < 0:
        print(f"- Ratchet Effect: Potential Reversal ({r_coef:.3f})")

if __name__ == "__main__":
    final_analysis()
