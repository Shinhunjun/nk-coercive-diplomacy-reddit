
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# SCORING CONFIG
SCALE_MAP = {
    'THREAT': -2,
    'DIPLOMACY': 2,
    'ECONOMIC': 0,
    'NEUTRAL': 0,
    'HUMANITARIAN': 0
}

def verify_iran_trends():
    print("Loading Data for NK vs Iran Parallel Trends Check...")
    
    # 1. Load NK P1
    nk_p1 = pd.read_csv('data/processed/nk_p1_framing_results.csv')
    nk_dates = pd.read_csv('data/processed/nk_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    nk_p1['id'] = nk_p1['id'].astype(str)
    nk_dates['id'] = nk_dates['id'].astype(str)
    nk = nk_p1.merge(nk_dates, on='id')
    
    # 2. Load Iran (Partial)
    iran_frames = pd.read_csv('data/processed/iran_framing_results.csv')
    iran_orig = pd.read_csv('data/processed/iran_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'], low_memory=False)
    iran_frames['id'] = iran_frames['id'].astype(str)
    iran_orig['id'] = iran_orig['id'].astype(str)
    iran_orig['created_utc'] = pd.to_numeric(iran_orig['created_utc'], errors='coerce')
    iran_orig = iran_orig.dropna(subset=['created_utc'])
    iran = iran_frames.merge(iran_orig, on='id')
    
    # 3. Apply Scale & Treatment
    nk['score'] = nk['frame'].map(SCALE_MAP).fillna(0)
    iran['score'] = iran['frame'].map(SCALE_MAP).fillna(0)
    nk['treated'] = 1
    iran['treated'] = 0
    
    # 4. Filter for P1 (< 2018-06-12)
    df = pd.concat([nk, iran])
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    df_p1 = df[df['date'] < '2018-06-12'].copy()
    
    print(f"Total P1 Records found: {len(df_p1)}")
    print(f"  - NK (P1): {len(df_p1[df_p1['treated']==1])}")
    print(f"  - Iran (P1): {len(df_p1[df_p1['treated']==0])}")
    
    # 5. Monthly Aggregation
    df_p1['month'] = df_p1['date'].dt.to_period('M')
    monthly = df_p1.groupby(['month', 'treated'])['score'].mean().reset_index()
    monthly['month_num'] = (monthly['month'].dt.year - 2017) * 12 + (monthly['month'].dt.month)
    
    # 6. Parallel Trends Test
    print("\n" + "="*50)
    print("PARALLEL TRENDS VERIFICATION: NK vs IRAN")
    print("="*50)
    model = smf.ols('score ~ month_num * treated', data=monthly).fit()
    print(model.summary().tables[1])
    
    p_val = model.pvalues['month_num:treated']
    print(f"\nInteraction P-Value: {p_val:.5f}")
    
    if p_val > 0.1:
        print("✅ PASS: NK and Iran had parallel framing trends in P1.")
    else:
        print("❌ FAIL: Trends are divergent.")

if __name__ == "__main__":
    verify_iran_trends()
