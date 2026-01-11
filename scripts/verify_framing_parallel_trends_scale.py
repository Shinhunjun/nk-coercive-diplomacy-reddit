
import pandas as pd
import numpy as np
import statsmodels.api as sm
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

def verify_trends():
    print("Loading Data...")
    
    # 1. Load Data
    nk_p1 = pd.read_csv('data/processed/nk_p1_framing_results.csv')
    china_partial = pd.read_csv('data/processed/china_framing_results.csv')
    
    # Load Dates
    nk_dates = pd.read_csv('data/processed/nk_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    china_dates = pd.read_csv('data/processed/china_comments_recursive_roberta_final.csv', usecols=['id', 'created_utc'])
    
    # Merge
    nk_p1['id'] = nk_p1['id'].astype(str)
    china_partial['id'] = china_partial['id'].astype(str)
    nk_dates['id'] = nk_dates['id'].astype(str)
    china_dates['id'] = china_dates['id'].astype(str)
    
    nk = nk_p1.merge(nk_dates, on='id')
    china = china_partial.merge(china_dates, on='id')
    
    # 2. Apply Scale
    nk['score'] = nk['frame'].map(SCALE_MAP).fillna(0)
    china['score'] = china['frame'].map(SCALE_MAP).fillna(0)
    
    nk['treated'] = 1
    china['treated'] = 0
    
    # 3. Combine & Filter for P1
    df = pd.concat([nk, china])
    df['date'] = pd.to_datetime(df['created_utc'], unit='s')
    
    # P1 Definition: Pre-Singapore (Before June 12, 2018)
    p1_end_date = '2018-06-12'
    df_p1 = df[df['date'] < p1_end_date].copy()
    
    print(f"Total P1 Records: {len(df_p1)}")
    print(f"  - NK (P1): {len(df_p1[df_p1['treated']==1])}")
    print(f"  - China (P1): {len(df_p1[df_p1['treated']==0])}")
    
    # 4. Aggregation
    df_p1['month'] = df_p1['date'].dt.to_period('M')
    monthly = df_p1.groupby(['month', 'treated'])['score'].mean().reset_index()
    monthly['month_num'] = monthly['month'].astype(int)
    
    # Sort for viewing
    monthly = monthly.sort_values(['treated', 'month'])
    
    print("\n--- Monthly Frame Scores (Scale: -2=War, +2=Peace) ---")
    print(monthly.pivot(index='month', columns='treated', values='score'))
    
    # 5. Parallel Trends Test (OLS)
    # Model: Score ~ Month + Treated + Month*Treated
    X = monthly[['month_num', 'treated']]
    X['interaction'] = X['month_num'] * X['treated']
    X = sm.add_constant(X)
    y = monthly['score']
    
    model = sm.OLS(y, X).fit()
    
    print("\n" + "="*50)
    print("PARALLEL TRENDS VERIFICATION (Pre-Summit)")
    print("="*50)
    print(model.summary().tables[1])
    
    p_val = model.pvalues['interaction']
    coef = model.params['interaction']
    
    print("\n--- CONCLUSION ---")
    print(f"Interaction Coefficient: {coef:.5f}")
    print(f"P-value: {p_val:.5f}")
    
    if p_val > 0.1:
        print("✅ PASS: Parallel Trends Assumption HOLDS (p > 0.1)")
        print("Interpertation: NK and China were moving on similar trends before the Summit.")
    else:
        print("❌ FAIL: Trends are Diverging (p < 0.1)")
        print("Interpretation: Pre-existing trends were different.")

if __name__ == "__main__":
    verify_trends()
