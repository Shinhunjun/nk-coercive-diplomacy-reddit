
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os

# Configuration
DATA_DIR = 'data/processed'
FIGURES_DIR = 'paper/figures'

FILES = {
    'NK': 'nk_comments_recursive_roberta_final.csv',
    'China': 'china_comments_recursive_roberta_final.csv',
    'Iran': 'iran_comments_recursive_roberta_final.csv',
    'Russia': 'russia_comments_recursive_roberta_final.csv'
}

TREATMENT_MONTH = '2018-06'

def load_data():
    dfs = []
    for country, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path, usecols=['roberta_compound', 'created_utc'], low_memory=False)
                df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
                df = df.dropna(subset=['created_utc'])
                df['date'] = pd.to_datetime(df['created_utc'], unit='s')
                df['month'] = df['date'].dt.to_period('M')
                df['topic'] = country
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {country}: {e}")
    return pd.concat(dfs)

def verify_pooled_trends(df):
    print("\n--- Event Study: NK vs POOLED CONTROL (China+Iran+Russia) ---")
    
    # Define Treated and Control
    df['is_treated'] = (df['topic'] == 'NK').astype(int)
    
    # Create relative time
    df['month_dt'] = df['month'].dt.to_timestamp()
    treatment_start_dt = pd.Timestamp(TREATMENT_MONTH + '-01')
    
    df['rel_month'] = ((df['month_dt'].dt.year - treatment_start_dt.year) * 12 + 
                      (df['month_dt'].dt.month - treatment_start_dt.month))
    
    # Filter Window: P1 (-17) to P2 start
    subset = df[(df['rel_month'] >= -17) & (df['rel_month'] <= 0)].copy()
    
    # Set Reference = -1
    subset['rel_month_cat'] = subset['rel_month'].astype('category')
    categories = sorted(subset['rel_month_cat'].unique())
    if -1 in categories:
        categories.remove(-1)
        categories = [-1] + categories
        subset['rel_month_cat'] = subset['rel_month_cat'].cat.reorder_categories(categories, ordered=True)
    
    # Run Regression
    formula = "roberta_compound ~ is_treated * C(rel_month_cat)"
    mod = smf.ols(formula, data=subset)
    res = mod.fit(cov_type='HC1')
    
    # Check Pre-trend coefficients (excluding reference -1)
    failures = 0
    total_checks = 0
    
    print("\nPre-trend Coefficients (should be close to 0):")
    for i in range(-17, 0): # Pre-period only
        if i == -1: continue
        term = f"is_treated:C(rel_month_cat)[T.{i}]"
        try:
            coef = res.params[term]
            p_val = res.pvalues[term]
            sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
            print(f"  Month {i}: {coef:.4f} {sig} (p={p_val:.4f})")
            
            if p_val < 0.05:
                failures += 1
            total_checks += 1
        except KeyError:
            pass
            
    print(f"\nResult: {failures} significant deviations out of {total_checks} months.")
    if failures <= 2: # Allow small tolerance
        print(">> POOLED CONTROL PASSES Parallel Trends Test.")
    else:
        print(">> POOLED CONTROL FAILS Parallel Trends Test.")

if __name__ == "__main__":
    df = load_data()
    verify_pooled_trends(df)
