
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Configuration
DATA_DIR = 'data/processed'

FILES = {
    'NK': 'nk_comments_recursive_roberta_final.csv',
    'China': 'china_comments_recursive_roberta_final.csv',
    'Iran': 'iran_comments_recursive_roberta_final.csv',
    'Russia': 'russia_comments_recursive_roberta_final.csv'
}

TREATMENT_MONTH = '2018-06'

def load_data():
    dfs = {}
    for country, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            print(f"Loading {country} from {path}...")
            try:
                df = pd.read_csv(path, usecols=['roberta_compound', 'created_utc'], low_memory=False)
                df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
                df = df.dropna(subset=['created_utc'])
                df['date'] = pd.to_datetime(df['created_utc'], unit='s')
                df['month'] = df['date'].dt.to_period('M')
                
                # Filter P1 only (Pre-Singapore: Up to 2018-05)
                # Singapore summit is June 2018
                df = df[df['month'] < '2018-06']
                
                # Filter crazy early data if any (e.g. before 2017) to match previous scope?
                # Previous P1 started 2017-01
                df = df[df['month'] >= '2017-01']
                
                df['topic'] = country
                dfs[country] = df
            except Exception as e:
                print(f"Error loading {country}: {e}")
    return dfs

def run_linear_test(nk_df, control_df, control_name):
    print(f"\n--- Linear Trend Test (Monthly Aggregated): NK vs {control_name} ---")
    
    # Prepare Data
    t = nk_df.copy()
    t['is_treat'] = 1
    
    c = control_df.copy()
    c['is_treat'] = 0
    
    combined = pd.concat([t, c], ignore_index=True)
    
    # AGGREGATE TO MONTHLY MEANS
    # Group by is_treat and month
    combined_agg = combined.groupby(['is_treat', 'month'])['roberta_compound'].mean().reset_index()
    
    # Create Linear Time Trend (0, 1, 2, ...)
    months = sorted(combined_agg['month'].unique())
    month_map = {m: i for i, m in enumerate(months)}
    combined_agg['time_trend'] = combined_agg['month'].map(month_map)
    
    # Interaction Term
    combined_agg['treat_time'] = combined_agg['is_treat'] * combined_agg['time_trend']
    
    print(f"  Data Points (Months x Groups): {len(combined_agg)}")
    
    # Run OLS on Aggregated Data
    # score_mean ~ is_treat + time_trend + treat_time
    try:
        mod = smf.ols('roberta_compound ~ is_treat + time_trend + treat_time', data=combined_agg)
        
        # Newey-West HAC errors might be better for time series, 
        # but sticking to simple HC1 or standard OLS for replication of simple linear trend check
        # Previous script on aggregated data likely just used OLS or HC1.
        res = mod.fit() 
        
        # Check interaction coefficient
        coef = res.params['treat_time']
        pval = res.pvalues['treat_time']
        
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.1 else ""))
        verdict = "PASS" if pval > 0.05 else "FAIL"
        
        print(f"  Interaction Coef (Slope Diff): {coef:.4f}")
        print(f"  P-value: {pval:.4f} {sig}")
        print(f"  Result: {verdict}")
        
    except Exception as e:
        print(f"Error running regression: {e}")

def main():
    dfs = load_data()
    nk = dfs['NK']
    
    for country in ['China', 'Iran', 'Russia']:
        if country in dfs:
            run_linear_test(nk, dfs[country], country)

if __name__ == "__main__":
    main()
