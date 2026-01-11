
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

FILES = {
    'NK': 'nk_comments_recursive_roberta_final.csv',
    'China': 'china_comments_recursive_roberta_final.csv',
    'Iran': 'iran_comments_recursive_roberta_final.csv',
    'Russia': 'russia_comments_recursive_roberta_final.csv'
}

DATA_DIR = 'data/processed'

# Periods
P1_START, P1_END = '2017-01-01', '2018-06-11'
P2_START, P2_END = '2018-06-12', '2019-02-27'
P3_START, P3_END = '2019-02-28', '2019-12-31'

def get_period(date):
    if P1_START <= str(date) <= P1_END: return 'P1'
    if P2_START <= str(date) <= P2_END: return 'P2'
    if P3_START <= str(date) <= P3_END: return 'P3'
    return None

def load_and_aggregate():
    print("Loading and Aggregating Data to Monthly Level...")
    monthly_data = []
    
    for country, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=['roberta_compound', 'created_utc'], low_memory=False)
            df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
            df.dropna(subset=['created_utc'], inplace=True)
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')
            
            # Filter Date Range
            df = df[(df['date'] >= P1_START) & (df['date'] <= P3_END)]
            
            # Add Month and Period
            df['month'] = df['date'].dt.to_period('M')
            df['period'] = df['date'].apply(get_period)
            df.dropna(subset=['period'], inplace=True)
            
            # Aggregate
            agg = df.groupby(['period', 'month'])['roberta_compound'].agg(['mean', 'count']).reset_index()
            agg['topic'] = country
            monthly_data.append(agg)
            
    return pd.concat(monthly_data)

def run_did(df, control_name):
    print(f"\n>>> DiD Analysis (Monthly Aggregated): NK vs {control_name}")
    
    # Filter
    subset = df[df['topic'].isin(['NK', control_name])].copy()
    subset['is_treated'] = (subset['topic'] == 'NK').astype(int)
    
    comparisons = [
        ('P1', 'P2', 'Singapore Summit'),
        ('P2', 'P3', 'Hanoi Collapse'),
        ('P1', 'P3', 'Long-term Effect')
    ]
    
    for pre, post, label in comparisons:
        sub_window = subset[subset['period'].isin([pre, post])].copy()
        sub_window['is_post'] = (sub_window['period'] == post).astype(int)
        
        # OLS: mean_score ~ Treated + Post + Treated*Post
        # We weigh by count? No, strictly monthly means as per "Parallel Trends" logic
        # But weighting by count is usually better. Let's do unweighted first to match visual check.
        
        mod = smf.ols("mean ~ is_treated * is_post", data=sub_window)
        res = mod.fit()
        
        did_coef = res.params['is_treated:is_post']
        p_val = res.pvalues['is_treated:is_post']
        
        sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else ""))
        
        # Get CI
        ci = res.conf_int().loc['is_treated:is_post']
        print(f"  {label} ({pre}->{post}): DiD = {did_coef:.4f} [{ci[0]:.3f}, {ci[1]:.3f}] {sig} (p={p_val:.4f})")

def test_parallel_trends(df, control_name):
    print(f"\n>>> Parallel Trends (Sentiment): NK vs {control_name}")
    subset = df[df['topic'].isin(['NK', control_name])].copy()
    subset['is_treated'] = (subset['topic'] == 'NK').astype(int)
    
    # Convert month to numerical time
    subset['ts'] = subset['month'].dt.to_timestamp()
    subset['month_num'] = (subset['ts'].dt.year - 2017) * 12 + subset['ts'].dt.month
    
    # P1 PT (Exclude Mar-May 2018 Buffer)
    p1 = subset[subset['ts'] < '2018-03-01'].copy()
    if len(p1) > 4:
        mod = smf.ols("mean ~ month_num * is_treated", data=p1)
        res = mod.fit()
        pval = res.pvalues['month_num:is_treated']
        print(f"  P1 (Pre-Summit) PT p-value: {pval:.4f}")
    
    # P2 PT (Exclude Feb 2019 Buffer)
    p2 = subset[(subset['ts'] >= '2018-06-01') & (subset['ts'] < '2019-02-01')].copy()
    if len(p2) > 3:
        mod = smf.ols("mean ~ month_num * is_treated", data=p2)
        res = mod.fit()
        pval = res.pvalues['month_num:is_treated']
        print(f"  P2 (Pre-Hanoi) PT p-value: {pval:.4f}")

def main():
    df_monthly = load_and_aggregate()
    
    for ctrl in ['China', 'Iran', 'Russia']:
        test_parallel_trends(df_monthly, ctrl)
        run_did(df_monthly, ctrl)

if __name__ == "__main__":
    main()
