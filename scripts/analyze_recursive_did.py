
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Configuration
DATA_DIR = 'data/processed'
RESULTS_DIR = 'data/results'
FIGURES_DIR = 'paper/figures'

FILES = {
    'NK': 'nk_comments_recursive_roberta_final.csv',
    'China': 'china_comments_recursive_roberta_final.csv',
    'Iran': 'iran_comments_recursive_roberta_final.csv',
    'Russia': 'russia_comments_recursive_roberta_final.csv'
}

PERIODS = {
    'P1': {'start': '2017-01-01', 'end': '2018-06-11'}, # Pre-Singapore
    'P2': {'start': '2018-06-12', 'end': '2019-02-27'}, # Singapore to Hanoi
    'P3': {'start': '2019-02-28', 'end': '2019-12-31'}  # Post-Hanoi
}

def load_data():
    dfs = {}
    for country, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            print(f"Loading {country} from {path}...")
            # Use strict types or handle bad lines?
            # Better to load, coerce, and drop bad rows
            try:
                # NEW: Load 'body' to filter removed/deleted
                df = pd.read_csv(path, usecols=['roberta_compound', 'created_utc', 'parent_post_id', 'body'], on_bad_lines='skip', low_memory=False)
                
                # Filter removed/deleted
                initial_len = len(df)
                df = df[~df['body'].isin(['[removed]', '[deleted]'])]
                # Also filter cases where body is NaN
                df = df.dropna(subset=['body'])
                print(f"  - Filtered {initial_len - len(df)} removed/deleted comments (Remaining: {len(df)})")
                
                # Coerce created_utc
                df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
                df = df.dropna(subset=['created_utc'])
                
                # Convert date
                df['date'] = pd.to_datetime(df['created_utc'], unit='s')
                df['topic'] = country
                dfs[country] = df
            except Exception as e:
                print(f"Error loading {country}: {e}")
        else:
            print(f"Warning: {path} not found.")
    return dfs

def assign_period(df):
    # Vectorized period assignment
    conditions = [
        (df['date'] >= PERIODS['P1']['start']) & (df['date'] < PERIODS['P2']['start']),
        (df['date'] >= PERIODS['P2']['start']) & (df['date'] < PERIODS['P3']['start']),
        (df['date'] >= PERIODS['P3']['start']) & (df['date'] <= PERIODS['P3']['end'])
    ]
    choices = ['P1', 'P2', 'P3']
    df['period'] = np.select(conditions, choices, default='Exclude')
    return df[df['period'] != 'Exclude']

def verify_parallel_trends(dfs):
    print("\n" + "="*60)
    print("VERIFYING PARALLEL TRENDS (VISUAL + STATISTICAL)")
    print("="*60)
    
    # 1. Aggregate to Monthly for Visualization
    all_data = []
    for country, df in dfs.items():
        df_p1 = df[df['period'] == 'P1'].copy()
        df_p1['month'] = df_p1['date'].dt.to_period('M')
        monthly = df_p1.groupby('month')['roberta_compound'].mean().reset_index()
        monthly['topic'] = country
        monthly['month_str'] = monthly['month'].astype(str)
        all_data.append(monthly)
        
    viz_df = pd.concat(all_data)
    
    # Plot
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.figure(figsize=(12, 6))
    
    # Plot lines
    for country in dfs.keys(): # NK and controls
        subset = viz_df[viz_df['topic'] == country]
        plt.plot(subset['month_str'], subset['roberta_compound'], marker='o', label=country)
        
    plt.title('Parallel Trends Verification: Monthly Sentiment (P1 Pre-Singapore)')
    plt.xlabel('Month')
    plt.ylabel('Average Sentiment (Compound)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'recursive_parallel_trends_p1.pdf'))
    print(f"saved parallel trends plot to {FIGURES_DIR}/recursive_parallel_trends_p1.pdf")
    
    # 2. Statistical Placebo Test (Interaction in Pre-Period)
    # Test if slope of NK differs from Control in P1
    # Model: Sentiment ~ Time(Month_Num) * Is_NK
    
    print("\n--- Statistical Parallel Trends Test (Slope Difference in P1) ---")
    results = []
    
    nk_p1 = dfs['NK'][dfs['NK']['period'] == 'P1'].copy()
    nk_p1['month_num'] = nk_p1['date'].dt.year * 12 + nk_p1['date'].dt.month
    nk_p1['is_treated'] = 1
    
    for country in ['China', 'Iran', 'Russia']:
        if country not in dfs: continue
        
        control_p1 = dfs[country][dfs[country]['period'] == 'P1'].copy()
        control_p1['month_num'] = control_p1['date'].dt.year * 12 + control_p1['date'].dt.month
        control_p1['is_treated'] = 0
        
        # Combine
        combined = pd.concat([nk_p1, control_p1])
        
        # Linear Trend Interaction Model
        # score ~ month_num + is_treated + month_num*is_treated
        mod = smf.ols("roberta_compound ~ month_num * is_treated", data=combined)
        res = mod.fit()
        
        interaction_pval = res.pvalues['month_num:is_treated']
        
        status = "PASS" if interaction_pval > 0.05 else "WARN (Trends Diverge)"
        print(f"NK vs {country}: Interaction p-value = {interaction_pval:.4f} [{status}]")
        results.append({'Control': country, 'P_Value': interaction_pval, 'Status': status})

def calculate_did(dfs):
    print("\n" + "="*60)
    print("ESTIMATING DID EFFECTS (BY PERIOD)")
    print("="*60)
    
    comparisons = [
        ('P1', 'P2', 'Singapore Summit Effect'),
        ('P2', 'P3', 'Hanoi Collapse Effect'),
        ('P1', 'P3', 'Long-term Effect')
    ]
    
    did_results = []
    
    nk_df = dfs['NK']
    
    for country in ['China', 'Iran', 'Russia']:
        ctrl_df = dfs[country]
        
        print(f"\n>>> Control Group: {country}")
        
        for p_pre, p_post, label in comparisons:
            # Filter Data
            t_pre = nk_df[nk_df['period'] == p_pre]['roberta_compound']
            t_post = nk_df[nk_df['period'] == p_post]['roberta_compound']
            c_pre = ctrl_df[ctrl_df['period'] == p_pre]['roberta_compound']
            c_post = ctrl_df[ctrl_df['period'] == p_post]['roberta_compound']
            
            if len(t_pre) == 0 or len(c_pre) == 0:
                print(f"  Skipping {label}: Missing data for {p_pre}")
                continue
                
            # Means
            mu_t_pre, mu_t_post = t_pre.mean(), t_post.mean()
            mu_c_pre, mu_c_post = c_pre.mean(), c_post.mean()
            
            # DiD
            did_coeff = (mu_t_post - mu_t_pre) - (mu_c_post - mu_c_pre)
            
            # Significance (Welch's t-test equivalent for DiD? Or Regression)
            # Regression is cleaner: Score ~ Treat + Post + Treat*Post
            
            # Convert to regression format for robust SE
            # Data setup
            df_reg_t = pd.DataFrame({'score': pd.concat([t_pre, t_post]), 'treat': 1, 'post': [0]*len(t_pre) + [1]*len(t_post)})
            df_reg_c = pd.DataFrame({'score': pd.concat([c_pre, c_post]), 'treat': 0, 'post': [0]*len(c_pre) + [1]*len(c_post)})
            df_reg = pd.concat([df_reg_t, df_reg_c])
            
            mod = smf.ols("score ~ treat * post", data=df_reg)
            res = mod.fit(cov_type='HC1') # Robust SE
            
            p_val = res.pvalues['treat:post']
            se = res.bse['treat:post']
            
            stars = "*" if p_val < 0.1 else ""
            stars = "**" if p_val < 0.05 else stars
            stars = "***" if p_val < 0.01 else stars
            
            print(f"  {label} ({p_pre}->{p_post}): DiD = {did_coeff:.4f} {stars} (p={p_val:.4f})")
            
            did_results.append({
                'Control': country,
                'Comparison': label,
                'Period_Pre': p_pre,
                'Period_Post': p_post,
                'DiD_Estimate': did_coeff,
                'SE': se,
                'P_Value': p_val,
                'Treat_Pre_Mean': mu_t_pre,
                'Treat_Post_Mean': mu_t_post,
                'Ctrl_Pre_Mean': mu_c_pre,
                'Ctrl_Post_Mean': mu_c_post
            })
            
    # Save Results
    res_df = pd.DataFrame(did_results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'recursive_did_results.csv'), index=False)
    print(f"\nSaved DiD results to {RESULTS_DIR}/recursive_did_results.csv")

def main():
    dfs = load_data()
    
    # Assign periods
    processed_dfs = {}
    for country, df in dfs.items():
        processed_dfs[country] = assign_period(df)
        print(f"{country}: {len(processed_dfs[country])} comments in analysis periods")
        
    # 1. Parallel Trends
    verify_parallel_trends(processed_dfs)
    
    # 2. DiD Estimation
    calculate_did(processed_dfs)

if __name__ == "__main__":
    main()
