
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

# Treatment Start: Singapore Summit June 2018
TREATMENT_MONTH = '2018-06'

def load_data():
    dfs = []
    for country, filename in FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            print(f"Loading {country} from {path}...")
            try:
                # Load with robust parsing
                # NEW: Load 'body' to filter removed/deleted
                df = pd.read_csv(path, usecols=['roberta_compound', 'created_utc', 'body'], on_bad_lines='skip', low_memory=False)
                
                # Filter removed/deleted
                initial_len = len(df)
                df = df[~df['body'].isin(['[removed]', '[deleted]'])]
                df = df.dropna(subset=['body'])
                print(f"  - Filtered {initial_len - len(df)} removed/deleted comments (Remaining: {len(df)})")
                
                df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
                df = df.dropna(subset=['created_utc'])
                
                df['date'] = pd.to_datetime(df['created_utc'], unit='s')
                df['month'] = df['date'].dt.to_period('M')
                df['topic'] = country
                df['is_treated'] = 1 if country == 'NK' else 0
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {country}: {e}")
    return pd.concat(dfs)

def run_event_study(df, control_name):
    print(f"\n--- Event Study: NK vs {control_name} ---")
    
    # Filter for just NK and current Control
    subset = df[df['topic'].isin(['NK', control_name])].copy()
    
    # Create relative time (months from treatment)
    # Treatment start: 2018-06 (Index 0)
    subset['month_dt'] = subset['month'].dt.to_timestamp()
    treatment_start_dt = pd.Timestamp(TREATMENT_MONTH + '-01')
    
    # Calculate difference in months
    subset['rel_month'] = ((subset['month_dt'].dt.year - treatment_start_dt.year) * 12 + 
                          (subset['month_dt'].dt.month - treatment_start_dt.month))
    
    # Limit window to relevant period (e.g., -12 to +12 months)
    # P1 starts roughly Jan 2017 (-17 months)
    subset = subset[(subset['rel_month'] >= -17) & (subset['rel_month'] <= 18)]
    
    # Set reference month to -1 (May 2018) by reordering categories
    subset['rel_month_cat'] = subset['rel_month'].astype('category')
    categories = sorted(subset['rel_month_cat'].unique())
    # Move -1 to front
    if -1 in categories:
        categories.remove(-1)
        categories = [-1] + categories
        subset['rel_month_cat'] = subset['rel_month_cat'].cat.reorder_categories(categories, ordered=True)
    
    print("Running Regression (this may take a moment)...")
    # Formula uses the ordered categorical directly
    formula = "roberta_compound ~ is_treated * C(rel_month_cat)"
    
    mod = smf.ols(formula, data=subset)
    res = mod.fit(cov_type='HC1') # Robust SE
    
    # Extract coefficients
    coefs = []
    cis = []
    months = []
    
    # Get sorted months for plotting
    plot_months = sorted(subset['rel_month'].unique())
    
    for i in plot_months:
        if i == -1:
            coefs.append(0)
            cis.append(0)
            months.append(i)
            continue
            
        term = f"is_treated:C(rel_month_cat)[T.{i}]"
        try:
            coef = res.params[term]
            ci = res.conf_int().loc[term]
            err = coef - ci[0] # roughly symmetric
            
            coefs.append(coef)
            cis.append(err)
            months.append(i)
        except KeyError:
            # Maybe dropped or ref?
            pass
            
    # Visualize
    plt.figure(figsize=(12, 6))
    plt.errorbar(months, coefs, yerr=cis, fmt='-o', color='b', ecolor='gray', capsize=5)
    plt.axvline(x=-0.5, color='r', linestyle='--', label='Singapore Summit')
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    
    # Labels
    plt.title(f'Event Study (Leads & Lags): North Korea vs {control_name}\n(Parallel Trends Validation)', fontsize=14)
    plt.xlabel('Months relative to Singapore Summit (Jun 2018)', fontsize=12)
    plt.ylabel('Treatment Effect Estimate (Diff-in-Diff coeff)', fontsize=12)
    
    # Shade Pre-Period check
    # Check if pre-period CIs overlap with 0
    pre_period_indices = [i for i, m in enumerate(months) if m < -1]
    pre_period_coefs = [coefs[i] for i in pre_period_indices]
    pre_period_cis = [cis[i] for i in pre_period_indices]
    
    violations = 0
    for c, err in zip(pre_period_coefs, pre_period_cis):
        if abs(c) > err: # Not crossing zero
            violations += 1
            
    status = "PASSED" if violations <= 2 else f"WARNING ({violations} significant pre-trends)"
    print(f"Parallel Trends Status: {status}")
    
    if violations == 0:
        plt.text(-10, max(coefs), "Strong Parallel Trends\n(Pre-period effect ~ 0)", color='green', fontweight='bold')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    out_path = os.path.join(FIGURES_DIR, f'event_study_{control_name.lower()}.pdf')
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

def main():
    print("Loading Dataset...")
    df = load_data()
    
    print("\nRunning Event Studies...")
    # Run for China (Best Control)
    run_event_study(df, 'China')
    
    # Run for Iran (Secondary)
    run_event_study(df, 'Iran')
    
    # Run for Russia
    run_event_study(df, 'Russia')

if __name__ == "__main__":
    main()
