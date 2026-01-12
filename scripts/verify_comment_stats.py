
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pathlib import Path

# Define dates
SINGAPORE_DATE = pd.to_datetime('2018-06-12').tz_localize('UTC')
HANOI_DATE = pd.to_datetime('2019-02-28').tz_localize('UTC')

DATA_DIR = Path('data/processed')

def assign_period(date):
    if date < SINGAPORE_DATE:
        return 'P1'
    elif date < HANOI_DATE:
        return 'P2'
    else:
        return 'P3'

def load_and_prep(filepath, group_name):
    try:
        # read_csv handles named columns. If there are malformed rows, we might need on_bad_lines='skip'
        # But for newlines in bodies, standard parser usually works if quotes are correct.
        df = pd.read_csv(filepath, on_bad_lines='skip', lineterminator='\n')
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

    initial_len = len(df)
    
    # Coerce created_utc to numeric, forcing errors to NaN
    df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
    
    # Drop rows with invalid created_utc
    df = df.dropna(subset=['created_utc'])
    
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"[{group_name}] Dropped {dropped} rows with invalid created_utc (parse error/misalignment).")

    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)
    df['period'] = df['created_utc'].apply(assign_period)
    df['group'] = group_name
    df['treated'] = 1 if group_name == 'NK' else 0
    return df[['period', 'treated', 'group', 'roberta_compound']]

def run_did_analysis():
    print("Loading data...")
    nk_df = load_and_prep(DATA_DIR / 'nk_comments_recursive_roberta_final.csv', 'NK')
    china_df = load_and_prep(DATA_DIR / 'china_comments_recursive_roberta_final.csv', 'China')
    iran_df = load_and_prep(DATA_DIR / 'iran_comments_recursive_roberta_final.csv', 'Iran')
    russia_df = load_and_prep(DATA_DIR / 'russia_comments_recursive_roberta_final.csv', 'Russia')
    
    controls = {'China': china_df, 'Iran': iran_df, 'Russia': russia_df}
    
    results = []
    
    print(f"{'Control':<10} | {'Term':<15} | {'Coef':<8} | {'P-value':<8} | {'CI Lower':<8} | {'CI Upper':<8}")
    print("-" * 75)

    for name, control_df in controls.items():
        combined = pd.concat([nk_df, control_df])
        
        # Create dummy variables
        combined['is_P2'] = (combined['period'] == 'P2').astype(int)
        combined['is_P3'] = (combined['period'] == 'P3').astype(int)
        
        # Define model: Sentiment ~ Treated + P2 + P3 + Treated*P2 + Treated*P3
        # Treated*P3 is the Net Effect (P1 -> P3) relative to control
        model = smf.ols('roberta_compound ~ treated + is_P2 + is_P3 + treated:is_P2 + treated:is_P3', data=combined)
        fit = model.fit()
        
        # Extract Treated:P3 (Net Effect)
        p3_coef = fit.params['treated:is_P3']
        p3_pval = fit.pvalues['treated:is_P3']
        p3_ci = fit.conf_int().loc['treated:is_P3']
        
        # Extract Treated:P2 (Summit Effect) for comparison
        p2_coef = fit.params['treated:is_P2']
        p2_pval = fit.pvalues['treated:is_P2']
        
        print(f"{name:<10} | {'Treated:P2':<15} | {p2_coef:.4f}   | {p2_pval:.4f}   | ...      | ...")
        print(f"{name:<10} | {'Treated:P3':<15} | {p3_coef:.4f}   | {p3_pval:.4f}   | {p3_ci[0]:.4f}   | {p3_ci[1]:.4f}")
        
        # Calculate reversion percentage (if P2 is positive)
        if p2_coef > 0:
            reversion = 1 - (p3_coef / p2_coef)
            print(f"{' ':<10} | {'Reversion %':<15} | {reversion*100:.1f}%")
        print("-" * 75)

    print("\n" + "="*30 + " Asymmetric Reversal Analysis (Ratio) " + "="*30)
    print(f"{'Control':<10} | {'|P1->P2|':<10} | {'|P2->P3|':<10} | {'Ratio':<10} | {'Interpretation':<20}")
    print("-" * 75)

    for name in controls.keys():
        # Retrieve stored coefficients
        # Note: We need to capture these from the loop above or re-run. 
        # For simplicity, let's re-run deeply or just store them in a dict.
        pass # Placeholder

    # Refactored loop to store results
    results = {}
    
    # ... (Re-running logic for clarity)
    for name, control_df in controls.items():
        # 1. P1 -> P2 (Singapore Effect) - using P1/P2 data only
        p1p2_df = pd.concat([nk_df, control_df])
        p1p2_df = p1p2_df[p1p2_df['period'].isin(['P1', 'P2'])].copy()
        p1p2_df['is_P2'] = (p1p2_df['period'] == 'P2').astype(int)
        
        model_p1p2 = smf.ols('roberta_compound ~ treated + is_P2 + treated:is_P2', data=p1p2_df)
        fit_p1p2 = model_p1p2.fit()
        coef_p1p2 = fit_p1p2.params['treated:is_P2']
        
        # 2. P2 -> P3 (Hanoi Reversal) - using P2/P3 data only
        p2p3_df = pd.concat([nk_df, control_df])
        p2p3_df = p2p3_df[p2p3_df['period'].isin(['P2', 'P3'])].copy()
        p2p3_df['is_P3'] = (p2p3_df['period'] == 'P3').astype(int)
        
        model_p2p3 = smf.ols('roberta_compound ~ treated + is_P3 + treated:is_P3', data=p2p3_df)
        fit_p2p3 = model_p2p3.fit()
        coef_p2p3 = fit_p2p3.params['treated:is_P3']
        pval_p2p3 = fit_p2p3.pvalues['treated:is_P3']
        
        ratio = abs(coef_p2p3) / abs(coef_p1p2)
        
        interp = "Full Reversion" if ratio > 0.8 else ("Partial Reversion" if ratio > 0.2 else "Ratchet (No Reversion)")
        if pval_p2p3 > 0.05:
            interp = "Ratchet (No Sig. Reversion)"
            
        print(f"{name:<10} | {abs(coef_p1p2):.4f}     | {abs(coef_p2p3):.4f}     | {ratio:.4f}     | {interp}")

if __name__ == "__main__":
    run_did_analysis()
