"""
Framing DiD Analysis with Clean Periods (Transition Excluded, through 2019-12)
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import json
import os

# Load monthly framing data
nk = pd.read_csv('data/framing/nk_monthly_framing.csv')
china = pd.read_csv('data/framing/china_monthly_framing.csv')
iran = pd.read_csv('data/framing/iran_monthly_framing.csv')

print(f"NK: {len(nk)} months, China: {len(china)} months, Iran: {len(iran)} months")

# Period definitions (with transition exclusion, through 2019-12)
def assign_period(month):
    if month <= '2018-02':  # Pre-Singapore
        return 'P1'
    elif month >= '2018-03' and month <= '2018-05':  # Transition
        return None
    elif month >= '2018-06' and month <= '2019-01':  # Singapore-Hanoi
        return 'P2'
    elif month == '2019-02':  # Hanoi month - exclude
        return None
    elif month >= '2019-03' and month <= '2019-12':  # Post-Hanoi
        return 'P3'
    return None

for df in [nk, china, iran]:
    df['period'] = df['month'].apply(assign_period)

# Filter out transition periods
nk = nk.dropna(subset=['period'])
china = china.dropna(subset=['period'])
iran = iran.dropna(subset=['period'])

print(f"\nNK periods (transition excluded): {nk.groupby('period').size().to_dict()}")
print(f"China periods: {china.groupby('period').size().to_dict()}")
print(f"Iran periods: {iran.groupby('period').size().to_dict()}")

def run_ols_did(df_nk, df_ctrl, ctrl_name, period1, period2, event_name):
    """Run OLS-based DID with clustered standard errors"""
    nk_sub = df_nk[df_nk['period'].isin([period1, period2])].copy()
    ctrl_sub = df_ctrl[df_ctrl['period'].isin([period1, period2])].copy()
    
    nk_sub['treat'] = 1
    ctrl_sub['treat'] = 0
    
    combined = pd.concat([nk_sub, ctrl_sub], ignore_index=True)
    combined['post'] = (combined['period'] == period2).astype(int)
    combined['treat_post'] = combined['treat'] * combined['post']
    
    # Run OLS with clustered SE
    model = smf.ols('framing_mean ~ treat + post + treat_post', data=combined).fit(
        cov_type='cluster', cov_kwds={'groups': combined['month']}
    )
    
    did = model.params['treat_post']
    se = model.bse['treat_post']
    p_val = model.pvalues['treat_post']
    ci_lower = did - 1.96 * se
    ci_upper = did + 1.96 * se
    
    sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
    print(f'{event_name} - {ctrl_name}: DiD={did:+.2f}, 95% CI=[{ci_lower:.2f}, {ci_upper:.2f}], p={p_val:.4f} {sig}')
    return {
        'event': event_name, 
        'control': ctrl_name, 
        'did_estimate': round(did, 2), 
        'ci_lower': round(ci_lower, 2), 
        'ci_upper': round(ci_upper, 2), 
        'p_value': round(p_val, 4), 
        'significant': sig
    }

print('\n' + '='*70)
print('FRAMING DiD - OLS REGRESSION (Clean Periods, 2019-12)')
print('='*70)

results = []
print('\nSingapore Effect (P1 -> P2):')
results.append(run_ols_did(nk, china, 'China', 'P1', 'P2', 'Singapore'))
results.append(run_ols_did(nk, iran, 'Iran', 'P1', 'P2', 'Singapore'))

print('\nHanoi Effect (P2 -> P3):')
results.append(run_ols_did(nk, china, 'China', 'P2', 'P3', 'Hanoi'))
results.append(run_ols_did(nk, iran, 'Iran', 'P2', 'P3', 'Hanoi'))

# Save results
os.makedirs('data/results', exist_ok=True)
with open('data/results/framing_did_clean_final.json', 'w') as f:
    json.dump(results, f, indent=2)
pd.DataFrame(results).to_csv('data/results/framing_did_clean_final.csv', index=False)
print('\n✓ Saved to data/results/framing_did_clean_final.json and .csv')
