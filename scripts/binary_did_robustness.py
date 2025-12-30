"""
Binary DID Robustness Check for Framing Analysis

This script runs DID analysis using binary framing indicators (THREAT=1/0, DIPLOMACY=1/0)
instead of the continuous diplomacy_scale (-2 to +2).

Purpose: Validate that the main results are robust to the choice of outcome scaling.
If binary DID shows the same direction of effects, it confirms that results are not
artifacts of the arbitrary -2/+2 scale choice.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from config import DATA_DIR, RESULTS_DIR


def load_and_combine_framing_data(topic: str) -> pd.DataFrame:
    """Load framing data and combine main + extended datasets."""
    main_file = DATA_DIR / 'framing' / f'{topic}_posts_framed.csv'
    extended_file = DATA_DIR / 'framing' / f'{topic}_posts_hanoi_extended_framed.csv'
    
    print(f"Loading {topic}...")
    
    dfs = []
    if main_file.exists():
        df_main = pd.read_csv(main_file)
        print(f"  Main: {len(df_main)} posts")
        dfs.append(df_main)
    
    if extended_file.exists():
        df_ext = pd.read_csv(extended_file)
        print(f"  Extended: {len(df_ext)} posts")
        dfs.append(df_ext)
    
    if not dfs:
        raise FileNotFoundError(f"No data found for {topic}")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Parse datetime
    df['created_utc'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['created_utc'].dt.to_period('M').astype(str)
    
    # Create binary indicators
    df['is_threat'] = (df['frame'] == 'THREAT').astype(int)
    df['is_diplomacy'] = (df['frame'] == 'DIPLOMACY').astype(int)
    
    print(f"  Total: {len(df)} posts, Date range: {df['created_utc'].min()} to {df['created_utc'].max()}")
    
    return df


def compute_monthly_binary(df: pd.DataFrame, topic: str) -> pd.DataFrame:
    """Compute monthly proportions for binary framing indicators."""
    monthly = df.groupby('month').agg({
        'is_threat': 'mean',      # Proportion of THREAT posts
        'is_diplomacy': 'mean',   # Proportion of DIPLOMACY posts
        'id': 'count'             # Total posts
    }).reset_index()
    
    monthly.columns = ['month', 'threat_prop', 'diplomacy_prop', 'post_count']
    monthly['topic'] = topic
    
    return monthly


def prepare_binary_did_data(nk_monthly: pd.DataFrame, control_monthly: pd.DataFrame, 
                            control_name: str) -> pd.DataFrame:
    """Prepare combined data for Binary DID analysis."""
    nk = nk_monthly.copy()
    ctrl = control_monthly.copy()
    
    nk['treat'] = 1
    ctrl['treat'] = 0
    
    combined = pd.concat([nk, ctrl], ignore_index=True)
    combined = combined.sort_values('month').reset_index(drop=True)
    
    # Create time variable
    months = sorted(combined['month'].unique())
    month_to_time = {m: i for i, m in enumerate(months)}
    combined['time'] = combined['month'].map(month_to_time)
    
    # Post-intervention indicator (Singapore Summit announced March 2018)
    # P1: 2017-01 to 2018-02, P2: 2018-06 to 2019-01, P3: 2019-03 to 2019-12
    # For simplicity: post = months >= 2018-03
    combined['post'] = (combined['month'] >= '2018-03').astype(int)
    
    # Interaction terms
    combined['treat_post'] = combined['treat'] * combined['post']
    
    print(f"\nBinary DID Data: NK vs {control_name}")
    print(f"  Total months: {len(combined)}")
    print(f"  Pre-period: {len(combined[combined['post']==0])}")
    print(f"  Post-period: {len(combined[combined['post']==1])}")
    
    return combined


def run_binary_did(combined: pd.DataFrame, outcome: str, control_name: str) -> dict:
    """
    Run Binary DID for a specific outcome variable.
    
    Model: outcome ~ treat + post + treat:post
    β₃ (treat:post) = DID estimate
    """
    print(f"\n{'='*60}")
    print(f"Binary DID: {outcome.upper()} | NK vs {control_name}")
    print(f"{'='*60}")
    
    # Run OLS regression
    formula = f'{outcome} ~ treat + post + treat:post'
    model = smf.ols(formula, data=combined).fit(cov_type='HC3')
    
    print(model.summary())
    
    # Extract DID estimate
    did_est = model.params['treat:post']
    se = model.bse['treat:post']
    p_val = model.pvalues['treat:post']
    ci_lower = did_est - 1.96 * se
    ci_upper = did_est + 1.96 * se
    
    # Calculate means
    pre_nk = combined[(combined['treat']==1) & (combined['post']==0)][outcome].mean()
    post_nk = combined[(combined['treat']==1) & (combined['post']==1)][outcome].mean()
    pre_ctrl = combined[(combined['treat']==0) & (combined['post']==0)][outcome].mean()
    post_ctrl = combined[(combined['treat']==0) & (combined['post']==1)][outcome].mean()
    
    print(f"\nDID Estimate: {did_est:+.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"P-value: {p_val:.4f}")
    
    print(f"\nMeans (Proportions):")
    print(f"  NK Pre:      {pre_nk:.3f} ({pre_nk*100:.1f}%)")
    print(f"  NK Post:     {post_nk:.3f} ({post_nk*100:.1f}%)")
    print(f"  Ctrl Pre:    {pre_ctrl:.3f} ({pre_ctrl*100:.1f}%)")
    print(f"  Ctrl Post:   {post_ctrl:.3f} ({post_ctrl*100:.1f}%)")
    
    sig = "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
    print(f"\n{'✓ SIGNIFICANT' if p_val < 0.05 else '○ Marginal' if p_val < 0.10 else '✗ Not significant'} {sig}")
    
    return {
        'outcome': outcome,
        'control': control_name,
        'did_estimate': float(did_est),
        'se': float(se),
        'p_value': float(p_val),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'means': {
            'nk_pre': float(pre_nk),
            'nk_post': float(post_nk),
            'control_pre': float(pre_ctrl),
            'control_post': float(post_ctrl)
        },
        'n_obs': len(combined),
        'r_squared': float(model.rsquared)
    }


def main():
    """Main execution."""
    print("="*80)
    print("BINARY DID ROBUSTNESS CHECK")
    print("="*80)
    print("\nPurpose: Validate that framing results are robust to outcome scaling")
    print("Outcomes: THREAT proportion, DIPLOMACY proportion")
    
    # Load data for all countries
    print("\n" + "="*80)
    print("Loading Data")
    print("="*80)
    
    nk_posts = load_and_combine_framing_data('nk')
    china_posts = load_and_combine_framing_data('china')
    iran_posts = load_and_combine_framing_data('iran')
    # Russia excluded due to parallel trends violation
    
    # Compute monthly proportions
    nk_monthly = compute_monthly_binary(nk_posts, 'nk')
    china_monthly = compute_monthly_binary(china_posts, 'china')
    iran_monthly = compute_monthly_binary(iran_posts, 'iran')
    
    print("\nMonthly data computed:")
    print(f"  NK: {len(nk_monthly)} months")
    print(f"  China: {len(china_monthly)} months")
    print(f"  Iran: {len(iran_monthly)} months")
    
    # Run Binary DID for each control and outcome
    results = {}
    
    for control_name, control_monthly in [('China', china_monthly), ('Iran', iran_monthly)]:
        combined = prepare_binary_did_data(nk_monthly, control_monthly, control_name)
        
        # THREAT proportion DID
        threat_result = run_binary_did(combined, 'threat_prop', control_name)
        results[f'threat_{control_name}'] = threat_result
        
        # DIPLOMACY proportion DID
        diplomacy_result = run_binary_did(combined, 'diplomacy_prop', control_name)
        results[f'diplomacy_{control_name}'] = diplomacy_result
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / 'binary_did_robustness_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved: {output_file}")
    
    # Summary
    print("\n" + "="*80)
    print("BINARY DID ROBUSTNESS SUMMARY")
    print("="*80)
    
    print("\n1. THREAT Proportion (expected: NEGATIVE = decrease after Singapore)")
    for key, res in results.items():
        if 'threat' in key:
            sig = "**" if res['p_value'] < 0.05 else "*" if res['p_value'] < 0.10 else ""
            print(f"   {res['control']:8s}: DID = {res['did_estimate']:+.4f} (p={res['p_value']:.4f}) {sig}")
    
    print("\n2. DIPLOMACY Proportion (expected: POSITIVE = increase after Singapore)")
    for key, res in results.items():
        if 'diplomacy' in key:
            sig = "**" if res['p_value'] < 0.05 else "*" if res['p_value'] < 0.10 else ""
            print(f"   {res['control']:8s}: DID = {res['did_estimate']:+.4f} (p={res['p_value']:.4f}) {sig}")
    
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("If THREAT ↓ and DIPLOMACY ↑ consistently across controls:")
    print("  → Main results are ROBUST to outcome scaling choice")
    print("  → The -2/+2 scale is not driving the findings")
    

if __name__ == "__main__":
    main()
