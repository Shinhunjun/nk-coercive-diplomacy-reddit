"""
Run Difference-in-Differences Analysis using Sentiment Scores
Treatment: NK (North Korea)
Control: Iran, Russia, China
Intervention: March 2018 (Trump accepts NK summit)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import json
import os


def load_sentiment_data(path: str = "data/sentiment/combined_monthly_did.csv") -> pd.DataFrame:
    """Load combined monthly sentiment data."""
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} monthly observations")
    print(f"Topics: {df['topic'].unique()}")
    print(f"Date range: {df['month'].min()} to {df['month'].max()}")
    return df


def run_did_analysis(df: pd.DataFrame, control_group: str = "all") -> dict:
    """
    Run DID regression analysis.

    Model: Y = β0 + β1*Treated + β2*Post + β3*(Treated*Post) + ε

    Where:
    - Y = sentiment_mean
    - Treated = 1 if NK, 0 if control
    - Post = 1 if month >= 2018-03, 0 otherwise
    - β3 = DID estimator (the causal effect of interest)
    """

    print("\n" + "=" * 60)
    print(f"DID ANALYSIS: NK vs {control_group.upper()}")
    print("=" * 60)

    # Filter data
    if control_group == "all":
        df_analysis = df.copy()
    else:
        df_analysis = df[(df['topic'] == 'nk') | (df['topic'] == control_group)].copy()

    # Remove rows with missing sentiment
    df_analysis = df_analysis.dropna(subset=['sentiment_mean'])

    print(f"\nObservations: {len(df_analysis)}")
    print(f"NK observations: {len(df_analysis[df_analysis['topic'] == 'nk'])}")
    print(f"Control observations: {len(df_analysis[df_analysis['topic'] != 'nk'])}")

    # Create DID variables
    df_analysis['treated'] = (df_analysis['topic'] == 'nk').astype(int)
    df_analysis['post'] = df_analysis['post_intervention']
    df_analysis['did'] = df_analysis['treated'] * df_analysis['post']

    # Pre/Post means
    pre_treatment = df_analysis[(df_analysis['treated'] == 1) & (df_analysis['post'] == 0)]['sentiment_mean'].mean()
    post_treatment = df_analysis[(df_analysis['treated'] == 1) & (df_analysis['post'] == 1)]['sentiment_mean'].mean()
    pre_control = df_analysis[(df_analysis['treated'] == 0) & (df_analysis['post'] == 0)]['sentiment_mean'].mean()
    post_control = df_analysis[(df_analysis['treated'] == 0) & (df_analysis['post'] == 1)]['sentiment_mean'].mean()

    print(f"\n--- Mean Sentiment by Group and Period ---")
    print(f"NK (Pre):      {pre_treatment:.4f}")
    print(f"NK (Post):     {post_treatment:.4f}")
    print(f"NK Change:     {post_treatment - pre_treatment:.4f}")
    print(f"\nControl (Pre): {pre_control:.4f}")
    print(f"Control (Post):{post_control:.4f}")
    print(f"Control Change:{post_control - pre_control:.4f}")

    # Simple DID calculation
    did_simple = (post_treatment - pre_treatment) - (post_control - pre_control)
    print(f"\n--- Simple DID Estimate ---")
    print(f"DID = (NK Post - NK Pre) - (Control Post - Control Pre)")
    print(f"DID = ({post_treatment:.4f} - {pre_treatment:.4f}) - ({post_control:.4f} - {pre_control:.4f})")
    print(f"DID = {did_simple:.4f}")

    # OLS Regression
    X = df_analysis[['treated', 'post', 'did']]
    X = sm.add_constant(X)
    y = df_analysis['sentiment_mean']

    model = sm.OLS(y, X).fit(cov_type='HC1')  # Robust standard errors

    print(f"\n--- OLS Regression Results ---")
    print(model.summary())

    # Extract results
    results = {
        'control_group': control_group,
        'n_observations': len(df_analysis),
        'n_nk': len(df_analysis[df_analysis['topic'] == 'nk']),
        'n_control': len(df_analysis[df_analysis['topic'] != 'nk']),
        'pre_treatment_mean': pre_treatment,
        'post_treatment_mean': post_treatment,
        'pre_control_mean': pre_control,
        'post_control_mean': post_control,
        'did_simple': did_simple,
        'did_coefficient': model.params['did'],
        'did_std_error': model.bse['did'],
        'did_t_stat': model.tvalues['did'],
        'did_p_value': model.pvalues['did'],
        'did_ci_lower': model.conf_int().loc['did', 0],
        'did_ci_upper': model.conf_int().loc['did', 1],
        'r_squared': model.rsquared,
        'r_squared_adj': model.rsquared_adj
    }

    # Interpretation
    print(f"\n--- Interpretation ---")
    if results['did_p_value'] < 0.05:
        direction = "increased" if results['did_coefficient'] > 0 else "decreased"
        print(f"The DID coefficient is statistically significant (p={results['did_p_value']:.4f})")
        print(f"After the March 2018 intervention, NK sentiment {direction} by {abs(results['did_coefficient']):.4f}")
        print(f"relative to the control group.")
    else:
        print(f"The DID coefficient is NOT statistically significant (p={results['did_p_value']:.4f})")
        print("No significant differential effect found for NK after the intervention.")

    return results


def run_parallel_trends_test(df: pd.DataFrame) -> dict:
    """Test parallel trends assumption in pre-intervention period."""

    print("\n" + "=" * 60)
    print("PARALLEL TRENDS TEST")
    print("=" * 60)

    # Filter to pre-intervention period
    df_pre = df[df['post_intervention'] == 0].copy()
    df_pre = df_pre.dropna(subset=['sentiment_mean'])

    # Create time trend
    df_pre['month_num'] = pd.to_datetime(df_pre['month']).dt.to_period('M').apply(lambda x: (x.year - 2017) * 12 + x.month)
    df_pre['treated'] = (df_pre['topic'] == 'nk').astype(int)
    df_pre['time_trend'] = df_pre['month_num']
    df_pre['treated_time'] = df_pre['treated'] * df_pre['time_trend']

    # Regression: Y = β0 + β1*Treated + β2*Time + β3*(Treated*Time) + ε
    # If β3 is significant, parallel trends assumption is violated
    X = df_pre[['treated', 'time_trend', 'treated_time']]
    X = sm.add_constant(X)
    y = df_pre['sentiment_mean']

    model = sm.OLS(y, X).fit(cov_type='HC1')

    print("\nPre-intervention Trend Regression:")
    print(model.summary())

    results = {
        'interaction_coef': model.params['treated_time'],
        'interaction_p_value': model.pvalues['treated_time'],
        'parallel_trends_holds': model.pvalues['treated_time'] > 0.05
    }

    print(f"\n--- Parallel Trends Test Result ---")
    if results['parallel_trends_holds']:
        print(f"Interaction term is NOT significant (p={results['interaction_p_value']:.4f})")
        print("Parallel trends assumption HOLDS - groups had similar trends before intervention")
    else:
        print(f"Interaction term IS significant (p={results['interaction_p_value']:.4f})")
        print("WARNING: Parallel trends assumption may be VIOLATED")

    return results


def main():
    """Run full DID analysis with sentiment scores."""

    output_dir = "data/results"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_sentiment_data()

    # Run parallel trends test
    pt_results = run_parallel_trends_test(df)

    # Run DID with all controls combined
    all_results = run_did_analysis(df, control_group="all")

    # Run DID with each control separately
    individual_results = {}
    for control in ['iran', 'russia', 'china']:
        individual_results[control] = run_did_analysis(df, control_group=control)

    # Summary
    print("\n" + "=" * 60)
    print("SENTIMENT DID ANALYSIS SUMMARY")
    print("=" * 60)

    print("\n--- All Controls Combined ---")
    print(f"DID Coefficient: {all_results['did_coefficient']:.4f}")
    print(f"P-value: {all_results['did_p_value']:.4f}")
    print(f"95% CI: [{all_results['did_ci_lower']:.4f}, {all_results['did_ci_upper']:.4f}]")

    print("\n--- Individual Control Groups ---")
    for control, res in individual_results.items():
        sig = "*" if res['did_p_value'] < 0.05 else ""
        print(f"{control.upper():8s}: DID = {res['did_coefficient']:+.4f}, p = {res['did_p_value']:.4f} {sig}")

    # Save results
    all_results_combined = {
        'parallel_trends': pt_results,
        'all_controls': all_results,
        'individual_controls': individual_results
    }

    with open(f"{output_dir}/sentiment_did_results.json", 'w') as f:
        json.dump(all_results_combined, f, indent=2, default=float)

    print(f"\nResults saved to: {output_dir}/sentiment_did_results.json")


if __name__ == '__main__':
    main()
