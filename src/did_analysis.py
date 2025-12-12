"""
Difference-in-Differences (DID) Analysis

Estimates the causal effect of NK-US summit announcement on Reddit sentiment
using Iran as control group.
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

from src.config import DID_CONFIG, INTERVENTION_DATE


class DIDAnalyzer:
    """Difference-in-Differences estimator for causal inference."""

    def __init__(self, treatment_data: pd.DataFrame, control_data: pd.DataFrame, control_name: str = 'Iran'):
        """
        Initialize DID analyzer.

        Args:
            treatment_data: NK monthly sentiment data
            control_data: Control group monthly sentiment data
            control_name: Name of control group
        """
        self.treatment_data = treatment_data.copy()
        self.control_data = control_data.copy()
        self.control_name = control_name

        # Prepare combined dataset
        self.treatment_data['treat'] = 1
        self.treatment_data['group'] = 'North Korea'
        self.control_data['treat'] = 0
        self.control_data['group'] = control_name

        self.combined = pd.concat([self.treatment_data, self.control_data], ignore_index=True)

        # Convert month to datetime (if not already present, for weekly data compatibility)
        if 'month_date' not in self.combined.columns:
            self.combined['month_date'] = pd.to_datetime(self.combined['month'] + '-01')

        # Create post-treatment indicator
        intervention_month = pd.to_datetime(DID_CONFIG['post_period_start'] + '-01')
        self.combined['post'] = (self.combined['month_date'] >= intervention_month).astype(int)

        # Create time variable
        min_month = self.combined['month_date'].min()
        self.combined['time'] = ((self.combined['month_date'] - min_month) /
                                 pd.Timedelta(days=30)).astype(int)

    def estimate_basic_did(self) -> dict:
        """
        Estimate ITS-style DID model with slope changes.

        Model: Y_it = β₀ + β₁*Treat_i + β₂*Time_t + β₃*Post_t
                    + β₄*(Treat_i × Time_t)           # Pre-intervention trend difference
                    + β₅*(Treat_i × Post_t)            # Level change at intervention
                    + β₆*(Treat_i × Time_t × Post_t)   # Post-intervention SLOPE change

        Key coefficients:
        - β₄: Tests if pre-trends are parallel (should be ~0 if parallel trends holds)
        - β₆: DID estimate of slope change (treatment effect on TREND)

        Returns:
            dict with DID results
        """
        print(f"\n{'='*60}")
        print(f"ITS-STYLE DID ESTIMATION: North Korea vs {self.control_name}")
        print(f"{'='*60}")

        # Remove missing values
        data = self.combined.dropna(subset=['sentiment_mean'])

        print(f"\nData Summary:")
        print(f"  Total observations: {len(data)}")
        print(f"  NK observations: {len(data[data['treat'] == 1])}")
        print(f"  {self.control_name} observations: {len(data[data['treat'] == 0])}")
        print(f"  Pre-intervention: {len(data[data['post'] == 0])}")
        print(f"  Post-intervention: {len(data[data['post'] == 1])}")

        # Fit ITS-style DID model
        print("\n--- ITS-Style DID Model (Slope Change) ---")
        print("Model: sentiment ~ treat + time + post + treat:time + treat:post + treat:time:post")

        model_ols = smf.ols(
            'sentiment_mean ~ treat + time + post + treat:time + treat:post + treat:time:post',
            data=data
        ).fit()

        # Extract key coefficients
        beta4_pre_trend_diff = model_ols.params['treat:time']
        beta5_level_change = model_ols.params['treat:post']
        beta6_slope_change = model_ols.params['treat:time:post']  # DID estimate

        print(f"\nOLS Results:")
        print(f"\n  β₄ (Treat×Time): {beta4_pre_trend_diff:+.4f}")
        print(f"     → Pre-intervention trend difference")
        print(f"     → P-value: {model_ols.pvalues['treat:time']:.4f}")
        print(f"     → {'✓ Parallel trends holds' if model_ols.pvalues['treat:time'] > 0.10 else '⚠️ Parallel trends violated'}")

        print(f"\n  β₅ (Treat×Post): {beta5_level_change:+.4f}")
        print(f"     → Level change at intervention")
        print(f"     → P-value: {model_ols.pvalues['treat:post']:.4f}")

        print(f"\n  β₆ (Treat×Time×Post): {beta6_slope_change:+.4f}")
        print(f"     → Post-intervention SLOPE change (DID estimate)")
        print(f"     → SE: {model_ols.bse['treat:time:post']:.4f}")
        print(f"     → P-value: {model_ols.pvalues['treat:time:post']:.4f}")
        print(f"     → 95% CI: [{model_ols.conf_int().loc['treat:time:post', 0]:.4f}, {model_ols.conf_int().loc['treat:time:post', 1]:.4f}]")

        print(f"\n  Model R²: {model_ols.rsquared:.4f}")

        # Fit with clustered standard errors
        print("\n--- Clustered SE (by month) ---")
        try:
            model_cluster = smf.ols(
                'sentiment_mean ~ treat + time + post + treat:time + treat:post + treat:time:post',
                data=data
            ).fit(
                cov_type='cluster',
                cov_kwds={'groups': data['month']}
            )

            print(f"\n  β₆ (DID) with Clustered SE:")
            print(f"     Coefficient: {model_cluster.params['treat:time:post']:+.4f}")
            print(f"     SE: {model_cluster.bse['treat:time:post']:.4f}")
            print(f"     P-value: {model_cluster.pvalues['treat:time:post']:.4f}")
            print(f"     95% CI: [{model_cluster.conf_int().loc['treat:time:post', 0]:.4f}, {model_cluster.conf_int().loc['treat:time:post', 1]:.4f}]")

            has_cluster = True
        except Exception as e:
            print(f"\n⚠️  Clustered SE failed: {e}")
            print("  Using OLS results only")
            has_cluster = False

        # Calculate slopes for interpretation
        # NK pre-slope: β₂ + β₄
        # NK post-slope: β₂ + β₄ + β₆ (after adding post×time effect)
        # Control pre-slope: β₂
        # Control post-slope: β₂ (unchanged for control)

        beta2_time = model_ols.params['time']
        beta3_post_time = model_ols.params.get('time:post', 0)  # If model includes time:post

        nk_pre_slope = beta2_time + beta4_pre_trend_diff
        nk_post_slope = beta2_time + beta4_pre_trend_diff + beta6_slope_change + beta3_post_time
        control_pre_slope = beta2_time
        control_post_slope = beta2_time + beta3_post_time

        print(f"\n{'='*60}")
        print("SLOPE DECOMPOSITION")
        print(f"{'='*60}")
        print(f"\nNorth Korea:")
        print(f"  Pre-intervention slope:  {nk_pre_slope:+.4f}")
        print(f"  Post-intervention slope: {nk_post_slope:+.4f}")
        print(f"  Slope change (Δ):        {nk_post_slope - nk_pre_slope:+.4f}")

        print(f"\n{self.control_name}:")
        print(f"  Pre-intervention slope:  {control_pre_slope:+.4f}")
        print(f"  Post-intervention slope: {control_post_slope:+.4f}")
        print(f"  Slope change (Δ):        {control_post_slope - control_pre_slope:+.4f}")

        print(f"\nDID Estimate (Slope Change Difference):")
        print(f"  (NK slope change) - (Control slope change)")
        print(f"  ({nk_post_slope - nk_pre_slope:+.4f}) - ({control_post_slope - control_pre_slope:+.4f}) = {beta6_slope_change:+.4f}")

        # Prepare results dictionary
        results = {
            'control_group': self.control_name,
            'n_observations': int(len(data)),
            'n_treatment': int(len(data[data['treat'] == 1])),
            'n_control': int(len(data[data['treat'] == 0])),

            'ols': {
                'did_estimate': float(beta6_slope_change),
                'se': float(model_ols.bse['treat:time:post']),
                'p_value': float(model_ols.pvalues['treat:time:post']),
                'ci_lower': float(model_ols.conf_int().loc['treat:time:post', 0]),
                'ci_upper': float(model_ols.conf_int().loc['treat:time:post', 1]),
                'r_squared': float(model_ols.rsquared),
                'beta4_pre_trend_diff': float(beta4_pre_trend_diff),
                'beta4_pvalue': float(model_ols.pvalues['treat:time']),
                'beta5_level_change': float(beta5_level_change),
                'beta5_pvalue': float(model_ols.pvalues['treat:post'])
            },

            'slopes': {
                'nk_pre_slope': float(nk_pre_slope),
                'nk_post_slope': float(nk_post_slope),
                'nk_slope_change': float(nk_post_slope - nk_pre_slope),
                'control_pre_slope': float(control_pre_slope),
                'control_post_slope': float(control_post_slope),
                'control_slope_change': float(control_post_slope - control_pre_slope)
            }
        }

        if has_cluster:
            results['clustered'] = {
                'did_estimate': float(model_cluster.params['treat:time:post']),
                'se': float(model_cluster.bse['treat:time:post']),
                'p_value': float(model_cluster.pvalues['treat:time:post']),
                'ci_lower': float(model_cluster.conf_int().loc['treat:time:post', 0]),
                'ci_upper': float(model_cluster.conf_int().loc['treat:time:post', 1])
            }

        return results

    def plot_did_visualization(self, save_path: str = None) -> None:
        """
        Create DID visualization showing treatment and control trends.

        Args:
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(14, 7))

        intervention_month = pd.to_datetime(DID_CONFIG['post_period_start'] + '-01')

        # Plot NK trend
        nk_data = self.combined[self.combined['treat'] == 1].dropna(subset=['sentiment_mean'])
        ax.plot(nk_data['month_date'], nk_data['sentiment_mean'],
               marker='o', linewidth=2.5, markersize=8, color='#2E86AB',
               label='North Korea (Treatment)', zorder=3)

        # Plot Control trend
        control_data = self.combined[self.combined['treat'] == 0].dropna(subset=['sentiment_mean'])
        ax.plot(control_data['month_date'], control_data['sentiment_mean'],
               marker='s', linewidth=2.5, markersize=8, color='#A23B72',
               label=f'{self.control_name} (Control)', zorder=3)

        # Add vertical line at intervention
        ax.axvline(intervention_month, color='red', linestyle='--', linewidth=2.5,
                  alpha=0.7, label='Intervention (2018-03)', zorder=2)

        # Shade pre/post periods
        ax.axvspan(self.combined['month_date'].min(), intervention_month,
                  alpha=0.1, color='blue', label='Pre-intervention', zorder=1)
        ax.axvspan(intervention_month, self.combined['month_date'].max(),
                  alpha=0.1, color='green', label='Post-intervention', zorder=1)

        # Labels and formatting
        ax.set_xlabel('Month', fontsize=13, fontweight='bold')
        ax.set_ylabel('Sentiment Score', fontsize=13, fontweight='bold')
        ax.set_title(f'DID Analysis: North Korea vs {self.control_name}',
                    fontsize=15, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved plot: {save_path}")

        plt.close()

    def run_full_analysis(self, save_dir: str = 'visualizations') -> dict:
        """
        Run complete DID analysis workflow.

        Args:
            save_dir: Directory to save outputs

        Returns:
            dict with all results
        """
        # Estimate DID
        results = self.estimate_basic_did()

        # Create visualization
        plot_path = os.path.join(save_dir, f'did_analysis_{self.control_name.lower()}.png')
        self.plot_did_visualization(save_path=plot_path)

        # Save results
        os.makedirs('data/results', exist_ok=True)
        results_path = f'data/results/did_{self.control_name.lower()}_results.json'

        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Results saved: {results_path}")

        return results


def run_did_analysis():
    """Run DID analysis with Iran as control group."""
    print("="*60)
    print("PHASE 4: DID ESTIMATION")
    print("="*60)

    # Load data
    nk_monthly = pd.read_csv('data/processed/nk_monthly.csv')
    iran_monthly = pd.read_csv('data/processed/iran_monthly.csv')

    print(f"\nLoaded NK data: {nk_monthly['post_count'].notna().sum()} months")
    print(f"Loaded Iran data: {iran_monthly['post_count'].notna().sum()} months")

    # Run DID analysis
    analyzer = DIDAnalyzer(nk_monthly, iran_monthly, control_name='Iran')
    results = analyzer.run_full_analysis()

    # Interpretation
    print(f"\n{'='*60}")
    print("INTERPRETATION")
    print(f"{'='*60}")

    did_estimate = results['ols']['did_estimate']
    p_value = results['ols']['p_value']

    if p_value < 0.05:
        significance = "statistically significant (p < 0.05)"
        interpretation = "✓ The summit announcement caused a significant sentiment improvement"
    elif p_value < 0.10:
        significance = "marginally significant (p < 0.10)"
        interpretation = "⚠️ The summit announcement may have caused sentiment improvement"
    else:
        significance = "not significant (p ≥ 0.10)"
        interpretation = "✗ No significant causal effect detected"

    print(f"\nDID Estimate: {did_estimate:+.4f}")
    print(f"P-value: {p_value:.4f} → {significance}")
    print(f"\n{interpretation}")

    if did_estimate > 0:
        print(f"\nThe summit announcement improved NK sentiment by {did_estimate:.4f} points")
        print(f"relative to Iran (counterfactual baseline).")
    else:
        print(f"\nThe summit announcement decreased NK sentiment by {abs(did_estimate):.4f} points")
        print(f"relative to Iran (counterfactual baseline).")

    return results


if __name__ == '__main__':
    run_did_analysis()
