"""
Parallel Trends Testing for DID Analysis

Tests the parallel trends assumption by comparing pre-intervention trends
between treatment (NK) and control (Iran/Russia/China) groups.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns
import os
import json

from src.config import DID_CONFIG


class ParallelTrendsTest:
    """Test parallel trends assumption for DID analysis."""

    def __init__(self, nk_data: pd.DataFrame, control_data: pd.DataFrame, control_name: str):
        """
        Initialize parallel trends tester.

        Args:
            nk_data: NK monthly sentiment data
            control_data: Control group monthly sentiment data
            control_name: Name of control group (e.g., 'Iran')
        """
        self.nk_data = nk_data.copy()
        self.control_data = control_data.copy()
        self.control_name = control_name

        # Prepare combined dataset
        self.nk_data['treat'] = 1
        self.nk_data['group'] = 'North Korea'
        self.control_data['treat'] = 0
        self.control_data['group'] = control_name

        self.combined = pd.concat([self.nk_data, self.control_data], ignore_index=True)
        self.combined['month'] = pd.to_datetime(self.combined['month'] + '-01')

        # Filter pre-intervention period
        pre_end = pd.to_datetime(DID_CONFIG['pre_period_end'] + '-01')
        self.pre_period = self.combined[self.combined['month'] <= pre_end].copy()

        # Create time variable (months since start)
        min_month = self.pre_period['month'].min()
        self.pre_period['time'] = ((self.pre_period['month'] - min_month) /
                                    pd.Timedelta(days=30)).astype(int)

    def test_parallel_trends(self) -> dict:
        """
        Formal statistical test of parallel trends.

        Model: sentiment ~ time + treat + time:treat
        H0: time:treat coefficient = 0 (parallel trends holds)

        Returns:
            dict with test results
        """
        # Remove rows with missing sentiment
        test_data = self.pre_period.dropna(subset=['sentiment_mean'])

        if len(test_data) < 10:
            return {
                'verdict': 'INSUFFICIENT_DATA',
                'coefficient': None,
                'p_value': None,
                'nk_slope': None,
                'control_slope': None,
                'message': f'Insufficient data for testing ({len(test_data)} observations)'
            }

        # Fit regression model
        try:
            model = smf.ols('sentiment_mean ~ time + treat + time:treat', data=test_data).fit()

            # Extract interaction coefficient
            interaction_coef = model.params['time:treat']
            interaction_pval = model.pvalues['time:treat']

            # Calculate individual slopes
            nk_slope = model.params['time'] + interaction_coef  # NK slope
            control_slope = model.params['time']  # Control slope

            # Determine verdict
            threshold = DID_CONFIG['parallel_trends_threshold']
            verdict = 'PASS' if interaction_pval > threshold else 'FAIL'

            return {
                'verdict': verdict,
                'coefficient': float(interaction_coef),
                'p_value': float(interaction_pval),
                'nk_slope': float(nk_slope),
                'control_slope': float(control_slope),
                'slope_difference': float(abs(nk_slope - control_slope)),
                'threshold': threshold,
                'model_summary': {
                    'r_squared': float(model.rsquared),
                    'n_obs': int(model.nobs)
                }
            }
        except Exception as e:
            return {
                'verdict': 'ERROR',
                'coefficient': None,
                'p_value': None,
                'error': str(e)
            }

    def plot_trends(self, save_path: str = None) -> None:
        """
        Visualize parallel trends with pre/post intervention periods.

        Args:
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Get intervention month
        intervention_month = pd.to_datetime(DID_CONFIG['post_period_start'] + '-01')

        # Plot data
        for group_name in ['North Korea', self.control_name]:
            group_data = self.combined[self.combined['group'] == group_name].dropna(subset=['sentiment_mean'])

            # Separate pre and post
            pre_data = group_data[group_data['month'] < intervention_month]
            post_data = group_data[group_data['month'] >= intervention_month]

            # Plot pre-intervention (solid line)
            ax.plot(pre_data['month'], pre_data['sentiment_mean'],
                   marker='o', linewidth=2, markersize=6,
                   label=f'{group_name} (Pre-intervention)')

            # Plot post-intervention (dashed line)
            if len(post_data) > 0:
                ax.plot(post_data['month'], post_data['sentiment_mean'],
                       marker='s', linewidth=2, markersize=6, linestyle='--',
                       label=f'{group_name} (Post-intervention)')

        # Add vertical line at intervention
        ax.axvline(intervention_month, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label='Intervention (2018-03)')

        # Shade pre-intervention period
        ax.axvspan(self.combined['month'].min(), intervention_month,
                  alpha=0.1, color='blue', label='Pre-intervention period')

        # Labels and formatting
        ax.set_xlabel('Month', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sentiment Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Parallel Trends Test: North Korea vs {self.control_name}',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot: {save_path}")

        plt.close()

    def run_full_test(self, save_dir: str = 'visualizations') -> dict:
        """
        Run complete parallel trends test and generate outputs.

        Args:
            save_dir: Directory to save visualizations

        Returns:
            dict with test results
        """
        print(f"\n{'='*60}")
        print(f"PARALLEL TRENDS TEST: North Korea vs {self.control_name}")
        print(f"{'='*60}")

        # Statistical test
        results = self.test_parallel_trends()

        # Print results
        print(f"\nPre-intervention Period: {DID_CONFIG['pre_period_start']} to {DID_CONFIG['pre_period_end']}")
        print(f"\nTest Results:")

        if results['verdict'] == 'INSUFFICIENT_DATA':
            print(f"  ⚠️  {results['message']}")
        elif results['verdict'] == 'ERROR':
            print(f"  ✗ Error: {results['error']}")
        else:
            print(f"  Time×Treat Coefficient: {results['coefficient']:.4f}")
            print(f"  P-value: {results['p_value']:.4f}")
            print(f"  Threshold: {results['threshold']:.2f}")
            print(f"\n  NK Slope: {results['nk_slope']:.4f}")
            print(f"  {self.control_name} Slope: {results['control_slope']:.4f}")
            print(f"  Slope Difference: {results['slope_difference']:.4f}")
            print(f"\n  Verdict: {'✓ PASS' if results['verdict'] == 'PASS' else '✗ FAIL'}")

            if results['verdict'] == 'PASS':
                print(f"  → Parallel trends assumption HOLDS")
                print(f"  → DID analysis is justified")
            else:
                print(f"  → Parallel trends assumption VIOLATED")
                print(f"  → DID estimates may be biased")

        # Generate plot
        plot_path = os.path.join(save_dir, f'parallel_trends_{self.control_name.lower()}.png')
        self.plot_trends(save_path=plot_path)

        # Add control name to results
        results['control_group'] = self.control_name

        return results


def run_all_parallel_trends_tests():
    """Run parallel trends tests for all control groups."""
    print("="*60)
    print("PARALLEL TRENDS TESTING - ALL CONTROL GROUPS")
    print("="*60)

    # Load NK data
    nk_monthly = pd.read_csv('data/processed/nk_monthly.csv')
    print(f"\nLoaded NK data: {nk_monthly['post_count'].notna().sum()} months with data")

    # Load control groups
    control_groups = {
        'Iran': pd.read_csv('data/processed/iran_monthly.csv'),
        # 'Russia': pd.read_csv('data/processed/russia_monthly.csv'),
        # 'China': pd.read_csv('data/processed/china_monthly.csv')
    }

    all_results = {}

    for control_name, control_data in control_groups.items():
        print(f"\nLoaded {control_name} data: {control_data['post_count'].notna().sum()} months with data")

        # Run test
        tester = ParallelTrendsTest(nk_monthly, control_data, control_name)
        results = tester.run_full_test()
        all_results[control_name] = results

    # Save results
    os.makedirs('data/results', exist_ok=True)
    results_path = 'data/results/parallel_trends_tests.json'

    # Convert numpy types to native Python for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    all_results = convert_types(all_results)

    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    for control_name, results in all_results.items():
        print(f"\n{control_name}:")
        if results['verdict'] in ['PASS', 'FAIL']:
            print(f"  Verdict: {results['verdict']}")
            print(f"  P-value: {results['p_value']:.4f}")
        else:
            print(f"  Status: {results['verdict']}")

    print(f"\n✓ Results saved: {results_path}")


if __name__ == '__main__':
    run_all_parallel_trends_tests()
