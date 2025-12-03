"""
Interrupted Time Series (ITS) Analysis for Causal Inference

This module implements ITS regression to estimate the causal effect
of the summit announcement on public sentiment.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import json

from config import INTERVENTION_DATE, RESULTS_DIR


class ITSAnalyzer:
    """Interrupted Time Series analyzer for causal inference."""

    def __init__(self, intervention_date: str = INTERVENTION_DATE):
        """
        Initialize the ITS analyzer.

        Args:
            intervention_date: Date of the intervention (YYYY-MM-DD format)
        """
        self.intervention_date = pd.to_datetime(intervention_date)

    def prepare_monthly_data(self, df: pd.DataFrame, date_column: str = 'created_utc',
                            sentiment_column: str = 'sentiment_score') -> pd.DataFrame:
        """
        Aggregate data to monthly level for ITS analysis.

        Args:
            df: DataFrame with dates and sentiment scores
            date_column: Name of the date column
            sentiment_column: Name of the sentiment score column

        Returns:
            Monthly aggregated DataFrame
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df[date_column], unit='s')
        df['month'] = df['date'].dt.to_period('M')

        monthly = df.groupby('month').agg({
            sentiment_column: ['mean', 'std', 'count']
        }).reset_index()

        monthly.columns = ['month', 'sentiment_mean', 'sentiment_std', 'count']
        monthly['month_date'] = monthly['month'].dt.to_timestamp()

        return monthly

    def fit_its_model(self, monthly_data: pd.DataFrame) -> dict:
        """
        Fit the ITS regression model.

        Model: Y_t = β₀ + β₁*T + β₂*X_t + β₃*(T×X_t) + ε_t

        Where:
        - T: Time variable (months)
        - X_t: Intervention indicator (0 before, 1 after)
        - β₁: Pre-intervention slope
        - β₂: Level change at intervention
        - β₃: Slope change post-intervention

        Args:
            monthly_data: Monthly aggregated data

        Returns:
            Dictionary with model results
        """
        df = monthly_data.copy()

        # Create ITS variables
        df['time'] = range(1, len(df) + 1)
        df['intervention'] = (df['month_date'] >= self.intervention_date).astype(int)
        df['time_after'] = df['time'] * df['intervention']

        # Fit OLS model
        X = sm.add_constant(df[['time', 'intervention', 'time_after']])
        y = df['sentiment_mean']

        model = sm.OLS(y, X).fit()

        # Extract coefficients
        results = {
            "coefficients": {
                "intercept": {
                    "estimate": float(model.params['const']),
                    "std_error": float(model.bse['const']),
                    "t_stat": float(model.tvalues['const']),
                    "p_value": float(model.pvalues['const'])
                },
                "beta1_pre_trend": {
                    "estimate": float(model.params['time']),
                    "std_error": float(model.bse['time']),
                    "t_stat": float(model.tvalues['time']),
                    "p_value": float(model.pvalues['time']),
                    "interpretation": "Pre-intervention slope (natural trend)"
                },
                "beta2_level_change": {
                    "estimate": float(model.params['intervention']),
                    "std_error": float(model.bse['intervention']),
                    "t_stat": float(model.tvalues['intervention']),
                    "p_value": float(model.pvalues['intervention']),
                    "interpretation": "Immediate effect at intervention"
                },
                "beta3_slope_change": {
                    "estimate": float(model.params['time_after']),
                    "std_error": float(model.bse['time_after']),
                    "t_stat": float(model.tvalues['time_after']),
                    "p_value": float(model.pvalues['time_after']),
                    "interpretation": "Change in trend post-intervention"
                }
            },
            "model_fit": {
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_statistic": float(model.fvalue),
                "f_p_value": float(model.f_pvalue)
            },
            "intervention_date": str(self.intervention_date.date())
        }

        return results

    def calculate_counterfactual(self, monthly_data: pd.DataFrame, model_results: dict) -> dict:
        """
        Calculate counterfactual (what would have happened without intervention).

        Args:
            monthly_data: Monthly aggregated data
            model_results: Results from fit_its_model

        Returns:
            Dictionary with counterfactual analysis
        """
        df = monthly_data.copy()
        df['month_date'] = pd.to_datetime(df['month_date'])

        # Get post-intervention data
        post_data = df[df['month_date'] >= self.intervention_date]

        if len(post_data) == 0:
            return {"error": "No post-intervention data available"}

        # Extract coefficients
        beta0 = model_results['coefficients']['intercept']['estimate']
        beta1 = model_results['coefficients']['beta1_pre_trend']['estimate']

        # Calculate counterfactual (extend pre-intervention trend)
        time_values = range(1, len(df) + 1)
        post_time_start = len(df) - len(post_data) + 1

        counterfactual_values = []
        for t in range(post_time_start, len(df) + 1):
            counterfactual_values.append(beta0 + beta1 * t)

        actual_mean = post_data['sentiment_mean'].mean()
        counterfactual_mean = np.mean(counterfactual_values)
        causal_effect = actual_mean - counterfactual_mean

        return {
            "post_intervention_actual_mean": float(actual_mean),
            "counterfactual_mean": float(counterfactual_mean),
            "estimated_causal_effect": float(causal_effect),
            "interpretation": f"The intervention improved sentiment by {causal_effect:.3f} compared to counterfactual"
        }


def interpret_its_results(results: dict) -> str:
    """
    Provide plain-language interpretation of ITS results.

    Args:
        results: Dictionary from fit_its_model

    Returns:
        String interpretation
    """
    beta1 = results['coefficients']['beta1_pre_trend']
    beta2 = results['coefficients']['beta2_level_change']
    beta3 = results['coefficients']['beta3_slope_change']

    interpretation = []

    # Pre-intervention trend
    if beta1['p_value'] < 0.05:
        direction = "improving" if beta1['estimate'] > 0 else "declining"
        interpretation.append(f"1. Pre-intervention: Sentiment was {direction} (β₁={beta1['estimate']:.3f}, p={beta1['p_value']:.3f})")
    else:
        interpretation.append(f"1. Pre-intervention: No significant trend (β₁={beta1['estimate']:.3f}, p={beta1['p_value']:.3f})")

    # Level change
    if beta2['p_value'] < 0.05:
        direction = "improved" if beta2['estimate'] > 0 else "declined"
        interpretation.append(f"2. Immediate effect: Sentiment {direction} by {abs(beta2['estimate']):.3f} (p={beta2['p_value']:.3f}) ***SIGNIFICANT***")
    else:
        interpretation.append(f"2. Immediate effect: No significant level change (β₂={beta2['estimate']:.3f}, p={beta2['p_value']:.3f})")

    # Slope change
    if beta3['p_value'] < 0.05:
        direction = "accelerated" if beta3['estimate'] > 0 else "decelerated"
        interpretation.append(f"3. Post-intervention trend: {direction} (β₃={beta3['estimate']:.3f}, p={beta3['p_value']:.3f})")
    else:
        interpretation.append(f"3. Post-intervention trend: No significant change in slope (β₃={beta3['estimate']:.3f}, p={beta3['p_value']:.3f})")

    return "\n".join(interpretation)


if __name__ == "__main__":
    print("=" * 60)
    print("Interrupted Time Series (ITS) Analysis")
    print("=" * 60)

    # Load pre-computed results
    results_path = RESULTS_DIR / "its_analysis_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        print(f"\nIntervention Date: {INTERVENTION_DATE}")
        print("\n" + "=" * 60)
        print("Model Coefficients")
        print("=" * 60)

        for name, coef in results['coefficients'].items():
            sig = "***" if coef['p_value'] < 0.05 else ""
            print(f"\n{name}:")
            print(f"  Estimate: {coef['estimate']:.4f}")
            print(f"  p-value:  {coef['p_value']:.4f} {sig}")

        print("\n" + "=" * 60)
        print("Interpretation")
        print("=" * 60)
        print(interpret_its_results(results))

    else:
        print("\nNo pre-computed results found.")
        print("Run full analysis with monthly sentiment data.")
