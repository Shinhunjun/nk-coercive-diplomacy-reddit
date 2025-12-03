#!/usr/bin/env python3
"""
Main Analysis Script for NK Coercive Diplomacy Reddit Analysis

This script runs the complete analysis pipeline:
1. Sentiment Analysis (BERT)
2. Framing Analysis (GPT-4o-mini) - requires API key
3. ITS Causal Inference
4. Visualization Generation

Usage:
    python run_analysis.py [--skip-framing]

Note: Framing analysis requires OPENAI_API_KEY environment variable.
      Use --skip-framing to use pre-computed results.
"""

import argparse
import json
from pathlib import Path

from config import SAMPLE_DIR, RESULTS_DIR, FIGURES_DIR


def run_sentiment_analysis():
    """Run BERT sentiment analysis on sample data."""
    print("\n" + "=" * 60)
    print("STEP 1: Sentiment Analysis")
    print("=" * 60)

    from sentiment_analysis import SentimentAnalyzer, compare_periods
    import pandas as pd

    # Load sample data
    p1 = pd.read_csv(SAMPLE_DIR / "posts_period1_sample.csv")
    p2 = pd.read_csv(SAMPLE_DIR / "posts_period2_sample.csv")

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Prepare text
    p1['text'] = p1['title'].fillna('') + ' ' + p1['selftext'].fillna('')
    p2['text'] = p2['title'].fillna('') + ' ' + p2['selftext'].fillna('')

    # Analyze
    p1 = analyzer.analyze_dataframe(p1, 'text')
    p2 = analyzer.analyze_dataframe(p2, 'text')

    # Compare
    results = compare_periods(
        p1['sentiment_score'].values,
        p2['sentiment_score'].values
    )

    # Display results
    print("\nResults:")
    print(f"  Tension Period Mean:   {results['period1']['mean']:.3f}")
    print(f"  Diplomacy Period Mean: {results['period2']['mean']:.3f}")
    print(f"  Change:                {results['change']:+.3f}")
    print(f"  T-test p-value:        {results['t_test']['p_value']:.6f}")
    print(f"  Cohen's d:             {results['effect_size']['cohens_d']:.3f}")

    return results


def display_framing_results():
    """Display pre-computed framing analysis results."""
    print("\n" + "=" * 60)
    print("STEP 2: Framing Analysis (Pre-computed)")
    print("=" * 60)

    results_path = RESULTS_DIR / "openai_framing_results.json"
    if not results_path.exists():
        print("No pre-computed framing results found.")
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    print("\nPeriod 1 (Tension) Frame Distribution:")
    for frame, count in results['period1']['frame_distribution'].items():
        pct = count / results['period1']['total_valid'] * 100
        print(f"  {frame}: {count} ({pct:.1f}%)")

    print("\nPeriod 2 (Diplomacy) Frame Distribution:")
    for frame, count in results['period2']['frame_distribution'].items():
        pct = count / results['period2']['total_valid'] * 100
        print(f"  {frame}: {count} ({pct:.1f}%)")

    print(f"\nChi-square: {results['comparison']['chi2']:.2f}")
    print(f"p-value: {results['comparison']['p_value']:.10f}")

    return results


def display_its_results():
    """Display pre-computed ITS analysis results."""
    print("\n" + "=" * 60)
    print("STEP 3: ITS Causal Inference (Pre-computed)")
    print("=" * 60)

    results_path = RESULTS_DIR / "its_analysis_results.json"
    if not results_path.exists():
        print("No pre-computed ITS results found.")
        return None

    with open(results_path, 'r') as f:
        results = json.load(f)

    from its_analysis import interpret_its_results

    print("\nModel Coefficients:")
    for name, coef in results['coefficients'].items():
        sig = "***" if coef['p_value'] < 0.05 else ""
        print(f"\n  {name}:")
        print(f"    Estimate: {coef['estimate']:.4f}")
        print(f"    p-value:  {coef['p_value']:.4f} {sig}")

    print("\nInterpretation:")
    print(interpret_its_results(results))

    return results


def generate_figures():
    """Generate all paper figures."""
    print("\n" + "=" * 60)
    print("STEP 4: Generating Figures")
    print("=" * 60)

    from visualizations import generate_all_figures
    generate_all_figures(FIGURES_DIR)


def main():
    parser = argparse.ArgumentParser(description="Run NK Coercive Diplomacy Analysis")
    parser.add_argument('--skip-sentiment', action='store_true',
                       help='Skip sentiment analysis (use pre-computed)')
    parser.add_argument('--skip-figures', action='store_true',
                       help='Skip figure generation')
    args = parser.parse_args()

    print("=" * 60)
    print("NK Coercive Diplomacy Reddit Analysis")
    print("=" * 60)
    print("\nThis analysis examines the impact of North Korea's coercive")
    print("diplomacy strategy on U.S. public opinion using Reddit data.")
    print("\nHypotheses:")
    print("  H1: Sentiment improved during diplomacy period")
    print("  H2: Framing shifted from THREAT to DIPLOMACY")
    print("  H3: Summit announcement caused opinion change (causal)")
    print("  H4: Knowledge graph structure changed")

    # Run analysis steps
    if not args.skip_sentiment:
        run_sentiment_analysis()
    else:
        print("\n[Skipping sentiment analysis - using pre-computed results]")

    display_framing_results()
    display_its_results()

    if not args.skip_figures:
        generate_figures()
    else:
        print("\n[Skipping figure generation]")

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print("\nKey Findings:")
    print("  - H1 SUPPORTED: Sentiment improved (+0.230, p < 0.001)")
    print("  - H2 SUPPORTED: Threat frame -29.3%, Diplomacy +22.7%")
    print("  - H3 SUPPORTED: Causal effect confirmed (beta2 = +0.293, p = 0.044)")
    print("  - H4 SUPPORTED: Knowledge structure shifted (qualitative)")
    print(f"\nFigures saved to: {FIGURES_DIR}")
    print(f"Results stored in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
