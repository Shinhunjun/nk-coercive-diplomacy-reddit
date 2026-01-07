"""
Parallel Trends Test - Current Data
Tests parallel trends assumption for sentiment and framing DiD analysis
"""
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import os

# Load monthly data
print("=" * 80)
print("PARALLEL TRENDS VERIFICATION")
print("=" * 80)

# Load Sentiment Data
print("\n1. SENTIMENT PARALLEL TRENDS")
print("-" * 80)

nk_sent = pd.read_csv('data/sentiment/nk_monthly_sentiment.csv')
china_sent = pd.read_csv('data/sentiment/china_monthly_sentiment.csv')
iran_sent = pd.read_csv('data/sentiment/iran_monthly_sentiment.csv')
russia_sent = pd.read_csv('data/sentiment/russia_monthly_sentiment.csv')

print(f"Loaded: NK={len(nk_sent)}, China={len(china_sent)}, Iran={len(iran_sent)}, Russia={len(russia_sent)} months")

def assign_period(month):
    if month <= '2018-02':
        return 'P1'
    elif '2018-03' <= month <= '2018-05':
        return None  # Transition
    elif '2018-06' <= month <= '2019-01':
        return 'P2'
    elif month == '2019-02':
        return None  # Hanoi month
    elif '2019-03' <= month <= '2019-12':
        return 'P3'
    return None

def test_parallel_trends(nk_df, ctrl_df, ctrl_name, period, outcome_col):
    """Test parallel trends in specified pre-treatment period."""
    nk_sub = nk_df[nk_df['period'] == period].copy()
    ctrl_sub = ctrl_df[ctrl_df['period'] == period].copy()
    
    if len(nk_sub) < 3 or len(ctrl_sub) < 3:
        return None, None
    
    nk_sub['treat'] = 1
    ctrl_sub['treat'] = 0
    
    combined = pd.concat([nk_sub, ctrl_sub], ignore_index=True)
    
    # Create time trend
    months = sorted(combined['month'].unique())
    month_map = {m: i for i, m in enumerate(months)}
    combined['time'] = combined['month'].map(month_map)
    combined['treat_time'] = combined['treat'] * combined['time']
    
    # Rename outcome column
    combined['y'] = combined[outcome_col]
    
    try:
        model = smf.ols('y ~ treat + time + treat_time', data=combined).fit(
            cov_type='cluster', cov_kwds={'groups': combined['month']}
        )
        p_val = model.pvalues['treat_time']
        coef = model.params['treat_time']
        return p_val, coef
    except Exception as e:
        print(f"  Error: {e}")
        return None, None

# Add period to sentiment data
for df in [nk_sent, china_sent, iran_sent, russia_sent]:
    df['period'] = df['month'].apply(assign_period)

results = []

# Test Sentiment P1 (Pre-Singapore)
print("\n[Sentiment] Pre-Singapore (P1):")
for ctrl_df, ctrl_name in [(china_sent, 'China'), (iran_sent, 'Iran'), (russia_sent, 'Russia')]:
    p_val, coef = test_parallel_trends(nk_sent, ctrl_df, ctrl_name, 'P1', 'sentiment_mean')
    if p_val is not None:
        status = "✅ PASS" if p_val > 0.05 else "❌ FAIL"
        print(f"  {ctrl_name}: p={p_val:.4f} {status}")
        results.append({'type': 'Sentiment', 'period': 'P1→P2', 'control': ctrl_name, 'p_value': round(p_val, 2)})

# Test Sentiment P2 (Pre-Hanoi)
print("\n[Sentiment] Pre-Hanoi (P2):")
for ctrl_df, ctrl_name in [(china_sent, 'China'), (iran_sent, 'Iran'), (russia_sent, 'Russia')]:
    p_val, coef = test_parallel_trends(nk_sent, ctrl_df, ctrl_name, 'P2', 'sentiment_mean')
    if p_val is not None:
        status = "✅ PASS" if p_val > 0.05 else "❌ FAIL"
        print(f"  {ctrl_name}: p={p_val:.4f} {status}")
        results.append({'type': 'Sentiment', 'period': 'P2→P3', 'control': ctrl_name, 'p_value': round(p_val, 2)})

# Load Framing Data
print("\n" + "=" * 80)
print("2. FRAMING PARALLEL TRENDS")
print("-" * 80)

nk_frame = pd.read_csv('data/framing/nk_monthly_framing.csv')
china_frame = pd.read_csv('data/framing/china_monthly_framing.csv')
iran_frame = pd.read_csv('data/framing/iran_monthly_framing.csv')
russia_frame = pd.read_csv('data/framing/russia_monthly_framing.csv')

print(f"Loaded: NK={len(nk_frame)}, China={len(china_frame)}, Iran={len(iran_frame)}, Russia={len(russia_frame)} months")

# Add period to framing data
for df in [nk_frame, china_frame, iran_frame, russia_frame]:
    df['period'] = df['month'].apply(assign_period)

# Test Framing P1 (Pre-Singapore)
print("\n[Framing] Pre-Singapore (P1):")
for ctrl_df, ctrl_name in [(china_frame, 'China'), (iran_frame, 'Iran'), (russia_frame, 'Russia')]:
    p_val, coef = test_parallel_trends(nk_frame, ctrl_df, ctrl_name, 'P1', 'framing_mean')
    if p_val is not None:
        status = "✅ PASS" if p_val > 0.05 else "❌ FAIL"
        print(f"  {ctrl_name}: p={p_val:.4f} {status}")
        results.append({'type': 'Framing', 'period': 'P1→P2', 'control': ctrl_name, 'p_value': round(p_val, 2)})

# Test Framing P2 (Pre-Hanoi)
print("\n[Framing] Pre-Hanoi (P2):")
for ctrl_df, ctrl_name in [(china_frame, 'China'), (iran_frame, 'Iran'), (russia_frame, 'Russia')]:
    p_val, coef = test_parallel_trends(nk_frame, ctrl_df, ctrl_name, 'P2', 'framing_mean')
    if p_val is not None:
        status = "✅ PASS" if p_val > 0.05 else "❌ FAIL"
        print(f"  {ctrl_name}: p={p_val:.4f} {status}")
        results.append({'type': 'Framing', 'period': 'P2→P3', 'control': ctrl_name, 'p_value': round(p_val, 2)})

# Summary
print("\n" + "=" * 80)
print("SUMMARY FOR LATEX TABLES")
print("=" * 80)

print("\nTable 2: Parallel Trends Test Results (Sentiment)")
print("-" * 60)
sent_results = [r for r in results if r['type'] == 'Sentiment']
for r in sent_results:
    status = "Valid" if r['p_value'] > 0.05 else "Invalid"
    print(f"{r['period']:12} & {r['control']:8} & {r['p_value']:.2f} & {status} \\\\")

print("\nTable 3: Parallel Trends Test Results (Framing)")
print("-" * 60)
frame_results = [r for r in results if r['type'] == 'Framing']
for r in frame_results:
    status = "Valid" if r['p_value'] > 0.05 else "Invalid"
    print(f"{r['period']:12} & {r['control']:8} & {r['p_value']:.2f} & {status} \\\\")

print("\n✓ Done!")
