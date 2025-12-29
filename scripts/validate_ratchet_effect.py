"""
Ratchet Effect Validation with Combined Framing Score

Uses combined metric: Framing Score = DIPLOMACY% - THREAT%
This captures the overall shift from threat-dominant to diplomacy-dominant framing.
"""
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("/Users/hunjunsin/Desktop/Jun/nk-coercive-diplomacy-reddit/data")
RESULTS_DIR = DATA_DIR / "results"

P1_END = datetime(2018, 6, 12)
P2_END = datetime(2019, 2, 28)

def framing_score(df):
    """Combined framing score: DIPLOMACY% - THREAT%"""
    diplo = (df['frame'] == 'DIPLOMACY').mean()
    threat = (df['frame'] == 'THREAT').mean()
    return diplo - threat

def bootstrap_ratio_test(data_p1, data_p2, data_p3, metric_func, n_bootstrap=1000):
    """Bootstrap test for ratchet effect."""
    ratios = []
    for _ in range(n_bootstrap):
        s1 = data_p1.sample(n=len(data_p1), replace=True)
        s2 = data_p2.sample(n=len(data_p2), replace=True)
        s3 = data_p3.sample(n=len(data_p3), replace=True)
        
        v1 = metric_func(s1)
        v2 = metric_func(s2)
        v3 = metric_func(s3)
        
        e1 = abs(v2 - v1)  # |P1→P2|
        e2 = abs(v3 - v2)  # |P2→P3|
        
        if e1 > 0.001:
            ratios.append(e2 / e1)
    
    ratios = np.array(ratios)
    return {
        'mean_ratio': np.mean(ratios),
        'ci_lower': np.percentile(ratios, 2.5),
        'ci_upper': np.percentile(ratios, 97.5),
        'p_value': np.mean(ratios >= 1.0)
    }

print("="*70)
print("RATCHET EFFECT VALIDATION: Combined Framing Score")
print("Framing Score = DIPLOMACY% - THREAT%")
print("="*70)

#=============================================================================
# 1. CONTENT FRAMING
#=============================================================================
print("\n" + "="*70)
print("1. CONTENT FRAMING (Post-level)")
print("="*70)

content_df = pd.read_csv(DATA_DIR / "processed" / "nk_posts_framing.csv")
content_df['datetime'] = pd.to_datetime(content_df['datetime'])

p1 = content_df[content_df['datetime'] < P1_END]
p2 = content_df[(content_df['datetime'] >= P1_END) & (content_df['datetime'] < P2_END)]
p3 = content_df[content_df['datetime'] >= P2_END]

print(f"\nSample sizes: P1={len(p1)}, P2={len(p2)}, P3={len(p3)}")

v1 = framing_score(p1)
v2 = framing_score(p2)
v3 = framing_score(p3)

print(f"\nFraming Score (DIPLOMACY - THREAT):")
print(f"  P1: {v1*100:+.1f}pp (threat-dominant)")
print(f"  P2: {v2*100:+.1f}pp (diplomacy-dominant)")
print(f"  P3: {v3*100:+.1f}pp")
print(f"\n  P1→P2 Change: {(v2-v1)*100:+.1f}pp")
print(f"  P2→P3 Change: {(v3-v2)*100:+.1f}pp")

result = bootstrap_ratio_test(p1, p2, p3, framing_score)
print(f"\nBootstrap Test (n=1000):")
print(f"  Mean ratio: {result['mean_ratio']:.3f}")
print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"  → {'RATCHET SUPPORTED' if result['ci_upper'] < 1.0 else 'CI includes 1.0'}")

content_result = {
    'p1_p2': abs(v2 - v1) * 100,
    'p2_p3': abs(v3 - v2) * 100,
    'ratio': result['mean_ratio'],
    'ci': f"[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]"
}

#=============================================================================
# 2. EDGE FRAMING
#=============================================================================
print("\n" + "="*70)
print("2. EDGE FRAMING (LLM-classified)")
print("="*70)

edge_p1 = pd.read_csv(RESULTS_DIR / "edge_framing_period1.csv")
edge_p2 = pd.read_csv(RESULTS_DIR / "edge_framing_period2.csv")
edge_p3 = pd.read_csv(RESULTS_DIR / "edge_framing_period3.csv")

print(f"\nSample sizes: P1={len(edge_p1)}, P2={len(edge_p2)}, P3={len(edge_p3)}")

v1 = framing_score(edge_p1)
v2 = framing_score(edge_p2)
v3 = framing_score(edge_p3)

print(f"\nFraming Score (DIPLOMACY - THREAT):")
print(f"  P1: {v1*100:+.1f}pp")
print(f"  P2: {v2*100:+.1f}pp")
print(f"  P3: {v3*100:+.1f}pp")
print(f"\n  P1→P2 Change: {(v2-v1)*100:+.1f}pp")
print(f"  P2→P3 Change: {(v3-v2)*100:+.1f}pp")

result = bootstrap_ratio_test(edge_p1, edge_p2, edge_p3, framing_score)
print(f"\nBootstrap Test (n=1000):")
print(f"  Mean ratio: {result['mean_ratio']:.3f}")
print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"  → {'RATCHET SUPPORTED' if result['ci_upper'] < 1.0 else 'CI includes 1.0'}")

edge_result = {
    'p1_p2': abs(v2 - v1) * 100,
    'p2_p3': abs(v3 - v2) * 100,
    'ratio': result['mean_ratio'],
    'ci': f"[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]"
}

#=============================================================================
# 3. COMMUNITY FRAMING
#=============================================================================
print("\n" + "="*70)
print("3. COMMUNITY FRAMING (LLM-classified)")
print("="*70)

comm = pd.read_csv(RESULTS_DIR / "community_framing_llm_results.csv")
comm_p1 = comm[comm['period'] == 'period1'].copy()
comm_p2 = comm[comm['period'] == 'period2'].copy()
comm_p3 = comm[comm['period'] == 'period3'].copy()

print(f"\nSample sizes: P1={len(comm_p1)}, P2={len(comm_p2)}, P3={len(comm_p3)}")

v1 = framing_score(comm_p1)
v2 = framing_score(comm_p2)
v3 = framing_score(comm_p3)

print(f"\nFraming Score (DIPLOMACY - THREAT):")
print(f"  P1: {v1*100:+.1f}pp")
print(f"  P2: {v2*100:+.1f}pp")
print(f"  P3: {v3*100:+.1f}pp")
print(f"\n  P1→P2 Change: {(v2-v1)*100:+.1f}pp")
print(f"  P2→P3 Change: {(v3-v2)*100:+.1f}pp")

result = bootstrap_ratio_test(comm_p1, comm_p2, comm_p3, framing_score)
print(f"\nBootstrap Test (n=1000):")
print(f"  Mean ratio: {result['mean_ratio']:.3f}")
print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"  → {'RATCHET SUPPORTED' if result['ci_upper'] < 1.0 else 'CI includes 1.0'}")

comm_result = {
    'p1_p2': abs(v2 - v1) * 100,
    'p2_p3': abs(v3 - v2) * 100,
    'ratio': result['mean_ratio'],
    'ci': f"[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]"
}

#=============================================================================
# 4. SENTIMENT
#=============================================================================
print("\n" + "="*70)
print("4. SENTIMENT (Post-level)")
print("="*70)

sentiment_df = pd.read_csv(DATA_DIR / "sentiment" / "nk_posts_sentiment.csv")
sentiment_df['datetime'] = pd.to_datetime(sentiment_df['created_utc'], unit='s')

p1 = sentiment_df[sentiment_df['datetime'] < P1_END]
p2 = sentiment_df[(sentiment_df['datetime'] >= P1_END) & (sentiment_df['datetime'] < P2_END)]
p3 = sentiment_df[sentiment_df['datetime'] >= P2_END]

print(f"\nSample sizes: P1={len(p1)}, P2={len(p2)}, P3={len(p3)}")

sent_func = lambda df: df['sentiment_compound'].mean()

v1 = sent_func(p1)
v2 = sent_func(p2)
v3 = sent_func(p3)

print(f"\nMean Sentiment Compound:")
print(f"  P1: {v1:.4f}")
print(f"  P2: {v2:.4f}")
print(f"  P3: {v3:.4f}")
print(f"\n  P1→P2 Change: {(v2-v1):+.4f}")
print(f"  P2→P3 Change: {(v3-v2):+.4f}")

result = bootstrap_ratio_test(p1, p2, p3, sent_func)
print(f"\nBootstrap Test (n=1000):")
print(f"  Mean ratio: {result['mean_ratio']:.3f}")
print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]")
print(f"  → {'RATCHET SUPPORTED' if result['ci_upper'] < 1.0 else 'CI includes 1.0'}")

sent_result = {
    'p1_p2': abs(v2 - v1),
    'p2_p3': abs(v3 - v2),
    'ratio': result['mean_ratio'],
    'ci': f"[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]"
}

#=============================================================================
# SUMMARY TABLE FOR PAPER
#=============================================================================
print("\n" + "="*70)
print("SUMMARY TABLE FOR PAPER")
print("="*70)
print(f"""
Dimension           |P1→P2|     |P2→P3|     Ratio   95% CI
------------------------------------------------------------------
Content Framing     {content_result['p1_p2']:.1f}pp      {content_result['p2_p3']:.1f}pp      {content_result['ratio']:.2f}    {content_result['ci']}
Edge Framing        {edge_result['p1_p2']:.1f}pp      {edge_result['p2_p3']:.1f}pp      {edge_result['ratio']:.2f}    {edge_result['ci']}
Community Framing   {comm_result['p1_p2']:.1f}pp      {comm_result['p2_p3']:.1f}pp      {comm_result['ratio']:.2f}    {comm_result['ci']}
Sentiment           {sent_result['p1_p2']:.3f}       {sent_result['p2_p3']:.3f}       {sent_result['ratio']:.2f}    {sent_result['ci']}

Note: Framing uses combined score (DIPLOMACY% - THREAT%)
      Ratios < 1.0 with 95% CI excluding 1.0 support ratchet effect
""")
