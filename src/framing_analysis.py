"""
LLM-based Framing Analysis for Reddit Posts

This module classifies posts into framing categories using OpenAI GPT-4o-mini.
Categories: THREAT, DIPLOMACY, NEUTRAL, ECONOMIC, HUMANITARIAN
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from scipy import stats
import json
from tqdm import tqdm
import os

from config import FRAME_CATEGORIES, OPENAI_MODEL, SAMPLE_DIR, RESULTS_DIR


class FramingAnalyzer:
    """LLM-based framing classifier for North Korea-related posts."""

    def __init__(self, api_key: str = None):
        """
        Initialize the framing analyzer.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env variable)
        """
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = OPENAI_MODEL
        self.categories = FRAME_CATEGORIES

    def classify_post(self, title: str, body: str = "") -> dict:
        """
        Classify a single post into a framing category.

        Args:
            title: Post title
            body: Post body text (optional)

        Returns:
            Dictionary with frame, confidence, and reason
        """
        text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"

        prompt = f"""Classify this Reddit post about North Korea into ONE of these frames:
- THREAT: Focus on military danger, nuclear weapons, missiles, war
- DIPLOMACY: Focus on negotiations, talks, peace, cooperation
- NEUTRAL: Factual information without clear framing
- ECONOMIC: Focus on sanctions, trade, economic aspects
- HUMANITARIAN: Focus on human rights, refugees, NK citizens

Post:
{text}

Respond in JSON format:
{{"frame": "CATEGORY", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            return {"frame": "NEUTRAL", "confidence": 0.5, "reason": f"Error: {str(e)}"}

    def analyze_dataframe(self, df: pd.DataFrame, sample_size: int = None) -> pd.DataFrame:
        """
        Classify framing for posts in a DataFrame.

        Args:
            df: DataFrame with 'title' and 'selftext' columns
            sample_size: Number of posts to sample (None for all)

        Returns:
            DataFrame with framing results
        """
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)

        results = []
        print(f"Classifying {len(df)} posts...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Framing Analysis"):
            title = str(row.get('title', ''))
            body = str(row.get('selftext', '')) if pd.notna(row.get('selftext')) else ''

            result = self.classify_post(title, body)
            results.append({
                'frame': result.get('frame', 'NEUTRAL'),
                'confidence': result.get('confidence', 0.5),
                'reason': result.get('reason', '')
            })

        df = df.copy()
        df['frame'] = [r['frame'] for r in results]
        df['frame_confidence'] = [r['confidence'] for r in results]
        df['frame_reason'] = [r['reason'] for r in results]

        return df


def calculate_frame_distribution(df: pd.DataFrame) -> dict:
    """
    Calculate framing distribution from classified posts.

    Args:
        df: DataFrame with 'frame' column

    Returns:
        Dictionary with frame counts and percentages
    """
    total = len(df)
    distribution = {}

    for frame in FRAME_CATEGORIES:
        count = (df['frame'] == frame).sum()
        distribution[frame] = {
            "count": int(count),
            "percentage": float(count / total * 100)
        }

    return distribution


def compare_framing(dist1: dict, dist2: dict) -> dict:
    """
    Compare framing distributions between two periods using chi-square test.

    Args:
        dist1: Framing distribution from period 1
        dist2: Framing distribution from period 2

    Returns:
        Dictionary with comparison results and chi-square test
    """
    # Prepare contingency table
    frames = FRAME_CATEGORIES
    observed1 = [dist1[f]['count'] for f in frames]
    observed2 = [dist2[f]['count'] for f in frames]

    contingency = np.array([observed1, observed2])

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    # Calculate changes
    changes = {}
    for frame in frames:
        change = dist2[frame]['percentage'] - dist1[frame]['percentage']
        changes[frame] = {
            "period1_pct": dist1[frame]['percentage'],
            "period2_pct": dist2[frame]['percentage'],
            "change_pct": change
        }

    return {
        "changes": changes,
        "chi_square": {
            "statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof)
        }
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Framing Analysis Example")
    print("=" * 60)
    print("\nNote: This requires OPENAI_API_KEY environment variable")
    print("Loading pre-computed results from data/results/")

    # Load pre-computed results
    results_path = RESULTS_DIR / "openai_framing_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        print("\n" + "=" * 60)
        print("Pre-computed Framing Results")
        print("=" * 60)
        print(f"\nPeriod 1 (Tension):")
        for frame, count in results['period1']['frame_distribution'].items():
            pct = count / results['period1']['total_valid'] * 100
            print(f"  {frame}: {count} ({pct:.1f}%)")

        print(f"\nPeriod 2 (Diplomacy):")
        for frame, count in results['period2']['frame_distribution'].items():
            pct = count / results['period2']['total_valid'] * 100
            print(f"  {frame}: {count} ({pct:.1f}%)")

        print(f"\nStatistical Significance:")
        print(f"  Chi-square: {results['comparison']['chi2']:.2f}")
        print(f"  p-value: {results['comparison']['p_value']:.6f}")
    else:
        print("\nNo pre-computed results found. Run analysis with API key.")
