"""
BERT-based Sentiment Analysis for Reddit Posts

This module provides sentiment analysis using DistilBERT model.
Sentiment scores range from -1 (very negative) to +1 (very positive).
"""

import pandas as pd
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import SENTIMENT_MODEL, SAMPLE_DIR, RESULTS_DIR


class SentimentAnalyzer:
    """BERT-based sentiment analyzer for text data."""

    def __init__(self, model_name: str = SENTIMENT_MODEL):
        """
        Initialize the sentiment analyzer.

        Args:
            model_name: HuggingFace model name for sentiment analysis
        """
        print(f"Loading sentiment model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=True,
            max_length=512
        )
        print("Model loaded successfully!")

    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text to analyze

        Returns:
            Sentiment score from -1 to +1
        """
        if pd.isna(text) or not str(text).strip():
            return 0.0

        try:
            result = self.pipeline(str(text)[:512])[0]
            # Convert 1-5 star rating to -1 to +1 scale
            star = int(result['label'].split()[0])
            return (star - 3) / 2
        except Exception as e:
            return 0.0

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment for all rows in a DataFrame.

        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text

        Returns:
            DataFrame with added 'sentiment_score' column
        """
        df = df.copy()

        print(f"Analyzing sentiment for {len(df)} texts...")
        sentiments = []

        for text in tqdm(df[text_column], desc="Sentiment Analysis"):
            sentiments.append(self.analyze_text(text))

        df['sentiment_score'] = sentiments
        return df


def compare_periods(period1_scores: np.ndarray, period2_scores: np.ndarray) -> dict:
    """
    Compare sentiment scores between two periods using statistical tests.

    Args:
        period1_scores: Sentiment scores from tension period
        period2_scores: Sentiment scores from diplomacy period

    Returns:
        Dictionary containing statistical test results
    """
    # T-test
    t_stat, t_pvalue = stats.ttest_ind(period1_scores, period2_scores)

    # Mann-Whitney U test (non-parametric)
    u_stat, u_pvalue = stats.mannwhitneyu(period1_scores, period2_scores, alternative='two-sided')

    # Cohen's d (effect size)
    pooled_std = np.sqrt(
        ((len(period1_scores) - 1) * np.std(period1_scores, ddof=1)**2 +
         (len(period2_scores) - 1) * np.std(period2_scores, ddof=1)**2) /
        (len(period1_scores) + len(period2_scores) - 2)
    )
    cohens_d = (np.mean(period2_scores) - np.mean(period1_scores)) / pooled_std

    return {
        "period1": {
            "mean": float(np.mean(period1_scores)),
            "std": float(np.std(period1_scores)),
            "n": len(period1_scores)
        },
        "period2": {
            "mean": float(np.mean(period2_scores)),
            "std": float(np.std(period2_scores)),
            "n": len(period2_scores)
        },
        "change": float(np.mean(period2_scores) - np.mean(period1_scores)),
        "t_test": {
            "statistic": float(t_stat),
            "p_value": float(t_pvalue)
        },
        "mann_whitney": {
            "statistic": float(u_stat),
            "p_value": float(u_pvalue)
        },
        "effect_size": {
            "cohens_d": float(cohens_d),
            "interpretation": interpret_cohens_d(cohens_d)
        }
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


if __name__ == "__main__":
    # Example usage
    import json

    print("=" * 60)
    print("Sentiment Analysis Example")
    print("=" * 60)

    # Load sample data
    p1 = pd.read_csv(SAMPLE_DIR / "posts_period1_sample.csv")
    p2 = pd.read_csv(SAMPLE_DIR / "posts_period2_sample.csv")

    # Initialize analyzer
    analyzer = SentimentAnalyzer()

    # Analyze (using title + selftext as text)
    p1['text'] = p1['title'].fillna('') + ' ' + p1['selftext'].fillna('')
    p2['text'] = p2['title'].fillna('') + ' ' + p2['selftext'].fillna('')

    p1 = analyzer.analyze_dataframe(p1, 'text')
    p2 = analyzer.analyze_dataframe(p2, 'text')

    # Compare periods
    results = compare_periods(
        p1['sentiment_score'].values,
        p2['sentiment_score'].values
    )

    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Tension Period Mean:   {results['period1']['mean']:.3f}")
    print(f"Diplomacy Period Mean: {results['period2']['mean']:.3f}")
    print(f"Change:                {results['change']:+.3f}")
    print(f"T-test p-value:        {results['t_test']['p_value']:.6f}")
    print(f"Cohen's d:             {results['effect_size']['cohens_d']:.3f} ({results['effect_size']['interpretation']})")
