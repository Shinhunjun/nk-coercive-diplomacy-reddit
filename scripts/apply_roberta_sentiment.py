"""
Apply twitter-roberta-base-sentiment-latest to posts
Ensures consistent sentiment analysis across treatment (NK) and control groups (Iran/Russia/China)
"""

import pandas as pd
import numpy as np
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RoBERTaSentimentAnalyzer:
    """Sentiment analyzer using Cardiff NLP's twitter-roberta-base-sentiment-latest."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize RoBERTa sentiment analyzer.

        Args:
            model_name: HuggingFace model identifier
        """
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        # Check if CUDA available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Labels: 0=Negative, 1=Neutral, 2=Positive
        self.labels = ['negative', 'neutral', 'positive']

    def analyze_text(self, text: str) -> dict:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text

        Returns:
            dict with sentiment scores
        """
        if pd.isna(text) or text.strip() == '':
            return {
                'label': 'neutral',
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33,
                'compound': 0.0
            }

        # Truncate to 512 tokens (RoBERTa limit)
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # Get predictions
        with torch.no_grad():
            output = self.model(**encoded)
            scores = output.logits[0].cpu().numpy()
            scores = softmax(scores)

        # Map to sentiment dict
        sentiment_dict = {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2])
        }

        # Determine label
        label_idx = np.argmax(scores)
        sentiment_dict['label'] = self.labels[label_idx]

        # Compute compound score (-1 to +1)
        # Formula: positive - negative
        sentiment_dict['compound'] = float(scores[2] - scores[0])

        return sentiment_dict

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Apply sentiment analysis to entire dataframe.

        Args:
            df: Input dataframe
            text_column: Column containing text to analyze
            batch_size: Batch size for processing

        Returns:
            DataFrame with sentiment columns added
        """
        print(f"\nAnalyzing {len(df)} texts...")

        results = []
        for idx in tqdm(range(len(df)), desc="Sentiment Analysis"):
            text = df.iloc[idx][text_column]
            sentiment = self.analyze_text(text)
            results.append(sentiment)

        # Add sentiment columns
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]

        # Primary sentiment score for analysis (-1 to +1)
        df['sentiment_score'] = df['sentiment_compound']

        print(f"\nSentiment Analysis Complete:")
        print(f"  Mean sentiment: {df['sentiment_score'].mean():.4f}")
        print(f"  Std sentiment: {df['sentiment_score'].std():.4f}")
        print(f"\nLabel distribution:")
        print(df['sentiment_label'].value_counts())

        return df


def reanalyze_nk_data():
    """Re-analyze NK data with RoBERTa sentiment model."""
    print("=" * 60)
    print("RE-ANALYZING NK DATA WITH TWITTER-ROBERTA-BASE-SENTIMENT-LATEST")
    print("=" * 60)

    # Load original NK data
    nk_full_path = '/Users/hunjunsin/Desktop/Jun/reddit_US_NK/data/processed/posts_final_bert_sentiment.csv'
    df = pd.read_csv(nk_full_path)

    print(f"\nLoaded {len(df)} NK posts")

    # Prepare text
    df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')

    # Apply RoBERTa sentiment
    analyzer = RoBERTaSentimentAnalyzer()
    df = analyzer.analyze_dataframe(df, text_column='text')

    # Save with RoBERTa sentiment
    output_path = 'data/processed/nk_posts_roberta_sentiment.csv'
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")

    return df


def reanalyze_control_data(control_name: str):
    """
    Re-analyze control group data with RoBERTa sentiment model.

    Args:
        control_name: 'iran', 'russia', or 'china'
    """
    print("=" * 60)
    print(f"RE-ANALYZING {control_name.upper()} DATA WITH TWITTER-ROBERTA-BASE-SENTIMENT-LATEST")
    print("=" * 60)

    # Load control data
    input_path = f'data/control/{control_name}_posts.csv'
    df = pd.read_csv(input_path)

    print(f"\nLoaded {len(df)} {control_name} posts")

    # Prepare text
    df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')

    # Apply RoBERTa sentiment
    analyzer = RoBERTaSentimentAnalyzer()
    df = analyzer.analyze_dataframe(df, text_column='text')

    # Save with RoBERTa sentiment
    output_path = f'data/control/{control_name}_posts_roberta.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")

    return df


def create_monthly_aggregation(df: pd.DataFrame, topic_name: str) -> pd.DataFrame:
    """
    Aggregate posts to monthly level.

    Args:
        df: Posts dataframe with sentiment_score column
        topic_name: 'northkorea', 'iran', 'russia', or 'china'

    Returns:
        Monthly aggregated dataframe
    """
    print(f"\nAggregating {topic_name} to monthly level...")

    # Convert created_utc to datetime
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['datetime'].dt.to_period('M').astype(str)

    # Filter to analysis period (2017-01 to 2019-06)
    df_filtered = df[(df['month'] >= '2017-01') & (df['month'] <= '2019-06')].copy()

    print(f"Posts in analysis period (2017-01 to 2019-06): {len(df_filtered)}")

    # Aggregate to monthly level
    monthly = df_filtered.groupby('month').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()

    monthly.columns = ['month', 'sentiment_mean', 'sentiment_std', 'post_count']

    # Fill missing months in the full time range (2017-01 to 2019-06)
    all_months = pd.period_range('2017-01', '2019-06', freq='M').astype(str)
    monthly_complete = pd.DataFrame({'month': all_months})
    monthly_complete = monthly_complete.merge(monthly, on='month', how='left')

    # Add topic identifier
    monthly_complete['topic'] = topic_name

    # Save to processed directory
    output_path = f'data/processed/{topic_name}_monthly_roberta.csv'
    monthly_complete.to_csv(output_path, index=False)

    print(f"✓ Saved to: {output_path}")
    print(f"\nMonthly Summary:")
    print(f"  Total months: {len(monthly_complete)}")
    print(f"  Months with data: {monthly_complete['post_count'].notna().sum()}")
    print(f"  Mean sentiment: {monthly_complete['sentiment_mean'].mean():.4f}")
    print(f"  Total posts used: {monthly_complete['post_count'].sum():.0f}")

    return monthly_complete


def main():
    """Re-analyze all data with twitter-roberta-base-sentiment-latest."""

    # 1. Re-analyze NK data
    print("\n" + "=" * 60)
    print("STEP 1: RE-ANALYZE NK DATA")
    print("=" * 60)
    nk_df = reanalyze_nk_data()
    nk_monthly = create_monthly_aggregation(nk_df, 'nk')

    # 2. Re-analyze control groups
    for control_name in ['iran', 'russia', 'china']:
        print("\n" + "=" * 60)
        print(f"STEP 2: RE-ANALYZE {control_name.upper()} DATA")
        print("=" * 60)
        control_df = reanalyze_control_data(control_name)
        control_monthly = create_monthly_aggregation(control_df, control_name)

    print("\n" + "=" * 60)
    print("✓ ALL DATA RE-ANALYZED WITH TWITTER-ROBERTA-BASE-SENTIMENT-LATEST")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run parallel trends test with new sentiment scores")
    print("2. Re-estimate DID with consistent sentiment measures")
    print("3. Compare with previous results")


if __name__ == '__main__':
    main()
