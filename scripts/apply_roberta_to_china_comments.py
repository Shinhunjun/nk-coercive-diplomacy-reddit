"""
Apply RoBERTa sentiment analysis to China comments
Uses twitter-roberta-base-sentiment-latest for consistency
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
        """Initialize RoBERTa sentiment analyzer."""
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
        """Analyze sentiment of a single text."""
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
        sentiment_dict['compound'] = float(scores[2] - scores[0])

        return sentiment_dict

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'body',
        batch_size: int = 32
    ) -> pd.DataFrame:
        """Apply sentiment analysis to entire dataframe."""
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


def main():
    """Apply RoBERTa sentiment to China comments."""
    print("=" * 60)
    print("APPLYING ROBERTA SENTIMENT TO CHINA COMMENTS")
    print("=" * 60)

    # Load China comments
    comments_path = 'data/control/china_comments.csv'
    df = pd.read_csv(comments_path)

    print(f"\nLoaded {len(df)} China comments")

    # Apply RoBERTa sentiment
    analyzer = RoBERTaSentimentAnalyzer()
    df = analyzer.analyze_dataframe(df, text_column='body')

    # Save with RoBERTa sentiment
    output_path = 'data/control/china_comments_roberta.csv'
    df.to_csv(output_path, index=False)

    print(f"\n✓ Saved to: {output_path}")

    # Show analysis period stats
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df_filtered = df[(df['datetime'] >= '2017-01-01') & (df['datetime'] <= '2019-06-30')]

    print(f"\nAnalysis period (2017-2019) stats:")
    print(f"  Comments: {len(df_filtered)}")
    print(f"  Mean sentiment: {df_filtered['sentiment_score'].mean():.4f}")
    print(f"  Std sentiment: {df_filtered['sentiment_score'].std():.4f}")

    return df


if __name__ == '__main__':
    main()
