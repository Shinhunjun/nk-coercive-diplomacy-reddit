"""
Apply RoBERTa sentiment analysis to merged posts (NK + Control groups)
Uses twitter-roberta-base-sentiment-latest for consistent analysis
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
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        self.labels = ['negative', 'neutral', 'positive']

    def analyze_text(self, text: str) -> dict:
        if pd.isna(text) or str(text).strip() == '':
            return {
                'label': 'neutral',
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33,
                'compound': 0.0
            }

        try:
            encoded = self.tokenizer(
                str(text),
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                output = self.model(**encoded)
                scores = output.logits[0].cpu().numpy()
                scores = softmax(scores)

            sentiment_dict = {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2])
            }

            label_idx = np.argmax(scores)
            sentiment_dict['label'] = self.labels[label_idx]
            sentiment_dict['compound'] = float(scores[2] - scores[0])

            return sentiment_dict

        except Exception as e:
            return {
                'label': 'neutral',
                'negative': 0.33,
                'neutral': 0.34,
                'positive': 0.33,
                'compound': 0.0
            }

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        print(f"\nAnalyzing {len(df)} texts...")

        results = []
        for idx in tqdm(range(len(df)), desc="Sentiment Analysis"):
            text = df.iloc[idx][text_column]
            sentiment = self.analyze_text(text)
            results.append(sentiment)

        df = df.copy()
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]
        df['sentiment_score'] = df['sentiment_compound']

        print(f"\nSentiment Analysis Complete:")
        print(f"  Mean sentiment: {df['sentiment_score'].mean():.4f}")
        print(f"  Std sentiment: {df['sentiment_score'].std():.4f}")
        print(f"\nLabel distribution:")
        print(df['sentiment_label'].value_counts())

        return df


def analyze_merged_posts(topic: str, input_path: str, output_dir: str = "data/sentiment"):
    """Analyze merged posts for a topic."""
    print("=" * 60)
    print(f"SENTIMENT ANALYSIS: {topic.upper()}")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{topic}_posts_sentiment.csv")

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} {topic} posts")

    # Prepare text (title + body)
    df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')

    # Apply RoBERTa sentiment
    analyzer = RoBERTaSentimentAnalyzer()
    df = analyzer.analyze_dataframe(df, text_column='text')

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df


def create_monthly_aggregation(df: pd.DataFrame, topic_name: str, output_dir: str = "data/sentiment") -> pd.DataFrame:
    """Aggregate posts to monthly level for DID analysis."""
    print(f"\nAggregating {topic_name} to monthly level...")

    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['datetime'].dt.to_period('M').astype(str)

    # Filter to analysis period (2017-01 to 2019-06)
    df_filtered = df[(df['month'] >= '2017-01') & (df['month'] <= '2019-06')].copy()
    print(f"Posts in analysis period (2017-01 to 2019-06): {len(df_filtered)}")

    # Aggregate
    monthly = df_filtered.groupby('month').agg({
        'sentiment_score': ['mean', 'std', 'count']
    }).reset_index()
    monthly.columns = ['month', 'sentiment_mean', 'sentiment_std', 'post_count']

    # Fill missing months
    all_months = pd.period_range('2017-01', '2019-06', freq='M').astype(str)
    monthly_complete = pd.DataFrame({'month': all_months})
    monthly_complete = monthly_complete.merge(monthly, on='month', how='left')
    monthly_complete['topic'] = topic_name

    # Intervention indicator (March 2018)
    monthly_complete['post_intervention'] = (monthly_complete['month'] >= '2018-03').astype(int)

    output_path = os.path.join(output_dir, f"{topic_name}_monthly_sentiment.csv")
    monthly_complete.to_csv(output_path, index=False)

    print(f"Saved to: {output_path}")
    print(f"  Total months: {len(monthly_complete)}")
    print(f"  Months with data: {monthly_complete['post_count'].notna().sum()}")
    if monthly_complete['sentiment_mean'].notna().any():
        print(f"  Mean sentiment: {monthly_complete['sentiment_mean'].mean():.4f}")
        print(f"  Total posts: {monthly_complete['post_count'].sum():.0f}")

    return monthly_complete


def main():
    """Apply sentiment analysis to all merged posts."""

    topics = {
        'nk': 'data/nk/nk_posts_merged.csv',
        'iran': 'data/control/iran_posts_merged.csv',
        'russia': 'data/control/russia_posts_merged.csv',
        'china': 'data/control/china_posts_merged.csv'
    }

    results = {}
    monthly_dfs = []

    for topic, path in topics.items():
        if os.path.exists(path):
            print("\n" + "=" * 60)
            df = analyze_merged_posts(topic, path)
            monthly = create_monthly_aggregation(df, topic)
            monthly_dfs.append(monthly)
            results[topic] = {
                'total': len(df),
                'mean_sentiment': df['sentiment_score'].mean(),
                'std_sentiment': df['sentiment_score'].std()
            }
        else:
            print(f"Warning: {path} not found, skipping {topic}")

    # Combine monthly data for DID
    if monthly_dfs:
        combined = pd.concat(monthly_dfs, ignore_index=True)
        combined['treated'] = (combined['topic'] == 'nk').astype(int)
        combined.to_csv('data/sentiment/combined_monthly_did.csv', index=False)
        print(f"\nCombined DID data saved to: data/sentiment/combined_monthly_did.csv")

    # Summary
    print("\n" + "=" * 60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)
    for topic, data in results.items():
        print(f"\n{topic.upper()}: {data['total']:,} posts")
        print(f"  Mean sentiment: {data['mean_sentiment']:.4f}")
        print(f"  Std sentiment: {data['std_sentiment']:.4f}")

    print("\n" + "=" * 60)
    print("All sentiment analysis complete!")
    print("Next: Run DID analysis with data/sentiment/combined_monthly_did.csv")


if __name__ == '__main__':
    main()
