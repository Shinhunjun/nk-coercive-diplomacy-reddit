"""
Apply RoBERTa sentiment analysis to extended data (July-Dec 2019)
All 4 groups: NK, China, Iran, Russia
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        self.labels = ['negative', 'neutral', 'positive']

    def analyze_text(self, text: str) -> dict:
        if pd.isna(text) or text.strip() == '':
            return {'label': 'neutral', 'negative': 0.33, 'neutral': 0.34, 'positive': 0.33, 'compound': 0.0}

        encoded = self.tokenizer(text, truncation=True, max_length=512, return_tensors='pt')
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            output = self.model(**encoded)
            scores = output.logits[0].cpu().numpy()
            scores = softmax(scores)

        sentiment_dict = {
            'negative': float(scores[0]),
            'neutral': float(scores[1]),
            'positive': float(scores[2]),
            'label': self.labels[np.argmax(scores)],
            'compound': float(scores[2] - scores[0])
        }
        return sentiment_dict

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        print(f"\nAnalyzing {len(df)} texts...")
        results = []
        for idx in tqdm(range(len(df)), desc="Sentiment"):
            text = df.iloc[idx][text_column]
            sentiment = self.analyze_text(text)
            results.append(sentiment)

        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]
        df['sentiment_score'] = df['sentiment_compound']

        print(f"\nMean sentiment: {df['sentiment_score'].mean():.4f}")
        print(f"Label distribution: {df['sentiment_label'].value_counts().to_dict()}")
        return df


def analyze_extended_data(input_path: str, output_path: str, topic: str):
    """Analyze extended data for one topic."""
    print(f"\n{'='*60}")
    print(f"SENTIMENT ANALYSIS: {topic.upper()}")
    print(f"{'='*60}")
    
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} posts from {input_path}")
    
    # Prepare text
    df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
    
    # Initialize analyzer (only once per session ideally)
    analyzer = RoBERTaSentimentAnalyzer()
    df = analyzer.analyze_dataframe(df, text_column='text')
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    return df


def main():
    print("=" * 70)
    print("EXTENDED DATA SENTIMENT ANALYSIS (July-Dec 2019)")
    print("=" * 70)
    
    # Files to process
    files = [
        ('data/nk/nk_posts_hanoi_extended.csv', 'data/sentiment/nk_posts_hanoi_extended_sentiment.csv', 'NK'),
        ('data/control/china_posts_hanoi_extended.csv', 'data/sentiment/china_posts_hanoi_extended_sentiment.csv', 'China'),
        ('data/control/iran_posts_hanoi_extended.csv', 'data/sentiment/iran_posts_hanoi_extended_sentiment.csv', 'Iran'),
        ('data/control/russia_posts_hanoi_extended.csv', 'data/sentiment/russia_posts_hanoi_extended_sentiment.csv', 'Russia'),
    ]
    
    results = {}
    for input_path, output_path, topic in files:
        if os.path.exists(input_path):
            df = analyze_extended_data(input_path, output_path, topic)
            results[topic] = {
                'count': len(df),
                'mean_sentiment': df['sentiment_score'].mean(),
                'std_sentiment': df['sentiment_score'].std()
            }
        else:
            print(f"File not found: {input_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for topic, stats in results.items():
        print(f"{topic}: {stats['count']} posts, Mean sentiment: {stats['mean_sentiment']:.4f}")
    
    print("\n✓ All sentiment analysis complete!")


if __name__ == '__main__':
    main()
