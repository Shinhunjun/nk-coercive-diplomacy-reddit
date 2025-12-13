"""
Create final unified datasets for analysis.
Combines existing full data (2017-2019.06) with extended data (2019.07-2019.12).
Applies sentiment analysis to all posts.
"""

import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RoBERTaSentimentAnalyzer:
    """Sentiment analyzer using Cardiff NLP's twitter-roberta-base-sentiment-latest."""

    def __init__(self):
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        self.labels = ['negative', 'neutral', 'positive']

    def analyze_batch(self, texts, batch_size=32):
        """Analyze a batch of texts."""
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
            batch = texts[i:i+batch_size]
            batch = [t if pd.notna(t) and t.strip() else "neutral" for t in batch]
            
            encoded = self.tokenizer(batch, truncation=True, max_length=512, padding=True, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                output = self.model(**encoded)
                scores = softmax(output.logits.cpu().numpy(), axis=1)
            
            for score in scores:
                results.append({
                    'negative': float(score[0]),
                    'neutral': float(score[1]),
                    'positive': float(score[2]),
                    'label': self.labels[np.argmax(score)],
                    'compound': float(score[2] - score[0])
                })
        return results


def assign_period(month):
    """Assign period based on month string."""
    if month <= '2018-05':
        return 'P1_PreSingapore'
    elif month <= '2019-02':
        return 'P2_SingaporeHanoi'
    else:
        return 'P3_PostHanoi'


def create_final_dataset(name, existing_path, extended_path, output_path, analyzer):
    """Create final unified dataset for one group."""
    print(f"\n{'='*60}")
    print(f"CREATING FINAL DATASET: {name.upper()}")
    print(f"{'='*60}")
    
    # Load existing data
    df_existing = pd.read_csv(existing_path)
    print(f"Existing: {len(df_existing)} posts")
    
    # Load extended data
    df_extended = pd.read_csv(extended_path)
    print(f"Extended: {len(df_extended)} posts")
    
    # Combine
    df = pd.concat([df_existing, df_extended], ignore_index=True)
    print(f"Combined: {len(df)} posts")
    
    # Remove duplicates by id
    df = df.drop_duplicates(subset=['id'], keep='first')
    print(f"After dedup: {len(df)} posts")
    
    # Parse datetime and add period
    df['datetime'] = pd.to_datetime(df['created_utc'], unit='s')
    df['month'] = df['datetime'].dt.strftime('%Y-%m')
    df['period'] = df['month'].apply(assign_period)
    df['topic'] = name.lower()
    
    # Filter to analysis period (2017-01 to 2019-12)
    df = df[(df['month'] >= '2017-01') & (df['month'] <= '2019-12')].copy()
    print(f"In analysis period (2017-2019): {len(df)} posts")
    
    # Check if sentiment already exists
    if 'sentiment_score' not in df.columns or df['sentiment_score'].isna().sum() > len(df) * 0.5:
        # Apply sentiment analysis
        df['text'] = df['title'].fillna('') + ' ' + df['selftext'].fillna('')
        results = analyzer.analyze_batch(df['text'].tolist())
        
        df['sentiment_negative'] = [r['negative'] for r in results]
        df['sentiment_neutral'] = [r['neutral'] for r in results]
        df['sentiment_positive'] = [r['positive'] for r in results]
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment_compound'] = [r['compound'] for r in results]
        df['sentiment_score'] = df['sentiment_compound']
    else:
        print("Sentiment already exists, skipping...")
    
    # Period statistics
    print(f"\nPeriod distribution:")
    for period in ['P1_PreSingapore', 'P2_SingaporeHanoi', 'P3_PostHanoi']:
        count = len(df[df['period'] == period])
        mean_sent = df[df['period'] == period]['sentiment_score'].mean()
        print(f"  {period}: {count} posts, mean sentiment: {mean_sent:.4f}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved: {output_path}")
    
    return df


def main():
    print("=" * 70)
    print("CREATING FINAL UNIFIED DATASETS")
    print("=" * 70)
    
    # Initialize sentiment analyzer
    analyzer = RoBERTaSentimentAnalyzer()
    
    # Define datasets to create
    datasets = [
        {
            'name': 'NK',
            'existing': 'data/processed/nk_posts_roberta_sentiment.csv',
            'extended': 'data/sentiment/nk_posts_hanoi_extended_sentiment.csv',
            'output': 'data/final/nk_final.csv'
        },
        {
            'name': 'China',
            'existing': 'data/control/china_posts_full.csv',
            'extended': 'data/control/china_posts_hanoi_extended.csv',
            'output': 'data/final/china_final.csv'
        },
        {
            'name': 'Iran',
            'existing': 'data/control/iran_posts_full.csv',
            'extended': 'data/control/iran_posts_hanoi_extended.csv',
            'output': 'data/final/iran_final.csv'
        },
        {
            'name': 'Russia',
            'existing': 'data/control/russia_posts_full.csv',
            'extended': 'data/control/russia_posts_hanoi_extended.csv',
            'output': 'data/final/russia_final.csv'
        }
    ]
    
    results = {}
    for ds in datasets:
        df = create_final_dataset(ds['name'], ds['existing'], ds['extended'], ds['output'], analyzer)
        results[ds['name']] = len(df)
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL DATASETS CREATED")
    print("=" * 70)
    for name, count in results.items():
        print(f"  {name}: {count:,} posts")
    
    print("\n✓ All final datasets created in data/final/")


if __name__ == '__main__':
    main()
