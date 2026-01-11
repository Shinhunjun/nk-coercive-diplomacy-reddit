"""
Apply sentiment analysis to collected comments using RoBERTa.
Processes NK (treatment) and control group (China, Iran, Russia) comments.
"""

import pandas as pd
import numpy as np
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from tqdm import tqdm

# Paths
COMMENT_FILES = {
    'nk': 'data/processed/nk_comments_top3_final.csv',
    'china': 'data/control/china_comments_top3_final.csv',
    'iran': 'data/control/iran_comments_top3_final.csv',
    'russia': 'data/control/russia_comments_top3_final.csv'
}

OUTPUT_SUFFIX = '_sentiment.csv'


class RoBERTaSentimentAnalyzer:
    """Sentiment analyzer using Cardiff NLP's twitter-roberta-base-sentiment-latest."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
    def analyze_text(self, text: str) -> dict:
        """Analyze sentiment of a single text."""
        if pd.isna(text) or len(str(text).strip()) < 3:
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33, 'sentiment_score': 0.0}
        
        text = str(text)[:512]  # Truncate to max length
        
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = softmax(outputs.logits.cpu().numpy()[0])
            
            # Score mapping: [-1, 0, 1] for [negative, neutral, positive]
            sentiment_score = scores[2] - scores[0]  # positive probability - negative probability
            
            return {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2]),
                'sentiment_score': float(sentiment_score)
            }
        except Exception as e:
            return {'negative': 0.33, 'neutral': 0.34, 'positive': 0.33, 'sentiment_score': 0.0}
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'body', batch_size: int = 32):
        """Apply sentiment analysis to entire dataframe."""
        results = []
        
        for idx in tqdm(range(len(df)), desc="Analyzing sentiment"):
            text = df.iloc[idx][text_column]
            result = self.analyze_text(text)
            results.append(result)
        
        # Add results to dataframe
        df_result = df.copy()
        df_result['sentiment_negative'] = [r['negative'] for r in results]
        df_result['sentiment_neutral'] = [r['neutral'] for r in results]
        df_result['sentiment_positive'] = [r['positive'] for r in results]
        df_result['sentiment_score'] = [r['sentiment_score'] for r in results]
        
        return df_result


def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS FOR COLLECTED COMMENTS")
    print("=" * 60)
    
    analyzer = RoBERTaSentimentAnalyzer()
    
    for topic, path in COMMENT_FILES.items():
        print(f"\nProcessing {topic.upper()}...")
        
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            continue
        
        df = pd.read_csv(path, low_memory=False)
        print(f"  Loaded {len(df):,} comments")
        
        # Filter out removed/deleted
        valid = df[~df['body'].astype(str).str.contains(r'\[removed\]|\[deleted\]', case=False, na=False, regex=True)]
        valid = valid[valid['body'].astype(str).str.len() > 10]
        print(f"  Valid comments: {len(valid):,}")
        
        # Analyze sentiment
        result = analyzer.analyze_dataframe(valid, text_column='body')
        
        # Save
        output_path = path.replace('.csv', OUTPUT_SUFFIX)
        result.to_csv(output_path, index=False)
        print(f"  ✅ Saved to: {output_path}")
        
        # Show summary
        mean_score = result['sentiment_score'].mean()
        print(f"  Mean sentiment: {mean_score:.3f}")


if __name__ == "__main__":
    main()
