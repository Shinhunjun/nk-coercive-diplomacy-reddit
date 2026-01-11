
import pandas as pd
import numpy as np
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
from tqdm import tqdm

import argparse

# Configuration
INPUT_FILE_DEFAULT = 'data/processed/nk_comments_recursive.csv'
OUTPUT_FILE_DEFAULT = 'data/processed/nk_comments_recursive_roberta.csv'
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32

class RoBERTaSentimentAnalyzer:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        self.labels = ['negative', 'neutral', 'positive']

    def analyze_batch(self, texts):
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.detach().cpu().numpy()
            scores = softmax(scores, axis=1)
            
        return scores

def main():
    parser = argparse.ArgumentParser(description='Run RoBERTa sentiment analysis on recursive comments.')
    parser.add_argument('--input', type=str, default=INPUT_FILE_DEFAULT, help='Input CSV file path')
    parser.add_argument('--output', type=str, default=OUTPUT_FILE_DEFAULT, help='Output CSV file path')
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output

    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    print(f"Reading {input_file}...")
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Total comments: {len(df)}")
    
    # Handle missing bodies
    df['body'] = df['body'].fillna('')
    
    analyzer = RoBERTaSentimentAnalyzer()
    
    # Prepare result columns
    df['roberta_neg'] = 0.0
    df['roberta_neu'] = 0.0
    df['roberta_pos'] = 0.0
    df['roberta_compound'] = 0.0
    df['roberta_label'] = ''

    # Process in chunks to save progress
    # But batching for inference is better done inside loop
    
    texts = df['body'].tolist()
    results = []
    
    print("Running Sentiment Analysis...")
    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i:i+BATCH_SIZE]
        if not batch_texts:
            continue
            
        scores = analyzer.analyze_batch(batch_texts)
        
        for score in scores:
            neg, neu, pos = score
            compound = pos - neg
            label_idx = np.argmax(score)
            label = analyzer.labels[label_idx]
            
            results.append({
                'roberta_neg': neg,
                'roberta_neu': neu,
                'roberta_pos': pos,
                'roberta_compound': compound,
                'roberta_label': label
            })
            
    # Assign back to DF
    result_df = pd.DataFrame(results)
    df['roberta_neg'] = result_df['roberta_neg']
    df['roberta_neu'] = result_df['roberta_neu']
    df['roberta_pos'] = result_df['roberta_pos']
    df['roberta_compound'] = result_df['roberta_compound']
    df['roberta_label'] = result_df['roberta_label']
    
    print(f"Saving to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
