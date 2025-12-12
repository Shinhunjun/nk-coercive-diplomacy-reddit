"""
Run OpenAI Framing Analysis on ALL posts (NK + Control Groups)
Processes posts in batches and saves progress incrementally.
"""

import pandas as pd
import os
import sys
import time
import json
from datetime import datetime
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import FRAME_CATEGORIES

# Load .env file
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class PostsFramingAnalyzer:
    """Analyze framing for all posts using OpenAI GPT-4o-mini."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.categories = FRAME_CATEGORIES

    def classify_post(self, title: str, body: str = "") -> dict:
        """Classify a single post."""
        text = f"Title: {title}\nBody: {body[:500] if body else 'N/A'}"

        prompt = f"""Classify this Reddit post into ONE of these 5 framing categories:
- THREAT: Military threat, nuclear weapons, missiles, war risk
- DIPLOMACY: Negotiation, dialogue, peace, cooperation
- NEUTRAL: Neutral information delivery
- ECONOMIC: Economic sanctions, trade aspects
- HUMANITARIAN: Human rights, refugees, civilian issues

Post:
{text}

Respond in JSON format:
{{"frame": "CATEGORY", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a political science researcher analyzing media framing."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content.strip())

            if result.get('frame') not in self.categories:
                result['frame'] = 'NEUTRAL'
                result['confidence'] = 0.5

            return result

        except Exception as e:
            return {
                "frame": "NEUTRAL",
                "confidence": 0.5,
                "reason": f"Error: {str(e)}"
            }

    def analyze_topic(
        self,
        topic: str,
        input_path: str,
        output_dir: str = "data/framing",
        batch_size: int = 100,
        delay: float = 0.05
    ) -> pd.DataFrame:
        """Analyze all posts for a topic with progress saving."""

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{topic}_posts_framed.csv")
        progress_path = os.path.join(output_dir, f"{topic}_progress.json")

        # Load posts
        df = pd.read_csv(input_path)
        print(f"\n{'='*60}")
        print(f"FRAMING ANALYSIS: {topic.upper()}")
        print(f"{'='*60}")
        print(f"Total posts: {len(df):,}")

        # Check for existing progress
        start_idx = 0
        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                start_idx = progress.get('last_processed', 0)
                print(f"Resuming from index {start_idx}")

        if os.path.exists(output_path) and start_idx > 0:
            df_existing = pd.read_csv(output_path)
            results = df_existing.to_dict('records')[:start_idx]
        else:
            results = []

        # Process remaining posts
        pbar = tqdm(range(start_idx, len(df)), desc=f"Analyzing {topic}")

        for idx in pbar:
            row = df.iloc[idx]
            title = str(row.get('title', ''))
            body = str(row.get('selftext', '')) if pd.notna(row.get('selftext')) else ''

            result = self.classify_post(title, body)

            # Add framing to row
            row_dict = row.to_dict()
            row_dict['frame'] = result.get('frame', 'NEUTRAL')
            row_dict['frame_confidence'] = result.get('confidence', 0.5)
            row_dict['frame_reason'] = result.get('reason', '')
            results.append(row_dict)

            # Save progress every batch_size
            if (idx + 1) % batch_size == 0:
                pd.DataFrame(results).to_csv(output_path, index=False)
                with open(progress_path, 'w') as f:
                    json.dump({'last_processed': idx + 1, 'timestamp': datetime.now().isoformat()}, f)
                pbar.set_postfix({'saved': idx + 1})

            time.sleep(delay)

        # Final save
        df_result = pd.DataFrame(results)
        df_result.to_csv(output_path, index=False)

        # Remove progress file on completion
        if os.path.exists(progress_path):
            os.remove(progress_path)

        # Print summary
        print(f"\n{topic.upper()} Framing Distribution:")
        print(df_result['frame'].value_counts())
        print(f"\nSaved to: {output_path}")

        return df_result


def main():
    """Run framing analysis on all topics."""

    api_key = OPENAI_API_KEY
    if not api_key:
        print("Error: OPENAI_API_KEY not set!")
        return

    analyzer = PostsFramingAnalyzer(api_key)

    # Define topics and paths
    topics = {
        'nk': 'data/nk/nk_posts_merged.csv',
        'iran': 'data/control/iran_posts_merged.csv',
        'russia': 'data/control/russia_posts_merged.csv',
        'china': 'data/control/china_posts_merged.csv'
    }

    results = {}

    for topic, path in topics.items():
        if os.path.exists(path):
            df = analyzer.analyze_topic(topic, path)
            results[topic] = {
                'total': len(df),
                'distribution': df['frame'].value_counts().to_dict()
            }
        else:
            print(f"Warning: {path} not found, skipping {topic}")

    # Print final summary
    print("\n" + "=" * 60)
    print("FRAMING ANALYSIS SUMMARY")
    print("=" * 60)

    for topic, data in results.items():
        print(f"\n{topic.upper()}: {data['total']:,} posts")
        for frame, count in data['distribution'].items():
            pct = count / data['total'] * 100
            print(f"  {frame}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("All framing analysis complete!")


if __name__ == '__main__':
    main()
