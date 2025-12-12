"""
Collect China posts for extended period (July-December 2019)
For Hanoi Summit 3-period DID analysis (Control Group)

Period: 2019-07-01 to 2019-12-31
"""

import pandas as pd
import requests
import time
from datetime import datetime
from typing import List, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# China keywords (from config.py + expanded)
CHINA_KEYWORDS = [
    # Core terms
    "china", "chinese", "beijing", "xi jinping",
    
    # Leaders
    "li keqiang", "wang yi",
    
    # Politics
    "china trade", "one china", "taiwan china",
    "south china sea", "china policy",
    
    # Military
    "PLA", "china military", "china navy", "china missile",
    
    # Economy
    "belt and road", "china economy", "china manufacturing",
    
    # Trade war (relevant for 2019)
    "trade war", "tariff china", "huawei"
]

SUBREDDITS = ['worldnews', 'geopolitics', 'news', 'politics', 'NeutralPolitics']


class ChinaExtendedCollector:
    """Collect China posts for July-December 2019 period."""

    def __init__(self):
        self.base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        self.session = requests.Session()

        # Extended period (July-December 2019)
        self.start_date = "2019-07-01"
        self.end_date = "2019-12-31"

    def search_posts(
        self,
        subreddit: str,
        keyword: str,
        start_date: str,
        end_date: str,
        limit: int = 100
    ) -> List[Dict]:
        """Search posts from Arctic Shift API."""

        params = {
            'subreddit': subreddit,
            'title': keyword,
            'after': start_date,
            'before': end_date,
            'limit': limit,
            'sort': 'desc'
        }

        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            posts = []
            for post in data.get('data', []):
                posts.append({
                    'id': post.get('id', ''),
                    'title': post.get('title', ''),
                    'selftext': post.get('selftext', ''),
                    'author': post.get('author', '[deleted]'),
                    'subreddit': post.get('subreddit', ''),
                    'score': post.get('score', 0),
                    'num_comments': post.get('num_comments', 0),
                    'created_utc': post.get('created_utc', 0),
                    'permalink': post.get('permalink', ''),
                    'url': post.get('url', '')
                })
            return posts

        except Exception as e:
            print(f"      Error: {e}")
            return []

    def collect_all(self) -> pd.DataFrame:
        """Collect all posts for the extended period."""

        all_posts = []
        seen_ids = set()

        print(f"\n--- CHINA EXTENDED PERIOD ({self.start_date} to {self.end_date}) ---")
        print(f"Total keywords: {len(CHINA_KEYWORDS)}")

        for subreddit in SUBREDDITS:
            print(f"\n  Searching r/{subreddit}...")

            for keyword in CHINA_KEYWORDS:
                posts = self.search_posts(
                    subreddit=subreddit,
                    keyword=keyword,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    limit=100
                )

                new_count = 0
                for post in posts:
                    if post['id'] not in seen_ids:
                        seen_ids.add(post['id'])
                        all_posts.append(post)
                        new_count += 1

                if new_count > 0:
                    print(f"    '{keyword}': +{new_count} (total: {len(all_posts)})")

                time.sleep(0.5)  # Rate limiting

        print(f"\n  Total collected: {len(all_posts)} posts")

        df = pd.DataFrame(all_posts) if all_posts else pd.DataFrame()
        if not df.empty:
            df['period'] = 'post_hanoi'
            df['topic'] = 'china'
            df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return df

    def save_results(self, df: pd.DataFrame, output_dir: str = 'data/control'):
        """Save collected data."""
        os.makedirs(output_dir, exist_ok=True)

        if df.empty:
            print("No data to save.")
            return None

        filepath = os.path.join(output_dir, 'china_posts_hanoi_extended.csv')
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved: {filepath}")

        # Print summary by month
        if 'created_utc' in df.columns:
            df['date'] = pd.to_datetime(df['created_utc'], unit='s')
            df['month'] = df['date'].dt.to_period('M')
            print("\n--- Posts by Month ---")
            for month, count in df.groupby('month').size().items():
                print(f"  {month}: {count} posts")

        return filepath


def main():
    print("=" * 70)
    print("CHINA EXTENDED DATA COLLECTION (Control Group)")
    print("Period: July-December 2019")
    print("=" * 70)
    
    print(f"\nKeywords ({len(CHINA_KEYWORDS)}):")
    for kw in CHINA_KEYWORDS:
        print(f"  - {kw}")

    collector = ChinaExtendedCollector()
    df = collector.collect_all()
    collector.save_results(df)

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Total posts collected: {len(df):,}")

    print("\n✓ China collection complete!")


if __name__ == '__main__':
    main()
