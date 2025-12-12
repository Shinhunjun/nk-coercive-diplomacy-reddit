"""
Collect Iran posts for extended period (July-December 2019)
For Hanoi Summit 3-period DID analysis (Control Group)
"""

import pandas as pd
import requests
import time
from datetime import datetime
from typing import List, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IRAN_KEYWORDS = [
    "iran", "iranian", "tehran",
    "JCPOA", "nuclear deal", "iran nuclear", "rouhani", "khamenei",
    "zarif", "soleimani", "ayatollah",
    "iran sanctions", "iran deal", "iran agreement",
    "IRGC", "revolutionary guard", "quds force", "iran military", "iran missile",
    "persian gulf", "strait of hormuz", "iran syria"
]

SUBREDDITS = ['worldnews', 'geopolitics', 'news', 'politics', 'NeutralPolitics']


class IranExtendedCollector:
    def __init__(self):
        self.base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        self.session = requests.Session()
        self.start_date = "2019-07-01"
        self.end_date = "2019-12-31"

    def search_posts(self, subreddit: str, keyword: str, start_date: str, end_date: str, limit: int = 100) -> List[Dict]:
        params = {'subreddit': subreddit, 'title': keyword, 'after': start_date, 'before': end_date, 'limit': limit, 'sort': 'desc'}
        try:
            response = self.session.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            posts = []
            for post in data.get('data', []):
                posts.append({
                    'id': post.get('id', ''), 'title': post.get('title', ''), 'selftext': post.get('selftext', ''),
                    'author': post.get('author', '[deleted]'), 'subreddit': post.get('subreddit', ''),
                    'score': post.get('score', 0), 'num_comments': post.get('num_comments', 0),
                    'created_utc': post.get('created_utc', 0), 'permalink': post.get('permalink', ''), 'url': post.get('url', '')
                })
            return posts
        except Exception as e:
            print(f"      Error: {e}")
            return []

    def collect_all(self) -> pd.DataFrame:
        all_posts, seen_ids = [], set()
        print(f"\n--- IRAN EXTENDED ({self.start_date} to {self.end_date}) ---")
        for subreddit in SUBREDDITS:
            print(f"\n  Searching r/{subreddit}...")
            for keyword in IRAN_KEYWORDS:
                posts = self.search_posts(subreddit, keyword, self.start_date, self.end_date, 100)
                for post in posts:
                    if post['id'] not in seen_ids:
                        seen_ids.add(post['id'])
                        all_posts.append(post)
                if len([p for p in posts if p['id'] not in (seen_ids - {p['id'] for p in posts})]) > 0:
                    print(f"    '{keyword}': total: {len(all_posts)}")
                time.sleep(0.5)
        df = pd.DataFrame(all_posts) if all_posts else pd.DataFrame()
        if not df.empty:
            df['period'], df['topic'] = 'post_hanoi', 'iran'
            df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return df

    def save_results(self, df: pd.DataFrame, output_dir: str = 'data/control'):
        os.makedirs(output_dir, exist_ok=True)
        if df.empty: return
        filepath = os.path.join(output_dir, 'iran_posts_hanoi_extended.csv')
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved: {filepath} ({len(df)} posts)")

def main():
    print("=" * 60)
    print("IRAN EXTENDED DATA COLLECTION")
    print("=" * 60)
    collector = IranExtendedCollector()
    df = collector.collect_all()
    collector.save_results(df)
    print(f"\n✓ Iran complete: {len(df)} posts")

if __name__ == '__main__':
    main()
