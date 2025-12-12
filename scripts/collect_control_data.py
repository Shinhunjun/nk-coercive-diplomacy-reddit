"""
Data Collection Script for Control Groups (DID Analysis)
Collects Reddit posts about Iran, Russia, and China for 2017-2019 period
"""

import pandas as pd
import requests
import time
from datetime import datetime
from typing import List, Dict
import json
import os
import sys

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONTROL_GROUPS


class ControlDataCollector:
    """Collects Reddit posts for control groups using Pushshift API."""

    def __init__(self):
        self.base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        self.session = requests.Session()
        self.collected_data = []

    def collect_control_posts(
        self,
        topic: str,
        start_date: str = '2017-01-01',
        end_date: str = '2019-06-30',
        subreddits: List[str] = ['worldnews', 'geopolitics', 'news'],
        target_count: int = 700
    ) -> pd.DataFrame:
        """
        Collect Reddit posts for a control group topic.

        Args:
            topic: 'iran', 'russia', or 'china'
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            subreddits: List of subreddits to search
            target_count: Target number of posts to collect

        Returns:
            DataFrame with collected posts
        """
        print(f"\n{'='*60}")
        print(f"Collecting {topic.upper()} posts")
        print(f"{'='*60}")
        print(f"Target: {target_count} posts")
        print(f"Period: {start_date} to {end_date}")
        print(f"Subreddits: {', '.join(subreddits)}")

        # Get keywords from config
        if topic not in CONTROL_GROUPS:
            raise ValueError(f"Unknown topic: {topic}. Must be one of {list(CONTROL_GROUPS.keys())}")

        keywords = CONTROL_GROUPS[topic]['keywords']
        print(f"Keywords: {', '.join(keywords)}")

        all_posts = []

        # Collect posts for each subreddit
        for subreddit in subreddits:
            print(f"\nSearching r/{subreddit}...")

            # Use ALL keywords for better coverage
            keywords_to_use = keywords if len(keywords) <= 10 else keywords[:10]
            for keyword in keywords_to_use:
                posts = self._search_posts(
                    subreddit=subreddit,
                    keyword=keyword,
                    start_date=start_date,
                    end_date=end_date,
                    limit=min(500, target_count // len(subreddits))
                )
                all_posts.extend(posts)

                # Rate limiting
                time.sleep(1)

                if len(all_posts) >= target_count * 3:  # Collect more for better coverage
                    break

            if len(all_posts) >= target_count * 2:
                break

        # Remove duplicates
        unique_posts = {post['id']: post for post in all_posts}.values()
        df = pd.DataFrame(list(unique_posts))

        # Limit to target count
        if len(df) > target_count:
            df = df.sample(n=target_count, random_state=42).reset_index(drop=True)

        print(f"\n✓ Collected {len(df)} unique posts")

        # Add metadata
        df['topic'] = topic
        df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return df

    def _search_posts(
        self,
        subreddit: str,
        keyword: str,
        start_date: str,
        end_date: str,
        limit: int = 500
    ) -> List[Dict]:
        """
        Search for posts using Arctic Shift API.

        Args:
            subreddit: Subreddit name
            keyword: Search keyword
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of posts to retrieve

        Returns:
            List of post dictionaries
        """
        params = {
            'subreddit': subreddit,
            'title': keyword,  # Arctic Shift uses 'title' for full-text search
            'after': start_date,
            'before': end_date,
            'limit': min(limit, 100),  # API limit per request
            'sort': 'desc'  # Newest first
        }

        posts = []
        attempts = 0
        max_attempts = 3

        for attempt in range(max_attempts):
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                # Arctic Shift returns {"data": [...]} format
                batch = data.get('data', [])

                if not batch:
                    break

                posts.extend(batch)
                print(f"  {keyword}: {len(posts)} posts collected...", end='\r')

                # Arctic Shift handles pagination internally with limit
                break  # Single request per query

            except requests.exceptions.RequestException as e:
                print(f"\n  Error on attempt {attempt + 1}: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(5)
                else:
                    return []

        # Clean and structure data
        cleaned_posts = []
        for post in posts:
            cleaned_post = {
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
            }
            cleaned_posts.append(cleaned_post)

        return cleaned_posts

    def save_to_csv(self, df: pd.DataFrame, topic: str, output_dir: str = 'data/control'):
        """Save collected posts to CSV."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f'{topic}_posts.csv')
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved to: {filepath}")
        return filepath


def main():
    """Main execution function."""
    print("="*60)
    print("CONTROL GROUP DATA COLLECTION")
    print("For DID Analysis: Iran, Russia, China")
    print("="*60)

    collector = ControlDataCollector()

    # Topics to collect
    topics = ['iran', 'russia', 'china']

    results = {}

    for topic in topics:
        try:
            df = collector.collect_control_posts(
                topic=topic,
                start_date='2017-01-01',
                end_date='2019-06-30',
                subreddits=['worldnews', 'geopolitics', 'news'],
                target_count=1500  # Increased for better coverage
            )

            # Save to CSV
            filepath = collector.save_to_csv(df, topic)
            results[topic] = {
                'count': len(df),
                'filepath': filepath
            }

            # Show sample
            print(f"\nSample posts:")
            print(df[['title', 'subreddit', 'score']].head(3))

        except Exception as e:
            print(f"\n✗ Error collecting {topic} data: {e}")
            results[topic] = {
                'count': 0,
                'error': str(e)
            }

    # Summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    for topic, result in results.items():
        if 'error' in result:
            print(f"{topic.upper()}: ✗ {result['error']}")
        else:
            print(f"{topic.upper()}: ✓ {result['count']} posts → {result['filepath']}")

    print("\n✓ Data collection complete!")
    print("\nNext steps:")
    print("1. Verify data quality")
    print("2. Apply sentiment analysis (Phase 2)")
    print("3. Create monthly aggregation")


if __name__ == '__main__':
    main()
