"""
Balanced Data Collection Script for Control Groups
Collects equal amounts of data for pre-event and post-event periods
"""

import pandas as pd
import requests
import time
from datetime import datetime
from typing import List, Dict
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import CONTROL_GROUPS, DID_CONFIG


class BalancedControlCollector:
    """Collects Reddit posts with balanced pre/post event coverage."""

    def __init__(self):
        self.base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        self.session = requests.Session()

        # Event date: March 8, 2018 (Trump accepts summit invitation)
        self.event_date = "2018-03-08"

        # Analysis periods from config (use YYYY-MM-DD format for API)
        self.pre_start = DID_CONFIG['pre_period_start'] + '-01'   # 2017-01-01
        self.pre_end = DID_CONFIG['pre_period_end'] + '-28'       # 2018-02-28
        self.post_start = DID_CONFIG['post_period_start'] + '-01' # 2018-03-01
        self.post_end = DID_CONFIG['post_period_end'] + '-30'     # 2019-06-30

    def collect_balanced_posts(
        self,
        topic: str,
        subreddits: List[str] = ['worldnews', 'geopolitics', 'news', 'politics', 'NeutralPolitics'],
        target_per_period: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect posts for both pre and post event periods.

        Args:
            topic: 'iran', 'russia', or 'china'
            subreddits: List of subreddits to search
            target_per_period: Target posts per period

        Returns:
            Dict with 'pre' and 'post' DataFrames
        """
        print(f"\n{'='*70}")
        print(f"BALANCED COLLECTION: {topic.upper()}")
        print(f"{'='*70}")

        keywords = CONTROL_GROUPS[topic]['keywords']
        print(f"Keywords: {keywords}")
        print(f"Subreddits: {subreddits}")
        print(f"Target per period: {target_per_period}")

        results = {}

        # Collect PRE-EVENT data
        print(f"\n--- PRE-EVENT PERIOD (2017-01-01 to 2018-02-28) ---")
        pre_posts = self._collect_period(
            keywords=keywords,
            subreddits=subreddits,
            start_date=self.pre_start,
            end_date=self.pre_end,
            target_count=target_per_period
        )
        pre_df = pd.DataFrame(pre_posts) if pre_posts else pd.DataFrame()
        if not pre_df.empty:
            pre_df['period'] = 'pre'
            pre_df['topic'] = topic
        results['pre'] = pre_df
        print(f"✓ PRE-EVENT: {len(pre_df)} posts collected")

        # Collect POST-EVENT data
        print(f"\n--- POST-EVENT PERIOD (2018-03-01 to 2019-06-30) ---")
        post_posts = self._collect_period(
            keywords=keywords,
            subreddits=subreddits,
            start_date=self.post_start,
            end_date=self.post_end,
            target_count=target_per_period
        )
        post_df = pd.DataFrame(post_posts) if post_posts else pd.DataFrame()
        if not post_df.empty:
            post_df['period'] = 'post'
            post_df['topic'] = topic
        results['post'] = post_df
        print(f"✓ POST-EVENT: {len(post_df)} posts collected")

        return results

    def _collect_period(
        self,
        keywords: List[str],
        subreddits: List[str],
        start_date: str,
        end_date: str,
        target_count: int
    ) -> List[Dict]:
        """Collect ALL posts for a specific period (no sampling)."""

        all_posts = []
        seen_ids = set()

        for subreddit in subreddits:
            print(f"\n  Searching r/{subreddit}...")

            for keyword in keywords:
                posts = self._search_posts(
                    subreddit=subreddit,
                    keyword=keyword,
                    start_date=start_date,
                    end_date=end_date,
                    limit=100  # API max is 100
                )

                # Add only unique posts
                for post in posts:
                    if post['id'] not in seen_ids:
                        seen_ids.add(post['id'])
                        all_posts.append(post)

                print(f"    '{keyword}': +{len(posts)} (total unique: {len(all_posts)})")
                time.sleep(1.5)  # Rate limiting - API has timeout issues

        # No sampling - return ALL collected posts
        print(f"  Total collected: {len(all_posts)} posts")
        return all_posts

    def _search_posts(
        self,
        subreddit: str,
        keyword: str,
        start_date: str,
        end_date: str,
        limit: int = 100
    ) -> List[Dict]:
        """Search Arctic Shift API for posts using YYYY-MM-DD date format.
        Note: API limit is 1-100 per request."""

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

    def save_results(self, results: Dict[str, pd.DataFrame], topic: str, output_dir: str = 'data/control'):
        """Save collected data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save combined file (full = no sampling)
        combined = pd.concat([results['pre'], results['post']], ignore_index=True)
        combined['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        filepath = os.path.join(output_dir, f'{topic}_posts_full.csv')
        combined.to_csv(filepath, index=False)
        print(f"\n✓ Saved: {filepath}")

        # Save separate files for each period
        for period, df in results.items():
            df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            period_path = os.path.join(output_dir, f'{topic}_posts_{period}_full.csv')
            df.to_csv(period_path, index=False)
            print(f"✓ Saved: {period_path}")

        return filepath


def main():
    """Main execution."""
    print("="*70)
    print("BALANCED CONTROL GROUP DATA COLLECTION")
    print("Pre-event (2017-01 to 2018-02) vs Post-event (2018-03 to 2019-06)")
    print("="*70)

    collector = BalancedControlCollector()

    # Collect for each control group
    topics = ['iran', 'russia', 'china']

    summary = {}

    for topic in topics:
        try:
            results = collector.collect_balanced_posts(
                topic=topic,
                subreddits=['worldnews', 'geopolitics', 'news', 'politics', 'NeutralPolitics'],
                target_per_period=500
            )

            collector.save_results(results, topic)

            summary[topic] = {
                'pre': len(results['pre']),
                'post': len(results['post'])
            }

        except Exception as e:
            print(f"\n✗ Error with {topic}: {e}")
            summary[topic] = {'error': str(e)}

    # Print summary
    print("\n" + "="*70)
    print("COLLECTION SUMMARY")
    print("="*70)
    print(f"{'Topic':<10} {'Pre-Event':<15} {'Post-Event':<15} {'Balance'}")
    print("-"*50)

    for topic, data in summary.items():
        if 'error' in data:
            print(f"{topic.upper():<10} ERROR: {data['error']}")
        else:
            pre = data['pre']
            post = data['post']
            ratio = f"{pre/post:.2f}" if post > 0 else "N/A"
            print(f"{topic.upper():<10} {pre:<15} {post:<15} {ratio}")

    print("\n✓ Collection complete!")
    print("\nOutput files:")
    print("  - {topic}_posts_balanced.csv  (combined)")
    print("  - {topic}_posts_pre.csv       (pre-event only)")
    print("  - {topic}_posts_post.csv      (post-event only)")


if __name__ == '__main__':
    main()
