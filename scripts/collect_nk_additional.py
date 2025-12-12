"""
Collect additional NK posts with expanded keywords
Period: 2017-01-01 to 2019-06-30 (same as DID analysis period)
"""

import pandas as pd
import requests
import time
from datetime import datetime
from typing import List, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 새로운 키워드 (기존 키워드 제외)
NK_NEW_KEYWORDS = [
    # 지도자/인물 (기존: kim jong un)
    "kim jong-un", "kim regime", "kim family north korea",

    # 핵/미사일 (기존: korea nuclear, korea missile)
    "icbm north korea", "hwasong", "nuclear test north korea",
    "hydrogen bomb north korea", "ballistic missile north korea",
    "north korea bomb", "north korea nuke",

    # 정상회담 관련
    "trump kim", "singapore summit", "hanoi summit",
    "denuclearization", "north korea talks", "north korea negotiation",

    # 제재/외교
    "north korea sanctions", "un sanctions north korea",
    "north korea diplomacy", "six party talks",

    # 군사/안보
    "dmz korea", "38th parallel", "demilitarized zone korea",
    "north korea military", "north korea army", "north korea threat",

    # 미국 관련
    "trump north korea", "us north korea", "america north korea",
    "trump pyongyang", "trump dprk",

    # 기타
    "north korean", "nk missile", "nk nuclear"
]

SUBREDDITS = ['worldnews', 'geopolitics', 'news', 'politics', 'NeutralPolitics']


class NKAdditionalCollector:
    """Collect additional NK posts with new keywords."""

    def __init__(self):
        self.base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        self.session = requests.Session()

        # Analysis period
        self.pre_start = "2017-01-01"
        self.pre_end = "2018-02-28"
        self.post_start = "2018-03-01"
        self.post_end = "2019-06-30"

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

    def collect_period(
        self,
        start_date: str,
        end_date: str,
        period_name: str
    ) -> pd.DataFrame:
        """Collect all posts for a specific period."""

        all_posts = []
        seen_ids = set()

        print(f"\n--- {period_name} ({start_date} to {end_date}) ---")

        for subreddit in SUBREDDITS:
            print(f"\n  Searching r/{subreddit}...")

            for keyword in NK_NEW_KEYWORDS:
                posts = self.search_posts(
                    subreddit=subreddit,
                    keyword=keyword,
                    start_date=start_date,
                    end_date=end_date,
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

                time.sleep(1.0)  # Rate limiting

        print(f"  Total {period_name}: {len(all_posts)} posts")

        df = pd.DataFrame(all_posts) if all_posts else pd.DataFrame()
        if not df.empty:
            df['period'] = 'pre' if 'PRE' in period_name else 'post'
            df['topic'] = 'nk'

        return df

    def collect_all(self) -> Dict[str, pd.DataFrame]:
        """Collect for both pre and post periods."""

        results = {}

        # PRE-EVENT
        results['pre'] = self.collect_period(
            self.pre_start, self.pre_end, "PRE-EVENT"
        )

        # POST-EVENT
        results['post'] = self.collect_period(
            self.post_start, self.post_end, "POST-EVENT"
        )

        return results

    def save_results(self, results: Dict[str, pd.DataFrame], output_dir: str = 'data/nk'):
        """Save collected data."""
        os.makedirs(output_dir, exist_ok=True)

        # Combined
        combined = pd.concat([results['pre'], results['post']], ignore_index=True)
        combined['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        filepath = os.path.join(output_dir, 'nk_posts_additional.csv')
        combined.to_csv(filepath, index=False)
        print(f"\n✓ Saved: {filepath}")

        # Separate files
        for period, df in results.items():
            if not df.empty:
                df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                period_path = os.path.join(output_dir, f'nk_posts_{period}_additional.csv')
                df.to_csv(period_path, index=False)
                print(f"✓ Saved: {period_path}")

        return filepath


def main():
    print("=" * 70)
    print("NK ADDITIONAL DATA COLLECTION")
    print("New keywords for 2017-01 to 2019-06 period")
    print("=" * 70)
    print(f"\nKeywords ({len(NK_NEW_KEYWORDS)}):")
    for kw in NK_NEW_KEYWORDS:
        print(f"  - {kw}")

    collector = NKAdditionalCollector()
    results = collector.collect_all()
    collector.save_results(results)

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    pre_count = len(results['pre']) if not results['pre'].empty else 0
    post_count = len(results['post']) if not results['post'].empty else 0
    print(f"PRE-EVENT:  {pre_count:,} posts")
    print(f"POST-EVENT: {post_count:,} posts")
    print(f"TOTAL:      {pre_count + post_count:,} posts")

    if post_count > 0:
        print(f"Pre/Post ratio: {pre_count/post_count:.2f}")

    print("\n✓ Collection complete!")


if __name__ == '__main__':
    main()
