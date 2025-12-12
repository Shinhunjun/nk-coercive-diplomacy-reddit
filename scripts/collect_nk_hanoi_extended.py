"""
Collect NK posts for extended period (July-December 2019)
For Hanoi Summit 3-period framing analysis

Period: 2019-07-01 to 2019-12-31
Keywords: Original + Expanded + Hanoi-specific
"""

import pandas as pd
import requests
import time
from datetime import datetime
from typing import List, Dict
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 기존 키워드 (Original keywords from existing analysis)
NK_ORIGINAL_KEYWORDS = [
    "north korea", "dprk", "pyongyang", "kim jong un"
]

# 확장 키워드 (From collect_nk_additional.py)
NK_EXPANDED_KEYWORDS = [
    # 지도자/인물
    "kim jong-un", "kim regime", "kim family north korea",

    # 핵/미사일
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

# 하노이 회담 결렬 관련 추가 키워드
HANOI_SPECIFIC_KEYWORDS = [
    # 하노이 회담 결렬 관련
    "hanoi failure", "hanoi collapse", "hanoi breakdown",
    "trump kim hanoi", "no deal hanoi", "hanoi walkout",
    
    # 회담 결렬 이후 상황
    "north korea stalemate", "nuclear talks failed", 
    "north korea deadlock", "denuclearization stalled",
    
    # 2019년 하반기 주요 이벤트
    "panmunjom meeting", "trump dmz", "trump kim dmz",
    "north korea short range", "north korea projectile",
    
    # 김정은 관련
    "kim jong un disappointed", "north korea angry",
    "north korea patience"
]

# 모든 키워드 통합
ALL_KEYWORDS = NK_ORIGINAL_KEYWORDS + NK_EXPANDED_KEYWORDS + HANOI_SPECIFIC_KEYWORDS
# 중복 제거
ALL_KEYWORDS = list(set(ALL_KEYWORDS))

SUBREDDITS = ['worldnews', 'geopolitics', 'news', 'politics', 'NeutralPolitics']


class NKHanoiExtendedCollector:
    """Collect NK posts for July-December 2019 period."""

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

        print(f"\n--- HANOI EXTENDED PERIOD ({self.start_date} to {self.end_date}) ---")
        print(f"Total keywords: {len(ALL_KEYWORDS)}")

        for subreddit in SUBREDDITS:
            print(f"\n  Searching r/{subreddit}...")

            for keyword in ALL_KEYWORDS:
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
            df['period'] = 'post_hanoi'  # Period 3: Post-Hanoi Stagnation
            df['topic'] = 'nk'
            df['collection_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return df

    def save_results(self, df: pd.DataFrame, output_dir: str = 'data/nk'):
        """Save collected data."""
        os.makedirs(output_dir, exist_ok=True)

        if df.empty:
            print("No data to save.")
            return None

        filepath = os.path.join(output_dir, 'nk_posts_hanoi_extended.csv')
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
    print("NK HANOI EXTENDED DATA COLLECTION")
    print("Period: July-December 2019 (Post-Hanoi Stagnation)")
    print("=" * 70)
    
    print(f"\nKeywords ({len(ALL_KEYWORDS)}):")
    print("\n  [Original Keywords]")
    for kw in NK_ORIGINAL_KEYWORDS:
        print(f"    - {kw}")
    print("\n  [Expanded Keywords]")
    for kw in NK_EXPANDED_KEYWORDS[:10]:
        print(f"    - {kw}")
    print(f"    ... and {len(NK_EXPANDED_KEYWORDS) - 10} more")
    print("\n  [Hanoi-Specific Keywords]")
    for kw in HANOI_SPECIFIC_KEYWORDS:
        print(f"    - {kw}")

    collector = NKHanoiExtendedCollector()
    df = collector.collect_all()
    collector.save_results(df)

    # Summary
    print("\n" + "=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)
    print(f"Total posts collected: {len(df):,}")
    if not df.empty:
        print(f"Unique subreddits: {df['subreddit'].nunique()}")
        print(f"Subreddit distribution:")
        for sub, count in df['subreddit'].value_counts().items():
            print(f"  r/{sub}: {count}")

    print("\n✓ Collection complete!")


if __name__ == '__main__':
    main()
