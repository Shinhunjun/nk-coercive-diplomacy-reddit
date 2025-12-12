"""
Collect Reddit COMMENTS for balanced control groups (Iran, Russia, China)
Uses the balanced post data (*_posts_balanced.csv) collected with new keywords

This ensures comments are collected proportionally for pre-event and post-event periods.
"""

import pandas as pd
import requests
import time
import sys
import os
from typing import List
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BalancedCommentsCollector:
    """Collects Reddit comments for balanced control groups using Arctic Shift API."""

    def __init__(self):
        self.comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
        self.session = requests.Session()
        self.request_delay = 0.5
        self.max_retries = 3

    def get_comments_for_post(self, link_id: str, limit: int = 100) -> List[dict]:
        """
        Get comments for a specific post using link_id

        Args:
            link_id: The post ID (e.g., 't3_abc123' or just 'abc123')
            limit: Max comments to retrieve (API max is 100)

        Returns:
            List of comment dictionaries
        """
        # Remove t3_ prefix if present
        if link_id.startswith('t3_'):
            link_id = link_id[3:]

        params = {
            "link_id": link_id,
            "limit": min(limit, 100),  # API max is 100
            "sort": "desc"
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(self.comments_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * 2)
                else:
                    return []

    def collect_comments_for_posts(
        self,
        posts_df: pd.DataFrame,
        comments_per_post: int = 50,
        max_posts: int = None
    ) -> pd.DataFrame:
        """
        Collect comments for a list of posts

        Args:
            posts_df: DataFrame with 'id' column
            comments_per_post: Max comments per post
            max_posts: Limit number of posts to process (None = all)

        Returns:
            DataFrame with all collected comments
        """
        all_comments = []
        seen_ids = set()

        posts_to_process = posts_df.head(max_posts) if max_posts else posts_df
        pbar = tqdm(total=len(posts_to_process), desc="Collecting Comments")

        for _, post in posts_to_process.iterrows():
            post_id = post.get("id")
            if pd.isna(post_id):
                pbar.update(1)
                continue

            comments = self.get_comments_for_post(post_id, limit=comments_per_post)

            for comment in comments:
                comment_id = comment.get("id")
                if comment_id and comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    comment["post_id"] = post_id
                    comment["post_title"] = post.get("title", "")
                    comment["topic"] = post.get("topic", "")
                    comment["period"] = post.get("period", "")  # pre or post
                    all_comments.append(comment)

            pbar.update(1)
            time.sleep(self.request_delay)

        pbar.close()

        if all_comments:
            comments_df = pd.DataFrame(all_comments)
            print(f"\n✓ Collected {len(comments_df)} unique comments from {len(posts_to_process)} posts")
            return comments_df
        else:
            return pd.DataFrame()


def collect_balanced_comments_for_topic(collector, topic):
    """Collect comments for balanced posts of a single topic."""

    posts_path = f'data/control/{topic}_posts_balanced.csv'
    if not os.path.exists(posts_path):
        print(f"Error: {posts_path} not found!")
        print("Please run collect_control_balanced.py first to collect balanced posts.")
        return None

    print("=" * 60)
    print(f"COLLECTING {topic.upper()} BALANCED COMMENTS")
    print("=" * 60)

    posts_df = pd.read_csv(posts_path)
    print(f"\nLoaded {len(posts_df)} {topic} balanced posts from {posts_path}")

    # Show period distribution
    if 'period' in posts_df.columns:
        print(f"Period distribution:")
        print(posts_df['period'].value_counts())

    # Collect comments (50 comments per post)
    comments = collector.collect_comments_for_posts(
        posts_df,
        comments_per_post=50,
        max_posts=None  # Process all posts
    )

    if len(comments) > 0:
        # Save
        os.makedirs('data/control', exist_ok=True)
        output_path = f'data/control/{topic}_comments_balanced.csv'
        comments.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(comments)} {topic} comments to {output_path}")

        # Show period distribution of comments
        if 'period' in comments.columns:
            print(f"\nComments by period:")
            print(comments['period'].value_counts())

        # Show date distribution
        if 'created_utc' in comments.columns:
            comments['datetime'] = pd.to_datetime(comments['created_utc'], unit='s')
            print(f"\nDate range: {comments['datetime'].min()} to {comments['datetime'].max()}")

            comments['month'] = comments['datetime'].dt.to_period('M')
            print(f"\nMonthly distribution (first 10):")
            print(comments['month'].value_counts().sort_index().head(10))

        return comments
    else:
        print(f"\n⚠️ No {topic} comments collected!")
        return None


def main():
    """Collect comments for all balanced control groups: Iran, Russia, China."""

    print("=" * 70)
    print("BALANCED COMMENT COLLECTION")
    print("For control groups with balanced pre/post event data")
    print("=" * 70)

    collector = BalancedCommentsCollector()

    topics = ['iran', 'russia', 'china']
    results = {}

    for topic in topics:
        try:
            comments = collect_balanced_comments_for_topic(collector, topic)
            if comments is not None:
                # Count by period
                if 'period' in comments.columns:
                    pre_count = len(comments[comments['period'] == 'pre'])
                    post_count = len(comments[comments['period'] == 'post'])
                    results[topic] = {'total': len(comments), 'pre': pre_count, 'post': post_count}
                else:
                    results[topic] = {'total': len(comments), 'pre': 0, 'post': 0}
            else:
                results[topic] = {'total': 0, 'pre': 0, 'post': 0}
        except Exception as e:
            print(f"\n✗ Error collecting {topic} comments: {e}")
            import traceback
            traceback.print_exc()
            results[topic] = {'total': 0, 'pre': 0, 'post': 0}

    # Summary
    print("\n" + "=" * 70)
    print("BALANCED COMMENT COLLECTION SUMMARY")
    print("=" * 70)
    print(f"{'Topic':<10} {'Total':<12} {'Pre-Event':<12} {'Post-Event':<12} {'Ratio'}")
    print("-" * 60)

    for topic, data in results.items():
        total = data['total']
        pre = data['pre']
        post = data['post']
        ratio = f"{pre/post:.2f}" if post > 0 else "N/A"
        if total > 0:
            print(f"{topic.upper():<10} {total:<12,} {pre:<12,} {post:<12,} {ratio}")
        else:
            print(f"{topic.upper():<10} ✗ No comments collected")

    print("\n✓ Balanced comment collection complete!")
    print("\nOutput files:")
    for topic in topics:
        if results[topic]['total'] > 0:
            print(f"  - data/control/{topic}_comments_balanced.csv")


if __name__ == '__main__':
    main()
