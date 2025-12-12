"""
Collect Reddit COMMENTS for control groups (Iran, Russia, China)
Comments reflect actual public opinion more than posts (which are often news links)

Uses same method as reddit_US_NK project:
1. Load existing posts
2. Collect comments for each post using link_id
"""

import pandas as pd
import requests
import time
import sys
import os
from typing import List
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ControlCommentsCollector:
    """Collects Reddit comments for control groups using Arctic Shift API."""

    def __init__(self):
        self.comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
        self.session = requests.Session()
        self.request_delay = 0.5
        self.max_retries = 3

    def get_comments_for_post(self, link_id: str, limit: int = 100) -> List[dict]:
        """
        Get comments for a specific post using link_id
        (Same as reddit_US_NK/src/data_collector.py)

        Args:
            link_id: The post ID (e.g., 't3_abc123' or just 'abc123')
            limit: Max comments to retrieve

        Returns:
            List of comment dictionaries
        """
        # Remove t3_ prefix if present
        if link_id.startswith('t3_'):
            link_id = link_id[3:]

        params = {
            "link_id": link_id,
            "limit": limit,
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
        (Same as reddit_US_NK/src/data_collector.py)

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


def collect_comments_for_topic(collector, topic):
    """Collect comments for a single topic (iran, russia, or china)."""

    posts_path = f'data/control/{topic}_posts.csv'
    if not os.path.exists(posts_path):
        print(f"Error: {posts_path} not found!")
        print("Please run collect_control_data.py first to collect posts.")
        return None

    print("=" * 60)
    print(f"COLLECTING {topic.upper()} COMMENTS")
    print("=" * 60)

    posts_df = pd.read_csv(posts_path)
    print(f"\nLoaded {len(posts_df)} {topic} posts from {posts_path}")

    # Filter to analysis period (2017-2019)
    posts_df['datetime'] = pd.to_datetime(posts_df['created_utc'], unit='s')
    posts_filtered = posts_df[
        (posts_df['datetime'] >= '2017-01-01') &
        (posts_df['datetime'] <= '2019-06-30')
    ].copy()

    print(f"Posts in analysis period (2017-2019): {len(posts_filtered)}")

    # Collect comments (50 comments per post, similar to NK project)
    comments = collector.collect_comments_for_posts(
        posts_filtered,
        comments_per_post=50,
        max_posts=None  # Process all posts
    )

    if len(comments) > 0:
        # Save
        os.makedirs('data/control', exist_ok=True)
        output_path = f'data/control/{topic}_comments.csv'
        comments.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(comments)} {topic} comments to {output_path}")

        # Show date distribution
        comments['datetime'] = pd.to_datetime(comments['created_utc'], unit='s')
        print(f"\nDate range: {comments['datetime'].min()} to {comments['datetime'].max()}")

        comments['month'] = comments['datetime'].dt.to_period('M')
        print(f"\nTop 10 months by comment count:")
        print(comments['month'].value_counts().sort_index().head(10))

        return comments
    else:
        print(f"\n⚠️ No {topic} comments collected!")
        return None


def main():
    """Collect comments for all control groups: Iran, Russia, China."""

    collector = ControlCommentsCollector()

    topics = ['iran', 'russia', 'china']
    results = {}

    for topic in topics:
        try:
            comments = collect_comments_for_topic(collector, topic)
            if comments is not None:
                results[topic] = len(comments)
            else:
                results[topic] = 0
        except Exception as e:
            print(f"\n✗ Error collecting {topic} comments: {e}")
            results[topic] = 0

    # Summary
    print("\n" + "=" * 60)
    print("COMMENT COLLECTION SUMMARY")
    print("=" * 60)
    for topic, count in results.items():
        if count > 0:
            print(f"{topic.upper()}: ✓ {count:,} comments")
        else:
            print(f"{topic.upper()}: ✗ No comments")

    print("\n✓ Comment collection complete!")


if __name__ == '__main__':
    main()
