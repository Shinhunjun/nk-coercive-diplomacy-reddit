"""
Collect ALL Reddit COMMENTS for control groups using pagination
Uses timestamp-based pagination to collect complete comment data

Pagination method:
- Use 'after' parameter with created_utc timestamp
- Sort by 'asc' to get oldest first, then paginate forward
"""

import pandas as pd
import requests
import time
import sys
import os
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FullCommentsCollector:
    """Collects ALL Reddit comments using timestamp-based pagination."""

    def __init__(self):
        self.comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
        self.session = requests.Session()
        self.request_delay = 0.5
        self.max_retries = 3

    def get_all_comments_for_post(
        self,
        link_id: str,
        max_comments: int = 10000  # Safety limit per post
    ) -> List[Dict]:
        """
        Get ALL comments for a specific post using pagination.

        Args:
            link_id: The post ID (e.g., 'abc123')
            max_comments: Maximum comments to collect per post (safety limit)

        Returns:
            List of all comment dictionaries
        """
        # Remove t3_ prefix if present
        if link_id.startswith('t3_'):
            link_id = link_id[3:]

        all_comments = []
        seen_ids = set()
        last_timestamp = None

        while len(all_comments) < max_comments:
            params = {
                "link_id": link_id,
                "limit": 100,  # API max per request
                "sort": "asc"  # Oldest first for forward pagination
            }

            if last_timestamp:
                params["after"] = last_timestamp + 1  # Start from next second

            for attempt in range(self.max_retries):
                try:
                    response = self.session.get(
                        self.comments_url, params=params, timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    comments = data.get("data", [])
                    break
                except requests.exceptions.RequestException as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.request_delay * 2)
                    else:
                        return all_comments

            if not comments:
                break  # No more comments

            new_count = 0
            for comment in comments:
                comment_id = comment.get("id")
                if comment_id and comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    all_comments.append(comment)
                    new_count += 1
                    last_timestamp = comment.get("created_utc")

            if new_count == 0:
                break  # No new comments (all duplicates)

            time.sleep(self.request_delay)

        return all_comments

    def collect_comments_for_posts(
        self,
        posts_df: pd.DataFrame,
        max_comments_per_post: int = 5000,
        max_posts: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Collect ALL comments for a list of posts.

        Args:
            posts_df: DataFrame with 'id' column
            max_comments_per_post: Safety limit per post
            max_posts: Limit number of posts to process (None = all)

        Returns:
            DataFrame with all collected comments
        """
        all_comments = []
        seen_ids = set()

        posts_to_process = posts_df.head(max_posts) if max_posts else posts_df
        pbar = tqdm(total=len(posts_to_process), desc="Collecting ALL Comments")

        for _, post in posts_to_process.iterrows():
            post_id = post.get("id")
            if pd.isna(post_id):
                pbar.update(1)
                continue

            num_comments = post.get("num_comments", 0)

            # Skip posts with very few comments
            if num_comments == 0:
                pbar.update(1)
                continue

            comments = self.get_all_comments_for_post(
                post_id,
                max_comments=max_comments_per_post
            )

            for comment in comments:
                comment_id = comment.get("id")
                if comment_id and comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    comment["post_id"] = post_id
                    comment["post_title"] = post.get("title", "")
                    comment["topic"] = post.get("topic", "")
                    comment["period"] = post.get("period", "")
                    all_comments.append(comment)

            pbar.set_postfix({
                "post_comments": len(comments),
                "total": len(all_comments)
            })
            pbar.update(1)

        pbar.close()

        if all_comments:
            comments_df = pd.DataFrame(all_comments)
            print(f"\n✓ Collected {len(comments_df):,} unique comments from {len(posts_to_process):,} posts")
            return comments_df
        else:
            return pd.DataFrame()


def collect_full_comments_for_topic(collector: FullCommentsCollector, topic: str):
    """Collect ALL comments for a single topic."""

    # Try full posts first, then balanced
    posts_path = f'data/control/{topic}_posts_full.csv'
    if not os.path.exists(posts_path):
        posts_path = f'data/control/{topic}_posts_balanced.csv'

    if not os.path.exists(posts_path):
        print(f"Error: No posts file found for {topic}!")
        return None

    print("=" * 70)
    print(f"COLLECTING ALL {topic.upper()} COMMENTS")
    print("=" * 70)

    posts_df = pd.read_csv(posts_path)
    print(f"\nLoaded {len(posts_df):,} {topic} posts from {posts_path}")

    # Show expected comment count
    total_expected = posts_df['num_comments'].sum() if 'num_comments' in posts_df.columns else 'Unknown'
    print(f"Expected total comments (from posts): {total_expected:,}" if isinstance(total_expected, int) else f"Expected total comments: {total_expected}")

    # Show period distribution
    if 'period' in posts_df.columns:
        print(f"\nPosts by period:")
        print(posts_df['period'].value_counts())

    # Collect ALL comments
    comments = collector.collect_comments_for_posts(
        posts_df,
        max_comments_per_post=5000,  # Safety limit
        max_posts=None  # Process all posts
    )

    if len(comments) > 0:
        # Save
        os.makedirs('data/control', exist_ok=True)
        output_path = f'data/control/{topic}_comments_full.csv'
        comments.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(comments):,} {topic} comments to {output_path}")

        # Show period distribution
        if 'period' in comments.columns:
            print(f"\nComments by period:")
            print(comments['period'].value_counts())

        # Show parent_id distribution (direct vs reply)
        if 'parent_id' in comments.columns:
            direct = comments['parent_id'].str.startswith('t3_').sum()
            replies = comments['parent_id'].str.startswith('t1_').sum()
            print(f"\nComment types:")
            print(f"  Direct comments (t3_): {direct:,} ({direct/len(comments)*100:.1f}%)")
            print(f"  Replies (t1_): {replies:,} ({replies/len(comments)*100:.1f}%)")

        # Show date range
        if 'created_utc' in comments.columns:
            comments['datetime'] = pd.to_datetime(comments['created_utc'], unit='s')
            print(f"\nDate range: {comments['datetime'].min()} to {comments['datetime'].max()}")

        return comments
    else:
        print(f"\n⚠️ No {topic} comments collected!")
        return None


def main():
    """Collect ALL comments for all control groups."""

    print("=" * 70)
    print("FULL COMMENT COLLECTION (WITH PAGINATION)")
    print("Collecting ALL comments for each post")
    print("=" * 70)

    collector = FullCommentsCollector()

    topics = ['iran', 'russia', 'china']
    results = {}

    for topic in topics:
        try:
            comments = collect_full_comments_for_topic(collector, topic)
            if comments is not None:
                if 'period' in comments.columns:
                    pre_count = len(comments[comments['period'] == 'pre'])
                    post_count = len(comments[comments['period'] == 'post'])
                    results[topic] = {
                        'total': len(comments),
                        'pre': pre_count,
                        'post': post_count
                    }
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
    print("FULL COMMENT COLLECTION SUMMARY")
    print("=" * 70)
    print(f"{'Topic':<10} {'Total':<15} {'Pre-Event':<15} {'Post-Event':<15} {'Ratio'}")
    print("-" * 65)

    for topic, data in results.items():
        total = data['total']
        pre = data['pre']
        post = data['post']
        ratio = f"{pre/post:.2f}" if post > 0 else "N/A"
        if total > 0:
            print(f"{topic.upper():<10} {total:<15,} {pre:<15,} {post:<15,} {ratio}")
        else:
            print(f"{topic.upper():<10} ✗ No comments collected")

    print("\n✓ Full comment collection complete!")


if __name__ == '__main__':
    main()
