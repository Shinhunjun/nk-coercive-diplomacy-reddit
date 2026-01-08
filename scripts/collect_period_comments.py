"""
Collect comments for posts within the study period (P1, P2, P3)
Uses Arctic Shift API to fetch comments for each post.

Study Periods:
- P1: 2017-01-01 ~ 2018-06-11 (Pre-Summit)
- P2: 2018-06-13 ~ 2019-02-27 (Summit Era)
- P3: 2019-03-01 ~ 2019-12-31 (Post-Hanoi)
"""

import pandas as pd
import requests
import time
import os
from datetime import datetime
from tqdm import tqdm
from typing import List, Dict, Optional

# Study period boundaries
P1_START = datetime(2017, 1, 1).timestamp()
P1_END = datetime(2018, 6, 11).timestamp()
P2_START = datetime(2018, 6, 13).timestamp()
P2_END = datetime(2019, 2, 27).timestamp()
P3_START = datetime(2019, 3, 1).timestamp()
P3_END = datetime(2019, 12, 31).timestamp()


def get_period(timestamp):
    """Assign period based on timestamp."""
    try:
        ts = float(timestamp)
        if P1_START <= ts <= P1_END:
            return 'P1'
        elif P2_START <= ts <= P2_END:
            return 'P2'
        elif P3_START <= ts <= P3_END:
            return 'P3'
        else:
            return 'Out'
    except:
        return 'Error'


class PeriodCommentsCollector:
    """Collects Reddit comments for posts within study periods using Arctic Shift API."""

    def __init__(self):
        self.comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
        self.session = requests.Session()
        self.request_delay = 0.5
        self.max_retries = 3

    def get_comments_for_post(
        self,
        link_id: str,
        max_comments: int = 100
    ) -> List[Dict]:
        """
        Get top comments for a specific post.

        Args:
            link_id: The post ID (e.g., 'abc123' or 't3_abc123')
            max_comments: Max comments to retrieve per post

        Returns:
            List of comment dictionaries
        """
        # Remove t3_ prefix if present
        if link_id.startswith('t3_'):
            link_id = link_id[3:]

        params = {
            "link_id": link_id,
            "limit": min(max_comments, 100),  # API max is 100
            "sort": "top"  # Get top comments by score
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.get(
                    self.comments_url, params=params, timeout=30
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.request_delay * 2)
                else:
                    print(f"  Failed to get comments for {link_id}: {e}")
                    return []

    def collect_comments_for_posts(
        self,
        posts_df: pd.DataFrame,
        comments_per_post: int = 3,
        output_path: Optional[str] = None,
        target_period: str = 'all'
    ) -> pd.DataFrame:
        """
        Collect top N comments for each post in the study period.

        Args:
            posts_df: DataFrame with posts (must have 'id', 'created_utc' columns)
            comments_per_post: Number of top comments to collect per post
            output_path: Path to save intermediate results (for resumability)
            target_period: 'P1', 'P2', 'P3', or 'all'

        Returns:
            DataFrame with all collected comments
        """
        all_comments = []
        seen_ids = set()

        # If splitting by period, modify output path to be independent
        if target_period != 'all' and output_path:
            base, ext = os.path.splitext(output_path)
            output_path = f"{base}_{target_period}{ext}"

        # Load existing if resuming
        if output_path and os.path.exists(output_path):
            existing = pd.read_csv(output_path)
            seen_ids = set(existing['id'].astype(str))
            all_comments = existing.to_dict('records')
            print(f"🔄 Resuming {target_period}: {len(seen_ids)} comments already collected")

        # Filter posts to study period
        posts_df = posts_df.copy()
        posts_df['period'] = posts_df['created_utc'].apply(get_period)
        
        # Apply target period filter
        if target_period == 'all':
            posts_in_period = posts_df[posts_df['period'].isin(['P1', 'P2', 'P3'])]
        else:
            posts_in_period = posts_df[posts_df['period'] == target_period]

        print(f"📊 Posts in {target_period}: {len(posts_in_period)}")
        print(f"   P1: {len(posts_in_period[posts_in_period['period'] == 'P1'])}")
        print(f"   P2: {len(posts_in_period[posts_in_period['period'] == 'P2'])}")
        print(f"   P3: {len(posts_in_period[posts_in_period['period'] == 'P3'])}")

        pbar = tqdm(total=len(posts_in_period), desc="Collecting Comments")
        batch_count = 0

        for _, post in posts_in_period.iterrows():
            post_id = post.get("id")
            if pd.isna(post_id):
                pbar.update(1)
                continue

            # Get comments
            comments = self.get_comments_for_post(post_id, max_comments=comments_per_post)

            for comment in comments[:comments_per_post]:  # Take top N
                comment_id = comment.get("id")
                if comment_id and str(comment_id) not in seen_ids:
                    seen_ids.add(str(comment_id))
                    
                    # Add post metadata
                    comment["parent_post_id"] = post_id
                    comment["parent_post_title"] = post.get("title", "")
                    comment["parent_post_period"] = post.get("period", "")
                    comment["parent_post_subreddit"] = post.get("subreddit", "")
                    
                    all_comments.append(comment)

            pbar.update(1)
            batch_count += 1

            # Save intermediate results every 100 posts
            if output_path and batch_count % 100 == 0:
                pd.DataFrame(all_comments).to_csv(output_path, index=False)

            time.sleep(self.request_delay)

        pbar.close()

        # Final save
        if all_comments:
            comments_df = pd.DataFrame(all_comments)
            if output_path:
                comments_df.to_csv(output_path, index=False)
            print(f"\n✓ Collected {len(comments_df):,} comments from {len(posts_in_period):,} posts")
            return comments_df
        else:
            return pd.DataFrame()


def collect_nk_comments(comments_per_post: int = 3, target_period: str = 'all'):
    """Collect comments for North Korea posts."""
    print("=" * 70)
    print(f"COLLECTING NK COMMENTS ({target_period}, TOP {comments_per_post} PER POST)")
    print("=" * 70)

    # Load NK posts (Final Merged Dataset)
    posts_path = 'data/processed/nk_posts_final.csv'
    
    if not os.path.exists(posts_path):
        print(f"Error: {posts_path} not found!")
        return None
        
    posts_df = pd.read_csv(posts_path, low_memory=False)
    print(f"Loaded {len(posts_df):,} NK posts from {os.path.basename(posts_path)}")

    # Collect comments
    collector = PeriodCommentsCollector()
    output_path = 'data/processed/nk_comments_top3.csv'
    
    comments = collector.collect_comments_for_posts(
        posts_df,
        comments_per_post=comments_per_post,
        output_path=output_path,
        target_period=target_period
    )

    if len(comments) > 0:
        print(f"\n✓ Saved to: {output_path}")
        
        # Show period distribution
        comments['period'] = comments['created_utc'].apply(get_period)
        print(f"\nComments by period:")
        print(comments['period'].value_counts())

    return comments


def collect_control_comments(topic: str, comments_per_post: int = 3, target_period: str = 'all'):
    """Collect comments for a control group topic."""
    print("=" * 70)
    print(f"COLLECTING {topic.upper()} COMMENTS ({target_period}, TOP {comments_per_post} PER POST)")
    print("=" * 70)

    # Load posts (Final Merged Dataset)
    posts_path = f'data/processed/{topic}_posts_final.csv'
    
    if not os.path.exists(posts_path):
        print(f"Error: {posts_path} not found!")
        return None
        
    posts_df = pd.read_csv(posts_path, low_memory=False)
    print(f"Loaded {len(posts_df):,} {topic} posts from {os.path.basename(posts_path)}")

    # Collect comments
    collector = PeriodCommentsCollector()
    output_path = f'data/control/{topic}_comments_top3.csv'
    
    comments = collector.collect_comments_for_posts(
        posts_df,
        comments_per_post=comments_per_post,
        output_path=output_path,
        target_period=target_period
    )

    if len(comments) > 0:
        print(f"\n✓ Saved to: {output_path}")

    return comments


def main():
    """Main execution function."""
    print("=" * 70)
    print("PERIOD-BASED COMMENT COLLECTION")
    print("Collecting Top 3 Comments per Post within Study Period")
    print("=" * 70)
    print(f"\nStudy Periods:")
    print(f"  P1: 2017-01-01 ~ 2018-06-11 (Pre-Summit)")
    print(f"  P2: 2018-06-13 ~ 2019-02-27 (Summit Era)")
    print(f"  P3: 2019-03-01 ~ 2019-12-31 (Post-Hanoi)")

    # Collect for all topics
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--topic', default='nk', 
                        help='Topic to collect: nk, china, iran, russia, or all')
    parser.add_argument('--period', default='all',
                        help='Specific period to collect: P1, P2, P3, or all')
    parser.add_argument('--n', type=int, default=3,
                        help='Number of top comments per post')
    args = parser.parse_args()

    if args.topic == 'all':
        topics = ['nk', 'china', 'iran', 'russia']
    else:
        topics = [args.topic]

    for topic in topics:
        if topic == 'nk':
            collect_nk_comments(comments_per_post=args.n, target_period=args.period)
        else:
            collect_control_comments(topic, comments_per_post=args.n, target_period=args.period)

    print("\n✓ Comment collection complete!")


if __name__ == '__main__':
    main()
