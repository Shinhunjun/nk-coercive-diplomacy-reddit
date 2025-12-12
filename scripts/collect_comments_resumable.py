"""
Collect ALL Reddit COMMENTS with RESUME support
Saves progress every 50 posts so it can resume if interrupted.
"""

import pandas as pd
import requests
import time
import sys
import os
import json
from typing import List, Dict, Optional
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ResumableCommentsCollector:
    """Collects ALL Reddit comments with resume capability."""

    def __init__(self):
        self.comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
        self.session = requests.Session()
        self.request_delay = 0.5
        self.max_retries = 3
        self.save_interval = 50  # Save every 50 posts

    def get_all_comments_for_post(
        self,
        link_id: str,
        max_comments: int = 5000
    ) -> List[Dict]:
        """Get ALL comments for a specific post using pagination."""
        if link_id.startswith('t3_'):
            link_id = link_id[3:]

        all_comments = []
        seen_ids = set()
        last_timestamp = None

        while len(all_comments) < max_comments:
            params = {
                "link_id": link_id,
                "limit": 100,
                "sort": "asc"
            }

            if last_timestamp:
                params["after"] = last_timestamp + 1

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
                break

            new_count = 0
            for comment in comments:
                comment_id = comment.get("id")
                if comment_id and comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    all_comments.append(comment)
                    new_count += 1
                    last_timestamp = comment.get("created_utc")

            if new_count == 0:
                break

            time.sleep(self.request_delay)

        return all_comments

    def collect_comments_for_topic(
        self,
        topic: str,
        posts_path: str,
        output_dir: str = "data/control"
    ) -> pd.DataFrame:
        """
        Collect ALL comments for a topic with resume support.
        """
        # Set output paths
        if topic == 'nk':
            output_dir = "data/nk"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f"{topic}_comments_full.csv")
        progress_path = os.path.join(output_dir, f"{topic}_comments_progress.json")

        # Load posts
        posts_df = pd.read_csv(posts_path)
        total_posts = len(posts_df)

        print("=" * 70)
        print(f"COLLECTING {topic.upper()} COMMENTS (RESUMABLE)")
        print("=" * 70)
        print(f"Total posts: {total_posts:,}")

        # Check for existing progress
        start_idx = 0
        all_comments = []
        processed_post_ids = set()

        if os.path.exists(progress_path):
            with open(progress_path, 'r') as f:
                progress = json.load(f)
                start_idx = progress.get('last_post_idx', 0)
                processed_post_ids = set(progress.get('processed_post_ids', []))
                print(f"Resuming from post {start_idx} ({len(processed_post_ids)} posts already processed)")

        # Load existing comments if resuming
        if os.path.exists(output_path) and start_idx > 0:
            existing_df = pd.read_csv(output_path)
            all_comments = existing_df.to_dict('records')
            print(f"Loaded {len(all_comments):,} existing comments")

        # Get seen comment IDs to avoid duplicates
        seen_comment_ids = set(c.get('id') for c in all_comments if c.get('id'))

        # Process posts
        pbar = tqdm(range(start_idx, total_posts), desc=f"Collecting {topic}")

        for idx in pbar:
            post = posts_df.iloc[idx]
            post_id = post.get("id")

            if pd.isna(post_id) or post_id in processed_post_ids:
                continue

            num_comments = post.get("num_comments", 0)
            if num_comments == 0:
                processed_post_ids.add(post_id)
                continue

            # Collect comments for this post
            comments = self.get_all_comments_for_post(post_id)

            new_count = 0
            for comment in comments:
                comment_id = comment.get("id")
                if comment_id and comment_id not in seen_comment_ids:
                    seen_comment_ids.add(comment_id)
                    comment["post_id"] = post_id
                    comment["post_title"] = post.get("title", "")
                    comment["topic"] = post.get("topic", topic)
                    comment["period"] = post.get("period", "")
                    all_comments.append(comment)
                    new_count += 1

            processed_post_ids.add(post_id)

            pbar.set_postfix({
                "post_comments": len(comments),
                "new": new_count,
                "total": len(all_comments)
            })

            # Save progress every N posts
            if (idx + 1) % self.save_interval == 0:
                self._save_progress(
                    all_comments, output_path,
                    idx + 1, processed_post_ids, progress_path
                )
                pbar.set_description(f"Collecting {topic} (saved at {idx+1})")

        # Final save
        self._save_progress(
            all_comments, output_path,
            total_posts, processed_post_ids, progress_path,
            final=True
        )

        print(f"\n✓ Collected {len(all_comments):,} comments for {topic}")

        return pd.DataFrame(all_comments) if all_comments else pd.DataFrame()

    def _save_progress(
        self,
        comments: List[Dict],
        output_path: str,
        last_idx: int,
        processed_ids: set,
        progress_path: str,
        final: bool = False
    ):
        """Save comments and progress."""
        # Save comments
        if comments:
            pd.DataFrame(comments).to_csv(output_path, index=False)

        # Save or remove progress file
        if final:
            if os.path.exists(progress_path):
                os.remove(progress_path)
            print(f"\n✓ Final save: {len(comments):,} comments to {output_path}")
        else:
            with open(progress_path, 'w') as f:
                json.dump({
                    'last_post_idx': last_idx,
                    'processed_post_ids': list(processed_ids),
                    'total_comments': len(comments),
                    'timestamp': datetime.now().isoformat()
                }, f)


def main():
    """Collect comments for Russia and NK."""

    collector = ResumableCommentsCollector()

    # Topics to collect
    topics = {
        'russia': 'data/control/russia_posts_merged.csv',
        'nk': 'data/nk/nk_posts_merged.csv'
    }

    for topic, posts_path in topics.items():
        if not os.path.exists(posts_path):
            print(f"Warning: {posts_path} not found, skipping {topic}")
            continue

        try:
            collector.collect_comments_for_topic(topic, posts_path)
        except Exception as e:
            print(f"\n✗ Error collecting {topic}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("COMMENT COLLECTION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
