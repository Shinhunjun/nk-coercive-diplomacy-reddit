
import pandas as pd
import os

# Define Periods
def assign_period(date):
    month = str(date)[:7]
    if '2018-06' <= month <= '2019-01':
        return 'P2 (Singapore)'
    elif '2019-03' <= month <= '2019-12':
        return 'P3 (Post-Hanoi)'
    return None

# Load Comments (Sampled Top 3)
comments_path = "data/comments_to_classify_top3.csv"
if not os.path.exists(comments_path):
    print("Comments file not found.")
    exit()

comments = pd.read_csv(comments_path)
print(f"Total Sampled Comments: {len(comments)}")

# Load Post Metadata (for Dates)
meta_files = [
    "data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv",
    "data/control/china_posts_merged.csv", "data/control/china_posts_hanoi_extended.csv",
    "data/control/iran_posts_merged.csv", "data/control/iran_posts_hanoi_extended.csv",
    "data/control/russia_posts_merged.csv", "data/control/russia_posts_hanoi_extended.csv"
]

print("Loading post metadata...")
dfs = []
for f in meta_files:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f, usecols=['id', 'created_utc'], low_memory=False))

posts = pd.concat(dfs, ignore_index=True)
posts['date'] = pd.to_datetime(posts['created_utc'], unit='s')
posts['period'] = posts['date'].apply(assign_period)

# Merge
merged = pd.merge(comments, posts, left_on='parent_post_id', right_on='id', how='inner')

# Count by Country and Period
print("\n" + "="*60)
print("COMMENT COUNT BY PERIOD (Sampled Top-3)")
print("="*60)
print(f"{'Country':<10} | {'P2 (Singapore)':<15} | {'P3 (Hanoi Fail)':<15} | {'Boost Factor'}")
print("-" * 60)

for country in ['nk', 'china', 'iran', 'russia']:
    subset = merged[merged['country'] == country]
    p2 = len(subset[subset['period'] == 'P2 (Singapore)'])
    p3 = len(subset[subset['period'] == 'P3 (Post-Hanoi)'])
    
    # Original Post Counts (approx) for comparison
    post_subset = posts[posts['id'].isin(subset['parent_post_id'])] # Posts that actually have comments
    post_p2 = len(post_subset[post_subset['period'] == 'P2 (Singapore)'])
    post_p3 = len(post_subset[post_subset['period'] == 'P3 (Post-Hanoi)'])

    # Approx boost
    avg_comments = (p2+p3) / (post_p2+post_p3) if (post_p2+post_p3) > 0 else 0
    
    print(f"{country:<10} | {p2:<15} | {p3:<15} | ~{avg_comments:.1f}x posts")

print("-" * 60)
print("Note: 'P3' includes March-Dec 2019. If count is high (>1000), statistical power is sufficient.")
