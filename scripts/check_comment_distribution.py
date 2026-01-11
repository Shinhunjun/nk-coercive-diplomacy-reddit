import pandas as pd
import glob
import os
from datetime import datetime

# Define P1, P2, P3 periods
# P1: 2017-01-01 to 2018-06-11 (Pre-Singapore)
# P2: 2018-06-13 to 2019-02-27 (Post-Singapore / Pre-Hanoi)
# P3: 2019-03-01 to 2019-12-31 (Post-Hanoi)

P1_START = datetime(2017, 1, 1).timestamp()
P1_END = datetime(2018, 6, 11).timestamp()
P2_START = datetime(2018, 6, 13).timestamp()
P2_END = datetime(2019, 2, 27).timestamp()
P3_START = datetime(2019, 3, 1).timestamp()
P3_END = datetime(2019, 12, 31).timestamp()

def get_period(timestamp):
    if P1_START <= timestamp <= P1_END:
        return 'P1'
    elif P2_START <= timestamp <= P2_END:
        return 'P2'
    elif P3_START <= timestamp <= P3_END:
        return 'P3'
    else:
        return 'Out'

print("Loading comments...")
comments_df = pd.read_csv('data/comments_to_classify_top3.csv')
print(f"Total comments: {len(comments_df)}")

# Load post data to get timestamps
post_files = {
    'North Korea': 'data/nk/nk_posts_full.csv', # Or processed/nk_posts_roberta.csv
    'China': 'data/control/china_posts_roberta.csv', # Try to find files with created_utc
    'Iran': 'data/control/iran_posts_roberta.csv',
    'Russia': 'data/control/russia_posts_roberta.csv'
}

# Fallback paths if above don't exist or don't generally cover everything
# Let's try to find best post files.
# Checking 'data/processed/' usually has the analysis ready files.

print("Loading post metadata...")
posts_metadata = []

# Function to load post data safely
def load_post_data(country):
    # Try different potential paths
    paths = [
        f'data/processed/{country.lower().replace(" ", "")}_posts_roberta.csv',
        f'data/control/{country.lower().replace(" ", "")}_posts_roberta.csv',
        f'data/nk/nk_posts_full.csv' if country == 'North Korea' else None
    ]
    
    df_list = []
    for p in paths:
        if p and os.path.exists(p):
            try:
                # Read only needed columns
                # 'id' is usually the post id. 'created_utc' is timestamp.
                df = pd.read_csv(p, usecols=['id', 'created_utc'])
                df['country'] = country
                df_list.append(df)
                print(f"Loaded {len(df)} posts from {p}")
            except ValueError:
                 print(f"Skipping {p}, missing columns")
            except Exception as e:
                print(f"Error loading {p}: {e}")
    
    if df_list:
        return pd.concat(df_list).drop_duplicates(subset=['id'])
    return pd.DataFrame()

all_posts = []
for country in ['North Korea', 'China', 'Iran', 'Russia']:
    country_posts = load_post_data(country)
    all_posts.append(country_posts)

posts_df = pd.concat(all_posts)
print(f"Total posts metadata loaded: {len(posts_df)}")

# Join
print("Joining comments with posts...")
# Comments has 'parent_post_id', Posts has 'id'
# Ensure types match
comments_df['parent_post_id'] = comments_df['parent_post_id'].astype(str)
posts_df['id'] = posts_df['id'].astype(str)

merged_df = comments_df.merge(posts_df[['id', 'created_utc']], left_on='parent_post_id', right_on='id', how='left')

print(f"Merged count: {len(merged_df)}")
print(f"Missing timestamps: {merged_df['created_utc'].isna().sum()}")

# Assign periods
merged_df['period'] = merged_df['created_utc'].apply(lambda x: get_period(x) if pd.notnull(x) else 'Unknown')

# Group by Country and Period
distribution = merged_df.groupby(['country', 'period']).size().unstack(fill_value=0)

print("\n=== Comment Distribution by Country and Period ===")
print(distribution)

# Calculate total coverage
valid_comments = merged_df[merged_df['period'].isin(['P1', 'P2', 'P3'])]
print(f"\nTotal analyzable comments (P1+P2+P3): {len(valid_comments)}")
