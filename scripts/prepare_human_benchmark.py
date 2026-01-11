import pandas as pd

# Load ground truth
gt = pd.read_csv('../data/anonymized/annotations/human_ground_truth.csv')
print(f"Ground Truth: {len(gt)} rows")

# Load posts to get titles
posts = pd.read_csv('../data/processed/nk_posts_framing.csv')
print(f"All Posts: {len(posts)} rows")

# Merge to get titles for the GT posts
merged = pd.merge(gt, posts[['id', 'title']], left_on='post_id', right_on='id', how='left')
print(f"Merged with titles: {len(merged)} rows")

# Check for missing titles
missing = merged['title'].isna().sum()
if missing > 0:
    print(f"⚠️ Warning: {missing} posts missing titles!")
    merged = merged.dropna(subset=['title'])

# Save as input for validation script
# We rename columns to match what the script expects or just use it as sample source
output = merged[['post_id', 'title', 'gold_label']]
output.to_csv('../data/results/human_benchmark_input.csv', index=False)
print(f"✅ Prepared input file: ../data/results/human_benchmark_input.csv ({len(output)} rows)")
