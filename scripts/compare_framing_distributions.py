
import pandas as pd
import os

# Load Old Monthly Framing (Pre-V2)
old_nk = pd.read_csv('data/framing/nk_monthly_framing.csv')

# Calculate OLD stats
print("="*80)
print("📊 OLD FRAMING DATA (Pre-V2 Prompt)")
print("="*80)
print(f"Months: {old_nk['month'].min()} ~ {old_nk['month'].max()}")
print(f"Total Posts: {old_nk['post_count'].sum()}")
print(f"Mean Framing Score: {old_nk['framing_mean'].mean():.3f}")
print(f"\nMonthly Framing Means:")
print(old_nk[['month', 'framing_mean', 'post_count']].to_string())

# Load NEW V2 Framing (Individual Posts)
print("\n" + "="*80)
print("📊 NEW FRAMING DATA (V2 Prompt)")
print("="*80)

SCALE = {
    'THREAT': -2, 'ECONOMIC': -1, 'NEUTRAL': 0, 'HUMANITARIAN': 1, 'DIPLOMACY': 2, 'ERROR': 0
}

META_FILES = {
    "nk": ["data/nk/nk_posts_merged.csv", "data/nk/nk_posts_hanoi_extended.csv"],
}

# Load V2 Results
v2 = pd.read_csv('data/results/final_framing_v2/nk_framing_v2.csv')
v2['score'] = v2['frame'].map(SCALE).fillna(0)

# Load Dates
meta_dfs = []
for p in META_FILES['nk']:
    if os.path.exists(p):
        meta_dfs.append(pd.read_csv(p, usecols=['id', 'created_utc']))
        
meta = pd.concat(meta_dfs, ignore_index=True).drop_duplicates(subset=['id'])
v2 = pd.merge(v2, meta, on='id', how='inner')
v2['date'] = pd.to_datetime(v2['created_utc'], unit='s')
v2['month'] = v2['date'].dt.to_period('M').astype(str)

# Calculate NEW monthly aggregates
new_monthly = v2.groupby('month').agg({'score': ['mean', 'count']}).reset_index()
new_monthly.columns = ['month', 'framing_mean', 'post_count']

print(f"Months: {new_monthly['month'].min()} ~ {new_monthly['month'].max()}")
print(f"Total Posts: {new_monthly['post_count'].sum()}")
print(f"Mean Framing Score: {new_monthly['framing_mean'].mean():.3f}")
print(f"\nMonthly Framing Means:")
print(new_monthly[['month', 'framing_mean', 'post_count']].to_string())

print("\n" + "="*80)
print("📊 FRAMING DISTRIBUTION COMPARISON")
print("="*80)
print(f"\nNEW V2 Frame Distribution:")
print(v2['frame'].value_counts().to_string())
print(f"\n% NEUTRAL: {(v2['frame'] == 'NEUTRAL').sum() / len(v2) * 100:.1f}%")
