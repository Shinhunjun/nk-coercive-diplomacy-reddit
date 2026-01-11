import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

OUTPUT_DIR = "paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    # Load Post Data (Ground Truth)
    # Assuming standard post framing files exist. Adjust paths if needed.
    posts = []
    # Try loading NK posts framing
    if os.path.exists("data/processed/nk_posts_framing.csv"):
        nk_posts = pd.read_csv("data/processed/nk_posts_framing.csv")
        nk_posts['type'] = 'Post'
        nk_posts['country'] = 'North Korea'
        posts.append(nk_posts)
    
    # Load Comment Data (New Results)
    comments = []
    comment_files = {
        'North Korea': "data/results/nk_comment_framing_final.csv",
        'China': "data/results/china_comment_framing_final.csv",
        'Iran': "data/results/iran_comment_framing_final.csv",
        'Russia': "data/results/russia_comment_framing_final.csv"
    }
    
    for country, path in comment_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['type'] = 'Comment'
            df['country'] = country
            comments.append(df)
            
    if not comments:
        print("Waiting for comment results...")
        return None, None
        
    comments_df = pd.concat(comments)
    
    # If posts loaded
    posts_df = pd.DataFrame()
    if posts:
        posts_df = pd.concat(posts)
        
    return posts_df, comments_df

def plot_framing_comparison(posts, comments, country='North Korea'):
    # Focus on one country
    c_comments = comments[comments['country'] == country].copy()
    
    # Needs 'month' or date
    # Convert timestamps
    if 'created_utc' in c_comments.columns:
        # Filter out error rows
        c_comments['created_utc'] = pd.to_numeric(c_comments['created_utc'], errors='coerce')
        c_comments = c_comments.dropna(subset=['created_utc'])
        
        c_comments['date'] = pd.to_datetime(c_comments['created_utc'], unit='s')
        c_comments['month'] = c_comments['date'].dt.to_period('M')

    # Aggregate by Month & Frame
    # Calculate % of THREAT and DIPLOMACY
    monthly = c_comments.groupby('month')['frame'].value_counts(normalize=True).unstack(fill_value=0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    if 'DIPLOMACY' in monthly.columns:
        plt.plot(monthly.index.astype(str), monthly['DIPLOMACY'], label='Comment: Diplomacy', linestyle='--', marker='o')
    if 'THREAT' in monthly.columns:
        plt.plot(monthly.index.astype(str), monthly['THREAT'], label='Comment: Threat', linestyle='--', marker='x')

    # Add Post data if available
    # (Simplified for now - assumes we just want to see Comment trends first)
    
    plt.title(f"Framing Trends in {country} Comments (Monthly)")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{country}_comment_framing_trends.pdf")
    print(f"Saved {country} plot.")

if __name__ == "__main__":
    posts, comments = load_data()
    if comments is not None:
        plot_framing_comparison(posts, comments, 'North Korea')
        # Can loop others
