
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams['font.family'] = 'sans-serif'

# Output directory
output_dir = "paper/figures"
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv('data/final/nk_final.csv')

# Helper to get ratios
def get_ratios(subset):
    # Sentiment (using 0.05 threshold)
    pos = len(subset[subset['sentiment_score'] > 0.05])
    neg = len(subset[subset['sentiment_score'] < -0.05])
    sent_ratio = pos / neg if neg > 0 else 0
    
    # Framing
    dip = len(subset[subset['frame'] == 'DIPLOMACY'])
    threat = len(subset[subset['frame'] == 'THREAT'])
    frame_ratio = dip / threat if threat > 0 else 0
    
    return sent_ratio, frame_ratio

# Define periods
p1 = df[df['period'] == 'P1_PreSingapore']
p2 = df[df['period'] == 'P2_SingaporeHanoi']
p3 = df[df['period'] == 'P3_PostHanoi']

# Calculate
s1, f1 = get_ratios(p1)
s2, f2 = get_ratios(p2)
s3, f3 = get_ratios(p3)

# Prepare data for plotting
# Pivot logic for line plot: Period on X, Ratio on Y, Hue is Metric
data = pd.DataFrame({
    'Period': ['Pre-Singapore\n(P1)', 'Singapore-Hanoi\n(P2)', 'Post-Hanoi\n(P3)'] * 2,
    'Metric': ['Sentiment Ratio\n(Positive / Negative)'] * 3 + ['Framing Ratio\n(Diplomacy / Threat)'] * 3,
    'Ratio': [s1, s2, s3, f1, f2, f3]
})

# Plot
plt.figure(figsize=(8, 5))

# Use a pointplot (line plot for categorical data)
ax = sns.pointplot(x='Period', y='Ratio', hue='Metric', data=data, 
                   palette=['#595959', '#4169E1'], markers=['o', 's'], scale=1.2)

# Annotate values
for i in range(len(data)):
    row = data.iloc[i]
    # Adjust alignment based on period index
    idx = i % 3
    offset = 0.08 if row['Metric'].startswith('Framing') else -0.08
    val = row['Ratio']
    
    # Manual offsetting for clarity
    if idx == 1: # Peak (P2)
        va = 'bottom'
        adjust = 0.05
    else:
        va = 'top'
        adjust = -0.05
        
    # Use color matching the line
    color = '#4169E1' if row['Metric'].startswith('Framing') else '#595959'
    weight = 'bold' if row['Metric'].startswith('Framing') else 'normal'
    
    ax.text(idx, val + adjust, f"{val:.2f}", 
            ha='center', va=va, color=color, fontweight=weight, fontsize=12)

plt.title('Discourse Trajectory: Pre-Singapore $\\rightarrow$ Post-Hanoi', fontsize=14, pad=20)
plt.ylabel('Ratio Value', fontsize=12)
plt.xlabel('')
plt.legend(title='', loc='upper left')
plt.ylim(0, 1.6)

# Save
output_path = os.path.join(output_dir, "fig_ratio_comparison_3p.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_path}")
