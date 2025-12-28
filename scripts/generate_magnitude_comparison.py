
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

# Calculate
s1, f1 = get_ratios(p1)
s2, f2 = get_ratios(p2)

s_change = ((s2 - s1) / s1) * 100
f_change = ((f2 - f1) / f1) * 100

# Prepare data for plotting
data = pd.DataFrame({
    'Metric': ['Sentiment Ratio\n(Pos/Neg)', 'Framing Ratio\n(Dip/Threat)'],
    'Change (%)': [s_change, f_change]
})

# Plot
plt.figure(figsize=(6, 5))
ax = sns.barplot(x='Metric', y='Change (%)', data=data, palette=['#95a5a6', '#2ecc71'])

# Annotate values
for i, v in enumerate([s_change, f_change]):
    ax.text(i, v + 2, f"+{v:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

plt.title('Magnitude of Shift: P1 (Pre-Summit) $\\rightarrow$ P2 (Summit)', fontsize=14, pad=20)
plt.ylabel('Percentage Increase (%)', fontsize=12)
plt.xlabel('')
plt.ylim(0, 250)  # Give some headroom

# Save
output_path = os.path.join(output_dir, "fig_magnitude_comparison.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved figure to {output_path}")
