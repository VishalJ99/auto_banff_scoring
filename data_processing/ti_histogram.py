import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
df = pd.read_csv("/data2/ac2220/data_handling/data.txt")

df['TI'] = pd.to_numeric(df['TI'], errors='coerce')
ti_scores = df['TI'].dropna()

# Style for thesis
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
bins = [-0.5, 0.5, 1.5, 2.5, 3.5]
sns.histplot(
    ti_scores, 
    bins=bins, 
    discrete=True, 
    kde=False, 
    color="gray", 
    edgecolor="black", 
    ax=ax
)

# Labels and layout
ax.set_title("Distribution of Total Inflammation (TI) Scores")
ax.set_xlabel("TI Score")
ax.set_ylabel("Number of Slides")
ax.set_xticks([0, 1, 2, 3])
ax.set_xlim(-0.5, 3.5)
ax.grid(False)

# Add counts on bars
for patch in ax.patches:
    height = patch.get_height()
    if height > 0:
        ax.text(patch.get_x() + patch.get_width() / 2, height - 2,  # <- place *inside* the bar
                int(height), ha='center', va='top', fontsize=12, color='white')

# Save
plt.tight_layout()
plt.savefig("ti_score_distribution.png", dpi=600, bbox_inches='tight')

