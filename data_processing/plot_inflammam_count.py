import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
import numpy as np

# Path to your .jsonl file
jsonl_path = "/data2/ac2220/macenko_pipeline_output/summary_with_banff_scores.jsonl"

# Load data
with open(jsonl_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Convert to DataFrame
df = pd.DataFrame(data)

# Extract metrics
df['inflamm_per_patch'] = df['normalised'].apply(lambda x: x['inflammatory_per_cortex_patch'])
df['TI'] = pd.to_numeric(df['TI'], errors='coerce')

# Filter valid rows
df_clean = df.dropna(subset=['TI', 'inflamm_per_patch'])

# Extract values
x = df_clean['TI']
y = df_clean['inflamm_per_patch']

# Linear regression
slope, intercept = np.polyfit(x, y, 1)
regression_line = slope * x + intercept

# Pearson correlation
r, p = pearsonr(x, y)

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue', alpha=0.7, label='Slide data')
plt.plot(x, regression_line, color='red', linestyle='--', label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')

# Top-left annotation with exponent notation for p-value
plt.annotate(f"r = {r:.2f}\np = {p:.2e}", xy=(0.05, 0.95), xycoords='axes fraction',
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray'))

plt.xlabel("TI Score")
plt.ylabel("Inflammatory Cells per Cortex Patch")
plt.title("Normalised Inflammatory Count vs TI Grade")
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig("macenko_inflamm_vs_TI.png", dpi=300)
plt.close()

print("✅ Plot saved as 'macenko_inflamm_vs_TI.png'")
