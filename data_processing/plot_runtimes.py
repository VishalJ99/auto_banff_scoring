import json
import csv
import numpy as np
from pathlib import Path
from shapely.geometry import Point, MultiPoint
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.metrics import cohen_kappa_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

def analyze_banff_performance(df):
    """Analyze correlation and classification performance"""
    # Convert banff predictions to numeric
    banff_to_numeric = {'ti0': 0, 'ti1': 1, 'ti2': 2, 'ti3': 3}
    df['banff_numeric'] = df['banff_ti_score'].map(banff_to_numeric)
    
    # Calculate metrics
    correlation = df['TI'].corr(df['inflammation_pct'])
    kappa = cohen_kappa_score(df['TI'], df['banff_numeric'])
    
    print(f"Simple Banff method performance:")
    print(f"Correlation: r = {correlation:.3f}")
    print(f"Cohen's Kappa: κ = {kappa:.3f}")
    
    return correlation, kappa

def plot_percentage_vs_ti(df, correlation, kappa):
    """Plot inflammation percentage vs TI scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot: TI vs percentage
    ax1.scatter(df['TI'], df['inflammation_pct'], alpha=0.6, s=50)
    
    # Add regression line
    z = np.polyfit(df['TI'], df['inflammation_pct'], 1)
    p = np.poly1d(z)
    ax1.plot(df['TI'], p(df['TI']), "b-", alpha=0.8, linewidth=2, 
             label=f'Linear fit (r={correlation:.3f})')
    
    # Add Banff threshold lines
    ax1.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='TI0/TI1 threshold (10%)')
    ax1.axhline(y=26, color='orange', linestyle='--', alpha=0.7, label='TI1/TI2 threshold (26%)')
    ax1.axhline(y=50, color='purple', linestyle='--', alpha=0.7, label='TI2/TI3 threshold (50%)')
    
    ax1.set_xlabel('TI Grade')
    ax1.set_ylabel('Inflammation Percentage (%)')
    ax1.set_title(f'Inflammation Percentage vs TI Grade\n(r={correlation:.3f}, κ={kappa:.3f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([0, 1, 2, 3])
    
    # Box plot: percentage distribution by TI grade
    box_data = [df[df['TI'] == ti]['inflammation_pct'].values for ti in [0, 1, 2, 3]]
    ax2.boxplot(box_data, labels=['0', '1', '2', '3'])
    
    # Add threshold lines to box plot
    ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7)
    ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7)
    ax2.axhline(y=50, color='purple', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('TI Grade')
    ax2.set_ylabel('Inflammation Percentage (%)')
    ax2.set_title('Percentage Distribution by TI Grade')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('inflammation_percentage_vs_ti_simple_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print confusion matrix
    print(f"\nConfusion Matrix (rows=Expert, cols=Banff):")
    confusion_matrix = pd.crosstab(df['TI'], df['banff_numeric'], margins=True)
    print(confusion_matrix)
    
    return correlation, kappa

# Load data and run analysis
df = pd.read_csv("/data2/ac2220/data_handling/banff_ti_scores_simple_method.csv")

# Calculate performance metrics
correlation, kappa = analyze_banff_performance(df)

# Create plots with metrics
plot_correlation, plot_kappa = plot_percentage_vs_ti(df, correlation, kappa)

print(f"\nFinal Results:")
print(f"Traditional Banff Method: r = {correlation:.3f}, κ = {kappa:.3f}")