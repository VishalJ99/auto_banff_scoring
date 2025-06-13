import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

def analyze_slide_agreement(df: pd.DataFrame, 
                          ti_col: str = 'TI',
                          predicted_col: str = 'optimized_inflammation_score',
                          slide_id_col: str = 'slide') -> Dict:
    """
    Analyze high and low agreement slides for each TI score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing slide data with TI scores and predictions
    ti_col : str
        Column name for expert TI scores
    predicted_col : str  
        Column name for predicted inflammation scores
    slide_id_col : str
        Column name for slide identifiers
        
    Returns:
    --------
    Dict containing analysis results
    """
    
    # Calculate residuals (prediction error)
    df = df.copy()
    df['residual'] = df[predicted_col] - df[ti_col]
    df['abs_residual'] = np.abs(df['residual'])
    
    results = {
        'summary': {},
        'high_agreement': {},
        'low_agreement': {},
        'statistics': {}
    }
    
    # Overall statistics
    results['statistics']['overall_mae'] = df['abs_residual'].mean()
    results['statistics']['overall_rmse'] = np.sqrt((df['residual']**2).mean())
    results['statistics']['overall_std'] = df['residual'].std()
    
    # Analysis by TI grade
    for ti_grade in sorted(df[ti_col].unique()):
        ti_subset = df[df[ti_col] == ti_grade].copy()
        n_slides = len(ti_subset)
        
        if n_slides == 0:
            continue
            
        # Calculate percentiles for this TI grade
        mae_25th = ti_subset['abs_residual'].quantile(0.25)
        mae_75th = ti_subset['abs_residual'].quantile(0.75)
        
        # High agreement: bottom 25% of absolute residuals for this TI grade
        high_agreement_mask = ti_subset['abs_residual'] <= mae_25th
        high_agreement_slides = ti_subset[high_agreement_mask]
        
        # Low agreement: top 25% of absolute residuals for this TI grade
        low_agreement_mask = ti_subset['abs_residual'] >= mae_75th
        low_agreement_slides = ti_subset[low_agreement_mask]
        
        # Store results
        results['high_agreement'][f'TI_{int(ti_grade)}'] = {
            'slides': high_agreement_slides[[slide_id_col, ti_col, predicted_col, 'residual', 'abs_residual']].to_dict('records'),
            'mean_residual': high_agreement_slides['residual'].mean(),
            'mean_abs_residual': high_agreement_slides['abs_residual'].mean(),
            'count': len(high_agreement_slides)
        }
        
        results['low_agreement'][f'TI_{int(ti_grade)}'] = {
            'slides': low_agreement_slides[[slide_id_col, ti_col, predicted_col, 'residual', 'abs_residual']].to_dict('records'),
            'mean_residual': low_agreement_slides['residual'].mean(), 
            'mean_abs_residual': low_agreement_slides['abs_residual'].mean(),
            'count': len(low_agreement_slides)
        }
        
        results['summary'][f'TI_{int(ti_grade)}'] = {
            'total_slides': n_slides,
            'mean_residual': ti_subset['residual'].mean(),
            'std_residual': ti_subset['residual'].std(),
            'mae': ti_subset['abs_residual'].mean(),
            'high_agreement_threshold': mae_25th,
            'low_agreement_threshold': mae_75th
        }
    
    return results

def get_extreme_cases(df: pd.DataFrame,
                     ti_col: str = 'TI', 
                     predicted_col: str = 'optimized_inflammation_score',
                     slide_id_col: str = 'slide',
                     n_cases: int = 3) -> Dict:
    """
    Get the most extreme high and low agreement cases overall.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing slide data
    n_cases : int
        Number of extreme cases to return for each category
        
    Returns:
    --------
    Dict with extreme cases
    """
    df = df.copy()
    df['residual'] = df[predicted_col] - df[ti_col] 
    df['abs_residual'] = np.abs(df['residual'])
    
    # Best agreement cases (smallest absolute residuals)
    best_cases = df.nsmallest(n_cases, 'abs_residual')
    
    # Worst agreement cases (largest absolute residuals)  
    worst_cases = df.nlargest(n_cases, 'abs_residual')
    
    # Systematic over-estimation cases (largest positive residuals)
    over_estimation = df.nlargest(n_cases, 'residual')
    
    # Systematic under-estimation cases (largest negative residuals)
    under_estimation = df.nsmallest(n_cases, 'residual')
    
    return {
        'best_agreement': best_cases[[slide_id_col, ti_col, predicted_col, 'residual', 'abs_residual']].to_dict('records'),
        'worst_agreement': worst_cases[[slide_id_col, ti_col, predicted_col, 'residual', 'abs_residual']].to_dict('records'),
        'over_estimation': over_estimation[[slide_id_col, ti_col, predicted_col, 'residual', 'abs_residual']].to_dict('records'),
        'under_estimation': under_estimation[[slide_id_col, ti_col, predicted_col, 'residual', 'abs_residual']].to_dict('records')
    }

def plot_agreement_analysis(df: pd.DataFrame,
                          ti_col: str = 'TI',
                          predicted_col: str = 'optimized_inflammation_score',
                          slide_id_col: str = 'slide'):
    """
    Create visualizations for agreement analysis.
    """
    df = df.copy()
    df['residual'] = df[predicted_col] - df[ti_col]
    df['abs_residual'] = np.abs(df['residual'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Residuals by TI grade
    sns.boxplot(data=df, x=ti_col, y='residual', ax=axes[0,0])
    axes[0,0].set_title('Prediction Residuals by TI Grade')
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0,0].set_ylabel('Residual (Predicted - Expert TI)')
    
    # 2. Absolute residuals by TI grade  
    sns.boxplot(data=df, x=ti_col, y='abs_residual', ax=axes[0,1])
    axes[0,1].set_title('Absolute Prediction Error by TI Grade')
    axes[0,1].set_ylabel('Absolute Residual')
    
    # 3. Scatter plot with residual coloring
    scatter = axes[1,0].scatter(df[ti_col], df[predicted_col], 
                              c=df['abs_residual'], cmap='viridis_r', alpha=0.7)
    axes[1,0].plot([df[ti_col].min(), df[ti_col].max()], 
                   [df[ti_col].min(), df[ti_col].max()], 'r--', alpha=0.7)
    axes[1,0].set_xlabel('Expert TI Score')
    axes[1,0].set_ylabel('Predicted Inflammation Score')
    axes[1,0].set_title('Predicted vs Expert TI (colored by error)')
    plt.colorbar(scatter, ax=axes[1,0], label='Absolute Error')
    
    # 4. Histogram of residuals
    axes[1,1].hist(df['residual'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[1,1].set_xlabel('Residual (Predicted - Expert TI)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Distribution of Prediction Residuals')
    
    plt.tight_layout()
    return fig

def print_agreement_summary(results: Dict):
    """
    Print a formatted summary of agreement analysis results.
    """
    print("="*60)
    print("SLIDE AGREEMENT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Mean Absolute Error: {results['statistics']['overall_mae']:.3f}")
    print(f"Root Mean Square Error: {results['statistics']['overall_rmse']:.3f}")
    print(f"Standard Deviation: {results['statistics']['overall_std']:.3f}")
    
    print(f"\nBY TI GRADE:")
    print("-" * 80)
    print(f"{'TI Grade':<10} {'N Slides':<10} {'Mean Error':<12} {'MAE':<10} {'High Agr.':<10} {'Low Agr.':<10}")
    print("-" * 80)
    
    for ti_grade in sorted(results['summary'].keys()):
        stats = results['summary'][ti_grade]
        high_count = results['high_agreement'][ti_grade]['count']
        low_count = results['low_agreement'][ti_grade]['count']
        
        print(f"{ti_grade:<10} {stats['total_slides']:<10} {stats['mean_residual']:<12.3f} "
              f"{stats['mae']:<10.3f} {high_count:<10} {low_count:<10}")
    
    print("\nHIGH AGREEMENT EXAMPLES (by TI grade):")
    for ti_grade in sorted(results['high_agreement'].keys()):
        slides = results['high_agreement'][ti_grade]['slides'][:2]  # Show top 2
        print(f"\n{ti_grade}:")
        for slide in slides:
            print(f"  Slide {slide['slide']}: Expert={slide['TI']:.1f}, "
                  f"Predicted={slide['optimized_inflammation_score']:.3f}, "
                  f"Error={slide['abs_residual']:.3f}")
    
    print("\nLOW AGREEMENT EXAMPLES (by TI grade):")
    for ti_grade in sorted(results['low_agreement'].keys()):
        slides = results['low_agreement'][ti_grade]['slides'][:2]  # Show top 2
        print(f"\n{ti_grade}:")
        for slide in slides:
            print(f"  Slide {slide['slide']}: Expert={slide['TI']:.1f}, "
                  f"Predicted={slide['optimized_inflammation_score']:.3f}, "
                  f"Error={slide['abs_residual']:.3f}")

# Example usage:
if __name__ == "__main__":
    # Load your actual data - update this path to your CSV file
    try:
        df = pd.read_csv('/data2/ac2220/data_handling/cluster_metrics_with_zscores.csv')  # UPDATE THIS PATH
        print("Available columns in your data:")
        print(df.columns.tolist())
        print(f"\nDataFrame shape: {df.shape}")
        print(f"\nFirst few rows:")
        print(df.head())
        
        # Updated with your actual column names:
        results = analyze_slide_agreement(
            df,
            ti_col='TI',  # Your expert TI score column
            predicted_col='optimized_inflammation_score',  # Your predicted score column
            slide_id_col='slide'  # Your slide ID column
        )
        print_agreement_summary(results)
        
    except FileNotFoundError:
        print("Please update the file path to your actual results CSV file")
        print("Also check the column names in your data and update the function call accordingly")
        
        # Fallback: create mock data for demonstration
        print("\nRunning with mock data for demonstration...")
        np.random.seed(42)
        n_slides = 93
        
        # Create mock data that roughly matches your distribution
        ti_scores = np.random.choice([0, 1, 2, 3], size=n_slides, p=[25/93, 22/93, 12/93, 34/93])
        predicted_scores = ti_scores + np.random.normal(0, 0.5, n_slides)  # Some noise
        slide_ids = [f"slide_{i:03d}" for i in range(n_slides)]
        
        df = pd.DataFrame({
            'slide': slide_ids,
            'TI': ti_scores,
            'optimized_inflammation_score': predicted_scores
        })
        
        # Run analysis with mock data
        results = analyze_slide_agreement(df)
        print_agreement_summary(results)
    
    # Get extreme cases
    extreme_cases = get_extreme_cases(df, n_cases=3)
    
    print("\n" + "="*60)
    print("EXTREME CASES ANALYSIS")
    print("="*60)
    
    print("\nBEST AGREEMENT CASES:")
    for i, slide in enumerate(extreme_cases['best_agreement'], 1):
        print(f"{i}. Slide {slide['slide']}: Expert={slide['TI']:.1f}, "
              f"Predicted={slide['optimized_inflammation_score']:.3f}, "
              f"Error={slide['abs_residual']:.3f}")
    
    print("\nWORST AGREEMENT CASES:")
    for i, slide in enumerate(extreme_cases['worst_agreement'], 1):
        print(f"{i}. Slide {slide['slide']}: Expert={slide['TI']:.1f}, "
              f"Predicted={slide['optimized_inflammation_score']:.3f}, "
              f"Error={slide['abs_residual']:.3f}")
    
    # Create plots
    fig = plot_agreement_analysis(df)
    plt.savefig("agreement_analysis.png")