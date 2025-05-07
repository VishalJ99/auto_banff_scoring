#!/usr/bin/env python3

import pandas as pd
import numpy as np

def main():
    # Read the PAS scores CSV
    df = pd.read_csv('pas_scores.csv')
    
    # Score columns to analyze
    score_columns = ['I', 'iIFTA', 'T', 'TI', 'V', 'Glomerulitis', 'PTC']
    
    print("PAS Scores Analysis")
    print("=" * 50)
    
    for col in score_columns:
        if col in df.columns:
            # Get value counts and sort by value
            value_counts = df[col].value_counts().sort_index()
            
            print(f"\n{col} Scores:")
            print("-" * 30)
            print("Value | Count")
            print("-" * 30)
            
            for value, count in value_counts.items():
                if isinstance(value, (int, float)) and not np.isnan(value):  # Skip NaN values
                    print(f"{value:5.1f} | {count:5d}")
            
            # Print total count of non-null values
            total = value_counts.sum()
            print("-" * 30)
            print(f"Total non-null values: {total}")
            
            # Print percentage of null values if any
            null_count = df[col].isna().sum()
            if null_count > 0:
                null_percentage = (null_count / len(df)) * 100
                print(f"Null values: {null_count} ({null_percentage:.1f}%)")

if __name__ == "__main__":
    main() 