#!/usr/bin/env python3

import pandas as pd
import re

# Score columns to extract
score_columns = ['I', 'iIFTA', 'T', 'TI', 'V', 'Glomerulitis', 'PTC']

def extract_filename(line):
    # Extract the filename from the line (everything before .svs)
    match = re.search(r'(anon_[^.]+\.svs)', line)
    if match:
        return match.group(1)
    return None

def main():
    # Read the PAS cases file
    with open('pas_cases.txt', 'r') as f:
        pas_cases = [line.strip() for line in f.readlines()]

    # Extract filenames from the PAS cases
    filenames = [extract_filename(case) for case in pas_cases if extract_filename(case)]

    # Read the Excel file
    df = pd.read_excel('data/batch_2_anon.xlsx')

    # Create a new DataFrame for results
    results = []
    
    # For each filename, find matching rows and extract scores
    for filename in filenames:
        if filename is None:
            continue
            
        # Find rows where the filename matches in the semicolon-separated list
        for index, row in df.iterrows():
            # Check if AnonymousFilename exists in the row
            if 'AnonymousFilename' in row and pd.notna(row['AnonymousFilename']):
                # Check each file in semicolon-separated list
                anon_files = row['AnonymousFilename'].split('; ')
                
                if filename in anon_files:
                    # Create a dictionary with filename and scores
                    result = {'filename': filename}
                    for col in score_columns:
                        result[col] = row.get(col, None)
                    
                    results.append(result)
                    break  # Found a match, no need to continue searching

    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('pas_scores.csv', index=False)
    print(f"Extracted scores for {len(results)} PAS cases to pas_scores.csv")

if __name__ == "__main__":
    main() 