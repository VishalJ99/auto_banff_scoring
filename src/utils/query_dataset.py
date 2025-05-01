import os
import pandas as pd
import csv
from pathlib import Path
import re

# Input directory with SVS files
svs_dir = "/vol/biomedic3/histopatho/win_share/"

# Output CSV file
output_csv = "svs_scores.csv"

# File to track unmatched files
unmatched_file = "unmatched.txt"

# Load Excel file with pandas
try:
    df = pd.read_excel("batch_2_anon.xlsx")
except Exception as e:
    print(f"Error loading Excel file: {e}")
    exit(1)

# Required score columns
score_columns = ['I', 'iIFTA', 'T', 'TI', 'V', 'Glomerulitis', 'PTC']

# Create CSV file for output
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write header
    csvwriter.writerow(['filepath'] + score_columns)
    
    # Store unmatched files
    unmatched_files = []
    
    # Walk through the directory
    for root, dirs, files in os.walk(svs_dir):
        for file in files:
            if file.lower().endswith('.svs'):
                filepath = os.path.join(root, file)
                print(f"Processing file: {file}")
                
                # Look for match in the AnonymousFilename column
                found_match = False
                
                # Extract the filename part without path
                filename = file
                
                for index, row in df.iterrows():
                    # Check if AnonymousFilename exists in the row
                    if 'AnonymousFilename' in row and pd.notna(row['AnonymousFilename']):
                        # Check each file in semicolon-separated list
                        anon_files = row['AnonymousFilename'].split('; ')
                        
                        if filename in anon_files:
                            found_match = True
                            
                            # Get scores for each column
                            scores = []
                            for col in score_columns:
                                if col in row and pd.notna(row[col]):
                                    score = str(row[col])
                                    
                                    # Remove ? if present
                                    score = score.replace('?', '')
                                    
                                    # Skip if score is 99
                                    if score == '99':
                                        score = ''
                                        
                                    scores.append(score)
                                else:
                                    scores.append('')
                            
                            # Write to CSV
                            csvwriter.writerow([filepath] + scores)
                            break
                
                if not found_match:
                    unmatched_files.append(filepath)
                    print(f"No match found for: {file}")

# Write unmatched files to a text file
with open(unmatched_file, 'w') as f:
    for filepath in unmatched_files:
        f.write(f"{filepath}\n")

print(f"Processing complete. Results saved to {output_csv}")
print(f"Unmatched files saved to {unmatched_file}")