import pandas as pd
import argparse
import sys

def find_files_by_score(score_name, score_value, csv_file="svs_scores.csv"):
    """
    Find all files with a specific score value.
    
    Args:
        score_name: The name of the score column (e.g., 'I', 'T', 'V')
        score_value: The value to search for (e.g., '3', '2')
        csv_file: Path to the CSV file with scores
    
    Returns:
        List of file paths matching the criteria
    """
    try:
        # Load the CSV file
        df = pd.read_csv(csv_file)
        
        # Check if the score column exists
        if score_name not in df.columns:
            print(f"Error: Score '{score_name}' not found in the CSV file.")
            print(f"Available scores: {', '.join([col for col in df.columns if col != 'filepath'])}")
            return []
        
        # Filter for rows where the score matches the value
        # Convert both to strings for comparison
        matches = df[df[score_name].astype(str) == str(score_value)]
        
        # Return the file paths
        return matches['filepath'].tolist()
    
    except Exception as e:
        print(f"Error: {e}")
        return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find files matching a specific score value")
    parser.add_argument("score", help="Score name (e.g., I, iIFTA, T, TI, V, Glomerulitis, PTC)")
    parser.add_argument("value", help="Score value to search for")
    parser.add_argument("--csv", default="svs_scores.csv", help="Path to CSV file with scores (default: svs_scores.csv)")
    
    args = parser.parse_args()
    
    matching_files = find_files_by_score(args.score, args.value, args.csv)
    
    if matching_files:
        print(f"Found {len(matching_files)} files with {args.score} = {args.value}:")
        for filepath in matching_files:
            print(filepath)
    else:
        print(f"No files found with {args.score} = {args.value}")