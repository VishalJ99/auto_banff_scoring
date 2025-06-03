#!/usr/bin/env python3
"""
ABOUTME: Filter bbox files to only include patches classified as cortex
ABOUTME: Takes bbox txt file and classification JSON, outputs filtered bbox txt file

Usage: python filter_cortex_bboxes.py bbox_file.txt classification_results.json output_file.txt
"""

import json
import argparse
import sys
from pathlib import Path


def load_classification_results(json_path):
    """Load classification results from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_bbox_file(bbox_path):
    """Load bbox coordinates from text file (ymin xmin ymax xmax format)"""
    bboxes = []
    with open(bbox_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    coords = list(map(int, line.split()))
                    if len(coords) != 4:
                        print(f"Warning: Line {line_num} has {len(coords)} coordinates, expected 4")
                        continue
                    ymin, xmin, ymax, xmax = coords
                    bboxes.append((ymin, xmin, ymax, xmax))
                except ValueError as e:
                    print(f"Warning: Could not parse line {line_num}: {line} - {e}")
                    continue
    return bboxes


def match_bbox_to_classification(bbox, classification_results):
    """
    Match bbox coordinates to classification results
    
    Args:
        bbox: (ymin, xmin, ymax, xmax) tuple
        classification_results: JSON data with patch_results
    
    Returns:
        dict or None: Matching classification result if found
    """
    ymin, xmin, ymax, xmax = bbox
    
    # Expected patch size (should be 1024x1024)
    expected_width = xmax - xmin
    expected_height = ymax - ymin
    
    for patch_result in classification_results['patch_results']:
        # Classification coordinates are [x, y] (top-left corner)
        class_x, class_y = patch_result['coordinates']
        
        # Check if classification coordinates match bbox top-left corner
        if class_x == xmin and class_y == ymin:
            return patch_result
    
    return None


def filter_cortex_bboxes(bbox_path, json_path, output_path, verbose=True):
    """
    Filter bbox file to only include cortex patches
    
    Args:
        bbox_path: Path to input bbox text file
        json_path: Path to classification JSON file
        output_path: Path to output filtered bbox file
        verbose: Print filtering statistics
    
    Returns:
        dict: Statistics about filtering process
    """
    
    # Load data
    if verbose:
        print(f"Loading bbox file: {bbox_path}")
    bboxes = load_bbox_file(bbox_path)
    
    if verbose:
        print(f"Loading classification results: {json_path}")
    classification_data = load_classification_results(json_path)
    
    # Filter bboxes
    cortex_bboxes = []
    filtered_examples = {
        'kept_cortex': [],
        'removed_medulla': [],
        'removed_non_kidney': [],
        'removed_transition': [],
        'no_match': []
    }
    
    if verbose:
        print(f"\nProcessing {len(bboxes)} bboxes...")
    
    for i, bbox in enumerate(bboxes):
        match = match_bbox_to_classification(bbox, classification_data)
        
        if match is None:
            filtered_examples['no_match'].append((i, bbox))
            if verbose and len(filtered_examples['no_match']) <= 3:
                print(f"  Bbox {i}: {bbox} - NO MATCH in classification results")
            continue
        
        class_name = match['class_name']
        
        if class_name == 'cortex':
            cortex_bboxes.append(bbox)
            if len(filtered_examples['kept_cortex']) < 3:
                filtered_examples['kept_cortex'].append((i, bbox, match))
        else:
            # Store examples of removed patches by type
            if class_name in filtered_examples and len(filtered_examples[f'removed_{class_name}']) < 3:
                filtered_examples[f'removed_{class_name}'].append((i, bbox, match))
    
    # Save filtered bboxes
    if verbose:
        print(f"\nSaving {len(cortex_bboxes)} cortex bboxes to: {output_path}")
    
    with open(output_path, 'w') as f:
        for ymin, xmin, ymax, xmax in cortex_bboxes:
            f.write(f"{ymin} {xmin} {ymax} {xmax}\n")
    
    # Print statistics and examples
    if verbose:
        print(f"\n{'='*60}")
        print("FILTERING RESULTS")
        print(f"{'='*60}")
        print(f"Total input bboxes: {len(bboxes)}")
        print(f"Cortex bboxes kept: {len(cortex_bboxes)}")
        print(f"Bboxes removed: {len(bboxes) - len(cortex_bboxes)}")
        print(f"Retention rate: {len(cortex_bboxes)/len(bboxes)*100:.1f}%")
        
        # Show classification breakdown
        class_counts = classification_data['class_counts']
        print(f"\nClassification breakdown:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} patches")
        
        # Show examples
        print(f"\n{'='*60}")
        print("EXAMPLES")
        print(f"{'='*60}")
        
        if filtered_examples['kept_cortex']:
            print(f"\n✅ KEPT (cortex examples):")
            for i, bbox, match in filtered_examples['kept_cortex']:
                print(f"  Bbox {i}: {bbox} -> {match['class_name']} (confidence: prediction {match['prediction']})")
        
        for class_type in ['medulla', 'non_kidney', 'transition']:
            examples = filtered_examples[f'removed_{class_type}']
            if examples:
                print(f"\n❌ REMOVED ({class_type} examples):")
                for i, bbox, match in examples:
                    print(f"  Bbox {i}: {bbox} -> {match['class_name']} (confidence: prediction {match['prediction']})")
        
        if filtered_examples['no_match']:
            print(f"\n⚠️  NO MATCH ({len(filtered_examples['no_match'])} total):")
            for i, bbox in filtered_examples['no_match'][:3]:
                print(f"  Bbox {i}: {bbox} -> No matching classification found")
            if len(filtered_examples['no_match']) > 3:
                print(f"  ... and {len(filtered_examples['no_match']) - 3} more")
    
    # Return statistics
    stats = {
        'total_input': len(bboxes),
        'cortex_kept': len(cortex_bboxes),
        'removed': len(bboxes) - len(cortex_bboxes),
        'no_match': len(filtered_examples['no_match']),
        'retention_rate': len(cortex_bboxes)/len(bboxes)*100 if bboxes else 0,
        'class_counts': classification_data['class_counts']
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Filter bbox file to only include cortex patches')
    parser.add_argument('bbox_file', help='Input bbox text file (ymin xmin ymax xmax format)')
    parser.add_argument('classification_json', help='Classification results JSON file')
    parser.add_argument('output_file', help='Output filtered bbox text file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not Path(args.bbox_file).exists():
        print(f"Error: Bbox file not found: {args.bbox_file}")
        sys.exit(1)
        
    if not Path(args.classification_json).exists():
        print(f"Error: Classification JSON not found: {args.classification_json}")
        sys.exit(1)
    
    # Run filtering
    try:
        stats = filter_cortex_bboxes(
            args.bbox_file, 
            args.classification_json, 
            args.output_file,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n✅ Filtering complete!")
            print(f"Input: {stats['total_input']} bboxes")
            print(f"Output: {stats['cortex_kept']} cortex bboxes ({stats['retention_rate']:.1f}% retention)")
            
    except Exception as e:
        print(f"Error during filtering: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()