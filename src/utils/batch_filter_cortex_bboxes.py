#!/usr/bin/env python3
"""
ABOUTME: Batch filter all bbox files to only include cortex patches using classification results
ABOUTME: Processes all _bbox.txt files in bbox_files directory and overwrites them with cortex-only versions

Usage: python batch_filter_cortex_bboxes.py [--dry-run] [--backup-dir BACKUP_DIR]
"""

import os
import argparse
import sys
import shutil
from pathlib import Path
from glob import glob
from filter_cortex_bboxes import filter_cortex_bboxes


def find_matching_files(bbox_files_dir, results_dir):
    """
    Find matching bbox and classification result files
    
    Args:
        bbox_files_dir: Directory containing bbox text files
        results_dir: Directory containing classification results
        
    Returns:
        list: List of (bbox_file, json_file) tuples for matching files
    """
    
    bbox_pattern = os.path.join(bbox_files_dir, "*_bbox.txt")
    bbox_files = glob(bbox_pattern)
    
    matching_pairs = []
    missing_results = []
    
    for bbox_file in bbox_files:
        # Extract case ID from bbox filename
        bbox_filename = os.path.basename(bbox_file)
        case_id = bbox_filename.replace("_bbox.txt", "")
        
        # Look for matching classification result
        json_file = os.path.join(results_dir, case_id, f"{case_id}_results.json")
        
        if os.path.exists(json_file):
            matching_pairs.append((bbox_file, json_file))
        else:
            missing_results.append((bbox_file, json_file))
    
    return matching_pairs, missing_results


def create_backup(file_path, backup_dir):
    """Create backup of original file"""
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, os.path.basename(file_path))
        shutil.copy2(file_path, backup_path)
        return backup_path
    return None


def batch_filter_cortex_bboxes(bbox_files_dir, results_dir, dry_run=False, backup_dir=None, verbose=True):
    """
    Batch process all bbox files to filter for cortex patches
    
    Args:
        bbox_files_dir: Directory containing bbox text files
        results_dir: Directory containing classification results
        dry_run: If True, show what would be done without making changes
        backup_dir: Directory to backup original files (optional)
        verbose: Print detailed progress
        
    Returns:
        dict: Summary statistics
    """
    
    # Find matching files
    if verbose:
        print(f"Scanning for bbox files in: {bbox_files_dir}")
        print(f"Looking for results in: {results_dir}")
    
    matching_pairs, missing_results = find_matching_files(bbox_files_dir, results_dir)
    
    if verbose:
        print(f"\nFound {len(matching_pairs)} matching pairs")
        if missing_results:
            print(f"Missing classification results for {len(missing_results)} files:")
            for bbox_file, expected_json in missing_results[:5]:
                print(f"  {os.path.basename(bbox_file)} -> {expected_json}")
            if len(missing_results) > 5:
                print(f"  ... and {len(missing_results) - 5} more")
    
    if not matching_pairs:
        print("No matching files found!")
        return {'processed': 0, 'errors': 0}
    
    # Process files
    processed_stats = []
    errors = []
    
    if dry_run:
        print(f"\n{'='*60}")
        print("DRY RUN - No files will be modified")
        print(f"{'='*60}")
    
    for i, (bbox_file, json_file) in enumerate(matching_pairs, 1):
        case_id = os.path.basename(bbox_file).replace("_bbox.txt", "")
        
        if verbose:
            print(f"\n[{i}/{len(matching_pairs)}] Processing: {case_id}")
        
        try:
            if dry_run:
                # Just check what would happen
                if verbose:
                    print(f"  Would filter: {bbox_file}")
                    print(f"  Using results: {json_file}")
                continue
            
            # Create backup if requested
            backup_path = None
            if backup_dir:
                backup_path = create_backup(bbox_file, backup_dir)
                if verbose:
                    print(f"  Backup created: {backup_path}")
            
            # Filter to temporary file first
            temp_file = bbox_file + ".tmp"
            
            # Run filtering (suppress verbose output for batch processing)
            stats = filter_cortex_bboxes(bbox_file, json_file, temp_file, verbose=False)
            
            # Replace original file with filtered version
            shutil.move(temp_file, bbox_file)
            
            processed_stats.append({
                'case_id': case_id,
                'original_count': stats['total_input'],
                'filtered_count': stats['cortex_kept'],
                'retention_rate': stats['retention_rate'],
                'backup_path': backup_path
            })
            
            if verbose:
                print(f"  ✅ Filtered: {stats['total_input']} → {stats['cortex_kept']} patches ({stats['retention_rate']:.1f}% retention)")
                
        except Exception as e:
            error_msg = f"Error processing {case_id}: {e}"
            errors.append(error_msg)
            if verbose:
                print(f"  ❌ {error_msg}")
            
            # Clean up temp file if it exists
            temp_file = bbox_file + ".tmp"
            if os.path.exists(temp_file):
                os.remove(temp_file)
    
    # Print summary
    if verbose and not dry_run:
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {len(processed_stats)}")
        print(f"Errors: {len(errors)}")
        
        if processed_stats:
            total_original = sum(s['original_count'] for s in processed_stats)
            total_filtered = sum(s['filtered_count'] for s in processed_stats)
            overall_retention = total_filtered / total_original * 100 if total_original > 0 else 0
            
            print(f"Total patches: {total_original} → {total_filtered} ({overall_retention:.1f}% retention)")
            
            # Show retention rate distribution
            retention_rates = [s['retention_rate'] for s in processed_stats]
            print(f"Retention rate range: {min(retention_rates):.1f}% - {max(retention_rates):.1f}%")
            print(f"Average retention rate: {sum(retention_rates)/len(retention_rates):.1f}%")
        
        if backup_dir and processed_stats:
            print(f"\nBackups created in: {backup_dir}")
        
        if errors:
            print(f"\nErrors encountered:")
            for error in errors:
                print(f"  {error}")
    
    return {
        'processed': len(processed_stats),
        'errors': len(errors),
        'stats': processed_stats,
        'error_messages': errors
    }


def main():
    parser = argparse.ArgumentParser(description='Batch filter bbox files to only include cortex patches')
    parser.add_argument('--bbox-dir', 
                       default='/data2/vj724/SwinTransformer_classification/bbox_files',
                       help='Directory containing bbox text files')
    parser.add_argument('--results-dir',
                       default='/data2/vj724/SwinTransformer_classification/inference_results', 
                       help='Directory containing classification results')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--backup-dir', 
                       help='Directory to backup original files (recommended)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Check input directories exist
    if not os.path.exists(args.bbox_dir):
        print(f"Error: Bbox directory not found: {args.bbox_dir}")
        sys.exit(1)
        
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Warning for non-dry-run without backup
    if not args.dry_run and not args.backup_dir and not args.quiet:
        response = input("\n⚠️  You're about to overwrite bbox files without backup. Continue? (y/N): ")
        if response.lower() != 'y':
            print("Aborted. Use --backup-dir to create backups or --dry-run to preview changes.")
            sys.exit(0)
    
    try:
        # Run batch processing
        summary = batch_filter_cortex_bboxes(
            args.bbox_dir,
            args.results_dir, 
            dry_run=args.dry_run,
            backup_dir=args.backup_dir,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            if args.dry_run:
                print(f"\n🔍 Dry run complete. Found {summary['processed']} files to process.")
                print("Use without --dry-run to apply filtering.")
            else:
                print(f"\n✅ Batch processing complete!")
                print(f"Successfully processed: {summary['processed']} files")
                if summary['errors'] > 0:
                    print(f"Errors: {summary['errors']} files")
                    sys.exit(1)
                    
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()