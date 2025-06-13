import argparse
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import json
from patch_extractor import extract_patches_from_wsi


def evaluate_patch_quality(patch_np):
    """
    Evaluate patch quality for PAS stain normalization reference selection.
    Returns quality score (0-10+) and detailed metrics.
    """
    if patch_np.ndim != 3 or patch_np.shape[2] != 3:
        return 0, {"error": "Invalid patch dimensions"}

    quality_score = 0
    metrics = {}

    # Convert to different color spaces
    gray = cv2.cvtColor(patch_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)

    # --- Inflammation filter ---
    purple_mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 150)
    sat_mask = hsv[:, :, 1] > 30
    val_mask = hsv[:, :, 2] < 200
    inflammatory_mask = purple_mask & sat_mask & val_mask
    inflammatory_fraction = np.sum(inflammatory_mask) / inflammatory_mask.size
    metrics['inflammatory_fraction'] = inflammatory_fraction
    if inflammatory_fraction > 0.15:
        return 0, metrics

    # --- PAS pink fraction reward ---
    pas_mask = ((hsv[:, :, 0] < 20) | (hsv[:, :, 0] > 320)) & (hsv[:, :, 1] > 80)
    pas_fraction = np.sum(pas_mask) / pas_mask.size
    metrics['pas_fraction'] = pas_fraction
    if 0.05 <= pas_fraction <= 0.25:
        quality_score += 1
    elif 0.02 <= pas_fraction <= 0.4:
        quality_score += 0.5

    # --- Brightness ---
    mean_brightness = np.mean(patch_np)
    metrics['brightness'] = mean_brightness
    if 100 <= mean_brightness <= 200:
        quality_score += 2
    elif 80 <= mean_brightness <= 220:
        quality_score += 1

    # --- Tissue coverage ---
    tissue_mask = gray < 240
    tissue_fraction = np.sum(tissue_mask) / patch_np.size
    metrics['tissue_fraction'] = tissue_fraction
    if 0.4 <= tissue_fraction <= 0.85:
        quality_score += 2
    elif 0.3 <= tissue_fraction <= 0.9:
        quality_score += 1

    # --- Stain separation ---
    try:
        od = -np.log10(np.maximum(patch_np / 255.0, 1e-6)).reshape(-1, 3)
        tissue_od = od[np.sum(od, axis=1) > 0.15]
        if len(tissue_od) >= 100:
            U, S, Vt = np.linalg.svd(tissue_od.T, full_matrices=False)
            stain_matrix = Vt[:2].T
            dot_product = np.clip(np.dot(stain_matrix[:, 0], stain_matrix[:, 1]), -1, 1)
            angle_deg = np.degrees(np.arccos(dot_product))
            metrics['stain_angle'] = angle_deg
            if 30 <= angle_deg <= 90:
                quality_score += 3
            elif 20 <= angle_deg <= 120:
                quality_score += 2
            elif 15 <= angle_deg <= 135:
                quality_score += 1

            metrics['stain_vector_1'] = stain_matrix[:, 0].tolist()
            metrics['stain_vector_2'] = stain_matrix[:, 1].tolist()

            # --- Balance index: reward PAS ≈ hematoxylin
            pas_od = np.mean(tissue_od[:, 0])  # red
            hem_od = np.mean(tissue_od[:, 2])  # blue
            balance_ratio = pas_od / (hem_od + 1e-6)
            metrics['pas_hematoxylin_balance'] = balance_ratio
            if 0.75 <= balance_ratio <= 1.3:
                quality_score += 0.5

        else:
            metrics['stain_angle'] = 0
            metrics['error'] = "Too few OD pixels"
    except Exception as e:
        metrics['stain_error'] = str(e)
        metrics['stain_angle'] = 0

    # --- Color variation ---
    std_rgb = np.std(patch_np, axis=(0, 1))
    total_std = np.mean(std_rgb)
    metrics['color_variation'] = total_std
    if 20 <= total_std <= 60:
        quality_score += 1
    elif 15 <= total_std <= 80:
        quality_score += 0.5

    # --- Artifact checks ---
    artifacts = 0
    bright_pixels = np.sum(gray > 250) / gray.size
    black_pixels = np.sum(gray < 10) / gray.size
    oversaturated = np.sum(hsv[:, :, 1] > 200) / hsv[:, :, 1].size
    if bright_pixels > 0.05: artifacts += 1
    if black_pixels > 0.02: artifacts += 1
    if oversaturated > 0.1: artifacts += 1
    metrics['artifacts'] = artifacts
    metrics['bright_pixels'] = bright_pixels
    metrics['black_pixels'] = black_pixels
    metrics['oversaturated'] = oversaturated
    if artifacts == 0:
        quality_score += 1
    elif artifacts == 1:
        quality_score += 0.5

    # --- Edge sharpness ---
    gray_float = gray.astype(float)
    grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mean_gradient = np.mean(gradient_magnitude)
    metrics['mean_gradient'] = mean_gradient
    if 8 <= mean_gradient <= 25:
        quality_score += 1
    elif 5 <= mean_gradient <= 35:
        quality_score += 0.5

    # --- Lighting consistency ---
    h, w = patch_np.shape[:2]
    quads = [
        patch_np[0:h//2, 0:w//2],
        patch_np[0:h//2, w//2:],
        patch_np[h//2:, 0:w//2],
        patch_np[h//2:, w//2:]
    ]
    brightnesses = [np.mean(q) for q in quads]
    consistency = np.std(brightnesses)
    metrics['brightness_consistency'] = consistency
    if consistency < 15:
        quality_score += 1
    elif consistency < 25:
        quality_score += 0.5

    try:
        od = -np.log10(np.maximum(patch_np / 255.0, 1e-6)).reshape(-1, 3)
        tissue_od = od[np.sum(od, axis=1) > 0.15]
        if len(tissue_od) >= 100:
            U, S, Vt = np.linalg.svd(tissue_od.T, full_matrices=False)
            stain_matrix = Vt[:2].T
            dot_product = np.clip(np.dot(stain_matrix[:, 0], stain_matrix[:, 1]), -1, 1)
            angle_deg = np.degrees(np.arccos(dot_product))
            metrics['stain_angle'] = angle_deg

            # Hard rejection if angle is too small (overlapping vectors)
            if angle_deg < 15:
                metrics['error'] = "Stain vectors too similar"
                return 0, metrics

            # Reward separation quality
            if 30 <= angle_deg <= 90:
                quality_score += 3
            elif 20 <= angle_deg <= 120:
                quality_score += 2
            elif 15 <= angle_deg <= 135:
                quality_score += 1

            metrics['stain_vector_1'] = stain_matrix[:, 0].tolist()
            metrics['stain_vector_2'] = stain_matrix[:, 1].tolist()

            # --- Balance index ---
            pas_od = np.mean(tissue_od[:, 0])  # red
            hem_od = np.mean(tissue_od[:, 2])  # blue
            balance_ratio = pas_od / (hem_od + 1e-6)
            metrics['pas_hematoxylin_balance'] = balance_ratio
            if 0.75 <= balance_ratio <= 1.3:
                quality_score += 0.5

        else:
            metrics['stain_angle'] = 0
            metrics['error'] = "Too few OD pixels"
    except Exception as e:
        metrics['stain_error'] = str(e)
        metrics['stain_angle'] = 0

    return quality_score, metrics


def extract_best_reference_patches(wsi_path, output_dir, patch_size=1024, 
                                   candidate_patches=20, best_patches=3, 
                                   min_quality_score=6.0):
    """
    Extract the best reference patches from a WSI based on quality assessment.
    """
    print(f"🔍 Analyzing {Path(wsi_path).name}...")
    
    # Extract candidate patches
    patches = extract_patches_from_wsi(
        wsi_path=wsi_path,
        patch_size=patch_size,
        overlap=0.25,
        level=0,
        tissue_threshold=0.15,
        create_debug_images=False,
        debug_output_dir=None,
        num_patches=candidate_patches,
        exclusion_conditions=[],
        exclusion_mode="any",
        extraction_mode="random",
        save_patches=False,
        output_dir=None,
        label=None
    )
    
    if not patches:
        print(f"❌ No patches extracted from {Path(wsi_path).name}")
        return []
    
    print(f"📊 Evaluating {len(patches)} candidate patches...")
    
    # Evaluate all patches
    patch_evaluations = []
    
    for i, (patch_np, x, y) in enumerate(patches):
        # Basic validation
        if patch_np.ndim != 3 or patch_np.shape[2] != 3:
            continue
            
        if patch_np.mean() < 5 or patch_np.std() < 5:
            continue
        
        # Ensure uint8
        if patch_np.dtype != np.uint8:
            patch_np = np.clip(patch_np * 255, 0, 255).astype(np.uint8)
        
        # Evaluate quality
        quality_score, metrics = evaluate_patch_quality(patch_np)
        
        patch_evaluations.append({
            'patch_np': patch_np,
            'x': x,
            'y': y,
            'quality_score': quality_score,
            'metrics': metrics,
            'index': i
        })
    
    if not patch_evaluations:
        print(f"❌ No valid patches found in {Path(wsi_path).name}")
        return []
    
    # Sort by quality score
    patch_evaluations.sort(key=lambda x: x['quality_score'], reverse=True)
    
    # Filter by minimum quality and select best
    good_patches = [p for p in patch_evaluations if p['quality_score'] >= min_quality_score]
    
    if not good_patches:
        print(f"⚠️ No patches meet quality threshold ({min_quality_score}) in {Path(wsi_path).name}")
        print(f"Best quality score: {patch_evaluations[0]['quality_score']:.1f}")
        # Lower threshold and take best available
        good_patches = patch_evaluations[:min(best_patches, len(patch_evaluations))]
        print(f"📝 Using top {len(good_patches)} patches with lower threshold")
    
    selected_patches = good_patches[:best_patches]
    
    # Save selected patches with metadata
    saved_patches = []
    wsi_name = Path(wsi_path).stem
    
    for i, patch_data in enumerate(selected_patches):
        patch_np = patch_data['patch_np']
        x, y = patch_data['x'], patch_data['y']
        quality_score = patch_data['quality_score']
        metrics = patch_data['metrics']
        
        # Save image
        img = Image.fromarray(patch_np)
        filename = f"{wsi_name}_ref_{i+1}_score{quality_score:.1f}_x{x}_y{y}.png"
        img_path = Path(output_dir) / filename
        img.save(img_path)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'wsi_source': Path(wsi_path).name,
            'coordinates': {'x': int(x), 'y': int(y)},
            'quality_score': float(quality_score),
            'rank': i + 1,
            'metrics': {k: (float(v) if isinstance(v, (int, float, np.number)) else v) 
                       for k, v in metrics.items()}
        }
        
        metadata_path = Path(output_dir) / f"{wsi_name}_ref_{i+1}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        saved_patches.append({
            'path': img_path,
            'metadata': metadata
        })
        
        print(f"✅ Saved reference {i+1}: {filename} (score: {quality_score:.1f}/10)")
        
        # Print key metrics
        if 'brightness' in metrics:
            print(f"   💡 Brightness: {metrics['brightness']:.1f}")
        if 'tissue_fraction' in metrics:
            print(f"   🧬 Tissue: {metrics['tissue_fraction']:.1%}")
        if 'stain_angle' in metrics:
            print(f"   🎨 Stain angle: {metrics['stain_angle']:.1f}°")
        if 'artifacts' in metrics:
            artifacts = metrics['artifacts']
            print(f"   ⚠️ Artifacts: {artifacts} {'❌' if artifacts > 0 else '✅'}")
    
    return saved_patches


def process_all_wsis_for_best_references(input_dir, output_dir, patch_size=1024, 
                                        candidate_patches=20, best_patches=3, 
                                        min_quality_score=6.0):
    """
    Process all WSIs in directory to extract best reference patches.
    """
    input_dir = Path(input_dir)
    wsi_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")) + \
                list(input_dir.glob("*.svs")) + list(input_dir.glob("*.ndpi"))
    
    print(f"🔍 Found {len(wsi_files)} WSI files to process.")
    
    if not wsi_files:
        print("❌ No WSI files found!")
        return
    
    all_references = []
    summary_stats = {
        'total_wsis': len(wsi_files),
        'successful_extractions': 0,
        'total_references': 0,
        'average_quality': 0.0,
        'quality_distribution': []
    }
    
    for wsi_path in tqdm(wsi_files, desc="Processing WSIs"):
        try:
            saved_patches = extract_best_reference_patches(
                wsi_path=str(wsi_path),
                output_dir=output_dir,
                patch_size=patch_size,
                candidate_patches=candidate_patches,
                best_patches=best_patches,
                min_quality_score=min_quality_score
            )
            
            if saved_patches:
                all_references.extend(saved_patches)
                summary_stats['successful_extractions'] += 1
                summary_stats['total_references'] += len(saved_patches)
                
                qualities = [p['metadata']['quality_score'] for p in saved_patches]
                summary_stats['quality_distribution'].extend(qualities)
                
        except Exception as e:
            print(f"❌ Error processing {wsi_path.name}: {e}")
            continue
    
    # Calculate summary statistics
    if summary_stats['quality_distribution']:
        summary_stats['average_quality'] = np.mean(summary_stats['quality_distribution'])
        summary_stats['median_quality'] = np.median(summary_stats['quality_distribution'])
        summary_stats['min_quality'] = np.min(summary_stats['quality_distribution'])
        summary_stats['max_quality'] = np.max(summary_stats['quality_distribution'])
    
    # Save summary
    summary_path = Path(output_dir) / "extraction_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"WSIs processed: {summary_stats['total_wsis']}")
    print(f"Successful extractions: {summary_stats['successful_extractions']}")
    print(f"Total reference patches: {summary_stats['total_references']}")
    
    if summary_stats['quality_distribution']:
        print(f"Average quality score: {summary_stats['average_quality']:.1f}/10")
        print(f"Quality range: {summary_stats['min_quality']:.1f} - {summary_stats['max_quality']:.1f}")
        
        # Count high-quality references
        high_quality = sum(1 for q in summary_stats['quality_distribution'] if q >= 8.0)
        good_quality = sum(1 for q in summary_stats['quality_distribution'] if q >= 6.0)
        
        print(f"High quality (≥8.0): {high_quality}")
        print(f"Good quality (≥6.0): {good_quality}")
    
    print(f"\n🎯 Recommendation: Use top {min(10, len(all_references))} patches as references")
    
    # Create ranked reference list
    all_references.sort(key=lambda x: x['metadata']['quality_score'], reverse=True)
    
    ranked_list = []
    for i, ref in enumerate(all_references[:10]):
        ranked_list.append({
            'rank': i + 1,
            'filename': ref['metadata']['filename'],
            'quality_score': ref['metadata']['quality_score'],
            'source_wsi': ref['metadata']['wsi_source']
        })
    
    ranked_path = Path(output_dir) / "top_references_ranked.json"
    with open(ranked_path, 'w') as f:
        json.dump(ranked_list, f, indent=2)
    
    print(f"💾 Saved ranking to: {ranked_path}")
    
    return all_references


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract best quality reference patches for stain normalization.")
    parser.add_argument("--wsi_dir", required=True, help="Directory containing WSI files")
    parser.add_argument("--out", required=True, help="Directory to save extracted reference patches")
    parser.add_argument("--patch_size", type=int, default=1024, help="Patch size (default: 1024)")
    parser.add_argument("--candidates", type=int, default=20, help="Number of candidate patches to evaluate per WSI")
    parser.add_argument("--best", type=int, default=3, help="Number of best patches to save per WSI")
    parser.add_argument("--min_quality", type=float, default=6.0, help="Minimum quality score (0-10)")
    parser.add_argument("--single_wsi", help="Process single WSI file instead of directory")
    
    args = parser.parse_args()
    
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    if args.single_wsi:
        # Process single WSI
        saved_patches = extract_best_reference_patches(
            wsi_path=args.single_wsi,
            output_dir=args.out,
            patch_size=args.patch_size,
            candidate_patches=args.candidates,
            best_patches=args.best,
            min_quality_score=args.min_quality
        )
        print(f"✅ Extracted {len(saved_patches)} reference patches from {Path(args.single_wsi).name}")
    else:
        # Process directory
        all_references = process_all_wsis_for_best_references(
            input_dir=args.wsi_dir,
            output_dir=args.out,
            patch_size=args.patch_size,
            candidate_patches=args.candidates,
            best_patches=args.best,
            min_quality_score=args.min_quality
        )
        print(f"🎉 Processing complete! Extracted {len(all_references)} total reference patches.")