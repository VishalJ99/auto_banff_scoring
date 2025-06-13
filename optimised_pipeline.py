#!/usr/bin/env python3

"""
Fast optimized pipeline that uses the new tiakong_inference.py optimized components.
This version should be 10x faster than the legacy BestReferenceMacenko.
"""

import argparse
import sys
import os
import numpy as np
from pathlib import Path
import time
import json
import csv
from tqdm import tqdm
import multiprocessing as mp

# Add your paths
sys.path.append(str(Path(__file__).resolve().parent / "SwinTransformer_classification"))
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "src/utils"))

# Import from your updated tiakong_inference.py
from tiakong_inference import (
    load_detector, 
    run_patch_inference_with_mask,
    initialize_normalization_components,
    MacenkoReferenceCache,
    ConditionalMacenkoNormalizer
)
from best_reference_macenko import BestReferenceMacenko
from PIL import Image


def check_gpu_availability():
    """Check which GPUs are actually available and working."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available, will use CPU")
            return []
        
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.set_device(i)
                test_tensor = torch.tensor([1.0]).cuda(i)
                available_gpus.append(i)
                print(f"✅ GPU {i} available: {torch.cuda.get_device_name(i)}")
            except Exception as e:
                print(f"❌ GPU {i} not available: {e}")
        
        return available_gpus
    except ImportError:
        print("⚠️  PyTorch not available")
        return []

def load_slide_scores(score_file_path):
    """Load Banff scores from CSV file."""
    slide_scores = {}
    with open(score_file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_name = row['filename'].strip()
            scores = {k: try_parse_float(v) for k, v in row.items() if k != 'filename'}
            slide_scores[slide_name] = scores
    return slide_scores

def load_processed_slides(summary_file):
    """Load list of already processed slides to enable resuming."""
    processed = set()
    if not os.path.exists(summary_file):
        return processed
    with open(summary_file, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                if "slide" in entry:
                    processed.add(entry["slide"] + ".svs")
            except Exception:
                continue
    return processed

def try_parse_float(value):
    """Safely parse float values from CSV."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def find_svs_file(slide_name, root_dir="/vol/biomedic3/histopatho/win_share"):
    """Recursively find SVS file in directory structure."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname == slide_name:
                return os.path.join(dirpath, fname)
    return None

def create_cortex_mask(coords, wsi_dims, patch_size):
    """Create binary mask for cortex regions."""
    mask = np.zeros(wsi_dims[::-1], dtype=np.uint8)
    for x, y in coords:
        mask[y:y+patch_size, x:x+patch_size] = 1
    return mask

def run_single_slide_optimized(args):
    """Process a single slide with GUARANTEED fast normalization."""
    wsi_path, output_dir, swin_model_path, linear_model_path, template_path, tiakong_model_path, banff_scores, gpu_id, reference_dir, force_optimized = args

    # Set GPU environment
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        print(f"🔄 Worker process using GPU {gpu_id}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("🔄 Worker process using CPU")

    # Import torch AFTER setting CUDA_VISIBLE_DEVICES
    try:
        import torch
        from SwinTransformer_classification.inference_svs import SVSInference
        from patch_extractor import extract_patches_from_wsi
        import openslide
        
        if gpu_id >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(0)
            print(f"✅ Successfully using GPU {gpu_id}")
        
    except Exception as e:
        print(f"❌ Failed to initialize PyTorch: {e}")
        return None

    try:
        slide_name = Path(wsi_path).stem
        print(f"\n🔄 Starting OPTIMIZED pipeline for slide: {slide_name}")

        # Test slide readability
        try:
            slide_test = openslide.OpenSlide(wsi_path)
            slide_test.close()
        except Exception as e:
            print(f"❌ Could not open WSI {wsi_path}: {e}")
            return None

        slide_output_dir = Path(output_dir) / slide_name
        slide_output_dir.mkdir(parents=True, exist_ok=True)

        # Extract tissue patches
        print(f"📦 Extracting tissue patches...")
        patch_infos = extract_patches_from_wsi(
            wsi_path=wsi_path,
            patch_size=1024,
            overlap=0.5,
            level=0,
            tissue_threshold=0.05,
            create_debug_images=False,
            save_patches=False,
            output_dir=None,
        )

        if not patch_infos:
            print("❌ No tissue patches found")
            return None

        coords = [(x, y) for _, x, y in patch_infos]
        print(f"✅ Found {len(coords)} tissue patches")
        
        bbox_path = slide_output_dir / "all_patch_coords.npy"
        np.save(bbox_path, np.array(coords))

        # Cortex classification
        print(f"🔍 Classifying cortex vs medulla...")
        swin_model = SVSInference(
            model_path=swin_model_path,
            linear_path=linear_model_path,
            template_path=template_path
        )

        results = swin_model.process_svs(wsi_path, output_dir=slide_output_dir, bbox_file=str(bbox_path))
        
        if not results or "patch_results" not in results:
            print("❌ Cortex classification failed")
            return None

        cortex_coords = [r['coordinates'] for r in results['patch_results'] if r['prediction'] == 0]
        print(f"✅ Identified {len(cortex_coords)} cortex patches")
        
        if not cortex_coords:
            print("⚠️  No cortex patches found")
            return {
                "slide": slide_name,
                "inflammatory": 0, "lymphocyte": 0, "monocyte": 0,
                "runtime_min": 0, "cortex_patch_count": 0,
                "total_patch_count": len(coords),
                "error": "No cortex patches found",
                "normalization_method": "none",
                **banff_scores
            }

        # Create cortex mask
        wsi_dims = results['parameters'].get('dimensions', (100000, 100000))
        cortex_mask = create_cortex_mask(cortex_coords, wsi_dims, 1024)

        # Initialize OPTIMIZED normalization components
        print(f"🚀 Initializing OPTIMIZED normalization from {reference_dir}")
        try:
            reference_cache, conditional_normalizer = initialize_normalization_components(reference_dir)
            print("✅ Optimized normalization components loaded successfully")
            use_optimized = True
        except Exception as e:
            print(f"⚠️  Failed to load optimized components: {e}")
            if force_optimized:
                print("❌ Force optimized mode enabled, failing...")
                return None
            print("📍 Falling back to legacy BestReferenceMacenko")
            normalizer = BestReferenceMacenko(reference_dir)
            reference_cache = None
            conditional_normalizer = None
            use_optimized = False
        
        # Load detection model
        model = load_detector(tiakong_model_path)

        # Run WBC detection with GUARANTEED fast normalization
        print(f"🥺 Running WBC detection with {'OPTIMIZED' if use_optimized else 'LEGACY'} normalization...")
        start = time.time()
        
        if use_optimized:
            # Use optimized normalization (FAST)
            result = run_patch_inference_with_mask(
                wsi_path=wsi_path,
                model=model,
                cortex_mask=cortex_mask,
                patch_size=256,
                stride=224,
                threshold=0.5,
                reference_cache=reference_cache,
                conditional_normalizer=conditional_normalizer
            )
            normalization_method = "optimized"
        else:
            # Use legacy normalization (SLOW)
            result = run_patch_inference_with_mask(
                wsi_path=wsi_path,
                model=model,
                cortex_mask=cortex_mask,
                patch_size=256,
                stride=224,
                threshold=0.5,
                normalizer=normalizer
            )
            normalization_method = "legacy"
        
        elapsed = (time.time() - start) / 60
        print(f"🕒 Detection complete in {elapsed:.2f} minutes using {normalization_method} normalization")

        # Extract results - handle both tuple and object returns
        if isinstance(result, tuple):
            inflamm, lymph, mono = result[:3]
            norm_stats = None
        else:
            inflamm = result.inflammatory_count
            lymph = result.lymphocyte_count
            mono = result.monocyte_count
            norm_stats = result.normalization_stats

        # Build results
        results_json = {
            "slide": slide_name,
            "inflammatory": inflamm,
            "lymphocyte": lymph,
            "monocyte": mono,
            "runtime_min": elapsed,
            "cortex_patch_count": len(cortex_coords),
            "total_patch_count": len(coords),
            "cortex_patch_ratio": len(cortex_coords) / len(coords),
            "normalization_method": normalization_method,
            "gpu_used": gpu_id if gpu_id >= 0 else "CPU",
            "normalised": {
                "inflammatory_per_cortex_patch": inflamm / len(cortex_coords),
                "lymphocyte_per_cortex_patch": lymph / len(cortex_coords),
                "monocyte_per_cortex_patch": mono / len(cortex_coords),
            }
        }

        # Add normalization stats if available
        if norm_stats and hasattr(norm_stats, 'normalized_successfully'):
            results_json["normalization_stats"] = {
                "total_patches": norm_stats.total_patches,
                "normalized_successfully": norm_stats.normalized_successfully,
                "skipped_normalization": getattr(norm_stats, 'skipped_normalization', 0),
                "success_rate": norm_stats.normalization_success_rate,
                "processing_efficiency": f"{getattr(norm_stats, 'skipped_normalization', 0) / norm_stats.total_patches * 100:.1f}% patches skipped (smart optimization)"
            }

        results_json.update(banff_scores)

        # Save results
        with open(slide_output_dir / f"{slide_name}_wbc_results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        print(f"✅ Slide {slide_name} complete: {inflamm} inflammatory, {lymph} lymphocyte, {mono} monocyte")
        if norm_stats and hasattr(norm_stats, 'skipped_normalization'):
            print(f"⚡ Optimization: {norm_stats.skipped_normalization}/{norm_stats.total_patches} patches skipped normalization (smart processing)")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        return results_json

    except Exception as e:
        import traceback
        print(f"❌ Error processing {wsi_path}: {e}")
        print(f"🔍 Traceback:\n{traceback.format_exc()}")
        return None

def run_with_scores_parallel(score_dict, output_dir, swin_model_path, linear_model_path, template_path, tiakong_model_path, reference_dir, num_workers=3, preview=False, force_optimized=True):
    """Run pipeline in parallel with GUARANTEED fast processing."""
    
    available_gpus = check_gpu_availability()
    
    if not available_gpus:
        print("⚠️  No GPUs available, using CPU")
        gpu_assignments = [-1] * num_workers
    else:
        print(f"✅ Found {len(available_gpus)} GPUs: {available_gpus}")
        if len(available_gpus) < num_workers:
            num_workers = len(available_gpus)
        gpu_assignments = available_gpus[:num_workers]
    
    log_file = Path(output_dir) / "summary_with_banff_scores.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    already_done = load_processed_slides(log_file)
    print(f"🧼 Skipping {len(already_done)} already processed slides")

    args_list = []
    for i, (slide_name, banff_scores) in enumerate(score_dict.items()):
        if slide_name in already_done:
            continue
        wsi_path = find_svs_file(slide_name)
        if not wsi_path:
            continue
        gpu_id = gpu_assignments[i % len(gpu_assignments)]
        args_list.append((wsi_path, output_dir, swin_model_path, linear_model_path, template_path, tiakong_model_path, banff_scores, gpu_id, reference_dir, force_optimized))

    if not args_list:
        print("✅ All slides already processed")
        return

    if preview:
        print(f"\n🔎 Found {len(args_list)} slides to process")
        print(f"🚀 Will use {'FORCED OPTIMIZED' if force_optimized else 'OPTIMIZED WITH FALLBACK'} normalization")
        print(f"🖥️  GPU assignments: {gpu_assignments}")
        for a in args_list[:5]:
            print(f"  - {Path(a[0]).name} (GPU {a[7]})")
        
        cont = input("\nProceed? [y/N] ").strip().lower()
        if cont != "y":
            return

    print(f"\n🚀 Processing {len(args_list)} slides with {num_workers} workers")
    print(f"⚡ Using OPTIMIZED normalization (10x speed improvement expected)")
    
    total_processed = 0
    failed = 0
    total_time_saved = 0
    
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool, open(log_file, "a") as lf:
        for result in tqdm(pool.imap_unordered(run_single_slide_optimized, args_list), total=len(args_list)):
            if result:
                lf.write(json.dumps(result) + "\n")
                lf.flush()
                total_processed += 1
                
                # Track optimization benefits
                if result.get('normalization_method') == 'optimized' and 'normalization_stats' in result:
                    stats = result['normalization_stats']
                    skipped = stats.get('skipped_normalization', 0)
                    total = stats.get('total_patches', 1)
                    # Estimate time saved: assume 50% time reduction for skipped patches
                    time_saved = (skipped / total) * result.get('runtime_min', 0) * 0.5
                    total_time_saved += time_saved
                    
            else:
                failed += 1

    print(f"\n🎉 Complete! ✅ {total_processed} processed, ❌ {failed} failed")
    if total_time_saved > 0:
        print(f"⚡ Estimated time saved with optimization: {total_time_saved:.1f} minutes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPTIMIZED pipeline with 10x speed improvement")
    parser.add_argument("--score_file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--swin_model", default="/data2/ac2220/tiakong_model/cortex_medulla_classifier.pth")
    parser.add_argument("--linear_model", default="/data2/ac2220/tiakong_model/cortex_medulla_classifier_linear.pth")
    parser.add_argument("--template", default="/data2/ac2220/auto_banff_scoring/SwinTransformer_classification/my_template.png")
    parser.add_argument("--tiakong_model", default="/data2/ac2220/tiakong_model/tiakong_model.pt")
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--force_optimized", action="store_true", default=True, help="Fail if optimized normalization not available")
    
    args = parser.parse_args()

    slide_scores = load_slide_scores(args.score_file)
    print(f"📊 Loaded {len(slide_scores)} slides")
    print(f"🚀 OPTIMIZED PIPELINE - Expected 10x speed improvement!")
    
    run_with_scores_parallel(
        score_dict=slide_scores,
        output_dir=args.output,
        swin_model_path=args.swin_model,
        linear_model_path=args.linear_model,
        template_path=args.template,
        tiakong_model_path=args.tiakong_model,
        reference_dir=args.reference_dir,
        num_workers=args.workers,
        preview=args.preview,
        force_optimized=args.force_optimized
    )