import argparse
import sys
import os
import numpy as np
from pathlib import Path
import time
import json
import csv
from tqdm import tqdm
from multiprocessing import Pool

sys.path.append(str(Path(__file__).resolve().parent / "SwinTransformer_classification"))
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "src/utils"))

# Import the optimized normalization components
from best_reference_macenko import BestReferenceMacenko  # Your existing normalizer
from staintools import read_image
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
                # Test if we can actually use this GPU
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

def initialize_optimized_normalizer(reference_dir):
    """Initialize the optimized normalization components."""
    try:
        # Import optimized components from the new inference script
        from optimized_inference_script import (
            MacenkoReferenceCache, 
            ConditionalMacenkoNormalizer,
            initialize_normalization_components
        )
        
        print(f"🚀 Initializing optimized normalization from {reference_dir}")
        reference_cache, conditional_normalizer = initialize_normalization_components(reference_dir)
        
        return {
            'type': 'optimized',
            'reference_cache': reference_cache,
            'conditional_normalizer': conditional_normalizer
        }
        
    except ImportError:
        print("⚠️  Optimized normalizer not available, falling back to BestReferenceMacenko")
        # Fallback to your existing normalizer
        normalizer = BestReferenceMacenko(reference_dir)
        return {
            'type': 'legacy',
            'normalizer': normalizer
        }

def run_pipeline_wrapper(args):
    """Wrapper function for multiprocessing pipeline execution."""
    wsi_path, output_dir, swin_model_path, linear_model_path, template_path, tiakong_model_path, banff_scores, gpu_id, reference_dir, use_optimized = args

    # Set GPU environment BEFORE importing torch
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
        from patch_extractor import extract_patches_from_wsi, extract_patch_array_from_wsi
        import openslide
        
        # Verify GPU is accessible
        if gpu_id >= 0:
            if torch.cuda.is_available():
                torch.cuda.set_device(0)  # Always use device 0 since CUDA_VISIBLE_DEVICES limits visibility
                print(f"✅ Successfully using GPU {gpu_id} (visible as device 0)")
            else:
                print(f"⚠️  GPU {gpu_id} requested but CUDA not available, falling back to CPU")
        
    except Exception as e:
        print(f"❌ Failed to initialize PyTorch with GPU {gpu_id}: {e}")
        return None

    try:
        slide_name = Path(wsi_path).stem
        print(f"\n🔄 Starting pipeline for slide: {slide_name}")

        # Early check for slide readability
        try:
            _ = openslide.OpenSlide(wsi_path)
        except Exception as e:
            print(f"❌ Could not open WSI {wsi_path}: {e}")
            return None

        slide_output_dir = Path(output_dir) / slide_name
        slide_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"📦 Extracting tissue patches...")
        swin_patch_size = 1024
        swin_stride = 512
        swin_overlap = 1 - (swin_stride / swin_patch_size)

        patch_infos = extract_patches_from_wsi(
            wsi_path=wsi_path,
            patch_size=swin_patch_size,
            overlap=swin_overlap,
            level=0,
            tissue_threshold=0.05,
            create_debug_images=False,
            debug_output_dir=None,
            num_patches=float("inf"),
            exclusion_conditions=[],
            exclusion_mode="any",
            extraction_mode="contiguous",
            save_patches=False,
            output_dir=None,
            label=None
        )

        if not patch_infos:
            print("❌ No tissue patches found. Skipping.")
            return None

        coords = [(x, y) for _, x, y in patch_infos]
        print(f"✅ Found {len(coords)} tissue patches.")
        bbox_path = slide_output_dir / "all_patch_coords.npy"
        np.save(bbox_path, np.array(coords))

        print(f"🔍 Classifying cortex vs medulla with SwinTransformer...")
        try:
            swin_model = SVSInference(
                model_path=swin_model_path,
                linear_path=linear_model_path,
                template_path=template_path
            )
        except Exception as e:
            print(f"❌ Failed to initialize SwinTransformer: {e}")
            return None

        results = swin_model.process_svs(wsi_path, output_dir=slide_output_dir, bbox_file=str(bbox_path))

        if not results or "patch_results" not in results:
            print("❌ Swin inference failed.")
            return None

        cortex_coords = [r['coordinates'] for r in results['patch_results'] if r['prediction'] == 0]
        print(f"✅ Identified {len(cortex_coords)} cortex patches.")
        
        if not cortex_coords:
            print("⚠️  No cortex patches found, skipping WBC detection")
            return {
                "slide": slide_name,
                "inflammatory": 0,
                "lymphocyte": 0,
                "monocyte": 0,
                "runtime_min": 0,
                "cortex_patch_count": 0,
                "total_patch_count": len(coords),
                "cortex_patch_ratio": 0,
                "error": "No cortex patches found",
                **banff_scores
            }
        
        cortex_coords_path = slide_output_dir / "cortex_coords.npy"
        np.save(cortex_coords_path, np.array(cortex_coords))

        print("🧩 Creating cortex mask...")
        wsi_dims = results['parameters'].get('dimensions', (100000, 100000))
        cortex_mask = create_cortex_mask(cortex_coords, wsi_dims, swin_patch_size)

        # Initialize normalizer based on optimization setting
        normalizer_config = initialize_optimized_normalizer(reference_dir)
        
        print(f"🥺 Running WBC detection with {'optimized' if use_optimized else 'legacy'} normalization...")
        
        start = time.time()
        patch_size = 256
        stride = 224
        threshold = 0.5

        # Choose inference method based on optimization setting
        if use_optimized and normalizer_config['type'] == 'optimized':
            # Use optimized inference
            try:
                from optimized_inference_script import run_patch_inference_with_mask, load_detector
                
                model = load_detector(tiakong_model_path)
                
                result = run_patch_inference_with_mask(
                    wsi_path=str(wsi_path),
                    model=model,
                    cortex_mask=cortex_mask,
                    patch_size=patch_size,
                    stride=stride,
                    threshold=threshold,
                    reference_cache=normalizer_config['reference_cache'],
                    conditional_normalizer=normalizer_config['conditional_normalizer']
                )
            except Exception as e:
                print(f"⚠️  Optimized inference failed: {e}, falling back to legacy")
                use_optimized = False
                normalizer_config = {'type': 'legacy', 'normalizer': BestReferenceMacenko(reference_dir)}
        
        if not use_optimized or normalizer_config['type'] == 'legacy':
            # Use legacy inference with original normalizer
            try:
                from tiakong_inference import run_patch_inference_with_mask, load_detector
                
                model = load_detector(tiakong_model_path)
                
                if normalizer_config['type'] == 'legacy':
                    normalizer = normalizer_config['normalizer']
                else:
                    normalizer = None
                    
                result = run_patch_inference_with_mask(
                    wsi_path=str(wsi_path),
                    model=model,
                    cortex_mask=cortex_mask,
                    patch_size=patch_size,
                    stride=stride,
                    threshold=threshold,
                    normalizer=normalizer
                )
            except Exception as e:
                print(f"❌ Legacy inference also failed: {e}")
                return None
        
        # Handle both old tuple format (for backward compatibility) and new InferenceResults format
        if hasattr(result, 'inflammatory_count'):
            # New InferenceResults object
            inflamm = result.inflammatory_count
            lymph = result.lymphocyte_count
            mono = result.monocyte_count
            inflamm_coords = result.inflammatory_coords
            lymph_coords = result.lymphocyte_coords
            mono_coords = result.monocyte_coords
            norm_stats = result.normalization_stats
        else:
            # Old tuple format (fallback)
            inflamm, lymph, mono, inflamm_coords, lymph_coords, mono_coords = result
            # Create dummy stats for consistency
            try:
                from optimized_inference_script import NormalizationStats
                norm_stats = NormalizationStats()
                norm_stats.total_patches = len(cortex_coords)  # Estimate
                norm_stats.skipped_normalization = len(cortex_coords)  # Assume no normalization
            except ImportError:
                norm_stats = None
        
        elapsed = (time.time() - start) / 60
        print(f"🕒 Detection complete in {elapsed:.2f} minutes.")

        # Display normalization statistics if available
        if norm_stats and hasattr(norm_stats, 'normalized_successfully'):
            print(f"📊 Normalization: {norm_stats.normalized_successfully}/{norm_stats.total_patches} patches ({norm_stats.normalization_success_rate:.1f}% success)")
            if hasattr(norm_stats, 'skipped_normalization'):
                print(f"⚡ Skipped normalization: {norm_stats.skipped_normalization} patches")

        cortex_ratio = len(cortex_coords) / len(coords)
        results_json = {
            "slide": slide_name,
            "inflammatory": inflamm,
            "lymphocyte": lymph,
            "monocyte": mono,
            "runtime_min": elapsed,
            "cortex_patch_count": len(cortex_coords),
            "total_patch_count": len(coords),
            "cortex_patch_ratio": cortex_ratio,
            "used_optimized_normalization": use_optimized and normalizer_config['type'] == 'optimized',
            "gpu_used": gpu_id if gpu_id >= 0 else "CPU",
            "normalised": {
                "inflammatory_per_cortex_patch": inflamm / len(cortex_coords) if cortex_coords else 0,
                "lymphocyte_per_cortex_patch": lymph / len(cortex_coords) if cortex_coords else 0,
                "monocyte_per_cortex_patch": mono / len(cortex_coords) if cortex_coords else 0,
            }
        }

        # Add normalization statistics to output if available
        if norm_stats and hasattr(norm_stats, 'normalized_successfully'):
            results_json["normalization_stats"] = {
                "total_patches_for_inference": norm_stats.total_patches,
                "patches_normalized_successfully": norm_stats.normalized_successfully,
                "patches_failed_normalization": norm_stats.normalization_failed,
                "patches_skipped_malformed": getattr(norm_stats, 'skipped_malformed', 0),
                "patches_skipped_low_contrast": getattr(norm_stats, 'skipped_low_contrast', 0),
                "patches_skipped_normalization": getattr(norm_stats, 'skipped_normalization', 0),
                "normalization_success_rate": norm_stats.normalization_success_rate,
                "overall_processing_rate": getattr(norm_stats, 'processing_success_rate', 100.0)
            }

        # Add Banff scores to results
        results_json.update(banff_scores)

        # Save results
        with open(slide_output_dir / f"{slide_name}_wbc_results.json", "w") as f:
            json.dump(results_json, f, indent=2)

        print(f"✅ Slide {slide_name} finished and saved.")
        
        # Print summary for this slide
        print(f"📋 Final counts: {inflamm} inflammatory, {lymph} lymphocyte, {mono} monocyte")
        if norm_stats and hasattr(norm_stats, 'normalized_successfully'):
            print(f"📊 Processing: {norm_stats.total_patches} patches, {norm_stats.normalization_success_rate:.1f}% normalized successfully")
            if hasattr(norm_stats, 'skipped_normalization'):
                print(f"⚡ Efficiency: {norm_stats.skipped_normalization} patches skipped normalization")
        
        return results_json

    except Exception as e:
        import traceback
        print(f"❌ Unexpected error in {wsi_path}: {e}")
        print(f"🔍 Full traceback:\n{traceback.format_exc()}")
        return None

def run_with_scores_parallel(score_dict, output_dir, swin_model_path, linear_model_path, template_path, tiakong_model_path, reference_dir, num_workers=3, preview=False, use_optimized=True):
    """Run pipeline on multiple slides in parallel."""
    
    # Check GPU availability first
    available_gpus = check_gpu_availability()
    
    if not available_gpus:
        print("⚠️  No GPUs available, running on CPU")
        # Use CPU for all workers
        gpu_assignments = [-1] * num_workers
    else:
        print(f"✅ Found {len(available_gpus)} available GPUs: {available_gpus}")
        # Reduce workers if we have fewer GPUs
        if len(available_gpus) < num_workers:
            print(f"📉 Reducing workers from {num_workers} to {len(available_gpus)} to match GPU count")
            num_workers = len(available_gpus)
        gpu_assignments = available_gpus[:num_workers]
    
    log_file = Path(output_dir) / "summary_with_banff_scores.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    already_done = load_processed_slides(log_file)
    print(f"🧼 Skipping {len(already_done)} already processed slides")

    args_list = []
    unresolved = []
    for i, (slide_name, banff_scores) in enumerate(score_dict.items()):
        if slide_name in already_done:
            continue  # skip processed
        wsi_path = find_svs_file(slide_name)
        if not wsi_path:
            unresolved.append(slide_name)
            continue
        gpu_id = gpu_assignments[i % len(gpu_assignments)]
        args_list.append((wsi_path, output_dir, swin_model_path, linear_model_path, template_path, tiakong_model_path, banff_scores, gpu_id, reference_dir, use_optimized))

    if not args_list:
        print("✅ All slides already processed.")
        return

    if preview:
        print(f"\n🔎 Found {len(args_list)} matching slides.")
        print(f"🚀 Will use {'optimized' if use_optimized else 'legacy'} normalization")
        print(f"🖥️  GPU assignment: {gpu_assignments}")
        
        if unresolved:
            print(f"⚠️ {len(unresolved)} slides could not be found:")
            for name in unresolved[:5]:
                print(f"  - {name}")
            if len(unresolved) > 5:
                print("  ...")

        print("\n📂 Example files to process:")
        for a in args_list[:5]:
            gpu_str = f"GPU {a[7]}" if a[7] >= 0 else "CPU"
            print(f"  - {Path(a[0]).name} ({gpu_str})")
        if len(args_list) > 5:
            print("  ...")

        cont = input("\nProceed with processing? [y/N] ").strip().lower()
        if cont != "y":
            print("❌ Aborted.")
            return

    print(f"\n🚀 Launching {len(args_list)} slides across {num_workers} workers")
    if available_gpus:
        print(f"🖥️  Using GPUs: {gpu_assignments}")
    else:
        print("🖥️  Using CPU only")
    print(f"⚡ Using {'optimized' if use_optimized else 'legacy'} normalization")
    
    # Track overall statistics
    total_slides_processed = 0
    failed_slides = 0
    total_time_saved = 0
    
    with Pool(processes=num_workers) as pool, open(log_file, "a") as lf:
        for result in tqdm(pool.imap_unordered(run_pipeline_wrapper, args_list), total=len(args_list)):
            if result:
                lf.write(json.dumps(result) + "\n")
                lf.flush()  # Ensure results are written immediately
                total_slides_processed += 1
                
                # Track performance if using optimized normalization
                if result.get('used_optimized_normalization') and 'normalization_stats' in result:
                    norm_stats = result['normalization_stats']
                    skipped = norm_stats.get('patches_skipped_normalization', 0)
                    total = norm_stats.get('total_patches_for_inference', 1)
                    time_saved_estimate = skipped / total * result.get('runtime_min', 0) * 0.5  # Estimate 50% time saving
                    total_time_saved += time_saved_estimate
                    
            else:
                failed_slides += 1

    # Print final summary
    print(f"\n🎉 Pipeline Complete!")
    print(f"✅ Successfully processed: {total_slides_processed} slides")
    if failed_slides > 0:
        print(f"❌ Failed slides: {failed_slides}")
    if use_optimized and total_time_saved > 0:
        print(f"⚡ Estimated time saved with optimization: {total_time_saved:.1f} minutes")
    print(f"📄 Results saved to: {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cortex classification + WBC detection on WSIs with robust GPU handling.")
    parser.add_argument("--score_file", help="CSV file containing .svs slide names and Banff scores")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--swin_model", default="/data2/ac2220/tiakong_model/cortex_medulla_classifier.pth")
    parser.add_argument("--linear_model", default="/data2/ac2220/tiakong_model/cortex_medulla_classifier_linear.pth")
    parser.add_argument("--template", default="/data2/ac2220/auto_banff_scoring/SwinTransformer_classification/my_template.png")
    parser.add_argument("--tiakong_model", default="/data2/ac2220/tiakong_model/tiakong_model.pt")
    parser.add_argument("--parallel", action="store_true", help="Enable multiprocessing for WSI directory")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers (default: 3)")
    parser.add_argument("--preview", action="store_true", help="Preview which files will be processed before running")
    parser.add_argument("--reference_dir", required=True, help="Directory containing reference stain images")
    parser.add_argument("--use_optimized", action="store_true", default=True, help="Use optimized normalization (default: True)")
    parser.add_argument("--use_legacy", action="store_true", help="Force use of legacy normalization for comparison")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU-only processing")
    
    args = parser.parse_args()

    # Handle optimization flag logic
    use_optimized = args.use_optimized and not args.use_legacy

    if args.score_file:
        slide_scores = load_slide_scores(args.score_file)
        print(f"📊 Loaded {len(slide_scores)} slides from score file")
        
        # Override GPU detection if CPU-only requested
        if args.cpu_only:
            print("🖥️  CPU-only mode requested")
        
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
            use_optimized=use_optimized
        )

    else:
        print("❌ You must specify --score_file.")
        sys.exit(1)