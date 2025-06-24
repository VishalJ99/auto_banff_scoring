import os
import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import concurrent.futures
from functools import partial

sys.path.append(str(Path(__file__).resolve().parent / "Monkey_TIAKong"))
sys.path.append(str(Path(__file__).resolve().parent / "src" / "utils"))

import torch
import ttach as tta
import numpy as np
from tiffslide import TiffSlide
import cv2

from monkey.config import PredictionIOConfig
from monkey.data.data_utils import (
    save_detection_records_monkey,
    imagenet_normalise_torch,
    slide_nms,
)
from patch_extractor import extract_patches_from_wsi
from monkey.model.utils import get_activation_function
from prediction.utils import binary_det_post_process

MODEL_PATH = Path("/data2/ac2220/tiakong_model")
TIAKONG_MODEL_NAME = "tiakong_model.pt"
OUTPUT_PATH = Path("/data2/ac2220/live_demo_output/macenko")
LOG_PATH = Path(OUTPUT_PATH / "inference_log.txt")

@dataclass
class NormalizationStats:
    total_patches: int = 0
    normalized_successfully: int = 0
    normalization_failed: int = 0
    skipped_malformed: int = 0
    skipped_low_contrast: int = 0
    skipped_normalization: int = 0
    
    @property
    def normalization_success_rate(self) -> float:
        if self.total_patches == 0:
            return 0.0
        return (self.normalized_successfully / self.total_patches) * 100
    
    @property
    def processing_success_rate(self) -> float:
        processed = self.normalized_successfully + self.normalization_failed
        if self.total_patches == 0:
            return 0.0
        return (processed / self.total_patches) * 100
    
    def __str__(self) -> str:
        return (
            f"Patch Processing Summary:\n"
            f"  Total patches: {self.total_patches}\n"
            f"  Successfully normalized: {self.normalized_successfully} ({self.normalized_successfully/self.total_patches*100:.1f}%)\n"
            f"  Failed normalization (using original): {self.normalization_failed} ({self.normalization_failed/self.total_patches*100:.1f}%)\n"
            f"  Skipped normalization (similar to ref): {self.skipped_normalization} ({self.skipped_normalization/self.total_patches*100:.1f}%)\n"
            f"  Skipped (malformed): {self.skipped_malformed}\n"
            f"  Skipped (low contrast): {self.skipped_low_contrast}\n"
            f"  Overall processing rate: {self.processing_success_rate:.1f}%"
        )

@dataclass 
class InferenceResults:
    inflammatory_count: int
    lymphocyte_count: int
    monocyte_count: int
    inflammatory_coords: List[Dict[str, Any]]
    lymphocyte_coords: List[Dict[str, Any]]
    monocyte_coords: List[Dict[str, Any]]
    normalization_stats: NormalizationStats
    processing_time_minutes: float

class PerformanceTimer:
    """Timer for tracking performance of different pipeline stages."""
    def __init__(self):
        self.times = {}
    
    def start(self, phase: str):
        self.times[phase] = time.time()
    
    def end(self, phase: str):
        if phase in self.times:
            duration = time.time() - self.times[phase]
            print(f"⏱️  {phase}: {duration:.2f} seconds")
            return duration
        return 0

class MacenkoReferenceCache:
    """Pre-computed reference cache for fast normalization."""
    def __init__(self, reference_dir: str):
        """Load and pre-compute all reference statistics."""
        self.reference_stats = []
        
        print("🔄 Pre-computing reference statistics...")
        reference_files = list(Path(reference_dir).glob("*.png"))
        
        if not reference_files:
            print("⚠️  No reference files found, creating default reference")
            # Create a default reference if none found
            default_ref = np.ones((256, 256, 3), dtype=np.uint8) * 128
            stain_matrix, max_conc = self._compute_stain_matrix(default_ref)
            self.reference_stats.append({
                'stain_matrix': stain_matrix,
                'max_conc': max_conc
            })
            return
        
        for ref_file in reference_files[:10]:  # Limit to 10 references
            try:
                ref_img = cv2.imread(str(ref_file))
                ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                
                stain_matrix, max_conc = self._compute_stain_matrix(ref_img)
                self.reference_stats.append({
                    'stain_matrix': stain_matrix,
                    'max_conc': max_conc
                })
            except Exception as e:
                print(f"⚠️  Failed to process reference {ref_file}: {e}")
        
        print(f"✅ Cached {len(self.reference_stats)} reference statistics")
    
    def _compute_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute stain matrix and concentrations from image."""
        try:
            # Downsample for faster computation
            small_img = cv2.resize(image, (128, 128))
            
            # Convert to OD space
            od = -np.log10((small_img.astype(np.float32) + 1) / 256.0)
            od_flat = od.reshape(-1, 3)
            
            # Remove background pixels
            mask = np.sum(od_flat, axis=1) > 0.15
            od_clean = od_flat[mask]
            
            if len(od_clean) < 10:
                # Default H&E stain matrix
                return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]), np.array([0.9, 0.9])
            
            # Use covariance matrix for faster computation than full SVD
            cov_matrix = np.cov(od_clean.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            
            # Take top 2 eigenvectors as stain directions
            stain_matrix = eigenvecs[:, -2:].T
            
            # Project data onto stain space
            concentrations = np.dot(od_clean, stain_matrix.T)
            max_conc = np.percentile(concentrations, 95, axis=0)
            
            return stain_matrix, max_conc
        except Exception:
            # Fallback to default
            return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]), np.array([0.9, 0.9])
    
    def get_random_reference(self):
        """Get a random pre-computed reference."""
        if not self.reference_stats:
            # Fallback to default
            return np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11]]), np.array([0.9, 0.9])
        
        ref = np.random.choice(self.reference_stats)
        return ref['stain_matrix'], ref['max_conc']

class ConditionalMacenkoNormalizer:
    """Only normalize patches that deviate significantly from reference."""
    
    def __init__(self, reference_patches: List[np.ndarray], deviation_threshold: float = 0.15):
        """Initialize with reference statistics and deviation threshold."""
        self.deviation_threshold = deviation_threshold
        self.reference_stats = self._compute_reference_stats(reference_patches)
        
    def _compute_reference_stats(self, reference_patches: List[np.ndarray]) -> dict:
        """Compute reference color statistics."""
        if not reference_patches:
            # Default stats
            return {
                'mean_lab': np.array([128, 128, 128]),
                'std_lab': np.array([20, 20, 20]),
                'mean_rgb': np.array([180, 140, 180]),
                'std_rgb': np.array([30, 30, 30])
            }
        
        all_means = []
        all_stds = []
        
        for patch in reference_patches:
            try:
                # Convert to LAB color space for better color representation
                lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
                all_means.append(np.mean(lab, axis=(0, 1)))
                all_stds.append(np.std(lab, axis=(0, 1)))
            except Exception:
                continue
        
        if not all_means:
            # Fallback to default
            return {
                'mean_lab': np.array([128, 128, 128]),
                'std_lab': np.array([20, 20, 20]),
                'mean_rgb': np.array([180, 140, 180]),
                'std_rgb': np.array([30, 30, 30])
            }
        
        return {
            'mean_lab': np.mean(all_means, axis=0),
            'std_lab': np.mean(all_stds, axis=0),
            'mean_rgb': np.mean([np.mean(p, axis=(0, 1)) for p in reference_patches], axis=0),
            'std_rgb': np.mean([np.std(p, axis=(0, 1)) for p in reference_patches], axis=0)
        }
    
    def needs_normalization(self, patch: np.ndarray) -> bool:
        """Determine if patch needs normalization based on color deviation."""
        try:
            # Quick LAB space comparison
            lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            patch_mean = np.mean(lab, axis=(0, 1))
            
            # Calculate deviation from reference
            deviation = np.linalg.norm(patch_mean - self.reference_stats['mean_lab'])
            
            return deviation > self.deviation_threshold
        except Exception:
            # If conversion fails, assume normalization is needed
            return True

class FastMacenkoNormalizer:
    """Simplified Macenko normalization with approximations for speed."""
    
    def __init__(self, target_stain_matrix: np.ndarray, target_max_conc: np.ndarray):
        """Initialize with pre-computed target statistics."""
        self.target_stain_matrix = target_stain_matrix
        self.target_max_conc = target_max_conc
    
    def _fast_stain_estimation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fast stain matrix estimation using approximations."""
        try:
            # Downsample for faster computation
            small_img = cv2.resize(image, (64, 64))  # Even smaller for speed
            
            # Convert to OD space
            od = -np.log10((small_img.astype(np.float32) + 1) / 256.0)
            od_flat = od.reshape(-1, 3)
            
            # Remove background (simplified threshold)
            mask = np.sum(od_flat, axis=1) > 0.15
            od_clean = od_flat[mask]
            
            if len(od_clean) < 5:
                return self.target_stain_matrix, self.target_max_conc
            
            # Simple PCA for speed
            cov_matrix = np.cov(od_clean.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
            stain_matrix = eigenvecs[:, -2:].T
            
            # Quick concentration estimation
            concentrations = np.dot(od_clean, stain_matrix.T)
            max_conc = np.percentile(concentrations, 90, axis=0)  # Use 90th percentile for speed
            
            return stain_matrix, max_conc
        except Exception:
            return self.target_stain_matrix, self.target_max_conc
    
    def normalize_fast(self, image: np.ndarray) -> np.ndarray:
        """Fast normalization with reduced precision."""
        try:
            # Work at lower resolution for speed
            h, w = image.shape[:2]
            if h > 256 or w > 256:
                # Downsample, normalize, then upsample
                scale = min(256/h, 256/w)
                new_h, new_w = int(h*scale), int(w*scale)
                small_img = cv2.resize(image, (new_w, new_h))
                
                normalized_small = self._normalize_core(small_img)
                return cv2.resize(normalized_small, (w, h))
            else:
                return self._normalize_core(image)
        except Exception:
            return image
    
    def _normalize_core(self, image: np.ndarray) -> np.ndarray:
        """Core normalization logic."""
        try:
            # Convert to OD space
            od = -np.log10((image.astype(np.float32) + 1) / 256.0)
            original_shape = od.shape
            od_flat = od.reshape(-1, 3)
            
            # Fast stain matrix estimation for source
            source_stain_matrix, source_max_conc = self._fast_stain_estimation(image)
            
            # Deconvolve and normalize
            source_concentrations = np.dot(od_flat, source_stain_matrix.T)
            concentration_ratio = self.target_max_conc / (source_max_conc + 1e-6)
            normalized_concentrations = source_concentrations * concentration_ratio
            
            # Reconvolve
            normalized_od = np.dot(normalized_concentrations, self.target_stain_matrix)
            normalized_od = normalized_od.reshape(original_shape)
            
            # Convert back to RGB
            normalized_rgb = 256 * (10 ** (-normalized_od))
            normalized_rgb = np.clip(normalized_rgb, 0, 255).astype(np.uint8)
            
            return normalized_rgb
        except Exception:
            return image

def process_patches_with_optimized_normalization(
    patches: List[Tuple], 
    reference_cache: MacenkoReferenceCache,
    conditional_normalizer: ConditionalMacenkoNormalizer,
    batch_size: int = 32
) -> Tuple[List[Tuple], NormalizationStats]:
    """
    Smart normalization that combines all optimization strategies.
    """
    stats = NormalizationStats()
    stats.total_patches = len(patches)
    
    if not patches:
        return [], stats
    
    print(f"🎨 Applying optimized stain normalization to {len(patches)} patches...")
    
    # Get random reference for this batch
    ref_stain_matrix, ref_max_conc = reference_cache.get_random_reference()
    fast_normalizer = FastMacenkoNormalizer(ref_stain_matrix, ref_max_conc)
    
    processed_patches = []
    
    # Process in batches for memory efficiency
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        
        for patch_data in batch:
            patch_img, x, y = patch_data
            
            # Validate patch format
            if patch_img.ndim != 3 or patch_img.shape[2] != 3:
                stats.skipped_malformed += 1
                continue
                
            # Check for low contrast patches
            if patch_img.mean() < 1 or patch_img.std() < 1:
                stats.skipped_low_contrast += 1
                continue
            
            # Check if normalization is needed
            if conditional_normalizer.needs_normalization(patch_img):
                # Attempt normalization
                try:
                    normalized_img = fast_normalizer.normalize_fast(patch_img)
                    processed_patches.append((normalized_img, x, y))
                    stats.normalized_successfully += 1
                except Exception:
                    # Normalization failed, use original patch
                    processed_patches.append((patch_img, x, y))
                    stats.normalization_failed += 1
            else:
                # Skip normalization - patch is already similar to reference
                processed_patches.append((patch_img, x, y))
                stats.skipped_normalization += 1
    
    # Print summary of normalization process
    print(f"✅ Normalization complete:")
    print(f"   📊 {stats.normalized_successfully}/{stats.total_patches} patches normalized successfully ({stats.normalization_success_rate:.1f}%)")
    print(f"   ⚡ {stats.skipped_normalization} patches skipped (already similar to reference)")
    if stats.normalization_failed > 0:
        print(f"   ⚠️  {stats.normalization_failed} patches failed normalization (using original)")
    if stats.skipped_malformed > 0 or stats.skipped_low_contrast > 0:
        print(f"   🗑️  {stats.skipped_malformed + stats.skipped_low_contrast} patches skipped (malformed: {stats.skipped_malformed}, low contrast: {stats.skipped_low_contrast})")
    
    return processed_patches, stats

def load_detector(model_path: str) -> torch.nn.Module:
    """Load TIAKong model with TTA wrapper."""
    model = torch.jit.load(model_path)
    model.eval().to("cuda")
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ])
    return tta.SegmentationTTAWrapper(model, transforms)

def get_slide_mpp(slide_path: str) -> float:
    """Extract microns per pixel from slide metadata."""
    try:
        slide = TiffSlide(slide_path)
        mpp_x = slide.properties.get("tiffslide.mpp-x") or slide.properties.get("openslide.mpp-x")
        return float(mpp_x) if mpp_x else 0.25
    except Exception:
        return 0.25

def create_normalization_debug_images(
    wsi_path: str, 
    coords: np.ndarray, 
    normalizer, 
    patch_size: int,
    output_dir: Path,
    num_samples: int = 5
) -> None:
    """Create debug images showing original vs normalized patches."""
    debug_dir = output_dir / "normalisation_debug"
    failed_dir = debug_dir / "failed_normalisation_patches"
    debug_dir.mkdir(parents=True, exist_ok=True)
    failed_dir.mkdir(parents=True, exist_ok=True)

    if len(coords) == 0:
        return

    sample_coords = random.sample(list(coords), min(num_samples, len(coords)))
    successful_examples = []

    slide = TiffSlide(wsi_path)
    for (x, y) in sample_coords:
        try:
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch)
            
            if hasattr(normalizer, 'normalize_fast'):
                norm_patch = normalizer.normalize_fast(patch_np)
            else:
                norm_patch = normalizer.normalize(patch_np)
            successful_examples.append((patch_np, norm_patch))
        except Exception:
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch.save(failed_dir / f"failed_patch_{x}_{y}.png")

    if successful_examples:
        fig, axs = plt.subplots(len(successful_examples), 2, figsize=(6, 2 * len(successful_examples)))
        if len(successful_examples) == 1:
            axs = axs.reshape(1, -1)
        
        for i, (orig, norm) in enumerate(successful_examples):
            axs[i, 0].imshow(orig)
            axs[i, 0].axis("off")
            axs[i, 0].set_title("Original")
            axs[i, 1].imshow(norm)
            axs[i, 1].axis("off")
            axs[i, 1].set_title("Normalized")
        
        fig.tight_layout()
        plt.savefig(debug_dir / "debug_normalisation_examples.png", dpi=200, bbox_inches='tight')
        plt.close()

def run_patch_inference_with_mask(
    wsi_path: str,
    model,
    cortex_mask: np.ndarray,
    patch_size: int = 256,
    stride: int = 224,
    threshold: float = 0.5,
    normalizer=None,  # <-- NEW: Legacy BestReferenceMacenko support
    reference_cache: MacenkoReferenceCache = None,
    conditional_normalizer: ConditionalMacenkoNormalizer = None
) -> Tuple[int, int, int, List, List, List]:
    """
    Run inference on patches within cortex mask regions with flexible normalization support.
    
    Args:
        normalizer: Legacy BestReferenceMacenko normalizer (for backward compatibility)
        reference_cache: Optimized reference cache (for new optimized pipeline)
        conditional_normalizer: Conditional normalizer (for new optimized pipeline)
    """
    # Generate coordinates within cortex mask
    coords = []
    for y in range(0, cortex_mask.shape[0] - patch_size + 1, stride):
        for x in range(0, cortex_mask.shape[1] - patch_size + 1, stride):
            if np.any(cortex_mask[y:y + patch_size, x:x + patch_size]):
                coords.append((x, y))

    coords = np.array(coords)
    print(f"🎯 Identified {len(coords)} potential cortex patch locations")

    # Save coordinates temporarily
    tmp_path = Path("/tmp") / "cortex_mask_coords.npy"
    np.save(tmp_path, coords)

    # Determine which normalization method to use
    if reference_cache and conditional_normalizer:
        # Use optimized normalization
        print("🚀 Using optimized normalization with conditional processing")
        results = run_patch_inference(
            wsi_path=wsi_path,
            model=model,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            include_coords_path=str(tmp_path),
            reference_cache=reference_cache,
            conditional_normalizer=conditional_normalizer
        )
    elif normalizer is not None:
        # Use legacy BestReferenceMacenko normalization
        print("🔧 Using legacy BestReferenceMacenko normalization")
        results = run_patch_inference_with_legacy_normalizer(
            wsi_path=wsi_path,
            model=model,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            include_coords_path=str(tmp_path),
            normalizer=normalizer
        )
    else:
        # No normalization
        print("⚪ Running without normalization")
        results = run_patch_inference(
            wsi_path=wsi_path,
            model=model,
            patch_size=patch_size,
            stride=stride,
            threshold=threshold,
            include_coords_path=str(tmp_path)
        )
    
    # Return in the format expected by the main pipeline
    return (
        results.inflammatory_count,
        results.lymphocyte_count, 
        results.monocyte_count,
        results.inflammatory_coords,
        results.lymphocyte_coords,
        results.monocyte_coords
    )

def run_patch_inference_with_legacy_normalizer(
    wsi_path: str,
    model,
    patch_size: int = 256,
    stride: int = 224,
    threshold: float = 0.5,
    include_coords_path: str = None,
    normalizer=None
) -> InferenceResults:
    """
    Run inference with legacy BestReferenceMacenko normalizer.
    This ensures normalization always works with your existing pipeline.
    """
    timer = PerformanceTimer()
    timer.start("total_inference")
    
    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
    output_path = OUTPUT_PATH / slide_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load coordinates if provided
    include_coords = None
    if include_coords_path is not None and os.path.exists(include_coords_path):
        include_coords = np.load(include_coords_path)
        print(f"📍 Using {len(include_coords)} specified coordinates")

    # Extract patches
    timer.start("patch_extraction")
    print(f"🔍 Extracting patches from {slide_name}...")
    overlap = 1 - (stride / patch_size)
    patches = extract_patches_from_wsi(
        wsi_path=wsi_path,
        patch_size=patch_size,
        overlap=overlap,
        level=0,
        tissue_threshold=0.05,
        create_debug_images=False,
        debug_output_dir="./debug",
        num_patches=10000,
        exclusion_conditions=[],
        exclusion_mode="any",
        extraction_mode="contiguous",
        save_patches=False,
        output_dir=str(output_path),
        label=None,
        include_coords=include_coords
    )
    timer.end("patch_extraction")

    if not patches:
        print(f"❌ No tissue patches found in {slide_name}")
        return InferenceResults(0, 0, 0, [], [], [], NormalizationStats(), 0.0)

    print(f"📦 Extracted {len(patches)} tissue patches")

    # Apply legacy normalization
    timer.start("normalization")
    norm_stats = NormalizationStats()
    norm_stats.total_patches = len(patches)
    
    if normalizer is not None:
        print(f"🎨 Applying BestReferenceMacenko normalization to {len(patches)} patches...")
        processed_patches = []
        
        for patch_data in tqdm(patches, desc="Normalizing patches"):
            patch_img, x, y = patch_data
            
            # Validate patch format
            if patch_img.ndim != 3 or patch_img.shape[2] != 3:
                norm_stats.skipped_malformed += 1
                continue
                
            # Check for low contrast patches
            if patch_img.mean() < 1 or patch_img.std() < 1:
                norm_stats.skipped_low_contrast += 1
                continue
            
            # Apply normalization
            try:
                normalized_img = normalizer.normalize(patch_img)
                processed_patches.append((normalized_img, x, y))
                norm_stats.normalized_successfully += 1
            except Exception:
                # Normalization failed, use original patch
                processed_patches.append((patch_img, x, y))
                norm_stats.normalization_failed += 1
    else:
        processed_patches = patches
        norm_stats.skipped_normalization = len(patches)
    
    norm_time = timer.end("normalization")
    
    if not processed_patches:
        print(f"❌ No valid patches remaining after processing in {slide_name}")
        return InferenceResults(0, 0, 0, [], [], [], norm_stats, timer.end("total_inference") / 60)

    print(f"🚀 {len(processed_patches)} patches ready for inference")
    print(f"✅ Normalization: {norm_stats.normalized_successfully}/{norm_stats.total_patches} patches normalized successfully")

    # Run inference (same as optimized version)
    timer.start("model_inference")
    batch_size = 16
    activation_dict = {f"head_{i+1}": get_activation_function("sigmoid") for i in range(3)}
    detected = {"inflamm": [], "lymph": [], "mono": []}

    print(f"🧠 Running inference on {len(processed_patches)} patches...")
    for i in tqdm(range(0, len(processed_patches), batch_size), desc=f"Processing {slide_name}"):
        batch = processed_patches[i:i+batch_size]
        imgs = [p[0] for p in batch]
        coords = [(p[1], p[2]) for p in batch]

        imgs_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
        imgs_tensor = imagenet_normalise_torch(imgs_tensor).to("cuda")

        with torch.no_grad():
            outputs = model(imgs_tensor)

        for j, out in enumerate(outputs):
            x, y = coords[j]
            out = out.cpu()
            for head_idx, label in enumerate(["inflamm", "lymph", "mono"]):
                seg_prob = activation_dict[f"head_{head_idx+1}"](out[head_idx * 3])
                det_prob = activation_dict[f"head_{head_idx+1}"](out[head_idx * 3 + 2])
                blended = 0.4 * seg_prob + 0.6 * det_prob

                processed_mask = binary_det_post_process(
                    blended.numpy(),
                    thresholds=threshold,
                    min_distances=[11, 11, 11][head_idx]
                )

                points = np.argwhere(processed_mask > 0)
                for r, c in points:
                    detected[label].append({
                        "x": x + c,
                        "y": y + r,
                        "type": {"inflamm": "inflammatory", "lymph": "lymphocyte", "mono": "monocyte"}[label],
                        "prob": float(blended[r, c].item())
                    })
    
    inf_time = timer.end("model_inference")

    # Apply NMS and save results
    timer.start("post_processing")
    max_y = max([p["y"] for v in detected.values() for p in v], default=0)
    max_x = max([p["x"] for v in detected.values() for p in v], default=0)
    binary_mask = np.ones((max_y + 100, max_x + 100), dtype=np.uint8)

    base_mpp = get_slide_mpp(wsi_path)
    config = PredictionIOConfig(
        wsi_dir=os.path.dirname(wsi_path),
        mask_dir=os.path.dirname(wsi_path),
        output_dir=str(output_path),
        patch_size=patch_size,
        resolution=0,
        units="level",
        stride=stride,
        thresholds=[0.5] * 3,
        min_distances=[11] * 3,
        nms_boxes=[11] * 3,
        nms_overlap_thresh=0.5,
    )

    inflamm_nms = slide_nms(None, binary_mask, detected["inflamm"], 4096, 11, 0.5)
    lymph_nms = slide_nms(None, binary_mask, detected["lymph"], 4096, 11, 0.5)
    mono_nms = slide_nms(None, binary_mask, detected["mono"], 4096, 11, 0.5)

    save_detection_records_monkey(config, inflamm_nms, lymph_nms, mono_nms, wsi_id=None, save_mpp=base_mpp)
    timer.end("post_processing")

    processing_time = timer.end("total_inference") / 60
    
    print(f"🎉 Completed {slide_name} in {processing_time:.2f} minutes")
    print(f"📋 Final counts: {len(inflamm_nms)} inflammatory, {len(lymph_nms)} lymphocyte, {len(mono_nms)} monocyte")
    print(f"📊 Normalization: {norm_stats.normalized_successfully}/{norm_stats.total_patches} patches ({norm_stats.normalization_success_rate:.1f}% success)")

    return InferenceResults(
        inflammatory_count=len(inflamm_nms),
        lymphocyte_count=len(lymph_nms),
        monocyte_count=len(mono_nms),
        inflammatory_coords=inflamm_nms,
        lymphocyte_coords=lymph_nms,
        monocyte_coords=mono_nms,
        normalization_stats=norm_stats,
        processing_time_minutes=processing_time
    )

def run_patch_inference(
    wsi_path: str,
    model,
    patch_size: int = 256,
    stride: int = 224,
    threshold: float = 0.5,
    include_coords_path: str = None,
    reference_cache: MacenkoReferenceCache = None,
    conditional_normalizer: ConditionalMacenkoNormalizer = None
) -> InferenceResults:
    """
    Main inference function with optimized normalization and comprehensive error handling.
    
    Returns:
        InferenceResults object containing counts, coordinates, and processing stats
    """
    timer = PerformanceTimer()
    timer.start("total_inference")
    
    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
    output_path = OUTPUT_PATH / slide_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load coordinates if provided
    include_coords = None
    if include_coords_path is not None and os.path.exists(include_coords_path):
        include_coords = np.load(include_coords_path)
        print(f"📍 Using {len(include_coords)} specified coordinates")

    # Extract patches
    timer.start("patch_extraction")
    print(f"🔍 Extracting patches from {slide_name}...")
    overlap = 1 - (stride / patch_size)
    patches = extract_patches_from_wsi(
        wsi_path=wsi_path,
        patch_size=patch_size,
        overlap=overlap,
        level=0,
        tissue_threshold=0.05,
        create_debug_images=False,
        debug_output_dir="./debug",
        num_patches=float("inf"),
        exclusion_conditions=[],
        exclusion_mode="any",
        extraction_mode="contiguous",
        save_patches=False,
        output_dir=str(output_path),
        label=None,
        include_coords=include_coords
    )
    timer.end("patch_extraction")

    if not patches:
        print(f"❌ No tissue patches found in {slide_name}")
        return InferenceResults(0, 0, 0, [], [], [], NormalizationStats(), 0.0)

    print(f"📦 Extracted {len(patches)} tissue patches")

    # Process patches with optimized normalization
    timer.start("normalization")
    if reference_cache and conditional_normalizer:
        processed_patches, norm_stats = process_patches_with_optimized_normalization(
            patches, reference_cache, conditional_normalizer
        )
    else:
        # Fallback to original processing if optimized components not available
        processed_patches, norm_stats = patches, NormalizationStats()
        norm_stats.total_patches = len(patches)
        norm_stats.skipped_normalization = len(patches)
    
    norm_time = timer.end("normalization")
    
    if not processed_patches:
        print(f"❌ No valid patches remaining after processing in {slide_name}")
        return InferenceResults(0, 0, 0, [], [], [], norm_stats, timer.end("total_inference") / 60)

    print(f"🚀 {len(processed_patches)} patches ready for inference")

    # Create debug images if normalizer is provided
    if reference_cache and len(include_coords) > 0:
        # Create a dummy normalizer for debug images
        ref_stain_matrix, ref_max_conc = reference_cache.get_random_reference()
        debug_normalizer = FastMacenkoNormalizer(ref_stain_matrix, ref_max_conc)
        create_normalization_debug_images(wsi_path, include_coords, debug_normalizer, patch_size, output_path)

    # Run inference
    timer.start("model_inference")
    batch_size = 16
    activation_dict = {f"head_{i+1}": get_activation_function("sigmoid") for i in range(3)}
    detected = {"inflamm": [], "lymph": [], "mono": []}

    print(f"🧠 Running inference on {len(processed_patches)} patches...")
    for i in tqdm(range(0, len(processed_patches), batch_size), desc=f"Processing {slide_name}"):
        batch = processed_patches[i:i+batch_size]
        imgs = [p[0] for p in batch]
        coords = [(p[1], p[2]) for p in batch]

        imgs_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
        imgs_tensor = imagenet_normalise_torch(imgs_tensor).to("cuda")

        with torch.no_grad():
            outputs = model(imgs_tensor)

        for j, out in enumerate(outputs):
            x, y = coords[j]
            out = out.cpu()
            for head_idx, label in enumerate(["inflamm", "lymph", "mono"]):
                seg_prob = activation_dict[f"head_{head_idx+1}"](out[head_idx * 3])
                det_prob = activation_dict[f"head_{head_idx+1}"](out[head_idx * 3 + 2])
                blended = 0.4 * seg_prob + 0.6 * det_prob

                processed_mask = binary_det_post_process(
                    blended.numpy(),
                    thresholds=threshold,
                    min_distances=[11, 11, 11][head_idx]
                )

                points = np.argwhere(processed_mask > 0)
                for r, c in points:
                    detected[label].append({
                        "x": x + c,
                        "y": y + r,
                        "type": {"inflamm": "inflammatory", "lymph": "lymphocyte", "mono": "monocyte"}[label],
                        "prob": float(blended[r, c].item())
                    })
    
    inf_time = timer.end("model_inference")

    # Apply NMS and save results
    timer.start("post_processing")
    max_y = max([p["y"] for v in detected.values() for p in v], default=0)
    max_x = max([p["x"] for v in detected.values() for p in v], default=0)
    binary_mask = np.ones((max_y + 100, max_x + 100), dtype=np.uint8)

    base_mpp = get_slide_mpp(wsi_path)
    config = PredictionIOConfig(
        wsi_dir=os.path.dirname(wsi_path),
        mask_dir=os.path.dirname(wsi_path),
        output_dir=str(output_path),
        patch_size=patch_size,
        resolution=0,
        units="level",
        stride=stride,
        thresholds=[0.5] * 3,
        min_distances=[11] * 3,
        nms_boxes=[11] * 3,
        nms_overlap_thresh=0.5,
    )

    inflamm_nms = slide_nms(None, binary_mask, detected["inflamm"], 4096, 11, 0.5)
    lymph_nms = slide_nms(None, binary_mask, detected["lymph"], 4096, 11, 0.5)
    mono_nms = slide_nms(None, binary_mask, detected["mono"], 4096, 11, 0.5)

    save_detection_records_monkey(config, inflamm_nms, lymph_nms, mono_nms, wsi_id=None, save_mpp=base_mpp)
    timer.end("post_processing")

    processing_time = timer.end("total_inference") / 60
    
    print(f"🎉 Completed {slide_name} in {processing_time:.2f} minutes")
    print(f"📋 Final counts: {len(inflamm_nms)} inflammatory, {len(lymph_nms)} lymphocyte, {len(mono_nms)} monocyte")
    
    # Performance breakdown
    print(f"📊 Performance breakdown:")
    print(f"   Normalization overhead: {norm_time/(norm_time+inf_time)*100:.1f}% of processing time")
    
    if reference_cache and conditional_normalizer:
        print(f"📊 {norm_stats}")

    return InferenceResults(
        inflammatory_count=len(inflamm_nms),
        lymphocyte_count=len(lymph_nms),
        monocyte_count=len(mono_nms),
        inflammatory_coords=inflamm_nms,
        lymphocyte_coords=lymph_nms,
        monocyte_coords=mono_nms,
        normalization_stats=norm_stats,
        processing_time_minutes=processing_time
    )

def initialize_normalization_components(reference_dir: str = None) -> Tuple[MacenkoReferenceCache, ConditionalMacenkoNormalizer]:
    """Initialize optimized normalization components."""
    print("🚀 Initializing optimized normalization components...")
    
    # Try to find reference directory
    possible_ref_dirs = [
        reference_dir,
        "/data2/ac2220/reference_patches",
        "./reference_patches",
        str(Path(__file__).parent / "reference_patches")
    ]
    
    ref_dir = None
    for dir_path in possible_ref_dirs:
        if dir_path and Path(dir_path).exists():
            ref_dir = dir_path
            break
    
    if ref_dir is None:
        print("⚠️  No reference directory found, using default settings")
        # Create minimal components with defaults
        reference_cache = MacenkoReferenceCache("")  # Will create default
        
        # Create default reference patches for conditional normalizer
        default_patches = [np.ones((256, 256, 3), dtype=np.uint8) * 180]
        conditional_normalizer = ConditionalMacenkoNormalizer(default_patches, deviation_threshold=0.2)
        
        return reference_cache, conditional_normalizer
    
    # Pre-compute reference cache
    reference_cache = MacenkoReferenceCache(ref_dir)
    
    # Load a few reference patches for conditional normalizer
    reference_patches = []
    ref_files = list(Path(ref_dir).glob("*.png"))[:5]
    
    if ref_files:
        for ref_file in ref_files:
            try:
                ref_img = cv2.imread(str(ref_file))
                if ref_img is not None:
                    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    reference_patches.append(ref_img)
            except Exception as e:
                print(f"⚠️  Failed to load reference {ref_file}: {e}")
    
    if not reference_patches:
        # Fallback to default
        reference_patches = [np.ones((256, 256, 3), dtype=np.uint8) * 180]
    
    conditional_normalizer = ConditionalMacenkoNormalizer(
        reference_patches, 
        deviation_threshold=0.12  # Adjust based on your needs
    )
    
    print("✅ Optimization components initialized")
    return reference_cache, conditional_normalizer

def main_optimized(args):
    """Main function with optimized normalization."""
    
    # Initialize normalization components once at startup
    reference_cache, conditional_normalizer = initialize_normalization_components()
    
    # Load model once
    model = load_detector(str(MODEL_PATH / TIAKONG_MODEL_NAME))

    with open(LOG_PATH, "w") as log_file:
        log_file.write("slide_name,time_minutes,inflammatory,lymphocyte,monocyte,patches_total,patches_normalized,patches_failed_norm,patches_skipped_norm,normalization_success_rate,normalization_overhead_pct\n")

        if args.wsi:
            print(f"🔬 Processing single slide: {args.wsi}")
            results = run_patch_inference(
                args.wsi, 
                model, 
                threshold=args.threshold,
                reference_cache=reference_cache,
                conditional_normalizer=conditional_normalizer
            )
            stats = results.normalization_stats
            
            # Calculate normalization overhead (estimate)
            norm_overhead = (stats.normalized_successfully / max(stats.total_patches, 1)) * 100 if stats.total_patches > 0 else 0
            
            log_file.write(f"{Path(args.wsi).name},{results.processing_time_minutes:.2f},{results.inflammatory_count},{results.lymphocyte_count},{results.monocyte_count},{stats.total_patches},{stats.normalized_successfully},{stats.normalization_failed},{stats.skipped_normalization},{stats.normalization_success_rate:.1f},{norm_overhead:.1f}\n")

        elif args.wsi_dir:
            wsi_dir = Path(args.wsi_dir)
            slide_files = sorted(wsi_dir.glob("*.svs"))
            
            print(f"🔬 Processing {len(slide_files)} slides from {wsi_dir}")
            
            for i, slide_path in enumerate(slide_files):
                print(f"\n[{i+1}/{len(slide_files)}] Processing {slide_path.name}")
                
                try:
                    results = run_patch_inference(
                        str(slide_path), 
                        model, 
                        threshold=args.threshold,
                        reference_cache=reference_cache,
                        conditional_normalizer=conditional_normalizer
                    )
                    stats = results.normalization_stats
                    
                    # Calculate normalization overhead
                    norm_overhead = (stats.normalized_successfully / max(stats.total_patches, 1)) * 100 if stats.total_patches > 0 else 0
                    
                    log_file.write(f"{slide_path.name},{results.processing_time_minutes:.2f},{results.inflammatory_count},{results.lymphocyte_count},{results.monocyte_count},{stats.total_patches},{stats.normalized_successfully},{stats.normalization_failed},{stats.skipped_normalization},{stats.normalization_success_rate:.1f},{norm_overhead:.1f}\n")
                    log_file.flush()
                    
                    print(f"✅ Completed {slide_path.name} - Time: {results.processing_time_minutes:.1f}min, Cells: {results.inflammatory_count + results.lymphocyte_count + results.monocyte_count}")
                    
                except Exception as e:
                    print(f"❌ Failed to process {slide_path.name}: {e}")
                    log_file.write(f"{slide_path.name},ERROR,0,0,0,0,0,0,0,0.0,0.0\n")
                    log_file.flush()
        else:
            print("❌ Please provide either --wsi or --wsi_dir.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run patch-based TIAKong inference with optimized Macenko normalization.")
    parser.add_argument("--wsi", type=str, help="Path to a single .svs file.")
    parser.add_argument("--wsi_dir", type=str, help="Directory containing .svs files.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--reference_dir", type=str, help="Directory containing reference patches for normalization")
    parser.add_argument("--disable_normalization", action="store_true", help="Disable stain normalization entirely")
    
    args = parser.parse_args()
    
    if args.disable_normalization:
        print("⚠️  Stain normalization disabled")
        # Run without normalization
        model = load_detector(str(MODEL_PATH / TIAKONG_MODEL_NAME))
        
        with open(LOG_PATH, "w") as log_file:
            log_file.write("slide_name,time_minutes,inflammatory,lymphocyte,monocyte,patches_total,patches_normalized,patches_failed_norm,patches_skipped_norm,normalization_success_rate,normalization_overhead_pct\n")

            if args.wsi:
                results = run_patch_inference(args.wsi, model, threshold=args.threshold)
                stats = results.normalization_stats
                log_file.write(f"{Path(args.wsi).name},{results.processing_time_minutes:.2f},{results.inflammatory_count},{results.lymphocyte_count},{results.monocyte_count},{stats.total_patches},0,0,{stats.total_patches},0.0,0.0\n")

            elif args.wsi_dir:
                wsi_dir = Path(args.wsi_dir)
                for slide_path in sorted(wsi_dir.glob("*.svs")):
                    try:
                        results = run_patch_inference(str(slide_path), model, threshold=args.threshold)
                        stats = results.normalization_stats
                        log_file.write(f"{slide_path.name},{results.processing_time_minutes:.2f},{results.inflammatory_count},{results.lymphocyte_count},{results.monocyte_count},{stats.total_patches},0,0,{stats.total_patches},0.0,0.0\n")
                        log_file.flush()
                    except Exception as e:
                        print(f"❌ Failed to process {slide_path.name}: {e}")
                        log_file.write(f"{slide_path.name},ERROR,0,0,0,0,0,0,0,0.0,0.0\n")
                        log_file.flush()
    else:
        main_optimized(args)