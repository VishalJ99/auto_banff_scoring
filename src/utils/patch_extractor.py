import json
import os
import random

import cv2
from tiffslide import TiffSlide
import numpy as np
import openslide
from PIL import Image, ImageDraw

MASK_SAT = 0
MASK_VAL = 245

def is_tissue_patch(patch_np, threshold=0.15):
    hsv = cv2.cvtColor(patch_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    tissue_mask = (saturation > MASK_SAT) & (value < MASK_VAL)
    tissue_percentage = np.sum(tissue_mask) / (patch_np.shape[0] * patch_np.shape[1])
    return tissue_percentage > threshold

def extract_patches_from_wsi(
    wsi_path,
    patch_size=1024,
    overlap=0.25,
    level=0,
    tissue_threshold=0.05,
    create_debug_images=True,
    debug_output_dir=None,
    num_patches=10000,
    exclusion_conditions=None,
    exclusion_mode="any",
    extraction_mode="contiguous",
    save_patches=False,
    output_dir=None,
    label=None,
    include_coords=None,
):
    if exclusion_conditions is None:
        exclusion_conditions = []

    if exclusion_mode not in ["any", "all"]:
        exclusion_mode = "any"
    if extraction_mode not in ["random", "contiguous"]:
        extraction_mode = "random"

    metadata = {}
    if save_patches:
        assert output_dir is not None
        os.makedirs(output_dir, exist_ok=True)
        slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
        if label is not None:
            label_dir = os.path.join(output_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            slide_output_dir = os.path.join(label_dir, slide_name)
        else:
            slide_output_dir = os.path.join(output_dir, slide_name)
        os.makedirs(slide_output_dir, exist_ok=True)
        metadata = {
            "slide_path": wsi_path,
            "slide_name": slide_name,
            "label": label,
            "patch_size": patch_size,
            "level": level,
            "overlap": overlap,
            "extraction_mode": extraction_mode,
            "patches": [],
        }

    if create_debug_images:
        assert debug_output_dir is not None
        os.makedirs(debug_output_dir, exist_ok=True)

    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.level_dimensions[level]

    patches = []
    count = 0
    if include_coords is not None:
        include_coords = set(tuple(map(int, coord)) for coord in include_coords)

    scale_factor = 1 / 16 if create_debug_images else 1 / min(32, width // 4000)
    thumb_width = int(width * scale_factor)
    thumb_height = int(height * scale_factor)

    thumbnail = slide.get_thumbnail((thumb_width, thumb_height)).convert("RGB")
    thumbnail_np = np.array(thumbnail)

    if create_debug_images:
        downsampled = thumbnail.copy()
        draw = ImageDraw.Draw(downsampled)

    hsv = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    tissue_mask = (saturation > MASK_SAT) & (value < MASK_VAL)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tissue_mask = cv2.morphologyEx(tissue_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)

    if create_debug_images:
        mask_path = os.path.join(debug_output_dir, "tissue_mask_thumbnail.png")
        Image.fromarray((tissue_mask * 255).astype(np.uint8)).save(mask_path)
        thumbnail.save(os.path.join(debug_output_dir, "thumbnail.png"))

    tissue_coords = np.where(tissue_mask)
    tissue_points = list(zip(tissue_coords[1], tissue_coords[0]))

    if not tissue_points:
        slide.close()
        return (patches, metadata) if save_patches else patches

    thumb_patch_size = int(patch_size * scale_factor)

    if extraction_mode == "random":
        sampled_regions = set()
        max_attempts = int(num_patches) * 100
        attempts = 0

        while count < num_patches and attempts < max_attempts:
            attempts += 1
            point_idx = np.random.randint(0, len(tissue_points))
            thumb_x, thumb_y = tissue_points[point_idx]
            if (
                thumb_x + thumb_patch_size >= thumbnail_np.shape[1]
                or thumb_y + thumb_patch_size >= thumbnail_np.shape[0]
            ):
                continue

            region = tissue_mask[
                thumb_y : thumb_y + thumb_patch_size,
                thumb_x : thumb_x + thumb_patch_size,
            ]
            if np.sum(region) / region.size <= tissue_threshold:
                continue

            full_x = int(thumb_x / scale_factor)
            full_y = int(thumb_y / scale_factor)

            should_exclude = False
            satisfied_conditions = []
            for condition in exclusion_conditions:
                coord, operator, value = condition
                coord_value = full_x if coord.lower() == "x" else full_y
                condition_met = eval(f"{coord_value} {operator} {value}")
                satisfied_conditions.append(condition_met)
            if exclusion_mode == "any" and any(satisfied_conditions):
                should_exclude = True
            elif exclusion_mode == "all" and all(satisfied_conditions):
                should_exclude = True
            if should_exclude:
                continue

            region_key = (full_x // (patch_size // 4), full_y // (patch_size // 4))
            if region_key in sampled_regions:
                continue
            sampled_regions.add(region_key)

            patch_pil = slide.read_region((full_x, full_y), level, (patch_size, patch_size)).convert("RGB")
            patch_np = np.array(patch_pil)
            if include_coords is not None and (full_x, full_y) not in include_coords:
                continue
            if not is_tissue_patch(patch_np, tissue_threshold):
                continue

            patches.append((patch_np, full_x, full_y))

            if save_patches:
                patch_filename = f"{slide_name}_x{full_x}_y{full_y}_l{level}.png"
                patch_path = os.path.join(slide_output_dir, patch_filename)
                patch_pil.save(patch_path)
                metadata["patches"].append({
                    "filename": patch_filename,
                    "x": full_x,
                    "y": full_y,
                    "level": level,
                    "tissue_percentage": 1.0,
                    "patch_index": count,
                })

            count += 1

    else:  # contiguous mode
        step_size = int(patch_size * (1 - overlap))
        num_patches_x = (width - patch_size) // step_size + 1
        num_patches_y = (height - patch_size) // step_size + 1
        dilated_mask = cv2.dilate(tissue_mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        for y_idx in range(num_patches_y):
            for x_idx in range(num_patches_x):
                if count >= num_patches:  # Check limit
                    break
                    
                thumb_x = int(x_idx * step_size * scale_factor)
                thumb_y = int(y_idx * step_size * scale_factor)
                if (
                    thumb_x + thumb_patch_size >= thumbnail_np.shape[1]
                    or thumb_y + thumb_patch_size >= thumbnail_np.shape[0]
                ):
                    continue
                patch_mask = dilated_mask[
                    thumb_y : thumb_y + thumb_patch_size,
                    thumb_x : thumb_x + thumb_patch_size,
                ]
                if np.sum(patch_mask) == 0:
                    continue
                    
                # CRITICAL: Create patch_np BEFORE using it
                full_x = x_idx * step_size
                full_y = y_idx * step_size
                patch_pil = slide.read_region((full_x, full_y), level, (patch_size, patch_size)).convert("RGB")
                patch_np = np.array(patch_pil)  # ← This must come before is_tissue_patch check
                
                if include_coords is not None and (full_x, full_y) not in include_coords:
                    continue
                    
                if is_tissue_patch(patch_np, tissue_threshold):  # ← Now patch_np exists
                    patches.append((patch_np, full_x, full_y))
                    count += 1
                    
            if count >= num_patches:  # Check limit for outer loop
                break

    slide.close()

    if save_patches:
        metadata_path = os.path.join(slide_output_dir, f"{slide_name}_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        return patches, metadata

    return patches

def extract_patch_array_from_wsi(wsi_path, x, y, patch_size, level=0):
    """Extract a patch as a NumPy array from a WSI at given coordinates using TiffSlide."""
    slide = TiffSlide(wsi_path)
    patch = slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")
    return np.array(patch)
