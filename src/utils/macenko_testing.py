import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tiffslide import TiffSlide
from skimage.io import imread

from staintools import (
    StainNormalizer,
    preprocess_image,
    read_image,
    convert_RGB_to_OD,
    get_concentrations,
)

# --- CONFIGURATION ---
SVS_PATH = "/data2/ac2220/real/ti0/svs/anon_47ad2dbb-296d-4295-b98d-d6b993a2f5aa.svs"
JSON_PATH = "/data2/ac2220/pipeline_output/anon_47ad2dbb-296d-4295-b98d-d6b993a2f5aa/anon_47ad2dbb-296d-4295-b98d-d6b993a2f5aa_results.json"
REFERENCE_PATCH_DIR = "/data2/ac2220/references"
PATCH_SIZE = 256
NUM_SAMPLES = 5
SAVE_PATH = "macenko_debug_plot_updated.png"

# --- HELPERS ---

def compute_multi_reference_normalizer(patch_dir):
    matrices = []
    max_concs = []
    for fname in sorted(os.listdir(patch_dir)):
        if not fname.endswith(".png"):
            continue
        img = read_image(os.path.join(patch_dir, fname))
        img = preprocess_image(img)
        normalizer = StainNormalizer()
        try:
            normalizer.fit(img)
            matrices.append(normalizer.target_stain_matrix)
            max_concs.append(normalizer.maxC_target)
        except Exception as e:
            print(f"⚠️ Skipping {fname}: {e}")
            continue
    median_stain_matrix = np.median(np.stack(matrices), axis=0)  # keep orientation consistent
    median_maxC = np.max(np.stack(max_concs), axis=0) 
    normalizer = StainNormalizer()
    normalizer.target_stain_matrix = median_stain_matrix
    normalizer.maxC_target = median_maxC
    return normalizer

def is_informative_patch(patch, threshold=0.15, min_frac=0.1):
    OD = convert_RGB_to_OD(patch)
    return (OD > threshold).any(axis=2).mean() > min_frac

def rgb_hist(ax, img, title):
    for i, c in enumerate(['r', 'g', 'b']):
        ax.hist(img[:, :, i].ravel(), bins=256, color=c, alpha=0.5, label=f'{c.upper()}')
    ax.set_xlim([0, 255])
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(axis='both', labelsize=6)

# --- Patch-normalizing version of StainNormalizer using fixed matrix ---
class FixedMatrixStainNormalizer(StainNormalizer):
    def normalize(self, source_image):
        OD = convert_RGB_to_OD(source_image)
        mask = (OD > 0.15).any(axis=2)
        if not np.any(mask):
            return source_image
        OD_flat = OD[mask]
        try:
            source_concentrations = get_concentrations(OD_flat, self.target_stain_matrix)
            maxC_source = np.percentile(source_concentrations, 99, axis=0)
            scale = np.clip(self.maxC_target / (maxC_source + 1e-8), 0.5, 2.0)
            norm_conc = source_concentrations * scale
            OD_recon = np.copy(OD)
            OD_recon[mask] = np.dot(norm_conc, self.target_stain_matrix)
            bg_mask = ~mask
            if np.any(bg_mask):
                light_eosin = np.array([0.01, 0.03])
                bg_OD = np.dot(light_eosin, self.target_stain_matrix)
                OD_recon[bg_mask] = bg_OD
            return np.clip(np.exp(-OD_recon) * 255, 0, 255).astype(np.uint8)
        except Exception:
            return source_image

# --- MAIN EXECUTION ---

# 1. Build normalizer using multi-reference fit
base_normalizer = compute_multi_reference_normalizer(REFERENCE_PATCH_DIR)
normalizer = FixedMatrixStainNormalizer()
normalizer.target_stain_matrix = base_normalizer.target_stain_matrix
normalizer.maxC_target = base_normalizer.maxC_target

# 2. Load JSON cortex coordinates
with open(JSON_PATH, "r") as f:
    data = json.load(f)
cortex_coords = [tuple(p["coordinates"]) for p in data["patch_results"] if p["class_name"] == "cortex"]
sample_coords = random.sample(cortex_coords, min(NUM_SAMPLES, len(cortex_coords)))

# 3. Extract and normalize patches
slide = TiffSlide(SVS_PATH)
examples = []
for (x, y) in sample_coords:
    region = slide.read_region((x, y), 0, (PATCH_SIZE, PATCH_SIZE)).convert("RGB")
    patch = np.array(region)
    patch_pre = preprocess_image(patch)

    informative = is_informative_patch(patch_pre)
    if informative:
        norm_patch = normalizer.normalize(patch_pre)
    else:
        print(f"⚠️ Skipping normalization for light patch at ({x},{y})")
        norm_patch = patch

    examples.append((patch, norm_patch, informative))

# 4. Plot results
fig, axs = plt.subplots(NUM_SAMPLES, 4, figsize=(12, 2.5 * NUM_SAMPLES))
for i, (orig, norm, informative) in enumerate(examples):
    axs[i, 0].imshow(orig)
    axs[i, 0].axis("off")
    axs[i, 0].set_title("Original")

    axs[i, 1].imshow(norm)
    axs[i, 1].axis("off")
    axs[i, 1].set_title("Normalized" if informative else "Skipped")

    rgb_hist(axs[i, 2], orig, "Original Histogram")
    rgb_hist(axs[i, 3], norm, "Normalized Histogram")

fig.tight_layout()
plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
print(f"✅ Saved updated debug plot to: {SAVE_PATH}")
