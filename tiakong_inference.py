import os
import argparse
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent / "Monkey_TIAKong"))
sys.path.append(str(Path(__file__).resolve().parent / "src" / "utils"))

import torch
import ttach as tta
import numpy as np
from tiffslide import TiffSlide

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
OUTPUT_PATH = Path("/data2/ac2220/real/ti2/output")
LOG_PATH = Path(OUTPUT_PATH / "inference_log.txt")

# Load the TorchScript model and wrap it in TTA for inference
def load_detector(model_path: str) -> torch.nn.Module:
    model = torch.jit.load(model_path)
    model.eval().to("cuda")
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
    ])
    return tta.SegmentationTTAWrapper(model, transforms)

# Extract microns-per-pixel metadata from slide
def get_slide_mpp(slide_path: str) -> float:
    try:
        slide = TiffSlide(slide_path)
        mpp_x = slide.properties.get("tiffslide.mpp-x") or slide.properties.get("openslide.mpp-x")
        return float(mpp_x) if mpp_x else 0.25
    except Exception as e:
        print(f"⚠️ Could not extract MPP from {slide_path}: {e}")
        return 0.25

# Main inference pipeline over all patches in a single WSI
def run_patch_inference(wsi_path: str, model, patch_size: int = 256, stride: int = 224, threshold: float = 0.5):
    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
    output_path = OUTPUT_PATH / slide_name
    output_path.mkdir(parents=True, exist_ok=True)

    overlap = 1 - (stride / patch_size)

    # Extract tissue-containing patches from the WSI
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
        label=None
    )

    if not patches:
        print(f"❌ No valid patches found in {slide_name}")
        return 0, 0, 0

    # Setup model heads and output dictionaries
    batch_size = 16
    activation_dict = {
        "head_1": get_activation_function("sigmoid"),
        "head_2": get_activation_function("sigmoid"),
        "head_3": get_activation_function("sigmoid"),
    }

    detected = {"inflamm": [], "lymph": [], "mono": []}

    # Iterate over patches in batches
    for i in tqdm(range(0, len(patches), batch_size), desc=f"Inference on {slide_name}"):
        batch = patches[i:i+batch_size]
        imgs = [p[0] for p in batch]
        coords = [(p[1], p[2]) for p in batch]

        # Prepare tensor for model input
        imgs_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
        imgs_tensor = imagenet_normalise_torch(imgs_tensor).to("cuda")

        # Run model inference
        with torch.no_grad():
            outputs = model(imgs_tensor)

        # Post-process detections
        for j, out in enumerate(outputs):
            x, y = coords[j]
            patch = batch[j][0]
            height, width = patch.shape[:2]
            out = out.cpu()

            for head_idx, label in enumerate(["inflamm", "lymph", "mono"]):
                seg_idx = head_idx * 3
                det_idx = seg_idx + 2

                # Blend segmentation and detection scores
                seg_prob = activation_dict[f"head_{head_idx+1}"](out[seg_idx])
                det_prob = activation_dict[f"head_{head_idx+1}"](out[det_idx])
                blended = 0.4 * seg_prob + 0.6 * det_prob

                # Apply post-processing to obtain final binary mask
                processed_mask = binary_det_post_process(
                    blended.cpu().numpy(),
                    thresholds=threshold,
                    min_distances=[11, 11, 11][head_idx]
                )

                # Extract coordinates of positive detections
                points = np.argwhere(processed_mask > 0)
                for r, c in points:
                    detected[label].append({
                        "x": x + c,
                        "y": y + r,
                        "type": {"inflamm": "inflammatory", "lymph": "lymphocyte", "mono": "monocyte"}[label],
                        "prob": float(blended[r, c].item())
                    })

    # Create a large mask image and apply NMS to final detections
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
        thresholds=[0.5, 0.5, 0.5],
        min_distances=[11, 11, 11],
        nms_boxes=[11, 11, 11],
        nms_overlap_thresh=0.5,
    )

    # Apply slide-level NMS
    inflamm_nms = slide_nms(None, binary_mask, detected["inflamm"], 4096, 11, 0.5)
    lymph_nms = slide_nms(None, binary_mask, detected["lymph"], 4096, 11, 0.5)
    mono_nms = slide_nms(None, binary_mask, detected["mono"], 4096, 11, 0.5)

    # Save detection results
    save_detection_records_monkey(
        config, inflamm_nms, lymph_nms, mono_nms, wsi_id=None, save_mpp=base_mpp
    )

    print(f"✅ Final saved: {len(inflamm_nms)} inflamm, {len(lymph_nms)} lymph, {len(mono_nms)} mono")
    return len(inflamm_nms), len(lymph_nms), len(mono_nms)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run patch-based TIAKong inference.")
    parser.add_argument("--wsi", type=str, help="Path to a single .svs file.")
    parser.add_argument("--wsi_dir", type=str, help="Directory containing .svs files.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    args = parser.parse_args()

    model = load_detector(str(MODEL_PATH / TIAKONG_MODEL_NAME))

    with open(LOG_PATH, "w") as log_file:
        log_file.write("slide_name,time_minutes,inflammatory,lymphocyte,monocyte\n")

        if args.wsi:
            start = time.time()
            inflamm, lymph, mono = run_patch_inference(args.wsi, model)
            elapsed = (time.time() - start) / 60
            log_file.write(f"{Path(args.wsi).name},{elapsed:.2f},{inflamm},{lymph},{mono}\n")

        elif args.wsi_dir:
            wsi_dir = Path(args.wsi_dir)
            for slide_path in sorted(wsi_dir.glob("*.svs")):
                start = time.time()
                inflamm, lymph, mono = run_patch_inference(str(slide_path), model, threshold=args.threshold)
                elapsed = (time.time() - start) / 60
                log_file.write(f"{slide_path.name},{elapsed:.2f},{inflamm},{lymph},{mono}\n")
                log_file.flush()
        else:
            print("❌ Please provide either --wsi or --wsi_dir.")