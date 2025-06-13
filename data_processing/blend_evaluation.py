import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score

# Ensure necessary paths are in sys.path
sys.path.append("/data2/ac2220/auto_banff_scoring")
sys.path.append("/data2/ac2220/auto_banff_scoring/Monkey_TIAKong")
sys.path.append("/data2/ac2220/auto_banff_scoring/src/utils")

from monkey.model.utils import get_activation_function
from monkey.data.data_utils import imagenet_normalise_torch, slide_nms
from monkey.config import PredictionIOConfig
from prediction.utils import binary_det_post_process
from patch_extractor import extract_patches_from_wsi

# CONFIG
WSI_DIR = Path("/data2/ac2220/monkey-data/input/pas-cpg")
MASK_DIR = Path("/data2/ac2220/monkey-data/input/tissue-mask")
ANNOTATION_DIR = Path("/data2/monkey-challenge/data/annotations/json")
MODEL_PATH = Path("/data2/ac2220/tiakong_model/tiakong_model.pt")
OUTPUT_CSV = Path("/data2/ac2220/data_handling/blend_eval_results.csv")
OUTPUT_DIR = Path("/data2/ac2220/blend_eval_tmp")
MATCH_RADIUS = 12
SLIDE_IDS = ["A_P000001", "A_P000002", "A_P000003", "A_P000004", "A_P000005"]

MODES = {
    "seg_only": (1.0, 0.0),
    "det_only": (0.0, 1.0),
    "blended": (0.4, 0.6)
}


def load_detector(model_path):
    import ttach as tta
    model = torch.jit.load(model_path)
    model.eval().to("cuda")
    transforms = tta.Compose([
        tta.HorizontalFlip(), tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270])
    ])
    return tta.SegmentationTTAWrapper(model, transforms)


def load_annotations(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return np.array([p["point"] for p in data["points"]], dtype=np.float32)


def match_predictions_to_gt(preds, gts, radius):
    matched_pred = set()
    matched_gt = set()
    for i, gt in enumerate(gts):
        for j, pred in enumerate(preds):
            if j in matched_pred:
                continue
            if np.linalg.norm(gt - pred) <= radius:
                matched_gt.add(i)
                matched_pred.add(j)
                break
    TP = len(matched_gt)
    FP = len(preds) - TP
    FN = len(gts) - TP
    return TP, FP, FN


def run_patch_inference(wsi_path, model, cortex_mask, blend_weights, patch_size=256, stride=224, threshold=0.5):
    slide_name = os.path.splitext(os.path.basename(wsi_path))[0]
    output_path = OUTPUT_DIR / slide_name
    output_path.mkdir(parents=True, exist_ok=True)

    coords = [
        (x, y)
        for y in range(0, cortex_mask.shape[0] - patch_size + 1, stride)
        for x in range(0, cortex_mask.shape[1] - patch_size + 1, stride)
        if np.any(cortex_mask[y:y + patch_size, x:x + patch_size])
    ]
    print(f"✅ Using {len(coords)} cortex coords")

    patches = extract_patches_from_wsi(
        wsi_path=wsi_path,
        patch_size=patch_size,
        overlap=1 - (stride / patch_size),
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
        include_coords=np.array(coords)
    )

    activation_dict = {f"head_{i+1}": get_activation_function("sigmoid") for i in range(3)}
    detected = {"inflamm": []}
    for i in tqdm(range(0, len(patches), 16), desc=f"Running eval inference: {slide_name}"):
        batch = patches[i:i+16]
        imgs = [p[0] for p in batch]
        coords = [(p[1], p[2]) for p in batch]
        imgs_tensor = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255.0
        imgs_tensor = imagenet_normalise_torch(imgs_tensor).to("cuda")

        with torch.no_grad():
            outputs = model(imgs_tensor)

        for j, out in enumerate(outputs):
            x, y = coords[j]
            out = out.cpu()
            seg_prob = activation_dict["head_1"](out[0])
            det_prob = activation_dict["head_1"](out[2])
            blended = blend_weights[0] * seg_prob + blend_weights[1] * det_prob

            processed_mask = binary_det_post_process(blended.numpy(), threshold, 11)
            for r, c in np.argwhere(processed_mask > 0):
                detected["inflamm"].append({"x": x + c, "y": y + r})

    return detected["inflamm"]


def main():
    model = load_detector(str(MODEL_PATH))
    all_results = ["slide,mode,tp,fp,fn,precision,recall,f1"]

    for slide_id in SLIDE_IDS:
        print(f"\n📄 {slide_id}")
        wsi_path = WSI_DIR / f"{slide_id}_PAS_CPG.tif"
        mask_path = MASK_DIR / f"{slide_id}_mask.tif"
        annot_path = ANNOTATION_DIR / f"{slide_id}_inflammatory-cells.json"
        gt_points = load_annotations(annot_path)

        Image.MAX_IMAGE_PIXELS = None
        tissue_mask = Image.open(mask_path).convert("L")
        cortex_mask = (np.array(tissue_mask) > 0).astype(np.uint8)

        for mode, blend in MODES.items():
            pred_points = run_patch_inference(wsi_path, model, cortex_mask, blend)
            pred_coords = np.array([[p["x"], p["y"]] for p in pred_points])
            TP, FP, FN = match_predictions_to_gt(pred_coords, gt_points, MATCH_RADIUS)
            prec = TP / (TP + FP + 1e-8)
            rec = TP / (TP + FN + 1e-8)
            f1 = 2 * prec * rec / (prec + rec + 1e-8)
            print(f"📊 {mode.upper()}: TP={TP}, FP={FP}, FN={FN}, P={prec:.3f}, R={rec:.3f}, F1={f1:.3f}")
            all_results.append(f"{slide_id},{mode},{TP},{FP},{FN},{prec:.4f},{rec:.4f},{f1:.4f}")

    with open(OUTPUT_CSV, "w") as f:
        f.write("\n".join(all_results))
    print("\n✅ Done.")


if __name__ == "__main__":
    main()
