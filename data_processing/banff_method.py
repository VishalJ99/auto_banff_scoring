import json
import csv
import numpy as np
from pathlib import Path
from shapely.geometry import Point
from shapely.ops import unary_union

# Config
INPUT_DIR = Path("/data2/ac2220/pipeline_output")
OUTPUT_CSV = Path("data_handling/banff_ti_scores_simple_method.csv")

# Parameters
CORTEX_PATCH_SIZE = 1024
CORTEX_PATCH_AREA = CORTEX_PATCH_SIZE ** 2
MICRONS_PER_PIXEL = 0.262719

# Cell area parameters (typical mononuclear cell)
CELL_RADIUS_MICRONS = 10  # Typical mononuclear cell radius
CELL_RADIUS_PIXELS = CELL_RADIUS_MICRONS / MICRONS_PER_PIXEL
CELL_AREA_PIXELS = np.pi * (CELL_RADIUS_PIXELS ** 2)

MAX_WBC_THRESHOLD = 200000

fieldnames = [
    "slide", "TI", "banff_ti_score", "inflammation_pct",
    "total_inflammatory_cells", "cortex_patch_count",
    "cortex_area_px2", "total_cell_area_px2"
]

# Write CSV header if needed
if not OUTPUT_CSV.exists():
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Iterate through each slide
for slide_dir in INPUT_DIR.iterdir():
    if not slide_dir.is_dir():
        continue

    slide_name = slide_dir.name
    cortex_json_path = slide_dir / f"{slide_name}_results.json"
    wbc_json_path = slide_dir / f"{slide_name}_wbc_results.json"

    if not cortex_json_path.exists() or not wbc_json_path.exists():
        print(f"⚠️ Skipping {slide_name}: missing cortex or WBC JSON")
        continue

    try:
        with open(cortex_json_path) as f:
            cortex_data = json.load(f)
        with open(wbc_json_path) as f:
            wbc_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON for {slide_name}: {e}")
        continue

    patch_results = cortex_data.get("patch_results", [])
    cortex_patches = [p for p in patch_results if p.get("class_name") == "cortex"]
    cortex_patch_count = len(cortex_patches)
    cortex_area_px2 = cortex_patch_count * CORTEX_PATCH_AREA

    wbc_coords = wbc_data.get("coordinates", {}).get("inflammatory", [])
    if len(wbc_coords) > MAX_WBC_THRESHOLD:
        print(f"⚠️ Skipping {slide_name}: too many WBCs ({len(wbc_coords)})")
        continue

    if cortex_patch_count == 0:
        print(f"⚠️ Skipping {slide_name}: no cortex patches found")
        continue

    # Simple Banff-style calculation: sum of individual cell areas
    total_inflammatory_cells = len(wbc_coords)
    total_cell_area_px2 = total_inflammatory_cells * CELL_AREA_PIXELS
    
    # Calculate inflammation percentage
    inflammation_pct = (total_cell_area_px2 / cortex_area_px2) * 100 if cortex_area_px2 else 0
    
    # Assign Banff score using official thresholds
    if inflammation_pct < 10:
        banff_ti = "ti0"
    elif inflammation_pct < 25:  # Updated to match official Banff (10-25%)
        banff_ti = "ti1"
    elif inflammation_pct <= 50:  # 26-50%
        banff_ti = "ti2"
    else:  # >50%
        banff_ti = "ti3"

    ti_score = wbc_data.get("TI", None)

    row = {
        "slide": slide_name,
        "TI": ti_score,
        "banff_ti_score": banff_ti,
        "inflammation_pct": round(inflammation_pct, 2),
        "total_inflammatory_cells": total_inflammatory_cells,
        "cortex_patch_count": cortex_patch_count,
        "cortex_area_px2": cortex_area_px2,
        "total_cell_area_px2": int(total_cell_area_px2)
    }

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

    print(f"✅ Processed {slide_name}: actual: {ti_score}, predicted: {banff_ti} ({inflammation_pct:.1f}%)")

print(f"\n📊 Results saved to: {OUTPUT_CSV}")