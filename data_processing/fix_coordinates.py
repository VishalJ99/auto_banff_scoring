import os
import json
from glob import glob

# === Paths ===
results_root = "/data2/ac2220/macenko_pipeline_output"
coords_root = "/data2/ac2220/real/ti2/output"

def convert_points(points):
    return [
        {
            "x": pt["point"][0],
            "y": pt["point"][1],
            "type": "inflammatory",
            "prob": pt["probability"]
        }
        for pt in points
    ]

# === Main Loop ===
for wbc_dir in glob(os.path.join(results_root, "anon_*")):
    slide_id = os.path.basename(wbc_dir)
    wbc_json_path = os.path.join(wbc_dir, f"{slide_id}_wbc_results.json")
    coord_json_path = os.path.join(coords_root, slide_id, "detected-inflammatory-cells.json")

    if not os.path.exists(wbc_json_path):
        print(f"❌ No WBC results found for {slide_id}")
        continue
    if not os.path.exists(coord_json_path):
        print(f"⚠️ No coordinate data found for {slide_id}")
        continue

    with open(wbc_json_path, "r") as f:
        wbc_data = json.load(f)

    with open(coord_json_path, "r") as f:
        coord_data = json.load(f)

    converted = convert_points(coord_data.get("points", []))

    if "coordinates" not in wbc_data:
        wbc_data["coordinates"] = {}
    wbc_data["coordinates"]["inflammatory"] = converted

    with open(wbc_json_path, "w") as f:
        json.dump(wbc_data, f, indent=2)

    print(f"✅ Updated {slide_id}")

print("🎯 All slides processed.")
