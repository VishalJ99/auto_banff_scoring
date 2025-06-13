import os
import json
import csv
from pathlib import Path

# Set your root directory here
input_root = Path("/data2/ac2220/pipeline_output") 
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
output_file = "inflammatory_threshold_summary.csv"

with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ['slide', 'cortex_patch_ratio', 'TI'] + [f"inflamm_norm_{t}" for t in thresholds]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for slide_dir in input_root.iterdir():
        if not slide_dir.is_dir():
            continue

        slide_name = slide_dir.name
        results_path = slide_dir / f"{slide_name}_wbc_results.json"
        if not results_path.exists():
            print(f"⚠️ Missing: {results_path}")
            continue

        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"❌ Error loading {results_path}: {e}")
            continue

        coords = results.get("coordinates", {}).get("inflammatory", [])
        cortex_patch_count = results.get("cortex_patch_count", 1) or 1
        cortex_patch_ratio = results.get("cortex_patch_ratio", 0)
        ti_score = results.get("TI", None)

        probs = [point["prob"] for point in coords if "prob" in point]
        row = {
            'slide': slide_name,
            'cortex_patch_ratio': cortex_patch_ratio,
            'TI': ti_score,
        }

        for thresh in thresholds:
            count = sum(p >= thresh for p in probs)
            norm = count / cortex_patch_count
            row[f"inflamm_norm_{thresh}"] = round(norm, 4)

        writer.writerow(row)

print(f"✅ Done. Saved to: {output_file}")
