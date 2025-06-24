import os
import json
import csv
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN

# Parameters
input_root = Path("/data2/ac2220/demo_output")  # 🔁 Update if needed
output_csv = Path("/data2/ac2220/demo_cluster_metrics_eps.csv")
MICRONS_PER_PIXEL = 0.262719
eps_microns = 37
eps = eps_microns / MICRONS_PER_PIXEL  # ≈ 140.8 pixels
min_samples = 3
LARGE_CLUSTER_THRESHOLD = 10  # min size for a "large" cluster
MAX_WBC_THRESHOLD = 100000

# Define fieldnames once
fieldnames = [
    "slide", "TI", "cortex_patch_ratio", "inflammatory_per_cortex_patch",
    "total_inflammatory_cells", "num_clusters", "clusters_per_patch",
    "mean_cluster_size", "max_cluster_size", "median_cluster_size",
    "cells_in_clusters", "fraction_clustered_cells",
    "large_cluster_count", "fraction_large_clusters",
    "wbc_clustering_index"
]

def compute_extended_cluster_metrics(coords, cortex_patch_count):
    if len(coords) == 0 or cortex_patch_count == 0:
        return dict.fromkeys([
            "num_clusters", "clusters_per_patch",
            "mean_cluster_size", "max_cluster_size", "median_cluster_size",
            "cells_in_clusters", "fraction_clustered_cells",
            "large_cluster_count", "fraction_large_clusters",
            "wbc_clustering_index"
        ], 0)

    points = np.array([[p["x"], p["y"]] for p in coords], dtype=np.float32)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan').fit(points)
    labels = clustering.labels_

    clustered = labels >= 0
    cluster_ids, counts = np.unique(labels[clustered], return_counts=True)

    if len(counts) == 0:
        return dict.fromkeys([
            "num_clusters", "clusters_per_patch",
            "mean_cluster_size", "max_cluster_size", "median_cluster_size",
            "cells_in_clusters", "fraction_clustered_cells",
            "large_cluster_count", "fraction_large_clusters",
            "wbc_clustering_index"
        ], 0)

    num_clusters = len(cluster_ids)
    clusters_per_patch = num_clusters / cortex_patch_count
    mean_cluster_size = np.mean(counts)
    max_cluster_size = np.max(counts)
    median_cluster_size = np.median(counts)
    cells_in_clusters = np.sum(counts)
    fraction_clustered_cells = cells_in_clusters / len(coords)

    large_clusters = counts[counts > LARGE_CLUSTER_THRESHOLD]
    large_cluster_count = len(large_clusters)
    fraction_large_clusters = large_cluster_count / num_clusters if num_clusters else 0

    # WBC Clustering Index: sum(n_k^2) / N^2
    wbc_clustering_index = np.sum(counts**2) / (len(coords)**2)

    return {
        "num_clusters": num_clusters,
        "clusters_per_patch": round(clusters_per_patch, 4),
        "mean_cluster_size": round(mean_cluster_size, 2),
        "max_cluster_size": int(max_cluster_size),
        "median_cluster_size": int(median_cluster_size),
        "cells_in_clusters": int(cells_in_clusters),
        "fraction_clustered_cells": round(fraction_clustered_cells, 3),
        "large_cluster_count": large_cluster_count,
        "fraction_large_clusters": round(fraction_large_clusters, 3),
        "wbc_clustering_index": round(wbc_clustering_index, 4)
    }

# Prepare to write header if needed
write_header = not output_csv.exists()
if write_header:
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Main execution
slide_dirs = list(input_root.iterdir())
print(f"Processing {len(slide_dirs)} slides...\n")

for i, slide_dir in enumerate(slide_dirs, start=1):
    if not slide_dir.is_dir():
        continue

    slide_name = slide_dir.name
    results_path = slide_dir / f"{slide_name}_wbc_results.json"
    if not results_path.exists():
        print(f"[{i}/{len(slide_dirs)}] ⚠️ Skipping {slide_name}: missing JSON")
        continue

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"[{i}/{len(slide_dirs)}] ❌ Error reading {results_path}: {e}")
        continue

    coords = results.get("coordinates", {}).get("inflammatory", [])
    if len(coords) > MAX_WBC_THRESHOLD:
        print(f"[{i}/{len(slide_dirs)}] ⚠️ Skipping {slide_name}: too many WBCs ({len(coords)})")
        continue

    cortex_patch_count = results.get("cortex_patch_count", 1) or 1
    cortex_patch_ratio = results.get("cortex_patch_ratio", 0)
    ti_score = results.get("TI", None)
    inflammatory_density = results.get("normalised", {}).get("inflammatory_per_cortex_patch", 0)

    print(f"[{i}/{len(slide_dirs)}] Processing {slide_name} with {len(coords)} WBCs...")

    metrics = compute_extended_cluster_metrics(coords, cortex_patch_count)
    row = {
        "slide": slide_name,
        "TI": ti_score,
        "cortex_patch_ratio": cortex_patch_ratio,
        "inflammatory_per_cortex_patch": inflammatory_density,
        "total_inflammatory_cells": len(coords),
        **metrics
    }

    with open(output_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)

print(f"\n✅ Done. Saved: {output_csv}")
