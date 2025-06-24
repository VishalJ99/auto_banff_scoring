import os
import json
import csv
import joblib
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
import pandas as pd

# ------------------- Paths & Parameters -------------------
input_root = Path("/data2/ac2220/demo_output")  # 🔁 Update if needed
output_csv = Path("/data2/ac2220/demo_predictions.csv")
MAX_WBC_THRESHOLD = 150000
MICRONS_PER_PIXEL = 0.262719
eps_microns = 37
eps = eps_microns / MICRONS_PER_PIXEL
min_samples = 3
LARGE_CLUSTER_THRESHOLD = 10

# Features used during model training
features_to_use = [
    "inflammatory_per_cortex_patch",
    "wbc_clustering_index"
]

fieldnames = [
    "slide", "TI", "TI_pred", "Inflamm_class_pred",
    "cortex_patch_ratio", "inflammatory_per_cortex_patch",
    "total_inflammatory_cells", "num_clusters", "clusters_per_patch",
    "mean_cluster_size", "max_cluster_size", "median_cluster_size",
    "cells_in_clusters", "fraction_clustered_cells",
    "large_cluster_count", "fraction_large_clusters",
    "wbc_clustering_index"
]

# ------------------- Load Models & Scalers -------------------
clf_multi = joblib.load("/data2/ac2220/logreg_multiclass.pkl")
scaler_multi = joblib.load("/data2/ac2220/scaler_multiclass.pkl")

clf_bin = joblib.load("/data2/ac2220/logreg_binary.pkl")
scaler_bin = joblib.load("/data2/ac2220/scaler_binary.pkl")

# ------------------- Metric Function -------------------
def compute_cluster_metrics(coords, cortex_patch_count):
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

# ------------------- Main Execution -------------------
write_header = not output_csv.exists()
slide_dirs = list(input_root.iterdir())
print(f"🔎 Processing {len(slide_dirs)} slides...\n")

with open(output_csv, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    for i, slide_dir in enumerate(slide_dirs, start=1):
        if not slide_dir.is_dir():
            continue

        slide_name = slide_dir.name
        results_path = slide_dir / f"{slide_name}_wbc_results.json"
        if not results_path.exists():
            print(f"[{i}/{len(slide_dirs)}] ⚠️ Missing JSON: {slide_name}")
            continue

        try:
            with open(results_path, 'r') as rf:
                results = json.load(rf)
        except Exception as e:
            print(f"[{i}/{len(slide_dirs)}] ❌ Error reading {slide_name}: {e}")
            continue

        coords = results.get("coordinates", {}).get("inflammatory", [])
        if len(coords) > MAX_WBC_THRESHOLD:
            print(f"[{i}/{len(slide_dirs)}] ⚠️ Skipping {slide_name} (too many WBCs)")
            continue

        cortex_patch_count = results.get("cortex_patch_count", 1) or 1
        cortex_patch_ratio = results.get("cortex_patch_ratio", 0)
        ti_score = results.get("TI", None)
        inflammatory_density = results.get("normalised", {}).get("inflammatory_per_cortex_patch", 0)

        metrics = compute_cluster_metrics(coords, cortex_patch_count)

        # Feature vector
        feature_vector = pd.DataFrame([{
            "inflammatory_per_cortex_patch": inflammatory_density,
            "wbc_clustering_index": metrics["wbc_clustering_index"]
        }])

        # Predict using both models
        try:
            X_multi = scaler_multi.transform(feature_vector)
            X_bin = scaler_bin.transform(feature_vector)

            ti_pred = int(clf_multi.predict(X_multi)[0])
            inflamm_class = int(clf_bin.predict(X_bin)[0])
        except Exception as e:
            print(f"[{i}/{len(slide_dirs)}] ❌ Prediction error on {slide_name}: {e}")
            ti_pred = -1
            inflamm_class = -1

        row = {
            "slide": slide_name,
            "TI": ti_score,
            "TI_pred": ti_pred,
            "Inflamm_class_pred": inflamm_class,
            "cortex_patch_ratio": cortex_patch_ratio,
            "inflammatory_per_cortex_patch": inflammatory_density,
            "total_inflammatory_cells": len(coords),
            **metrics
        }

        writer.writerow(row)
        print(f"[{i}/{len(slide_dirs)}] ✅ {slide_name}: TI_actual={ti_score}, TI_pred={ti_pred}, Inflamm_class={inflamm_class}")

print(f"\n✅ Done. Results saved to: {output_csv}")
