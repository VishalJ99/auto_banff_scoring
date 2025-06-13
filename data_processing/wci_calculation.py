import os
import json
import csv
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.stats import pearsonr
import pandas as pd

# Parameters
input_root = Path("/data2/ac2220/pipeline_output")  # 🔁 Update if needed
MICRONS_PER_PIXEL = 0.262719
min_samples = 3
eps_micron_range = [27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
output_csv = "data_handling/eps_wci_correlation_summary.csv"

# Load TI scores and coordinates from all slides
def load_slide_data(slide_dir):
    slide_name = slide_dir.name
    results_path = slide_dir / f"{slide_name}_wbc_results.json"
    if not results_path.exists():
        return None
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        coords = results.get("coordinates", {}).get("inflammatory", [])
        ti_score = results.get("TI", None)
        return slide_name, coords, ti_score
    except Exception:
        return None

# Compute WBC Clustering Index
def compute_wci(coords, eps_pixels):
    if len(coords) == 0:
        return 0
    points = np.array([[p["x"], p["y"]] for p in coords])
    clustering = DBSCAN(eps=eps_pixels, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clustered = labels >= 0
    cluster_ids, counts = np.unique(labels[clustered], return_counts=True)
    if len(counts) == 0:
        return 0
    return np.sum(counts**2) / (len(coords) ** 2)

# Load all valid slides
slides = []
for slide_dir in input_root.iterdir():
    if not slide_dir.is_dir():
        continue
    data = load_slide_data(slide_dir)
    if data is None:
        continue
    slide_name, coords, ti = data
    if ti is None:
        continue
    slides.append({"slide": slide_name, "coords": coords, "TI": ti})

# Sweep eps and compute correlation
summary = []
for eps_microns in eps_micron_range:
    eps_pixels = eps_microns / MICRONS_PER_PIXEL
    ti_scores = []
    wci_values = []

    for slide in slides:
        ti = slide["TI"]
        wci = compute_wci(slide["coords"], eps_pixels)
        ti_scores.append(ti)
        wci_values.append(wci)

    r, p = pearsonr(ti_scores, wci_values)
    print(f"eps = {eps_microns} μm → r = {r:.3f}, p = {p:.2e}")
    summary.append({"eps_microns": eps_microns, "r": round(r, 3), "p": p})

# Save summary CSV
summary_df = pd.DataFrame(summary)
summary_df.to_csv(output_csv, index=False)
print(f"✅ Correlation summary saved to: {output_csv}")
