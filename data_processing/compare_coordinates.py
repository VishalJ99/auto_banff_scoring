import json
import matplotlib.pyplot as plt
from pathlib import Path

# === File paths ===
pre_file = Path("/data2/ac2220/pipeline_output/anon_00b1a986-94b9-436d-9520-c87343949154/anon_00b1a986-94b9-436d-9520-c87343949154_wbc_results.json")
post_file = Path("/data2/ac2220/macenko_pipeline_output/anon_00b1a986-94b9-436d-9520-c87343949154/anon_00b1a986-94b9-436d-9520-c87343949154_wbc_results.json")

# === Load JSON files ===
with open(pre_file) as f:
    pre_data = json.load(f)

with open(post_file) as f:
    post_data = json.load(f)

# === Extract inflammatory coordinates ===
pre_coords = pre_data.get("coordinates", {}).get("inflammatory", [])
post_coords = post_data.get("coordinates", {}).get("inflammatory", [])

pre_xy = [(p["x"], p["y"]) for p in pre_coords]
post_xy = [(p["x"], p["y"]) for p in post_coords]

# === Plot side-by-side ===
fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)

# Pre-Macenko
if pre_xy:
    x_pre, y_pre = zip(*pre_xy)
    axes[0].scatter(x_pre, y_pre, s=1, c='blue', alpha=0.5)
axes[0].set_title("Pre-Macenko")
axes[0].set_xlabel("X (pixels)")
axes[0].set_ylabel("Y (pixels)")
axes[0].grid(True)

# Post-Macenko
if post_xy:
    x_post, y_post = zip(*post_xy)
    axes[1].scatter(x_post, y_post, s=1, c='red', alpha=0.5)
axes[1].set_title("Post-Macenko")
axes[1].set_xlabel("X (pixels)")
axes[1].grid(True)

plt.suptitle("Inflammatory Cell Coordinates: Pre vs Post Macenko", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("coordinates_comparison_side_by_side.png", dpi=300)
