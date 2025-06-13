import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.linalg import svd
from glob import glob


def rgb_to_od(rgb):
    rgb = np.clip(rgb, 1, 255).astype(np.float32)
    return -np.log10(rgb / 255.0 + 1e-8)


def get_stain_matrix(img, beta=0.15, alpha=1.0, min_pixels=500):
    od = rgb_to_od(img.reshape((-1, 3)))
    od = od[np.all(od > beta, axis=1)]
    if len(od) < min_pixels:
        raise ValueError("Too few valid OD pixels")
    _, _, Vt = svd(od, full_matrices=False)
    V = Vt[:2, :].T
    proj = od @ V
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, [np.cos(min_phi), np.sin(min_phi)])
    v2 = np.dot(V, [np.cos(max_phi), np.sin(max_phi)])
    HE = np.stack([v1, v2], axis=1)
    return HE / np.linalg.norm(HE, axis=0)


def visualize_stain_vectors(ref_dir):
    paths = sorted(glob(os.path.join(ref_dir, "*.png")))
    if not paths:
        print("❌ No reference .png files found.")
        return

    fig_od, ax_od = plt.subplots(figsize=(6, 6))
    ax_od.set_title("Stain Vectors in OD Space")
    ax_od.set_xlim(-1, 1)
    ax_od.set_ylim(-1, 1)
    ax_od.axhline(0, color='gray', lw=0.5)
    ax_od.axvline(0, color='gray', lw=0.5)
    ax_od.set_aspect('equal')
    ax_od.grid(True)

    rgb_bars = []

    for path in paths:
        try:
            img = imread(path)
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            stain_matrix = get_stain_matrix(img)
            vectors = stain_matrix.T

            # Plot OD directions
            for vec in vectors:
                ax_od.plot(vec[0], vec[1], 'o', alpha=0.7)

            # Convert OD → RGB
            stain_rgb = 255 * np.exp(-vectors)
            rgb_bars.append(np.clip(stain_rgb / 255.0, 0, 1))

        except Exception as e:
            print(f"⚠️ {os.path.basename(path)}: {e}")

    plt.tight_layout()
    plt.savefig("macenko_visualisation1.png", dpi=150, bbox_inches="tight")
    print(f"✅ Saved debug plot to: macenko_visualisation1.png")

    # Show RGB bar chart of stain colors
    fig_rgb, ax_rgb = plt.subplots(figsize=(5, 0.3 * len(rgb_bars)))
    for i, colors in enumerate(rgb_bars):
        for j in range(2):
            ax_rgb.add_patch(plt.Rectangle((j, i), 1, 1, color=colors[j]))
    ax_rgb.set_xlim(0, 2)
    ax_rgb.set_ylim(0, len(rgb_bars))
    ax_rgb.set_xticks([0.5, 1.5])
    ax_rgb.set_xticklabels(["Stain 1", "Stain 2"])
    ax_rgb.set_yticks([])
    ax_rgb.set_title("Reference Stain Colors (RGB)")
    plt.tight_layout()
    plt.savefig("macenko_visualisation2.png", dpi=150, bbox_inches="tight")
    print(f"✅ Saved debug plot to: macenko_visualisation2.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_dir", required=True, help="Path to directory of reference PNG patches")
    args = parser.parse_args()

    visualize_stain_vectors(args.ref_dir)
