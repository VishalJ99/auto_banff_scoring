import os
import random
import numpy as np
from pathlib import Path
from staintools import StainNormalizer, read_image

class MultiReferenceMacenko:
    def __init__(self, reference_dir, n_refs=5, seed=42):
        self.reference_dir = Path(reference_dir)
        self.reference_images = list(self.reference_dir.glob("*.png"))
        if len(self.reference_images) < n_refs:
            raise ValueError("Not enough reference slides for the requested number.")
        random.seed(seed)
        self.selected_refs = random.sample(self.reference_images, n_refs)
        self.normalizers = []
        for ref_path in self.selected_refs:
            ref_img = read_image(str(ref_path))
            norm = StainNormalizer(method='macenko')
            norm.fit(ref_img)
            self.normalizers.append(norm)

    def normalize(self, img):
        # Skip normalization for low-content patches
        if img.mean() > 240 or img.std() < 5:
            raise ValueError("Patch is too light or low contrast for reliable normalization")

        # Select the best normalizer based on mean intensity similarity
        patch_mean = img.mean()
        scores = [
            abs(patch_mean - np.mean(norm.target_concentrations))
            for norm in self.normalizers
        ]
        best_idx = int(np.argmin(scores))
        return self.normalizers[best_idx].normalize(img)

def normalize_patch_or_slide(input_path, output_path, normalizer):
    img = read_image(str(input_path))
    try:
        norm_img = normalizer.normalize(img)
        os.makedirs(Path(output_path).parent, exist_ok=True)
        norm_img.save(output_path)
    except Exception as e:
        print(f"⚠️ Skipping {input_path} due to normalization failure: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Apply Macenko stain normalization using multiple references.")
    parser.add_argument("--input_dir", required=True, help="Input directory of .png patches or slides")
    parser.add_argument("--output_dir", required=True, help="Where to save normalized images")
    parser.add_argument("--reference_dir", required=True, help="Directory containing reference slides (.png)")
    parser.add_argument("--n_refs", type=int, default=5, help="Number of reference slides to sample")
    args = parser.parse_args()

    normalizer = MultiReferenceMacenko(reference_dir=args.reference_dir, n_refs=args.n_refs)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    all_imgs = list(input_dir.rglob("*.png"))

    print(f"🔎 Found {len(all_imgs)} images to normalize.")

    for img_path in all_imgs:
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        normalize_patch_or_slide(img_path, out_path, normalizer)
