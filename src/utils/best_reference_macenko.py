import numpy as np
from pathlib import Path
from PIL import Image
from staintools import StainNormalizer, convert_RGB_to_OD  # Your existing functions

class BestReferenceMacenko:
    def __init__(self, reference_dir):
        self.reference_dir = Path(reference_dir)
        self.reference_paths = list(self.reference_dir.glob("*.png"))
        if not self.reference_paths:
            raise ValueError("No reference images found in the provided directory.")

        self.references = [self._load_image(p) for p in self.reference_paths]
        self.normalizers = []
        for img in self.references:
            norm = StainNormalizer()
            norm.fit(img)
            self.normalizers.append(norm)

    def _load_image(self, path):
        return np.array(Image.open(path).convert("RGB"))

    def _get_best_index(self, img):
        img_mean = img.mean(axis=(0, 1))
        dists = [np.linalg.norm(img_mean - ref.mean(axis=(0, 1))) for ref in self.references]
        return int(np.argmin(dists))

    def normalize(self, img):
        if img.mean() > 240 or img.std() < 5:
            raise ValueError("Patch is too light or low contrast for reliable normalization")

        best_idx = self._get_best_index(img)
        return self.normalizers[best_idx].normalize(img)