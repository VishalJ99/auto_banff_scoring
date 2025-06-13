import numpy as np
from PIL import Image

class StainNormalizer:
    def __init__(self, method='macenko', verbose=False):
        if method != 'macenko':
            raise NotImplementedError("Only Macenko normalization is implemented.")
        self.target_concentrations = None
        self.target_stain_matrix = None
        self.verbose = verbose

    def fit(self, target_image):
        OD = convert_RGB_to_OD(target_image)
        mask = (OD > 0.15).any(axis=2)
        OD_flat = OD[mask]
        stain_matrix = get_stain_matrix_macenko(OD_flat, use_reference_fallback=True)
        concentrations = get_concentrations(OD_flat, stain_matrix)
        self.target_stain_matrix = stain_matrix
        self.target_concentrations = concentrations

    def normalize(self, source_image):
        OD = convert_RGB_to_OD(source_image)
        mask = (OD > 0.15).any(axis=2)
        if not np.any(mask):
            raise ValueError("Patch is too light or low contrast for reliable normalization")

        OD_flat = OD[mask]
        
        # Try to estimate stain matrix, but be more lenient about using references
        try:
            source_stain_matrix = get_stain_matrix_macenko(OD_flat, use_reference_fallback=True)
        except ValueError:
            # If estimation fails completely, use the target stain matrix
            if self.verbose:
                print("Using target stain matrix for source image")
            source_stain_matrix = self.target_stain_matrix
            
        source_concentrations = get_concentrations(OD_flat, source_stain_matrix)
        
        # Clip concentrations to reasonable bounds to prevent extreme values
        source_concentrations = np.clip(source_concentrations, 0, 3.0)

        # Use percentile-based normalization for better histogram matching
        normalized_concentrations = np.zeros_like(source_concentrations)
        
        for i in range(2):  # For each stain (H&E)
            source_perc = np.percentile(source_concentrations[:, i], [1, 50, 99])
            target_perc = np.percentile(self.target_concentrations[:, i], [1, 50, 99])
            
            # Map source percentiles to target percentiles
            normalized_concentrations[:, i] = np.interp(
                source_concentrations[:, i], source_perc, target_perc
            )

        # Fallback to scaling method if percentile method fails
        try:
            if np.any(np.isnan(normalized_concentrations)) or np.any(np.isinf(normalized_concentrations)):
                raise ValueError("Percentile normalization failed")
        except:
            # Fallback: use improved scaling method
            maxC_source = np.percentile(source_concentrations, 99, axis=0)
            maxC_target = np.percentile(self.target_concentrations, 99, axis=0)
            raw_scale = maxC_target / (maxC_source + 1e-8)
            scale = np.clip(raw_scale, 0.5, 2.0)  # Relaxed scaling bounds
            normalized_concentrations = source_concentrations * scale

        # Reconstruct OD from normalized concentrations
        reconstructed_OD_flat = np.dot(normalized_concentrations, self.target_stain_matrix)

        # Fill background as white (OD = 0 → RGB = 255)
        reconstructed_OD = np.zeros_like(OD)
        reconstructed_OD[mask] = reconstructed_OD_flat
        
        normalized_rgb = convert_OD_to_RGB(reconstructed_OD)
        
        # Apply subtle contrast enhancement
        enhanced_rgb = enhance_contrast(normalized_rgb, alpha=1.05)
        
        return enhanced_rgb

def read_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img)

def convert_RGB_to_OD(I):
    I = I.astype(np.float32)
    I = np.maximum(I, 1)  # avoid log(0)
    return -np.log(I / 255.0)

def convert_OD_to_RGB(OD):
    # Keep background pixels (OD=0) as pure white, but clamp stained regions
    OD_clamped = np.clip(OD, 0.0, 2.0)  # Allow true background (OD=0 → RGB=255)
    RGB = np.exp(-OD_clamped) * 255.0
    return np.clip(RGB, 0, 255).astype(np.uint8)

def get_stain_matrix_macenko(I, alpha=0.1, use_reference_fallback=True):
    if I.shape[0] < 10:
        raise ValueError("Not enough valid pixels for stain matrix estimation")

    # Check if we have sufficient color variation
    color_range = np.ptp(I, axis=0)  # Peak-to-peak (max - min) for each channel
    if np.max(color_range) < 0.3:  # Very low color variation
        if use_reference_fallback:
            # Use reference vectors without warning for low-variation images
            v1 = np.array([0.65, 0.70, 0.29])  # Hematoxylin reference
            v2 = np.array([0.07, 0.99, 0.11])  # Eosin reference
            stain_matrix = np.array([v1, v2])
            return normalize_rows(stain_matrix)
        else:
            raise ValueError("Insufficient color variation for stain matrix estimation")

    _, _, Vt = np.linalg.svd(I, full_matrices=False)
    V = Vt.T  # shape (3, 3)
    proj = np.dot(I, V[:, :2])
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    
    # Use more robust percentile selection
    min_phi = np.percentile(phi, alpha * 100)
    max_phi = np.percentile(phi, (1 - alpha) * 100)
    
    # Ensure we have sufficient angular separation
    angle_diff = abs(max_phi - min_phi)
    if angle_diff < np.pi/4:  # Less than 45 degrees separation
        # Expand the angle range
        phi_mean = np.mean([min_phi, max_phi])
        min_phi = phi_mean - np.pi/3
        max_phi = phi_mean + np.pi/3

    v1 = np.dot(V[:, :2], [np.cos(min_phi), np.sin(min_phi)])
    v2 = np.dot(V[:, :2], [np.cos(max_phi), np.sin(max_phi)])

    if v1[0] < 0: v1 = -v1
    if v2[0] < 0: v2 = -v2

    # More lenient validation - only use reference if vectors are nearly identical
    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    if abs(dot_product) > 0.95:  # Only for very similar vectors (was 0.9)
        if use_reference_fallback:
            v1 = np.array([0.65, 0.70, 0.29])  # Hematoxylin reference
            v2 = np.array([0.07, 0.99, 0.11])  # Eosin reference
        else:
            # Try to create orthogonal vectors
            v2 = v1 + np.array([0.1, -0.1, 0.05])  # Small perturbation
            v2 = v2 / np.linalg.norm(v2)

    stain_matrix = np.array([v1, v2])
    return normalize_rows(stain_matrix)

def normalize_rows(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

def get_concentrations(I, stain_matrix):
    concentrations, _, _, _ = np.linalg.lstsq(stain_matrix.T, I.T, rcond=None)
    return concentrations.T  # shape (N, 2)

def enhance_contrast(img, alpha=1.05):
    """
    Apply subtle contrast enhancement to improve visual quality
    
    Args:
        img: Input RGB image as numpy array
        alpha: Contrast enhancement factor (1.0 = no change, >1.0 = more contrast)
    
    Returns:
        Enhanced image as numpy array
    """
    if alpha == 1.0:
        return img
        
    img_float = img.astype(np.float32)
    
    # Apply contrast enhancement per channel to preserve color balance
    enhanced = np.zeros_like(img_float)
    for c in range(3):  # RGB channels
        channel = img_float[:, :, c]
        mean = channel.mean()
        enhanced[:, :, c] = mean + alpha * (channel - mean)
    
    return np.clip(enhanced, 0, 255).astype(np.uint8)