import numpy as np
from skimage import exposure
from skimage.morphology import white_tophat, disk

def preprocess(gray01: np.ndarray,
               preproc_variant: str,
               clahe_kernel=None,
               clahe_clip: float = 0.01,
               tophat_radius: int = 5) -> np.ndarray:
    x = np.clip(gray01.astype(np.float32), 0.0, 1.0)
    x = 1.0 - x  # invert always

    v = preproc_variant.upper()
    if v in ("CLAHE", "CLAHE_TOPHAT"):
        x = exposure.equalize_adapthist(x, kernel_size=clahe_kernel, clip_limit=clahe_clip)

    if v == "CLAHE_TOPHAT":
        x = white_tophat(x, footprint=disk(tophat_radius))

    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x