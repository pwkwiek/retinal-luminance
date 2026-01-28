import numpy as np
from skimage.filters import threshold_otsu, threshold_yen, threshold_isodata

def threshold_moments_tsai(image01: np.ndarray, nbins: int = 256) -> float:
    img = image01.astype(np.float64).ravel()
    img = img[np.isfinite(img)]
    hist, _ = np.histogram(img, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    s = hist.sum()
    if s == 0:
        return 0.5
    p = hist / s
    i = np.arange(nbins, dtype=np.float64)

    m1 = np.sum(i * p)
    m2 = np.sum((i ** 2) * p)
    m3 = np.sum((i ** 3) * p)

    cd = m2 - m1 * m1
    if abs(cd) < 1e-12:
        return float(m1 / (nbins - 1))

    c0 = (-m2 * m2 + m1 * m3) / cd
    c1 = (-m3 + m2 * m1) / cd

    disc = c1 * c1 - 4.0 * c0
    if disc < 0:
        disc = 0.0

    z0 = 0.5 * (-c1 - np.sqrt(disc))
    z1 = 0.5 * (-c1 + np.sqrt(disc))

    p0 = (z1 - m1) / (z1 - z0 + 1e-12)
    p0 = float(np.clip(p0, 0.0, 1.0))

    cdf = np.cumsum(p)
    idx = int(np.searchsorted(cdf, p0, side="left"))
    idx = max(0, min(nbins - 1, idx))
    return idx / (nbins - 1)

def threshold_image(x01: np.ndarray, method: str, global_t: float = 0.5) -> np.ndarray:
    m = method.lower()
    if m == "otsu":    return x01 > threshold_otsu(x01)
    if m == "yen":     return x01 > threshold_yen(x01)
    if m == "isodata": return x01 > threshold_isodata(x01)
    if m == "moments": return x01 > threshold_moments_tsai(x01)
    if m == "global":  return x01 > global_t
    raise ValueError(f"Unknown threshold method: {method}")