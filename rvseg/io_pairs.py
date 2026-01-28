from pathlib import Path
import re
import cv2
import numpy as np
from .config import IMG_EXTS

def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img

def read_mask(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Could not read mask: {path}")
    return (m > 0)

def extract_id(stem: str):
    m = re.match(r"^(\d+)", stem)
    return int(m.group(1)) if m else None

def find_pairs(images_dir: Path, masks_dir: Path, masks2_dir: Path | None = None):
    image_paths = [p for p in images_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    mask_paths  = [p for p in masks_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found in {images_dir.resolve()}")
    if len(mask_paths) == 0:
        raise FileNotFoundError(f"No masks found in {masks_dir.resolve()}")

    mask_by_id = {}
    for mp in mask_paths:
        mid = extract_id(mp.stem)
        if mid is None:
            continue
        if mid not in mask_by_id:
            mask_by_id[mid] = mp
        else:
            cur = mask_by_id[mid]
            if ("manual" in mp.stem.lower()) and ("manual" not in cur.stem.lower()):
                mask_by_id[mid] = mp

    mask2_by_id = None
    if masks2_dir is not None:
        mask2_paths = [p for p in Path(masks2_dir).rglob("*") if p.suffix.lower() in IMG_EXTS]
        mask2_by_id = {}
        for mp in mask2_paths:
            mid = extract_id(mp.stem)
            if mid is None:
                continue
            mask2_by_id[mid] = mp

    paired = []
    for ip in image_paths:
        iid = extract_id(ip.stem)
        if iid is None:
            continue
        m1 = mask_by_id.get(iid, None)
        if m1 is None:
            continue
        m2 = mask2_by_id.get(iid, None) if mask2_by_id is not None else None
        paired.append((ip, m1, m2))

    if len(paired) == 0:
        raise ValueError("No (image, mask) pairs found. Check filenames / extract_id().")
    return paired