from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathsConfig:
    images_dir: Path
    masks_dir: Path
    masks2_dir: Path | None = None

@dataclass
class PreprocConfig:
    clahe_kernel: int | None = None
    clahe_clip: float = 0.01
    tophat_radius: int = 5
    preproc_variants: tuple[str, ...] = ("NONE", "CLAHE", "CLAHE_TOPHAT")
    default_preproc: str = "CLAHE_TOPHAT"

@dataclass
class PostprocConfig:
    remove_small: bool = True
    min_obj_size: int = 50

@dataclass
class ThresholdConfig:
    global_t: float = 0.50
    thresh_methods: tuple[str, ...] = ("otsu", "yen", "isodata", "moments", "global")

@dataclass
class WeightSearchConfig:
    min_g: float = 0.55
    max_b: float = 0.10
    coarse_step: float = 0.05
    refine_step: float = 0.01
    refine_span: float = 0.05
    force_include: tuple[tuple[float,float,float], ...] = ((0.20, 0.75, 0.05),)

@dataclass
class UncertaintyConfig:
    uncertainty_threshold: float = 0.70
    save_uncertainty_maps: bool = True
    unc_out_dir: Path = Path("uncertainty_outputs")
    coverage_out_dir: Path = Path("uncertainty_outputs/coverage_panels")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}