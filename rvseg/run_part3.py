# calculate uncertainties and convergence

from __future__ import annotations

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, distance_transform_edt
from skimage.filters import frangi

from .io_pairs import read_bgr, read_mask, find_pairs
from .channels import ch_gray, ch_rgb_G, ch_yuv_Y, ch_weighted_rgb
from .run_part1 import segment_from_channel
from .metrics import metrics_binary


# UNCERTAINTY HELPERS
def compute_gradient_uncertainty(img01: np.ndarray) -> np.ndarray:
    img = (img01 * 255.0).astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    return 1.0 - mag_norm

def compute_local_contrast_uncertainty(img01: np.ndarray, window: int = 7) -> np.ndarray:
    img = img01.astype(np.float32)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img * img, size=window)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    std_norm = (std - std.min()) / (std.max() - std.min() + 1e-8)
    return 1.0 - std_norm

def compute_snr_uncertainty(img01: np.ndarray, window: int = 5) -> np.ndarray:
    img = img01.astype(np.float32)
    sm = cv2.GaussianBlur(img, (5, 5), 1)
    noise = img - sm
    noise_var = uniform_filter(noise * noise, size=window)
    mean = uniform_filter(img, size=window)
    mean_sq = uniform_filter(img * img, size=window)
    sig_var = np.maximum(mean_sq - mean * mean, 1e-8)
    snr = sig_var / (noise_var + 1e-8)
    snr_norm = (snr - snr.min()) / (snr.max() - snr.min() + 1e-8)
    return np.clip(1.0 - snr_norm, 0, 1)

def compute_vesselness_uncertainty(img01: np.ndarray, sigmas=(0.5, 3.0)) -> np.ndarray:
    x = 1.0 - np.clip(img01, 0, 1)
    v = frangi(x, sigmas=np.linspace(sigmas[0], sigmas[1], 6), black_ridges=False)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return 1.0 - v_norm

def compute_aleatoric_uncertainty(img01: np.ndarray, w=(0.35, 0.30, 0.10, 0.25)) -> np.ndarray:
    w_grad, w_con, w_snr, w_ves = w
    u1 = compute_gradient_uncertainty(img01)
    u2 = compute_local_contrast_uncertainty(img01)
    u3 = compute_snr_uncertainty(img01)
    u4 = compute_vesselness_uncertainty(img01)
    u = w_grad * u1 + w_con * u2 + w_snr * u3 + w_ves * u4
    u = (u - u.min()) / (u.max() - u.min() + 1e-8)
    return np.clip(u, 0, 1)

def compute_geometric_uncertainty(vessel_mask: np.ndarray) -> np.ndarray:
    m = vessel_mask.astype(bool)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.float32)

    dist_in = distance_transform_edt(m).astype(np.float32)

    s = 1.5
    u_boundary = np.exp(-dist_in / s).astype(np.float32)  # 1 at edges, falls inside

    thickness = 2.0 * dist_in
    t_norm = (thickness - thickness.min()) / (thickness.max() - thickness.min() + 1e-8)
    u_thick = 1.0 - t_norm

    u = (0.4 * u_boundary + 0.6 * u_thick).astype(np.float32)

    out = np.zeros_like(u, dtype=np.float32)
    out[m] = u[m]
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    return np.clip(out, 0, 1)

def compute_inter_observer_uncertainty(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
    return np.abs(mask1.astype(np.float32) - mask2.astype(np.float32))

def combine_uncertainties(u_ale: np.ndarray, u_geo: np.ndarray, u_inter=None, weights=(0.4, 0.4, 0.2)) -> np.ndarray:
    u_ale = np.clip(u_ale, 0, 1)
    u_geo = np.clip(u_geo, 0, 1)

    if u_inter is None:
        u = 0.7 * u_ale + 0.3 * u_geo
    else:
        wa, wg, wi = weights
        s = wa + wg + wi + 1e-8
        u = (wa * u_ale + wg * u_geo + wi * np.clip(u_inter, 0, 1)) / s

    u = (u - u.min()) / (u.max() - u.min() + 1e-8)
    return np.clip(u, 0, 1)

def save_uncertainty_png(u01: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(u01, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(out_path), u8)


# COVERAGE PANEL HELPERS
def keep_by_coverage(pred_mask: np.ndarray, u_map: np.ndarray, coverage: float):
    vessel = pred_mask.astype(bool)
    if vessel.sum() == 0:
        return np.zeros_like(vessel, dtype=bool), np.nan
    if coverage >= 1.0:
        return vessel.copy(), 1.0
    if coverage <= 0.0:
        return np.zeros_like(vessel, dtype=bool), 0.0

    u_vals = u_map[vessel]
    thr = float(np.quantile(u_vals, coverage))  # keep lowest-U vessel pixels
    kept = vessel & (u_map <= thr)
    return kept, thr

def overlay_mask_on_bgr(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.65, bg_alpha: float = 0.5):
    rgb = bgr[..., ::-1].astype(np.float32) / 255.0
    base = bg_alpha * rgb
    m = mask.astype(bool)
    white = np.ones_like(base)
    out = base.copy()
    out[m] = (1 - alpha) * out[m] + alpha * white[m]
    return np.clip(out, 0, 1)

def save_coverage_panel(
    bgr: np.ndarray,
    pred: np.ndarray,
    u_comb: np.ndarray,
    out_path: Path,
    coverages=(0.1, 0.6, 1.0),
    title: str = "",
):
    pred_bool = pred.astype(bool)
    u = np.clip(u_comb.astype(np.float32), 0, 1)

    ncols = 2 + len(coverages)
    plt.figure(figsize=(4 * ncols, 4))

    ax = plt.subplot(1, ncols, 1)
    ax.imshow(bgr[..., ::-1])
    ax.set_title("Input")
    ax.axis("off")

    ax = plt.subplot(1, ncols, 2)
    im = ax.imshow(u, vmin=0, vmax=1)
    ax.set_title("Ucomb (0=confident)")
    ax.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    total = int(pred_bool.sum())
    for i, c in enumerate(coverages, start=3):
        kept, thr = keep_by_coverage(pred_bool, u, c)
        ov = overlay_mask_on_bgr(bgr, kept, alpha=0.95, bg_alpha=0.45)

        ax = plt.subplot(1, ncols, i)
        ax.imshow(ov)
        ax.axis("off")
        ax.set_title(f"Cov {c:.1f}\nthr={thr:.3f}\n{int(kept.sum())}/{total}")

    if title:
        plt.suptitle(title, y=1.03, fontsize=14)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close()


# MAIN PART 3
def run_part3(
    paths_cfg,
    preproc_cfg,
    post_cfg,
    thr_cfg,
    unc_cfg,
    best_w: tuple[float, float, float],
    thresh_method: str,
    selected_channels=("GRAY", "RGB_G", "YUV_Y", "W_RGB_BEST"),
    out_per_image="uncertainty_selected_channels_per_image.csv",
    out_summary="uncertainty_selected_channels_summary.csv",
):
    paired = find_pairs(paths_cfg.images_dir, paths_cfg.masks_dir, paths_cfg.masks2_dir)

    unc_cfg.unc_out_dir.mkdir(parents=True, exist_ok=True)
    unc_cfg.coverage_out_dir.mkdir(parents=True, exist_ok=True)

    # channel selection (names match file naming)
    selected = {}
    for ch in selected_channels:
        if ch in ("GRAY", "RGB_G", "YUV_Y"):
            selected[ch] = ("fixed", None)
        elif ch == "W_RGB_BEST":
            selected[ch] = ("weighted", best_w)
        else:
            raise ValueError(f"Unknown selected channel: {ch}")

    records = []

    for idx, (img_path, m1_path, m2_path) in enumerate(tqdm(paired, desc="Part3: Uncertainty")):
        bgr = read_bgr(img_path)
        gt1 = read_mask(m1_path)
        gt2 = read_mask(m2_path) if m2_path is not None else None

        for name, (kind, param) in selected.items():
            if kind == "fixed":
                if name == "GRAY":
                    ch = ch_gray(bgr)
                elif name == "RGB_G":
                    ch = ch_rgb_G(bgr)
                elif name == "YUV_Y":
                    ch = ch_yuv_Y(bgr)
                else:
                    raise ValueError("Unknown fixed channel")
            else:
                wR, wG, wB = param
                ch = ch_weighted_rgb(bgr, wR, wG, wB)

            pred = segment_from_channel(
                ch, thresh_method, preproc_cfg.default_preproc,
                preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                thr_cfg.global_t,
                post_cfg.remove_small, post_cfg.min_obj_size
            )

            u_ale = compute_aleatoric_uncertainty(ch)
            u_geo = compute_geometric_uncertainty(pred)
            u_inter = compute_inter_observer_uncertainty(gt1, gt2) if gt2 is not None else None
            u_comb = combine_uncertainties(u_ale, u_geo, u_inter=u_inter)

            vessel_pixels = pred.astype(bool)
            mean_u_vessel = float(u_comb[vessel_pixels].mean()) if vessel_pixels.sum() else float(u_comb.mean())
            mean_u_all = float(u_comb.mean())

            err = (pred.astype(np.uint8) != gt1.astype(np.uint8)).astype(np.uint8)
            mean_u_error = float(u_comb[err.astype(bool)].mean()) if err.sum() else 0.0

            high_unc = (u_comb > unc_cfg.uncertainty_threshold) & vessel_pixels
            frac_high_unc_vessel = float(high_unc.sum() / (vessel_pixels.sum() + 1e-8))

            met = metrics_binary(pred, gt1)

            row = {
                "image": img_path.stem,
                "channel": name,
                "thresh": thresh_method,
                "dice": met["dice"],
                "accuracy": met["accuracy"],
                "precision": met["precision"],
                "recall": met["recall"],
                "specificity": met["specificity"],
                "mean_uncertainty_all": mean_u_all,
                "mean_uncertainty_vessel": mean_u_vessel,
                "mean_uncertainty_error": mean_u_error,
                "high_uncertainty_fraction_vessel": frac_high_unc_vessel,
            }

            if kind == "weighted":
                row.update({"wR": param[0], "wG": param[1], "wB": param[2]})
            else:
                row.update({"wR": np.nan, "wG": np.nan, "wB": np.nan})

            records.append(row)

            # Save maps + coverage panels for first 5 images
            if unc_cfg.save_uncertainty_maps and idx < 5:
                save_uncertainty_png(u_comb, unc_cfg.unc_out_dir / f"{img_path.stem}_{name}_Ucomb.png")
                save_uncertainty_png(u_ale,  unc_cfg.unc_out_dir / f"{img_path.stem}_{name}_Uale.png")
                save_uncertainty_png(u_geo,  unc_cfg.unc_out_dir / f"{img_path.stem}_{name}_Ugeo.png")
                if u_inter is not None:
                    save_uncertainty_png(u_inter, unc_cfg.unc_out_dir / f"{img_path.stem}_{name}_Uinter.png")

                save_coverage_panel(
                    bgr=bgr,
                    pred=pred,
                    u_comb=u_comb,
                    out_path=unc_cfg.coverage_out_dir / f"{img_path.stem}_{name}_coverage.png",
                    coverages=(0.1, 0.6, 1.0),
                    title=f"{img_path.stem} | {name} | thresh={thresh_method}",
                )

    df_unc = pd.DataFrame(records)
    df_unc.to_csv(out_per_image, index=False)

    summary_unc = (
        df_unc.groupby("channel")[[
            "dice", "accuracy", "precision", "recall", "specificity",
            "mean_uncertainty_all", "mean_uncertainty_vessel", "mean_uncertainty_error",
            "high_uncertainty_fraction_vessel"
        ]]
        .mean()
        .reset_index()
        .sort_values("dice", ascending=False)
    )
    summary_unc.to_csv(out_summary, index=False)

    return df_unc, summary_unc
