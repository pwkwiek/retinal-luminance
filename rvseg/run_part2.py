# generate best weighted RGB channel

import numpy as np
import pandas as pd
from tqdm import tqdm

from .io_pairs import read_bgr, read_mask, find_pairs
from .channels import ch_rgb_G, ch_yuv_Y, ch_gray, ch_weighted_rgb
from .run_part1 import segment_from_channel
from .metrics import metrics_binary

def generate_weight_grid(step=0.05, min_g=0.55, max_b=0.10):
    vals = np.arange(0.0, 1.0 + 1e-9, step)
    ws = []
    for wB in vals:
        if wB > max_b:
            continue
        for wG in vals:
            if wG < min_g:
                continue
            wR = 1.0 - wG - wB
            if wR < -1e-9:
                continue
            if abs(wR) < 1e-9:
                wR = 0.0
            if abs((wR + wG + wB) - 1.0) < 1e-6:
                ws.append((round(float(wR),4), round(float(wG),4), round(float(wB),4)))
    return sorted(set(ws))

def generate_local_grid(center, step=0.01, span=0.05, min_g=0.55, max_b=0.10):
    cR, cG, cB = center
    ws = []
    r_vals = np.arange(max(0.0, cR - span), min(1.0, cR + span) + 1e-9, step)
    g_vals = np.arange(max(0.0, cG - span), min(1.0, cG + span) + 1e-9, step)
    for wR in r_vals:
        for wG in g_vals:
            wB = 1.0 - wR - wG
            if wB < 0: continue
            if wG < min_g: continue
            if wB > max_b: continue
            wR2, wG2, wB2 = round(float(wR),4), round(float(wG),4), round(float(wB),4)
            if abs((wR2+wG2+wB2)-1.0) < 1e-4:
                ws.append((wR2,wG2,wB2))
    return sorted(set(ws))

def pick_thresh_for_rgb_g(summary_default_csv: str) -> str:
    df = pd.read_csv(summary_default_csv)
    best_row = df[df["channel"]=="RGB_G"].sort_values("dice", ascending=False).iloc[0]
    return str(best_row["thresh"])

def run_part2(paths_cfg, preproc_cfg, post_cfg, thr_cfg, ws_cfg,
              summary_default_csv="segmentation_all_channels_summary.csv",
              out_coarse_per_image="weight_search_coarse_per_image.csv",
              out_coarse_summary="weight_search_coarse_summary.csv",
              out_ref_per_image="weight_search_refined_per_image.csv",
              out_ref_summary="weight_search_refined_summary.csv"):
    paired = find_pairs(paths_cfg.images_dir, paths_cfg.masks_dir, paths_cfg.masks2_dir)
    weight_search_thresh = pick_thresh_for_rgb_g(summary_default_csv)

    # COARSE
    coarse_candidates = generate_weight_grid(ws_cfg.coarse_step, ws_cfg.min_g, ws_cfg.max_b)
    for w in ws_cfg.force_include:
        if w not in coarse_candidates:
            coarse_candidates.append(w)
    coarse_candidates = sorted(set(coarse_candidates))

    coarse_recs = []
    for (img_path, m1_path, _) in tqdm(paired, desc="Part2: Coarse"):
        bgr = read_bgr(img_path)
        gt  = read_mask(m1_path)

        for label, ch in [("RGB_G", ch_rgb_G(bgr)), ("YUV_Y", ch_yuv_Y(bgr)), ("GRAY", ch_gray(bgr))]:
            pred = segment_from_channel(
                ch, weight_search_thresh, preproc_cfg.default_preproc,
                preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
            )
            met = metrics_binary(pred, gt)
            met.update({"image": img_path.stem, "channel": label, "wR": np.nan, "wG": np.nan, "wB": np.nan, "phase": "baseline"})
            coarse_recs.append(met)

        for (wR, wG, wB) in coarse_candidates:
            ch = ch_weighted_rgb(bgr, wR, wG, wB)
            pred = segment_from_channel(
                ch, weight_search_thresh, preproc_cfg.default_preproc,
                preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
            )
            met = metrics_binary(pred, gt)
            met.update({"image": img_path.stem, "channel": "W_RGB", "wR": wR, "wG": wG, "wB": wB, "phase": "coarse"})
            coarse_recs.append(met)

    df_coarse = pd.DataFrame(coarse_recs)
    df_coarse.to_csv(out_coarse_per_image, index=False)

    sum_coarse = (
        df_coarse[df_coarse["phase"]=="coarse"]
        .groupby(["channel","wR","wG","wB"])[["dice","iou","precision","recall","accuracy","specificity","f1"]]
        .mean().reset_index().sort_values("dice", ascending=False)
    )
    sum_coarse.to_csv(out_coarse_summary, index=False)
    best = sum_coarse.iloc[0]
    best_w = (float(best["wR"]), float(best["wG"]), float(best["wB"]))

    # REFINE
    refined_candidates = generate_local_grid(best_w, ws_cfg.refine_step, ws_cfg.refine_span, ws_cfg.min_g, ws_cfg.max_b)

    ref_recs = []
    for (img_path, m1_path, _) in tqdm(paired, desc="Part2: Refine"):
        bgr = read_bgr(img_path)
        gt  = read_mask(m1_path)

        for label, ch in [("RGB_G", ch_rgb_G(bgr)), ("YUV_Y", ch_yuv_Y(bgr)), ("GRAY", ch_gray(bgr))]:
            pred = segment_from_channel(
                ch, weight_search_thresh, preproc_cfg.default_preproc,
                preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
            )
            met = metrics_binary(pred, gt)
            met.update({"image": img_path.stem, "channel": label, "wR": np.nan, "wG": np.nan, "wB": np.nan, "phase": "baseline"})
            ref_recs.append(met)

        for (wR, wG, wB) in refined_candidates:
            ch = ch_weighted_rgb(bgr, wR, wG, wB)
            pred = segment_from_channel(
                ch, weight_search_thresh, preproc_cfg.default_preproc,
                preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
            )
            met = metrics_binary(pred, gt)
            met.update({"image": img_path.stem, "channel": "W_RGB", "wR": wR, "wG": wG, "wB": wB, "phase": "refine"})
            ref_recs.append(met)

    df_ref = pd.DataFrame(ref_recs)
    df_ref.to_csv(out_ref_per_image, index=False)

    sum_ref = (
        df_ref[df_ref["phase"]=="refine"]
        .groupby(["channel","wR","wG","wB"])[["dice","iou","precision","recall","accuracy","specificity","f1"]]
        .mean().reset_index().sort_values("dice", ascending=False)
    )
    sum_ref.to_csv(out_ref_summary, index=False)

    return weight_search_thresh, best_w