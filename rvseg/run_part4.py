# plot analysis

from __future__ import annotations

import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

from .io_pairs import read_bgr, read_mask, find_pairs
from .channels import ch_rgb_G, ch_yuv_Y, ch_weighted_rgb
from .run_part1 import segment_from_channel


def _load_unc_png(unc_dir: Path, stem: str, ch_name: str, kind="Ucomb"):
    p = unc_dir / f"{stem}_{ch_name}_{kind}.png"
    if not p.exists():
        return None
    u = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if u is None:
        return None
    return u.astype(np.float32) / 255.0


def run_part4(
    paths_cfg,
    preproc_cfg,
    post_cfg,
    thr_cfg,
    fig_dir: Path = Path("figs"),
    # inputs from earlier parts:
    seg_summary_csv: str = "segmentation_all_channels_summary.csv",
    seg_per_image_csv: str = "segmentation_all_channels_per_image_with_preproc.csv",
    weight_refined_summary_csv: str = "weight_search_refined_summary.csv",
    weight_refined_per_image_csv: str = "weight_search_refined_per_image.csv",
    unc_per_image_csv: str = "uncertainty_selected_channels_per_image.csv",
    unc_summary_csv: str = "uncertainty_selected_channels_summary.csv",
    uncertainty_outputs_dir: Path = Path("uncertainty_outputs"),
    # controls:
    n_examples: int = 3,
    filter_channels=("RGB_G", "YUV_Y", "W_RGB_BEST"),
    taus=np.linspace(0.0, 0.95, 20),
    out_filter_curve_csv: str = "uncertainty_filtering_curves.csv",
):
    fig_dir.mkdir(parents=True, exist_ok=True)

    seg_sum = pd.read_csv(seg_summary_csv)
    seg_img = pd.read_csv(seg_per_image_csv)
    unc_img = pd.read_csv(unc_per_image_csv)
    unc_sum = pd.read_csv(unc_summary_csv)

    wref_sum = pd.read_csv(weight_refined_summary_csv) if Path(weight_refined_summary_csv).exists() else None
    wref_img = pd.read_csv(weight_refined_per_image_csv) if Path(weight_refined_per_image_csv).exists() else None

    # threshold method used in uncertainty file
    default_thresh_method = str(unc_img["thresh"].iloc[0])

    best_w = None
    if wref_sum is not None and len(wref_sum):
        r = wref_sum.iloc[0]
        best_w = (float(r["wR"]), float(r["wG"]), float(r["wB"]))


    # A) TOP (channel, thresh) BY DICE (selected channels only)
    allowed = set(filter_channels)
    topk = seg_sum[seg_sum["channel"].isin(allowed)].sort_values("dice", ascending=False).head(15).copy()

    plt.figure(figsize=(10, 5))
    labels = [f"{r.channel}\n{r.thresh}" for r in topk.itertuples()]
    plt.plot(range(len(topk)), topk["dice"].values, marker="o")
    plt.xticks(range(len(topk)), labels, rotation=60, ha="right")
    plt.ylabel("Mean Dice")
    plt.title("Top 15 (channel, threshold) by Dice (selected channels)")
    plt.tight_layout()
    plt.savefig(fig_dir / "A_top15_channel_thresh_dice.png", dpi=200)
    plt.show()


    # B) BASELINES vs BEST CUSTOM (Dice distribution)
    if wref_img is not None and wref_sum is not None and len(wref_sum):
        base = wref_img[wref_img["phase"] == "baseline"].copy()
        bw = wref_sum.iloc[0][["wR", "wG", "wB"]].to_dict()
        custom = wref_img[
            (wref_img["phase"] == "refine") &
            (np.isclose(wref_img["wR"], bw["wR"])) &
            (np.isclose(wref_img["wG"], bw["wG"])) &
            (np.isclose(wref_img["wB"], bw["wB"]))
        ].copy()

        keep = base[base["channel"].isin(["RGB_G", "YUV_Y"])]
        keep_custom = custom.copy()
        keep_custom["channel"] = "W_RGB_BEST"

        dice_df = pd.concat(
            [keep[["image", "channel", "dice"]], keep_custom[["image", "channel", "dice"]]],
            ignore_index=True
        )
    else:
        dice_df = unc_img[["image", "channel", "dice"]].copy()

    channels_order = [c for c in filter_channels if c in set(dice_df["channel"])]
    data = [dice_df[dice_df["channel"] == c]["dice"].values for c in channels_order]

    plt.figure(figsize=(8, 5))
    plt.boxplot(data, labels=channels_order, showfliers=False)
    plt.ylabel("Dice (per-image)", fontsize=18)
    plt.title("Dice distribution: RGB_G vs YUV_Y vs Best custom", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(fig_dir / "B_dice_boxplot_selected.png", dpi=200)
    plt.show()


    # C) UNCERTAINTY vs DICE (scatter, per-image)
    plt.figure(figsize=(7, 5))
    styles = {
        "RGB_G": dict(marker="o", alpha=0.8),
        "YUV_Y": dict(marker="^", alpha=0.8),
        "W_RGB_BEST": dict(marker="s", alpha=0.9),
    }
    for ch in channels_order:
        sub = unc_img[unc_img["channel"] == ch]
        plt.scatter(sub["mean_uncertainty_vessel"], sub["dice"], label=ch, **styles.get(ch, dict(alpha=0.8)))
    plt.xlabel("Mean uncertainty on predicted vessels")
    plt.ylabel("Dice")
    plt.title("Dice vs Uncertainty (per-image)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "C_dice_vs_uncertainty_scatter.png", dpi=200)
    plt.show()


    # D) EXAMPLE OVERLAYS (Original + GT + pred + uncertainty)
    paired = find_pairs(paths_cfg.images_dir, paths_cfg.masks_dir, paths_cfg.masks2_dir)

    if len(paired) == 0:
        print("[WARN] No pairs found; skipping overlays.")
    else:
        for i in range(min(n_examples, len(paired))):
            img_path, m1_path, _ = paired[i]
            bgr = read_bgr(img_path)
            gt = read_mask(m1_path)

            chG = ch_rgb_G(bgr)
            chY = ch_yuv_Y(bgr)
            chW = ch_weighted_rgb(bgr, *best_w) if best_w is not None else None

            predG = segment_from_channel(
                chG, default_thresh_method, preproc_cfg.default_preproc,
                preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
            )
            predY = segment_from_channel(
                chY, default_thresh_method, preproc_cfg.default_preproc,
                preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
            )
            predW = None
            if chW is not None:
                predW = segment_from_channel(
                    chW, default_thresh_method, preproc_cfg.default_preproc,
                    preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                    thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
                )

            uG = _load_unc_png(uncertainty_outputs_dir, img_path.stem, "RGB_G", "Ucomb")
            uY = _load_unc_png(uncertainty_outputs_dir, img_path.stem, "YUV_Y", "Ucomb")
            uW = _load_unc_png(uncertainty_outputs_dir, img_path.stem, "W_RGB_BEST", "Ucomb") if predW is not None else None

            if predW is not None:
                fig = plt.figure(figsize=(16, 8))
                fig.suptitle(f"Example {img_path.stem} (thresh={default_thresh_method})")

                ax1 = fig.add_subplot(2, 4, 1)
                ax1.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                ax1.set_title("Original"); ax1.axis("off")

                ax2 = fig.add_subplot(2, 4, 2)
                ax2.imshow(gt, cmap="gray")
                ax2.set_title("GT (manual1)"); ax2.axis("off")

                ax3 = fig.add_subplot(2, 4, 3)
                ax3.imshow(predG, cmap="gray")
                ax3.set_title("Pred: RGB_G"); ax3.axis("off")

                ax4 = fig.add_subplot(2, 4, 4)
                ax4.imshow(uG if uG is not None else np.zeros_like(gt, dtype=np.float32), cmap="gray")
                ax4.set_title("Ucomb: RGB_G"); ax4.axis("off")

                ax5 = fig.add_subplot(2, 4, 5)
                ax5.imshow(predY, cmap="gray")
                ax5.set_title("Pred: YUV_Y"); ax5.axis("off")

                ax6 = fig.add_subplot(2, 4, 6)
                ax6.imshow(uY if uY is not None else np.zeros_like(gt, dtype=np.float32), cmap="gray")
                ax6.set_title("Ucomb: YUV_Y"); ax6.axis("off")

                ax7 = fig.add_subplot(2, 4, 7)
                ax7.imshow(predW, cmap="gray")
                ax7.set_title("Pred: W_RGB_BEST"); ax7.axis("off")

                ax8 = fig.add_subplot(2, 4, 8)
                ax8.imshow(uW if uW is not None else np.zeros_like(gt, dtype=np.float32), cmap="gray")
                ax8.set_title("Ucomb: W_RGB_BEST"); ax8.axis("off")

                plt.tight_layout()
                plt.savefig(fig_dir / f"D_example_{img_path.stem}_G_Y_W.png", dpi=200)
                plt.show()
            else:
                fig = plt.figure(figsize=(12, 8))
                fig.suptitle(f"Example {img_path.stem} (thresh={default_thresh_method})")

                ax1 = fig.add_subplot(2, 3, 1)
                ax1.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                ax1.set_title("Original"); ax1.axis("off")

                ax2 = fig.add_subplot(2, 3, 2)
                ax2.imshow(gt, cmap="gray")
                ax2.set_title("GT (manual1)"); ax2.axis("off")

                ax3 = fig.add_subplot(2, 3, 3)
                ax3.imshow(predG, cmap="gray")
                ax3.set_title("Pred: RGB_G"); ax3.axis("off")

                ax4 = fig.add_subplot(2, 3, 4)
                ax4.imshow(uG if uG is not None else np.zeros_like(gt, dtype=np.float32), cmap="gray")
                ax4.set_title("Ucomb: RGB_G"); ax4.axis("off")

                ax5 = fig.add_subplot(2, 3, 5)
                ax5.imshow(predY, cmap="gray")
                ax5.set_title("Pred: YUV_Y"); ax5.axis("off")

                ax6 = fig.add_subplot(2, 3, 6)
                ax6.imshow(uY if uY is not None else np.zeros_like(gt, dtype=np.float32), cmap="gray")
                ax6.set_title("Ucomb: YUV_Y"); ax6.axis("off")

                plt.tight_layout()
                plt.savefig(fig_dir / f"D_example_{img_path.stem}_G_Y.png", dpi=200)
                plt.show()

    # E) UNCERTAINTY-AWARE FILTERING: coverage vs Dice
    def dice_on_mask_region(pred: np.ndarray, gt: np.ndarray, region_mask: np.ndarray) -> float:
        pred_r = pred & region_mask
        gt_r = gt & region_mask
        tp = np.logical_and(pred_r, gt_r).sum()
        fp = np.logical_and(pred_r, ~gt_r).sum()
        fn = np.logical_and(~pred_r, gt_r).sum()
        return float((2 * tp) / (2 * tp + fp + fn + 1e-9))

    def load_U_for(image_stem: str, channel_name: str):
        p = uncertainty_outputs_dir / f"{image_stem}_{channel_name}_Ucomb.png"
        if not p.exists():
            return None
        u = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if u is None:
            return None
        return u.astype(np.float32) / 255.0

    file_ch_map = {
        "RGB_G": "RGB_G",
        "YUV_Y": "YUV_Y",
        "W_RGB_BEST": "W_RGB_BEST",
    }

    if "W_RGB_BEST" in filter_channels and best_w is None:
        print("[WARN] Best weights not found; W_RGB_BEST filtering curve will be skipped.")

    curves = []
    for ch_name in filter_channels:
        # quick check: uncertainty map exists
        test_u = load_U_for(paired[0][0].stem, file_ch_map.get(ch_name, ch_name)) if len(paired) else None
        if test_u is None:
            print(f"[WARN] Missing uncertainty maps for {ch_name}. Skipping.")
            continue
        if ch_name == "W_RGB_BEST" and best_w is None:
            continue

        dice_means, cover_means = [], []

        for tau in taus:
            d_list, c_list = [], []
            for (img_path, m1_path, _) in paired:
                bgr = read_bgr(img_path)
                gt = read_mask(m1_path)

                if ch_name == "RGB_G":
                    ch = ch_rgb_G(bgr)
                elif ch_name == "YUV_Y":
                    ch = ch_yuv_Y(bgr)
                elif ch_name == "W_RGB_BEST":
                    ch = ch_weighted_rgb(bgr, *best_w)
                else:
                    continue

                pred = segment_from_channel(
                    ch, default_thresh_method, preproc_cfg.default_preproc,
                    preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                    thr_cfg.global_t, post_cfg.remove_small, post_cfg.min_obj_size
                )

                u = load_U_for(img_path.stem, file_ch_map.get(ch_name, ch_name))
                if u is None:
                    continue

                region = (u <= tau)
                coverage = float(region.mean())
                d = dice_on_mask_region(pred, gt, region)

                d_list.append(d)
                c_list.append(coverage)

            dice_means.append(np.mean(d_list) if len(d_list) else np.nan)
            cover_means.append(np.mean(c_list) if len(c_list) else np.nan)

        for tau, d, c in zip(taus, dice_means, cover_means):
            curves.append({"channel": ch_name, "tau": float(tau), "dice_filtered": float(d), "coverage": float(c)})

    curve_df = pd.DataFrame(curves)
    curve_df.to_csv(out_filter_curve_csv, index=False)

    plt.figure(figsize=(7, 5))
    for ch_name in curve_df["channel"].unique():
        sub = curve_df[curve_df["channel"] == ch_name].sort_values("coverage")
        plt.plot(sub["coverage"], sub["dice_filtered"], linestyle="-", label=ch_name)
    plt.xlabel("Coverage (fraction of pixels kept)", fontsize=14)
    plt.ylabel("Dice on kept region", fontsize=14)
    plt.title("Uncertainty-aware filtering trade-off", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "E_uncertainty_filtering_tradeoff.png", dpi=200)
    plt.show()

    return {
        "default_thresh_method": default_thresh_method,
        "best_w": best_w,
        "topk": topk,
        "curve_df": curve_df,
        "fig_dir": fig_dir,
    }
