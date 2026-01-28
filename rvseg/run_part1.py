# run segmentation and analysis of best channel from 6 color space models

import pandas as pd
from tqdm import tqdm

from .io_pairs import read_bgr, read_mask, find_pairs
from .channels import CHANNELS
from .preproc import preprocess
from .thresholding import threshold_image
from skimage.morphology import remove_small_objects

def cleanup(mask, remove_small: bool, min_obj_size: int):
    mask = mask.astype(bool)
    if remove_small and min_obj_size > 0:
        mask = remove_small_objects(mask, min_obj_size)
    return mask

def segment_from_channel(gray01, thresh_method: str, preproc_variant: str,
                         clahe_kernel, clahe_clip, tophat_radius,
                         global_t: float,
                         remove_small: bool, min_obj_size: int):
    x = preprocess(gray01, preproc_variant, clahe_kernel, clahe_clip, tophat_radius)
    m = threshold_image(x, thresh_method, global_t=global_t)
    return cleanup(m, remove_small, min_obj_size)

def run_part1(paths_cfg, preproc_cfg, post_cfg, thr_cfg,
              out_per_image="segmentation_all_channels_per_image_with_preproc.csv",
              out_summary_all="segmentation_all_channels_summary_with_preproc.csv",
              out_summary_default="segmentation_all_channels_summary.csv"):
    paired = find_pairs(paths_cfg.images_dir, paths_cfg.masks_dir, paths_cfg.masks2_dir)
    records = []

    for (img_path, m1_path, _) in tqdm(paired, desc="Part1: Channel eval (+preproc)"):
        bgr = read_bgr(img_path)
        gt  = read_mask(m1_path)

        for ch_name, ch_fun in CHANNELS.items():
            ch = ch_fun(bgr)

            for preproc in preproc_cfg.preproc_variants:
                for tm in thr_cfg.thresh_methods:
                    pred = segment_from_channel(
                        ch, tm, preproc,
                        preproc_cfg.clahe_kernel, preproc_cfg.clahe_clip, preproc_cfg.tophat_radius,
                        thr_cfg.global_t,
                        post_cfg.remove_small, post_cfg.min_obj_size
                    )
                    from .metrics import metrics_binary
                    met = metrics_binary(pred, gt)
                    met.update({
                        "image": img_path.stem,
                        "channel": ch_name,
                        "preproc": preproc,
                        "thresh": tm
                    })
                    records.append(met)

    df_eval = pd.DataFrame(records)
    df_eval.to_csv(out_per_image, index=False)

    summary_eval = (
        df_eval.groupby(["channel","thresh","preproc"])[["dice","iou","precision","recall","accuracy","specificity","f1"]]
              .mean().reset_index()
              .sort_values("dice", ascending=False)
    )
    summary_eval.to_csv(out_summary_all, index=False)

    df_default = df_eval[df_eval["preproc"] == preproc_cfg.default_preproc].copy()
    summary_default = (
        df_default.groupby(["channel","thresh"])[["dice","iou","precision","recall","accuracy","specificity","f1"]]
                  .mean().reset_index()
                  .sort_values("dice", ascending=False)
    )
    summary_default.to_csv(out_summary_default, index=False)

    return df_eval, summary_eval, summary_default
