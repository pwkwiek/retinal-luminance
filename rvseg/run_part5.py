# comparative statistical analysis of best channels

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from scipy.stats import pearsonr, spearmanr, ttest_rel, wilcoxon


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return np.nan, np.nan
    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return float(r), float(p)


def cohens_dz_paired(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(d) / sd)


def cliffs_delta_fast(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return np.nan

    x_sorted = np.sort(x)
    y_sorted = np.sort(y)

    # count(y < x) efficiently
    j = 0
    lt_count = 0
    for xi in x_sorted:
        while j < len(y_sorted) and y_sorted[j] < xi:
            j += 1
        lt_count += j

    # count(x < y) via swap
    j = 0
    lt_yx = 0
    for yi in y_sorted:
        while j < len(x_sorted) and x_sorted[j] < yi:
            j += 1
        lt_yx += j

    gt_count = lt_yx
    n = len(x) * len(y)
    return float((gt_count - lt_count) / n)


def paired_test_table(df, value_col, a, b):
    A = df[df["channel"] == a][["image", value_col]].rename(columns={value_col: "A"})
    B = df[df["channel"] == b][["image", value_col]].rename(columns={value_col: "B"})
    M = A.merge(B, on="image", how="inner").dropna()

    x = M["A"].values.astype(float)
    y = M["B"].values.astype(float)

    n = len(M)
    if n < 3:
        return {
            "comparison": f"{a} vs {b}",
            "metric": value_col,
            "n": int(n),
            "mean_A": float(np.nanmean(x)) if n else np.nan,
            "mean_B": float(np.nanmean(y)) if n else np.nan,
            "mean_diff(A-B)": float(np.nanmean(x - y)) if n else np.nan,
            "t_stat": np.nan, "t_p": np.nan,
            "wilcoxon_stat": np.nan, "wilcoxon_p": np.nan,
            "cohens_dz": np.nan,
            "cliffs_delta": np.nan,
        }

    t = ttest_rel(x, y, nan_policy="omit")
    t_stat, t_p = float(t.statistic), float(t.pvalue)

    diffs = x - y
    if np.all(np.isclose(diffs, 0.0)):
        w_stat, w_p = np.nan, np.nan
    else:
        try:
            w = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided", mode="auto")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except Exception:
            w_stat, w_p = np.nan, np.nan

    dz = cohens_dz_paired(x, y)
    cd = cliffs_delta_fast(x, y)

    return {
        "comparison": f"{a} vs {b}",
        "metric": value_col,
        "n": int(n),
        "mean_A": float(np.mean(x)),
        "mean_B": float(np.mean(y)),
        "mean_diff(A-B)": float(np.mean(x - y)),
        "t_stat": t_stat, "t_p": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_p": w_p,
        "cohens_dz": dz,
        "cliffs_delta": cd,
    }


def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n, dtype=float)

    for k, idx in enumerate(order):
        adj[idx] = min(1.0, (n - k) * p[idx])

    # enforce monotonicity
    adj_sorted = adj[order]
    for i in range(1, n):
        adj_sorted[i] = max(adj_sorted[i], adj_sorted[i - 1])
    adj[order] = adj_sorted
    return adj


def run_part5(
    unc_per_image_csv: str = "uncertainty_selected_channels_per_image.csv",
    out_corr_csv: str = "stats_correlations.csv",
    out_pairwise_dice_csv: str = "stats_pairwise_dice.csv",
    out_pairwise_unc_csv: str = "stats_pairwise_uncertainty.csv",
    channels=("RGB_G", "YUV_Y", "W_RGB_BEST", "GRAY"),
    pairs=(("RGB_G", "YUV_Y"), ("RGB_G", "W_RGB_BEST"), ("YUV_Y", "W_RGB_BEST")),
):
    unc = pd.read_csv(unc_per_image_csv)

    available = set(unc["channel"].unique())
    use_channels = [c for c in channels if c in available]

    # 1) correlations per channel: Dice vs mean_uncertainty_vessel
    corr_rows = []
    for ch in use_channels:
        sub = unc[unc["channel"] == ch].sort_values("image")
        r_p, p_p = safe_corr(sub["dice"], sub["mean_uncertainty_vessel"], method="pearson")
        r_s, p_s = safe_corr(sub["dice"], sub["mean_uncertainty_vessel"], method="spearman")
        corr_rows.append({
            "channel": ch,
            "n": int(len(sub)),
            "pearson_r(dice, Uvessel)": r_p,
            "pearson_p": p_p,
            "spearman_rho(dice, Uvessel)": r_s,
            "spearman_p": p_s,
        })

    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r(dice, Uvessel)")
    corr_df.to_csv(out_corr_csv, index=False)

    # 2) paired comparisons: Dice + mean_uncertainty_vessel
    dice_tests = []
    unc_tests = []

    for a, b in pairs:
        if a not in available or b not in available:
            continue
        dice_tests.append(paired_test_table(unc, "dice", a, b))
        unc_tests.append(paired_test_table(unc, "mean_uncertainty_vessel", a, b))

    dice_df = pd.DataFrame(dice_tests)
    unc_df = pd.DataFrame(unc_tests)

    # Holm correction inside each table
    if len(dice_df):
        dice_df["t_p_holm"] = holm_bonferroni(dice_df["t_p"].values)
        dice_df["wilcoxon_p_holm"] = holm_bonferroni(dice_df["wilcoxon_p"].fillna(1.0).values)
    if len(unc_df):
        unc_df["t_p_holm"] = holm_bonferroni(unc_df["t_p"].values)
        unc_df["wilcoxon_p_holm"] = holm_bonferroni(unc_df["wilcoxon_p"].fillna(1.0).values)

    dice_df.to_csv(out_pairwise_dice_csv, index=False)
    unc_df.to_csv(out_pairwise_unc_csv, index=False)

    return corr_df, dice_df, unc_df