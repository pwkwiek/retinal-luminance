import numpy as np

def metrics_binary(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    dice      = (2 * tp) / (2 * tp + fp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return {
        "dice": float(dice), "iou": float(iou), "precision": float(precision),
        "recall": float(recall), "accuracy": float(accuracy),
        "specificity": float(specificity), "f1": float(f1),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }