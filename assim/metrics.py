import functools

import numpy as np


def init_metrics(mask=None, sparse_gt=False):
    """Initialize generic scalar metrics.

    This intentionally follows the compact wrapper style from
    paper_vae_borey/3d_var/src/logger/metrics.py, but removes ice-specific
    threshold metrics.
    """
    metrics = []

    def extract_values(func):
        @functools.wraps(func)
        def fix_values(pred, gt):
            pred = np.asarray(pred, dtype=np.float64).copy()
            gt = np.asarray(gt, dtype=np.float64).copy()

            if mask is None:
                eval_mask = np.ones_like(pred, dtype=bool)
            else:
                eval_mask = ~np.asarray(mask, dtype=bool)
                eval_mask = np.broadcast_to(eval_mask, pred.shape)

            nans_pred = np.isnan(pred)
            nans_gt = np.isnan(gt)

            if sparse_gt:
                correct_mask = eval_mask & (~nans_gt) & (~nans_pred)
            else:
                pred[nans_pred & eval_mask] = 0.0
                gt[nans_gt & eval_mask] = 0.0
                correct_mask = eval_mask & (~np.isnan(pred)) & (~np.isnan(gt))

            if not np.any(correct_mask):
                return float("nan")
            return float(func(pred[correct_mask], gt[correct_mask]))

        return fix_values

    @extract_values
    def mae(pred, gt):
        return np.mean(np.abs(gt - pred))

    metrics.append(mae)

    @extract_values
    def rmse(pred, gt):
        return np.sqrt(np.mean((gt - pred) ** 2))

    metrics.append(rmse)

    @extract_values
    def mse(pred, gt):
        return np.mean((gt - pred) ** 2)

    metrics.append(mse)
    return metrics
