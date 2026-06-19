"""Forecast accuracy and uncertainty-calibration metrics.

All functions take arrays shaped (n_samples, horizon) unless noted.
"""
import numpy as np


def rmse(y_true, y_pred):
    """Root mean squared error over all samples and horizon steps."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true, y_pred):
    """Symmetric mean absolute percentage error (%) -- scale-free, robust near 0."""
    denom = np.abs(y_true) + np.abs(y_pred) + 1e-8
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def per_horizon_rmse(y_true, y_pred):
    """RMSE at each forecast step -> array of shape (horizon,).

    Shows how error grows the further ahead the model predicts.
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))


def picp(y_true, lower, upper):
    """Prediction Interval Coverage Probability: fraction of truths inside [lo, hi].

    A well-calibrated 95% interval should yield ~0.95.
    """
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def mpiw(lower, upper):
    """Mean Prediction Interval Width -- sharpness (smaller is better, given coverage)."""
    return float(np.mean(upper - lower))
