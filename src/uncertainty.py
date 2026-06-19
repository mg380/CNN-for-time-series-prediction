"""Uncertainty quantification via Monte-Carlo Dropout.

Dropout is normally disabled at inference. By keeping it *active* (training=True)
and running many stochastic forward passes, the spread of predictions
approximates the model's predictive uncertainty (Gal & Ghahramani, 2016).
"""
import numpy as np
import tensorflow as tf


def mc_dropout_predict(model, X, n_passes=50):
    """Return (mean, std) over ``n_passes`` dropout-active forward passes.

    Each has shape (n_samples, horizon).
    """
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    preds = np.stack([model(X, training=True).numpy() for _ in range(n_passes)])
    return preds.mean(axis=0), preds.std(axis=0)


def intervals(mean, std, z=1.96):
    """Symmetric Gaussian prediction interval; z=1.96 -> nominal 95%."""
    return mean - z * std, mean + z * std
