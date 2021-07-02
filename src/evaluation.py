"""
Implements evaluation metrics
"""
import numpy as np
import torch


def tonumpy(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    return x


def rmse(y_true, y_pred):
    """Root mean squared error"""
    y_true, y_pred = map(tonumpy, (y_true, y_pred))
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def picp(Lpi, Upi, y_true):
    """Prediction interval coverage probability"""
    Lpi, Upi, y_true = map(tonumpy, (Lpi, Upi, y_true))
    within = (Lpi < y_true).astype(int) + (Upi > y_true).astype(int)
    within = within == 2  # higher than lower bound, and smaller than upper bound
    return np.mean(within)


def mpiw(Lpi, Upi):
    """Mean prediction interval width"""
    Lpi, Upi = map(tonumpy, (Lpi, Upi))
    return np.mean(Upi - Lpi)
