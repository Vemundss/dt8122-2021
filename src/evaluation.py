"""
Implements evaluation metrics
"""
import numpy
import torch


def rmse(y_true, y_pred):
	"""Root mean squared error"""
	if isinstance(y_pred, torch.Tensor):
		y_pred = y_pred.numpy()

	return np.sqrt(np.mean((y_true - y_pred)**2))

def picp():
	"""Prediction interval coverage probability"""
	pass

def mpiw():
	"""Mean prediction interval width"""
	pass