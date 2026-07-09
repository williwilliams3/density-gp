import math

import numpy as np
import torch


def rmse(pred, truth, mask):
    return torch.sqrt(torch.mean((pred[mask] - truth[mask]).pow(2)))


def nlpd(y_true, mean, var, mask):
    pred_var = torch.clamp(var, min=1e-8)
    return 0.5 * (
        torch.log(2.0 * math.pi * pred_var[mask])
        + (y_true[mask] - mean[mask]).pow(2) / pred_var[mask]
    ).mean()


def coverage(y_true, mean, var, mask):
    std = torch.sqrt(torch.clamp(var, min=0.0))
    inside = (y_true >= mean - 2.0 * std) & (y_true <= mean + 2.0 * std)
    return inside[mask].to(y_true.dtype).mean()


def evaluate_predictions(y_true, mean, var, masks):
    return {
        name: {
            "rmse": float(rmse(mean, y_true, mask).cpu()),
            "nlpd": float(nlpd(y_true, mean, var, mask).cpu()),
            "coverage": float(coverage(y_true, mean, var, mask).cpu()),
        }
        for name, mask in masks.items()
    }


def average_pair_correlation(corr, pairs):
    return torch.stack([corr[i, j] for i, j in pairs]).mean()


def average_pair_distance(points, pairs):
    return float(np.mean([np.linalg.norm(points[i] - points[j]) for i, j in pairs]))


def average_pair_label_difference(y_true, pairs):
    return float(torch.stack([torch.abs(y_true[i] - y_true[j]) for i, j in pairs]).mean().cpu())
