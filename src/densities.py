import numpy as np
import torch
from scipy.integrate import quad


def build_notched_gaussian_density(
    x_min=-3.0,
    x_max=3.0,
    scale=1.5,
    valley_width=0.15,
    eps=1e-3,
):
    def unnormalized_numpy(x):
        envelope = np.exp(-0.5 * (x / scale) ** 2)
        notch = eps + 1.0 - np.exp(-0.5 * (x / valley_width) ** 2)
        return envelope * notch

    normalizer = quad(unnormalized_numpy, x_min, x_max)[0]

    def p_torch(x):
        envelope = torch.exp(-0.5 * (x / scale) ** 2)
        notch = eps + 1.0 - torch.exp(-0.5 * (x / valley_width) ** 2)
        return envelope * notch / normalizer

    def p_numpy(x):
        return unnormalized_numpy(x) / normalizer

    return p_torch, p_numpy


def sample_notched_gaussian_density(
    n=1000,
    x_min=-3.0,
    x_max=3.0,
    scale=1.5,
    valley_width=0.15,
    eps=1e-3,
    device=None,
    dtype=torch.float64,
    generator=None,
):
    device = device or torch.device("cpu")
    accepted = []
    total = 0

    while total < n:
        candidates = torch.rand(
            2 * n,
            dtype=dtype,
            device=device,
            generator=generator,
        ) * (x_max - x_min) + x_min
        envelope = torch.exp(-0.5 * (candidates / scale) ** 2)
        notch = eps + 1.0 - torch.exp(-0.5 * (candidates / valley_width) ** 2)
        accept_prob = envelope * notch / (1.0 + eps)
        uniforms = torch.rand(
            candidates.shape,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        batch = candidates[uniforms < accept_prob]
        accepted.append(batch)
        total += len(batch)

    return torch.cat(accepted)[:n]
