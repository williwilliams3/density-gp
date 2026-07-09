import math

import torch


def neg_mll_loss(y, kernel, noise_var=1e-3, jitter=1e-5):
    n = len(y)
    eye = torch.eye(n, dtype=kernel.dtype, device=kernel.device)
    kernel_noisy = kernel + (noise_var + jitter) * eye
    chol = torch.linalg.cholesky(kernel_noisy)
    alpha = torch.cholesky_solve(y.unsqueeze(1), chol).squeeze()
    return (
        0.5 * torch.dot(y, alpha)
        + torch.log(torch.diag(chol)).sum()
        + 0.5 * n * math.log(2 * math.pi)
    )


def gp_posterior(kernel_full, idx_train, y_train, noise_var=1e-3, jitter=1e-5):
    n_train = len(y_train)
    eye_train = torch.eye(n_train, dtype=kernel_full.dtype, device=kernel_full.device)
    kernel_train = kernel_full[idx_train][:, idx_train] + (noise_var + jitter) * eye_train
    chol = torch.linalg.cholesky(kernel_train)
    alpha = torch.cholesky_solve(y_train.unsqueeze(1), chol)

    kernel_cross = kernel_full[:, idx_train]
    mean = kernel_cross @ alpha

    v = torch.linalg.solve_triangular(chol, kernel_cross.T, upper=False)
    eye_full = torch.eye(len(kernel_full), dtype=kernel_full.dtype, device=kernel_full.device)
    cov = kernel_full - v.T @ v + jitter * eye_full
    return mean.squeeze(), cov.diag()


def gp_posterior_subset(kernel_full, idx_train, y_train, idx_test, noise_var=1e-3, jitter=1e-5):
    n_train = len(y_train)
    eye_train = torch.eye(n_train, dtype=kernel_full.dtype, device=kernel_full.device)
    kernel_train = kernel_full[idx_train][:, idx_train] + (noise_var + jitter) * eye_train
    chol = torch.linalg.cholesky(kernel_train)
    alpha = torch.cholesky_solve(y_train.unsqueeze(1), chol)

    kernel_cross = kernel_full[idx_test][:, idx_train]
    mean = (kernel_cross @ alpha).squeeze()
    v = torch.linalg.solve_triangular(chol, kernel_cross.T, upper=False)
    cov = kernel_full[idx_test][:, idx_test] - v.T @ v
    cov = 0.5 * (cov + cov.T)
    return mean, cov


def gaussian_nll(y, mean, cov, noise_var=1e-3, jitter=1e-5):
    n = len(y)
    eye = torch.eye(n, dtype=cov.dtype, device=cov.device)
    predictive_cov = cov + (noise_var + jitter) * eye
    chol = torch.linalg.cholesky(predictive_cov)
    diff = y - mean
    alpha = torch.cholesky_solve(diff.unsqueeze(1), chol).squeeze()
    return (
        0.5 * torch.dot(diff, alpha)
        + torch.log(torch.diag(chol)).sum()
        + 0.5 * n * math.log(2.0 * math.pi)
    )


def marginal_nll(y, mean, cov, noise_var=1e-3):
    predictive_var = torch.clamp(torch.diag(cov) + noise_var, min=1e-8)
    return 0.5 * (
        torch.log(2.0 * math.pi * predictive_var)
        + (y - mean).pow(2) / predictive_var
    ).mean()
