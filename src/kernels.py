import torch
import torch.nn.functional as F


def density_amplitude(p_vals, beta=0.5, eps=1e-3, max_amp=3.0):
    q = p_vals / p_vals.max()
    amp = (q + eps).pow(-0.5 * beta)
    return torch.clamp(amp, max=max_amp)


def heat_kernel_weights(tau_raw, eigenvalues):
    tau = F.softplus(tau_raw)
    return torch.exp(-tau * eigenvalues)


def density_heat_kernel(tau_raw, sigma_f_raw, eigenvalues, eigenvectors, amp=None):
    sigma_f = F.softplus(sigma_f_raw)
    weights = heat_kernel_weights(tau_raw, eigenvalues)

    if amp is not None:
        eigenvectors = amp.unsqueeze(1) * eigenvectors

    return (sigma_f**2) * (eigenvectors * weights.unsqueeze(0)) @ eigenvectors.T


def density_matern_kernel(kappa_raw, sigma_f_raw, eigenvalues, eigenvectors, amp=None, alpha=1.5):
    kappa = F.softplus(kappa_raw)
    sigma_f = F.softplus(sigma_f_raw)
    weights = ((kappa**2) / (kappa**2 + eigenvalues)).pow(alpha)

    if amp is not None:
        eigenvectors = amp.unsqueeze(1) * eigenvectors

    base_kernel = (eigenvectors * weights.unsqueeze(0)) @ eigenvectors.T
    base_kernel = base_kernel / torch.diag(base_kernel).mean().clamp_min(1e-12)
    return (sigma_f**2) * base_kernel


def rbf_kernel(lengthscale_raw, sigma_f_raw, x_grid):
    lengthscale = F.softplus(lengthscale_raw)
    sigma_f = F.softplus(sigma_f_raw)
    diff = x_grid.unsqueeze(1) - x_grid.unsqueeze(0)
    return (sigma_f**2) * torch.exp(-0.5 * (diff / lengthscale) ** 2)


def rbf_kernel_2d(lengthscale_raw, sigma_f_raw, points):
    lengthscale = F.softplus(lengthscale_raw)
    sigma_f = F.softplus(sigma_f_raw)
    dist2 = torch.cdist(points, points).pow(2)
    return (sigma_f**2) * torch.exp(-0.5 * dist2 / lengthscale**2)


def kernel_to_correlation(kernel, eps=1e-12):
    std = torch.sqrt(torch.clamp(torch.diag(kernel), min=eps))
    return kernel / (std.unsqueeze(1) * std.unsqueeze(0))
