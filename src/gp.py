import math
from dataclasses import dataclass

import gpytorch
import torch


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


def canonicalize_inputs(x):
    """Represent unbatched GP inputs as ``(n_points, n_dimensions)``."""
    return x.unsqueeze(-1) if x.ndim == 1 else x


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact zero-mean GP whose covariance module is supplied by the caller."""

    def __init__(self, train_x, train_y, likelihood, covariance_module, mean_module=None):
        train_x = canonicalize_inputs(train_x)
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or gpytorch.means.ZeroMean()
        self.covar_module = covariance_module

    def forward(self, x):
        x = canonicalize_inputs(x)
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


def build_euclidean_kernel(kind="rbf", *, num_dims=None, ard=False, nu=1.5):
    """Build a dimension-independent Euclidean covariance module."""
    ard_num_dims = num_dims if ard else None
    kind = kind.lower()
    if kind == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
    elif kind == "matern":
        base_kernel = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=ard_num_dims)
    else:
        raise ValueError(f"Unsupported Euclidean kernel: {kind!r}")
    return gpytorch.kernels.ScaleKernel(base_kernel)


def fixed_noise_likelihood(targets, noise_var):
    """Create a fixed-noise likelihood with the targets' dtype and device."""
    noise = torch.as_tensor(noise_var, dtype=targets.dtype, device=targets.device)
    noise = noise.expand_as(targets).clone()
    return gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise,
        learn_additional_noise=False,
    )


@dataclass(frozen=True)
class GPPrediction:
    latent_mean: torch.Tensor
    latent_variance: torch.Tensor
    observed_mean: torch.Tensor
    observed_variance: torch.Tensor
    latent_covariance: torch.Tensor | None = None
    observed_covariance: torch.Tensor | None = None


def predict_exact_gp(model, x, noise_var, *, full_covariance=False):
    """Return clearly separated latent and noisy predictive quantities."""
    x = canonicalize_inputs(x)
    model.eval()
    model.likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var(False):
        latent = model(x)
        test_noise = torch.as_tensor(
            noise_var,
            dtype=latent.mean.dtype,
            device=latent.mean.device,
        )
        test_noise = test_noise.expand_as(latent.mean).clone()
        observed = model.likelihood(latent, noise=test_noise)

        latent_covariance = latent.covariance_matrix if full_covariance else None
        observed_covariance = observed.covariance_matrix if full_covariance else None
        return GPPrediction(
            latent_mean=latent.mean,
            latent_variance=latent.variance,
            observed_mean=observed.mean,
            observed_variance=observed.variance,
            latent_covariance=latent_covariance,
            observed_covariance=observed_covariance,
        )


def dense_prior_covariance(model, x):
    """Materialize a prior covariance only for diagnostics that require it."""
    x = canonicalize_inputs(x)
    with torch.no_grad():
        return model.covar_module(x).to_dense()
