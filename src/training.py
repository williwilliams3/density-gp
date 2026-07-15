from dataclasses import dataclass
from itertools import product

import gpytorch
import torch
import torch.nn.functional as F

from .gp import (
    ExactGPModel,
    build_euclidean_kernel,
    canonicalize_inputs,
    fixed_noise_likelihood,
)
from .kernels import build_grid_density_kernel


@dataclass(frozen=True)
class GPFitResult:
    model: gpytorch.models.ExactGP
    total_negative_mll: float


def fit_exact_gp_multistart(model_factory, starts, *, steps=300, lr=0.1):
    """Fit models built by ``model_factory`` and retain the best final state.

    ``model_factory(start)`` owns kernel-specific initialization. This keeps the
    optimization loop shared without coupling it to a particular kernel's
    constrained parameter names.
    """
    best = None
    starts = tuple(starts)
    if not starts:
        raise ValueError("At least one initialization is required")

    for start in starts:
        model = model_factory(start)
        model.train()
        model.likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        for _ in range(steps):
            optimizer.zero_grad()
            output = model(*model.train_inputs)
            loss = -mll(output, model.train_targets)
            loss.backward()
            optimizer.step()

        model.train()
        model.likelihood.train()
        with torch.no_grad():
            output = model(*model.train_inputs)
            total_negative_mll = float(
                (-mll(output, model.train_targets) * len(model.train_targets)).cpu()
            )

        if best is None or total_negative_mll < best.total_negative_mll:
            best = GPFitResult(model=model, total_negative_mll=total_negative_mll)

    return best


def fit_euclidean_gp(
    y_train,
    x_train,
    noise_var,
    *,
    kind="rbf",
    lengthscale_raw_inits=(0.0,),
    sigma_raw_inits=(0.0,),
    steps=300,
    lr=0.1,
    ard=False,
    nu=1.5,
):
    """Fit a Euclidean exact GP for inputs of any spatial dimension."""
    x_train = canonicalize_inputs(x_train)
    starts = tuple(product(lengthscale_raw_inits, sigma_raw_inits))

    def model_factory(start):
        lengthscale_raw, sigma_raw = start
        covariance_module = build_euclidean_kernel(
            kind,
            num_dims=x_train.shape[-1],
            ard=ard,
            nu=nu,
        ).to(dtype=y_train.dtype, device=y_train.device)
        covariance_module.base_kernel.lengthscale = F.softplus(
            torch.as_tensor(lengthscale_raw, dtype=y_train.dtype, device=y_train.device)
        )
        covariance_module.outputscale = F.softplus(
            torch.as_tensor(sigma_raw, dtype=y_train.dtype, device=y_train.device)
        ).square()
        likelihood = fixed_noise_likelihood(y_train, noise_var)
        return ExactGPModel(x_train, y_train, likelihood, covariance_module)

    return fit_exact_gp_multistart(model_factory, starts, steps=steps, lr=lr)


def fit_density_gp(
    y_train,
    train_grid_indices,
    noise_var,
    eigenvalues,
    eigenvectors,
    *,
    kind,
    amplitude=None,
    spectral_raw_inits=(0.0,),
    sigma_raw_inits=(0.0,),
    steps=300,
    lr=0.1,
    alpha=1.5,
):
    """Fit a low-rank density GP without constructing the full grid kernel."""
    train_grid_indices = canonicalize_inputs(
        train_grid_indices.to(dtype=y_train.dtype, device=y_train.device)
    )
    starts = tuple(product(spectral_raw_inits, sigma_raw_inits))

    def model_factory(start):
        spectral_raw, sigma_raw = start
        covariance_module = build_grid_density_kernel(
            kind,
            eigenvalues,
            eigenvectors,
            amplitude=amplitude,
            alpha=alpha,
        ).to(dtype=y_train.dtype, device=y_train.device)
        spectral_value = F.softplus(
            torch.as_tensor(spectral_raw, dtype=y_train.dtype, device=y_train.device)
        )
        if kind.lower() == "heat":
            covariance_module.base_kernel.tau = spectral_value
        elif kind.lower() == "matern":
            covariance_module.base_kernel.kappa = spectral_value
        else:
            raise ValueError(f"Unsupported density spectral kernel: {kind!r}")
        covariance_module.outputscale = F.softplus(
            torch.as_tensor(sigma_raw, dtype=y_train.dtype, device=y_train.device)
        ).square()
        likelihood = fixed_noise_likelihood(y_train, noise_var)
        return ExactGPModel(train_grid_indices, y_train, likelihood, covariance_module)

    return fit_exact_gp_multistart(model_factory, starts, steps=steps, lr=lr)
