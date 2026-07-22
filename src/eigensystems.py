"""Shared linear algebra and diagnostics for weighted eigensystems."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def normalize_probability_weights(
    weights: torch.Tensor,
    *,
    length: int | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    weights = torch.as_tensor(weights, dtype=dtype, device=device)
    if weights.ndim != 1 or (length is not None and len(weights) != length):
        expected = "(N,)" if length is None else f"({length},)"
        raise ValueError(f"probability_weights must have shape {expected}")
    if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("probability_weights must be finite and non-negative")
    total = weights.sum()
    if not bool(total > 0):
        raise ValueError("probability_weights must have positive sum")
    return weights / total


def weighted_center(
    values: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = normalize_probability_weights(
        weights, length=len(values), dtype=values.dtype, device=values.device
    )
    mean = torch.sum(weights.unsqueeze(1) * values, dim=0)
    return mean, values - mean


def weighted_mass(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = normalize_probability_weights(
        weights, length=len(values), dtype=values.dtype, device=values.device
    )
    mass = values.T @ (weights.unsqueeze(1) * values)
    return 0.5 * (mass + mass.T)


def normalize_eigenfunction_columns(
    functions: torch.Tensor,
    weights: torch.Tensor,
    *,
    constant_mode: bool = True,
) -> torch.Tensor:
    """Center and normalize columns without mixing distinct spectral modes."""

    weights = normalize_probability_weights(
        weights, length=len(functions), dtype=functions.dtype, device=functions.device
    )
    normalized = functions.clone()
    start = 0
    if constant_mode:
        normalized[:, 0] = 1.0
        start = 1
    if start < normalized.shape[1]:
        columns = normalized[:, start:]
        columns = columns - torch.sum(weights.unsqueeze(1) * columns, dim=0)
        norms = torch.sqrt(
            torch.sum(weights.unsqueeze(1) * columns.square(), dim=0)
        ).clamp_min(torch.finfo(functions.dtype).eps)
        normalized[:, start:] = columns / norms
    return normalized


@dataclass(frozen=True)
class GeneralizedEigenResult:
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    jitter: float
    condition_number: float
    residuals: torch.Tensor


def solve_generalized_eigenproblem(
    stiffness: torch.Tensor,
    mass: torch.Tensor,
    *,
    num_modes: int | None = None,
    relative_jitter: float = 1e-8,
    max_attempts: int = 8,
) -> GeneralizedEigenResult:
    """Solve ``S u = lambda M u`` using Cholesky whitening in Torch."""

    if stiffness.ndim != 2 or stiffness.shape[0] != stiffness.shape[1]:
        raise ValueError("stiffness must be square")
    if mass.shape != stiffness.shape:
        raise ValueError("mass and stiffness must have equal square shapes")
    if relative_jitter <= 0 or max_attempts <= 0:
        raise ValueError("relative_jitter and max_attempts must be positive")

    stiffness = 0.5 * (stiffness + stiffness.T)
    mass = 0.5 * (mass + mass.T)
    dimension = len(mass)
    if num_modes is None:
        num_modes = dimension
    if not 0 < num_modes <= dimension:
        raise ValueError("num_modes must lie between one and the matrix dimension")

    scale = torch.trace(mass).abs() / max(dimension, 1)
    scale = scale.clamp_min(torch.finfo(mass.dtype).eps)
    identity = torch.eye(dimension, dtype=mass.dtype, device=mass.device)
    jitter_tensor = relative_jitter * scale
    chol = None
    regularized_mass = None
    for _ in range(max_attempts):
        regularized_mass = mass + jitter_tensor * identity
        candidate, info = torch.linalg.cholesky_ex(regularized_mass)
        if int(info.max().detach().cpu()) == 0:
            chol = candidate
            break
        jitter_tensor = 10.0 * jitter_tensor
    if chol is None or regularized_mass is None:
        raise RuntimeError("mass matrix is not positive definite after adaptive jitter")

    left_whitened = torch.linalg.solve_triangular(
        chol, stiffness, upper=False
    )
    whitened = torch.linalg.solve_triangular(
        chol, left_whitened.T, upper=False
    ).T
    whitened = 0.5 * (whitened + whitened.T)
    eigenvalues, whitened_vectors = torch.linalg.eigh(whitened)
    eigenvalues = eigenvalues[:num_modes].clamp_min(0.0)
    whitened_vectors = whitened_vectors[:, :num_modes]
    eigenvectors = torch.linalg.solve_triangular(
        chol.T, whitened_vectors, upper=True
    )

    residual_matrix = (
        stiffness @ eigenvectors
        - (mass @ eigenvectors) * eigenvalues.unsqueeze(0)
    )
    residual_scale = (
        torch.linalg.matrix_norm(stiffness)
        + eigenvalues.abs()
        * torch.linalg.matrix_norm(mass)
    ).clamp_min(torch.finfo(stiffness.dtype).eps)
    residuals = torch.linalg.vector_norm(residual_matrix, dim=0) / residual_scale
    mass_spectrum = torch.linalg.eigvalsh(regularized_mass)
    condition_number = float(
        (mass_spectrum[-1] / mass_spectrum[0]).detach().cpu()
    )
    return GeneralizedEigenResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        jitter=float(jitter_tensor.detach().cpu()),
        condition_number=condition_number,
        residuals=residuals,
    )


@dataclass(frozen=True)
class EigensystemComparison:
    eigenvalue_relative_errors: torch.Tensor
    mode_correlations: torch.Tensor
    principal_cosines: torch.Tensor
    mass_orthogonality_error: float
    heat_kernel_relative_error: float
    projector_error: float

    @property
    def mean_eigenvalue_relative_error(self) -> float:
        return float(self.eigenvalue_relative_errors.mean().cpu())

    @property
    def mean_mode_correlation(self) -> float:
        return float(self.mode_correlations.mean().cpu())

    @property
    def minimum_principal_cosine(self) -> float:
        return float(self.principal_cosines.min().cpu())


def _weighted_orthonormalize(
    functions: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    gram = weighted_mass(functions, weights)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    tolerance = torch.finfo(gram.dtype).eps * len(gram) * eigenvalues[-1].abs()
    if bool((eigenvalues <= tolerance).any()):
        raise ValueError("eigenfunctions are linearly dependent under the weights")
    inverse_sqrt = eigenvectors @ torch.diag(eigenvalues.rsqrt()) @ eigenvectors.T
    return functions @ inverse_sqrt


def _weighted_heat_kernel_error(
    reference_functions: torch.Tensor,
    reference_eigenvalues: torch.Tensor,
    learned_functions: torch.Tensor,
    learned_eigenvalues: torch.Tensor,
    weights: torch.Tensor,
    heat_time: float,
) -> float:
    sqrt_weights = weights.sqrt().unsqueeze(1)
    reference_features = (
        sqrt_weights
        * reference_functions
        * torch.exp(-0.5 * heat_time * reference_eigenvalues).unsqueeze(0)
    )
    learned_features = (
        sqrt_weights
        * learned_functions
        * torch.exp(-0.5 * heat_time * learned_eigenvalues).unsqueeze(0)
    )
    reference_gram = reference_features.T @ reference_features
    learned_gram = learned_features.T @ learned_features
    cross_gram = reference_features.T @ learned_features
    difference_squared = (
        reference_gram.square().sum()
        + learned_gram.square().sum()
        - 2.0 * cross_gram.square().sum()
    ).clamp_min(0.0)
    denominator = reference_gram.square().sum().clamp_min(
        torch.finfo(reference_gram.dtype).eps
    )
    return float(torch.sqrt(difference_squared / denominator).cpu())


def compare_eigensystems(
    reference_eigenvalues: torch.Tensor,
    reference_eigenfunctions: torch.Tensor,
    learned_eigenvalues: torch.Tensor,
    learned_eigenfunctions: torch.Tensor,
    probability_weights: torch.Tensor,
    *,
    heat_time: float = 1.0,
) -> EigensystemComparison:
    """Compare ordered modes and their invariant weighted subspaces."""

    if reference_eigenfunctions.shape != learned_eigenfunctions.shape:
        raise ValueError("reference and learned eigenfunctions must have equal shape")
    if reference_eigenvalues.shape != learned_eigenvalues.shape:
        raise ValueError("reference and learned eigenvalues must have equal shape")
    weights = normalize_probability_weights(
        probability_weights,
        length=len(reference_eigenfunctions),
        dtype=reference_eigenfunctions.dtype,
        device=reference_eigenfunctions.device,
    )
    learned_gram = weighted_mass(learned_eigenfunctions, weights)
    identity = torch.eye(
        learned_eigenfunctions.shape[1],
        dtype=learned_eigenfunctions.dtype,
        device=learned_eigenfunctions.device,
    )
    orthogonality_error = float(
        torch.linalg.matrix_norm(learned_gram - identity).cpu()
    )
    reference = _weighted_orthonormalize(reference_eigenfunctions, weights)
    learned = _weighted_orthonormalize(learned_eigenfunctions, weights)
    cross = reference.T @ (weights.unsqueeze(1) * learned)
    nonconstant_cross = cross[1:, 1:]
    mode_correlations = torch.abs(torch.diagonal(nonconstant_cross))
    principal_cosines = torch.linalg.svdvals(nonconstant_cross).clamp(0.0, 1.0)
    projector_error = float(
        torch.sqrt((1.0 - principal_cosines.square()).mean().clamp_min(0.0)).cpu()
    )
    denominator = reference_eigenvalues[1:].abs().clamp_min(
        torch.finfo(reference_eigenvalues.dtype).eps
    )
    eigenvalue_errors = (
        learned_eigenvalues[1:] - reference_eigenvalues[1:]
    ).abs() / denominator
    kernel_error = _weighted_heat_kernel_error(
        reference_eigenfunctions,
        reference_eigenvalues,
        learned_eigenfunctions,
        learned_eigenvalues,
        weights,
        heat_time,
    )
    return EigensystemComparison(
        eigenvalue_relative_errors=eigenvalue_errors,
        mode_correlations=mode_correlations,
        principal_cosines=principal_cosines,
        mass_orthogonality_error=orthogonality_error,
        heat_kernel_relative_error=kernel_error,
        projector_error=projector_error,
    )


__all__ = [
    "EigensystemComparison",
    "GeneralizedEigenResult",
    "compare_eigensystems",
    "normalize_probability_weights",
    "solve_generalized_eigenproblem",
    "weighted_center",
    "weighted_mass",
]
