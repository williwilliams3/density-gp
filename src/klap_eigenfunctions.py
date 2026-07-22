"""Torch implementation of Gaussian KLAP/Galerkin eigenfunctions.

The matrix formulas are adapted from the MIT-licensed KLAP implementation at
``code/laplacian`` and Cabannes & Bach (AISTATS 2024).
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .eigensystems import (
    normalize_probability_weights,
    solve_generalized_eigenproblem,
    weighted_mass,
)


@dataclass(frozen=True)
class KlapConfig:
    num_nonconstant_modes: int = 6
    num_centers: int = 256
    bandwidth_candidates: tuple[float, ...] = (0.15, 0.25, 0.40, 0.65)
    relative_mass_jitter: float = 1e-8
    max_condition_number: float = 1e12
    seed: int = 12

    def __post_init__(self) -> None:
        if self.num_nonconstant_modes <= 0 or self.num_centers <= 0:
            raise ValueError("mode and center counts must be positive")
        if self.num_centers <= self.num_nonconstant_modes:
            raise ValueError("num_centers must exceed num_nonconstant_modes")
        if not self.bandwidth_candidates or any(
            value <= 0 for value in self.bandwidth_candidates
        ):
            raise ValueError("bandwidth_candidates must be positive")
        if self.relative_mass_jitter <= 0 or self.max_condition_number <= 1:
            raise ValueError("invalid mass regularization or condition threshold")


class KlapEigenfunctionSystem(torch.nn.Module):
    """Evaluable constant plus ordered Gaussian Galerkin eigenfunctions."""

    def __init__(
        self,
        centers: torch.Tensor,
        bandwidth: float,
        feature_mean: torch.Tensor,
        coefficients: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("centers", centers)
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("coefficients", coefficients)
        self.register_buffer("eigenvalues", eigenvalues)
        self.bandwidth = float(bandwidth)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        values = gaussian_trial_values(points, self.centers, self.bandwidth)
        nonconstant = (values - self.feature_mean) @ self.coefficients
        constant = torch.ones(
            (*points.shape[:-1], 1), dtype=points.dtype, device=points.device
        )
        return torch.cat((constant, nonconstant), dim=-1)


@dataclass(frozen=True)
class KlapEigenfunctionFit:
    system: KlapEigenfunctionSystem
    mass_matrix: torch.Tensor
    stiffness_matrix: torch.Tensor
    residuals: torch.Tensor
    selected_jitter: float
    condition_number: float
    validation_score: float
    fitting_time: float

    @property
    def eigenvalues(self) -> torch.Tensor:
        return self.system.eigenvalues

    @property
    def bandwidth(self) -> float:
        return self.system.bandwidth


def gaussian_trial_values(
    points: torch.Tensor, centers: torch.Tensor, bandwidth: float
) -> torch.Tensor:
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("points and centers must have shapes (N,d) and (m,d)")
    squared_distances = torch.cdist(points, centers).square()
    return torch.exp(-squared_distances / bandwidth**2)


def gaussian_trial_moments(
    points: torch.Tensor,
    probability_weights: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
    *,
    feature_mean: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return feature mean and weighted mass/stiffness matrices."""

    weights = normalize_probability_weights(
        probability_weights,
        length=len(points),
        dtype=points.dtype,
        device=points.device,
    )
    values = gaussian_trial_values(points, centers, bandwidth)
    if feature_mean is None:
        feature_mean = torch.sum(weights.unsqueeze(1) * values, dim=0)
    centered = values - feature_mean
    mass = weighted_mass(centered, weights)

    point_center_products = points @ centers.T
    center_products = centers @ centers.T
    point_norms = points.square().sum(dim=1)
    weighted_values = weights.unsqueeze(1) * values
    value_mass = values.T @ weighted_values
    norm_term = values.T @ (
        (weights * point_norms).unsqueeze(1) * values
    )
    cross_term = (values * point_center_products).T @ weighted_values
    stiffness = (
        norm_term
        - cross_term
        - cross_term.T
        + center_products * value_mass
    )
    stiffness = (4.0 / bandwidth**4) * stiffness
    stiffness = 0.5 * (stiffness + stiffness.T)
    return feature_mean, mass, stiffness


def _select_centers(
    points: torch.Tensor, weights: torch.Tensor, count: int, seed: int
) -> torch.Tensor:
    if count > len(points):
        raise ValueError("num_centers cannot exceed the number of reference points")
    generator = torch.Generator(device=points.device)
    generator.manual_seed(seed)
    indices = torch.multinomial(
        weights, count, replacement=False, generator=generator
    )
    return points[indices].detach().clone()


def fit_klap_eigenfunctions(
    reference_points: torch.Tensor,
    probability_weights: torch.Tensor,
    config: KlapConfig,
    *,
    validation_points: torch.Tensor,
    validation_probability_weights: torch.Tensor,
    verbose: bool = True,
) -> KlapEigenfunctionFit:
    """Fit and select a Gaussian Nyström Galerkin eigensystem."""

    if reference_points.ndim != 2 or not reference_points.dtype.is_floating_point:
        raise ValueError("reference_points must be a floating tensor of shape (N,d)")
    validation_points = torch.as_tensor(
        validation_points,
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    if validation_points.ndim != 2 or validation_points.shape[1] != reference_points.shape[1]:
        raise ValueError("validation_points must have shape (N_validation,d)")
    weights = normalize_probability_weights(
        probability_weights,
        length=len(reference_points),
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    validation_weights = normalize_probability_weights(
        validation_probability_weights,
        length=len(validation_points),
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    centers = _select_centers(
        reference_points, weights, config.num_centers, config.seed
    )
    start = time.perf_counter()
    best = None

    for bandwidth in config.bandwidth_candidates:
        train_mean, train_mass, train_stiffness = gaussian_trial_moments(
            reference_points, weights, centers, bandwidth
        )
        try:
            train_solution = solve_generalized_eigenproblem(
                train_stiffness,
                train_mass,
                num_modes=config.num_nonconstant_modes,
                relative_jitter=config.relative_mass_jitter,
            )
        except RuntimeError:
            continue
        if train_solution.condition_number > config.max_condition_number:
            continue

        validation_mean, validation_mass, validation_stiffness = gaussian_trial_moments(
            validation_points, validation_weights, centers, bandwidth
        )
        coefficients = train_solution.eigenvectors
        restricted_mass = coefficients.T @ validation_mass @ coefficients
        restricted_stiffness = coefficients.T @ validation_stiffness @ coefficients
        try:
            validation_solution = solve_generalized_eigenproblem(
                restricted_stiffness,
                restricted_mass,
                num_modes=config.num_nonconstant_modes,
                relative_jitter=config.relative_mass_jitter,
            )
        except RuntimeError:
            continue
        final_coefficients = coefficients @ validation_solution.eigenvectors
        final_mass = final_coefficients.T @ validation_mass @ final_coefficients
        final_stiffness = (
            final_coefficients.T @ validation_stiffness @ final_coefficients
        )
        score = float(validation_solution.eigenvalues.sum().detach().cpu())
        candidate = (
            score,
            bandwidth,
            validation_mean,
            final_coefficients,
            validation_solution,
            final_mass,
            final_stiffness,
            max(train_solution.condition_number, validation_solution.condition_number),
            max(train_solution.jitter, validation_solution.jitter),
        )
        if best is None or score < best[0]:
            best = candidate
        if verbose:
            print(
                f"  KLAP sigma={bandwidth:.3f}: validation trace={score:.5f}, "
                f"condition={candidate[7]:.3e}"
            )

    if best is None:
        raise RuntimeError("no KLAP bandwidth produced a stable Galerkin system")
    (
        score,
        bandwidth,
        feature_mean,
        coefficients,
        solution,
        mass,
        stiffness,
        condition_number,
        jitter,
    ) = best
    eigenvalues = torch.cat(
        (
            torch.zeros(1, dtype=reference_points.dtype, device=reference_points.device),
            solution.eigenvalues,
        )
    )
    system = KlapEigenfunctionSystem(
        centers=centers,
        bandwidth=bandwidth,
        feature_mean=feature_mean,
        coefficients=coefficients,
        eigenvalues=eigenvalues,
    )
    return KlapEigenfunctionFit(
        system=system,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        residuals=solution.residuals,
        selected_jitter=jitter,
        condition_number=condition_number,
        validation_score=score,
        fitting_time=time.perf_counter() - start,
    )


__all__ = [
    "KlapConfig",
    "KlapEigenfunctionFit",
    "KlapEigenfunctionSystem",
    "fit_klap_eigenfunctions",
    "gaussian_trial_moments",
    "gaussian_trial_values",
]
