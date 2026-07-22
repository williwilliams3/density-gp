"""Joint neural approximation of low-frequency weighted-Laplacian eigenpairs.

This implements Idea 1 from the project notes: a shared network learns an
orthonormal low-frequency subspace with the weak Dirichlet form and an
augmented-Lagrangian mass constraint.  The individual ordered modes are
recovered only after training through a small generalized eigendecomposition.
Only first spatial derivatives of the network are used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import LinAlgError, eigh

from .eigensystems import EigensystemComparison, compare_eigensystems


@dataclass(frozen=True)
class NeuralEigenfunctionConfig:
    num_nonconstant_modes: int = 8
    hidden_width: int = 96
    hidden_layers: int = 3
    fourier_frequencies: int = 4
    steps: int = 1_500
    batch_size: int = 1_024
    learning_rate: float = 2e-3
    constraint_strength: float = 20.0
    multiplier_update_every: int = 100
    gradient_clip: float = 10.0
    seed: int = 0
    log_every: int = 100

    def __post_init__(self) -> None:
        if self.num_nonconstant_modes <= 0:
            raise ValueError("num_nonconstant_modes must be positive")
        if self.hidden_width <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden_width and hidden_layers must be positive")
        if self.fourier_frequencies < 0:
            raise ValueError("fourier_frequencies must be non-negative")
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.constraint_strength <= 0:
            raise ValueError("learning_rate and constraint_strength must be positive")
        if self.multiplier_update_every <= 0:
            raise ValueError("multiplier_update_every must be positive")


class SharedEigenfunctionNetwork(torch.nn.Module):
    """A shared MLP with one output for each candidate nonconstant mode."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_width: int,
        hidden_layers: int,
        fourier_frequencies: int,
        coordinate_center: torch.Tensor,
        coordinate_scale: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("coordinate_center", coordinate_center)
        self.register_buffer("coordinate_scale", coordinate_scale)
        self.register_buffer(
            "frequencies",
            torch.arange(
                1,
                fourier_frequencies + 1,
                dtype=coordinate_center.dtype,
                device=coordinate_center.device,
            )
            * torch.pi,
        )
        layers: list[torch.nn.Module] = []
        layer_input = input_dim * (1 + 2 * fourier_frequencies)
        for _ in range(hidden_layers):
            layers.extend((torch.nn.Linear(layer_input, hidden_width), torch.nn.Tanh()))
            layer_input = hidden_width
        layers.append(torch.nn.Linear(layer_input, output_dim))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        normalized = (points - self.coordinate_center) / self.coordinate_scale
        if len(self.frequencies) == 0:
            features = normalized
        else:
            phases = normalized.unsqueeze(-1) * self.frequencies
            features = torch.cat(
                (normalized, torch.sin(phases).flatten(-2), torch.cos(phases).flatten(-2)),
                dim=-1,
            )
        return self.network(features)


class NeuralEigenfunctionSystem(torch.nn.Module):
    """Evaluable ordered eigenfunctions obtained from a trained shared basis."""

    def __init__(
        self,
        network: SharedEigenfunctionNetwork,
        output_mean: torch.Tensor,
        rotation: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("rotation", rotation)
        self.register_buffer("eigenvalues", eigenvalues)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        raw = self.network(points)
        nonconstant = (raw - self.output_mean) @ self.rotation
        constant = torch.ones(
            (*points.shape[:-1], 1), dtype=points.dtype, device=points.device
        )
        return torch.cat((constant, nonconstant), dim=-1)


@dataclass(frozen=True)
class NeuralEigenfunctionFit:
    system: NeuralEigenfunctionSystem
    mass_matrix: torch.Tensor
    stiffness_matrix: torch.Tensor
    losses: tuple[float, ...]
    constraint_errors: tuple[float, ...]

    @property
    def eigenvalues(self) -> torch.Tensor:
        return self.system.eigenvalues


def fit_neural_eigenfunctions(
    reference_points: torch.Tensor,
    probability_weights: torch.Tensor,
    config: NeuralEigenfunctionConfig,
    *,
    validation_points: torch.Tensor | None = None,
    validation_probability_weights: torch.Tensor | None = None,
    verbose: bool = True,
) -> NeuralEigenfunctionFit:
    """Fit a joint nonconstant eigenspace from weighted reference points.

    ``probability_weights`` is normalized internally and represents samples
    from the density ``p`` on the reference quadrature/grid.  The full weighted
    reference set estimates the mass matrix, while stochastic draws from the
    same distribution estimate the Dirichlet energy.  If an independent
    validation quadrature is supplied, it is used to recompute the final mass
    and stiffness matrices before the generalized eigendecomposition.
    """

    if reference_points.ndim != 2:
        raise ValueError("reference_points must have shape (N, input_dim)")
    if not reference_points.dtype.is_floating_point:
        raise TypeError("reference_points must have a floating-point dtype")
    weights = torch.as_tensor(
        probability_weights,
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    if weights.shape != (len(reference_points),):
        raise ValueError("probability_weights must have shape (N,)")
    if bool((weights < 0).any()) or not bool((weights.sum() > 0)):
        raise ValueError("probability_weights must be non-negative with positive sum")
    weights = weights / weights.sum()
    if (validation_points is None) != (validation_probability_weights is None):
        raise ValueError(
            "validation_points and validation_probability_weights must be supplied together"
        )
    if validation_points is None:
        final_points = reference_points
        final_weights = weights
    else:
        final_points = torch.as_tensor(
            validation_points,
            dtype=reference_points.dtype,
            device=reference_points.device,
        )
        if final_points.ndim != 2 or final_points.shape[1] != reference_points.shape[1]:
            raise ValueError("validation_points must have shape (N_validation, input_dim)")
        final_weights = torch.as_tensor(
            validation_probability_weights,
            dtype=reference_points.dtype,
            device=reference_points.device,
        )
        if final_weights.shape != (len(final_points),):
            raise ValueError(
                "validation_probability_weights must have shape (N_validation,)"
            )
        if bool((final_weights < 0).any()) or not bool((final_weights.sum() > 0)):
            raise ValueError(
                "validation_probability_weights must be non-negative with positive sum"
            )
        final_weights = final_weights / final_weights.sum()

    center = 0.5 * (
        reference_points.amin(dim=0) + reference_points.amax(dim=0)
    )
    scale = 0.5 * (
        reference_points.amax(dim=0) - reference_points.amin(dim=0)
    )
    if bool((scale <= 0).any()):
        raise ValueError("every input coordinate must have nonzero range")
    fork_devices = (
        [reference_points.device.index]
        if reference_points.device.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(config.seed)
        network = SharedEigenfunctionNetwork(
            reference_points.shape[1],
            config.num_nonconstant_modes,
            hidden_width=config.hidden_width,
            hidden_layers=config.hidden_layers,
            fourier_frequencies=config.fourier_frequencies,
            coordinate_center=center,
            coordinate_scale=scale,
        ).to(dtype=reference_points.dtype, device=reference_points.device)

    optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
    multiplier = torch.zeros(
        (config.num_nonconstant_modes, config.num_nonconstant_modes),
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    identity = torch.eye(
        config.num_nonconstant_modes,
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    generator = torch.Generator(device=reference_points.device)
    generator.manual_seed(config.seed + 1)
    losses: list[float] = []
    constraint_errors: list[float] = []

    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        raw_reference = network(reference_points)
        output_mean, centered_reference, mass = _center_and_mass(
            raw_reference, weights
        )

        sample_indices = torch.multinomial(
            weights,
            config.batch_size,
            replacement=True,
            generator=generator,
        )
        energy_points = reference_points[sample_indices].detach().requires_grad_(True)
        raw_energy = network(energy_points)
        energy = _dirichlet_trace(raw_energy, energy_points, create_graph=True)

        constraint = mass - identity
        loss = (
            energy
            + torch.sum(multiplier * constraint)
            + 0.5 * config.constraint_strength * constraint.square().sum()
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
        optimizer.step()

        should_record = step == 1 or step % config.log_every == 0 or step == config.steps
        if should_record:
            constraint_error = float(torch.linalg.matrix_norm(constraint).detach().cpu())
            losses.append(float(loss.detach().cpu()))
            constraint_errors.append(constraint_error)
            if verbose:
                print(
                    f"  neural eigensolver step {step:4d}/{config.steps}: "
                    f"loss={loss.detach().cpu():.5f}, energy={energy.detach().cpu():.5f}, "
                    f"||M-I||={constraint_error:.5f}"
                )

        if step % config.multiplier_update_every == 0:
            with torch.no_grad():
                updated_raw = network(reference_points)
                _, _, updated_mass = _center_and_mass(updated_raw, weights)
                multiplier.add_(config.constraint_strength * (updated_mass - identity))
                multiplier.copy_(0.5 * (multiplier + multiplier.T))

    output_mean, mass, stiffness = _reference_moments(
        network, final_points, final_weights
    )
    mass_numpy = mass.detach().cpu().double().numpy()
    stiffness_numpy = stiffness.detach().cpu().double().numpy()
    try:
        eigenvalues_numpy, rotation_numpy = eigh(stiffness_numpy, mass_numpy)
    except LinAlgError as error:
        raise RuntimeError(
            "learned mass matrix is not positive definite; increase training steps "
            "or constraint_strength"
        ) from error
    eigenvalues = torch.cat(
        (
            torch.zeros(1, dtype=reference_points.dtype, device=reference_points.device),
            torch.as_tensor(
                np.maximum(eigenvalues_numpy, 0.0),
                dtype=reference_points.dtype,
                device=reference_points.device,
            ),
        )
    )
    rotation = torch.as_tensor(
        rotation_numpy,
        dtype=reference_points.dtype,
        device=reference_points.device,
    )
    system = NeuralEigenfunctionSystem(
        network=network,
        output_mean=output_mean,
        rotation=rotation,
        eigenvalues=eigenvalues,
    )
    return NeuralEigenfunctionFit(
        system=system,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        losses=tuple(losses),
        constraint_errors=tuple(constraint_errors),
    )


def _center_and_mass(
    raw_outputs: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_mean = torch.sum(weights.unsqueeze(1) * raw_outputs, dim=0)
    centered = raw_outputs - output_mean
    mass = centered.T @ (weights.unsqueeze(1) * centered)
    return output_mean, centered, 0.5 * (mass + mass.T)


def _dirichlet_trace(
    outputs: torch.Tensor,
    points: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    energy = torch.zeros((), dtype=outputs.dtype, device=outputs.device)
    for mode in range(outputs.shape[1]):
        gradient = torch.autograd.grad(
            outputs[:, mode].sum(),
            points,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        energy = energy + gradient.square().sum(dim=1).mean()
    return energy


def _reference_moments(
    network: SharedEigenfunctionNetwork,
    points: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    evaluation_points = points.detach().requires_grad_(True)
    raw = network(evaluation_points)
    output_mean, _, mass = _center_and_mass(raw, weights)
    gradients = []
    for mode in range(raw.shape[1]):
        gradient = torch.autograd.grad(
            raw[:, mode].sum(),
            evaluation_points,
            create_graph=False,
            retain_graph=True,
        )[0]
        gradients.append(gradient)
    jacobian = torch.stack(gradients, dim=1)
    stiffness = torch.einsum("n,nid,njd->ij", weights, jacobian, jacobian)
    stiffness = 0.5 * (stiffness + stiffness.T)
    return output_mean.detach(), mass.detach(), stiffness.detach()


__all__ = [
    "EigensystemComparison",
    "NeuralEigenfunctionConfig",
    "NeuralEigenfunctionFit",
    "NeuralEigenfunctionSystem",
    "SharedEigenfunctionNetwork",
    "compare_eigensystems",
    "fit_neural_eigenfunctions",
]
