"""PyTorch SpIN for low-frequency weighted-Laplacian eigenfunctions.

The masked Cholesky gradient is translated from DeepMind's Apache-2.0
``spectral_inference_networks/src/spin.py``.  Unlike the TensorFlow 1 online
implementation, this version computes the mass matrix on a complete fixed
reference set, so no moving average of its parameter Jacobian is required.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from .eigensystems import normalize_probability_weights, solve_generalized_eigenproblem
from .neural_eigenfunctions import SharedEigenfunctionNetwork


class _MaskedSpINObjective(torch.autograd.Function):
    """Rayleigh trace with the ordered triangular gradient from SpIN."""

    @staticmethod
    def forward(ctx, mass: torch.Tensor, stiffness: torch.Tensor):
        chol = torch.linalg.cholesky(mass)
        chol_inverse = torch.linalg.inv(chol)
        rayleigh = chol_inverse @ stiffness @ chol_inverse.T
        diagonal = torch.diagonal(rayleigh)
        ctx.save_for_backward(chol_inverse, rayleigh)
        return torch.trace(rayleigh), diagonal

    @staticmethod
    def backward(ctx, grad_loss, grad_diagonal):
        del grad_diagonal
        chol_inverse, rayleigh = ctx.saved_tensors
        diagonal_inverse = torch.diag(torch.diagonal(chol_inverse))
        triangular = torch.triu(rayleigh @ diagonal_inverse)
        mass_gradient = -chol_inverse.T @ triangular
        stiffness_gradient = chol_inverse.T @ diagonal_inverse
        return grad_loss * mass_gradient, grad_loss * stiffness_gradient


def masked_spin_objective(
    mass: torch.Tensor, stiffness: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return _MaskedSpINObjective.apply(mass, stiffness)


@dataclass(frozen=True)
class SpINConfig:
    num_nonconstant_modes: int = 6
    hidden_width: int = 96
    hidden_layers: int = 3
    fourier_frequencies: int = 4
    steps: int = 2_500
    batch_size: int = 1_024
    learning_rate: float = 1e-3
    gradient_clip: float = 10.0
    relative_mass_jitter: float = 1e-8
    seed: int = 12
    log_every: int = 500

    def __post_init__(self) -> None:
        if self.num_nonconstant_modes <= 0:
            raise ValueError("num_nonconstant_modes must be positive")
        if self.hidden_width <= 0 or self.hidden_layers <= 0:
            raise ValueError("hidden dimensions must be positive")
        if self.fourier_frequencies < 0:
            raise ValueError("fourier_frequencies must be non-negative")
        if self.steps <= 0 or self.batch_size <= 0:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip <= 0:
            raise ValueError("optimizer settings must be positive")
        if self.relative_mass_jitter <= 0 or self.log_every <= 0:
            raise ValueError("jitter and log_every must be positive")


class SpINEigenfunctionSystem(torch.nn.Module):
    """Constant plus the directly ordered, Cholesky-whitened SpIN outputs."""

    def __init__(
        self,
        network: SharedEigenfunctionNetwork,
        output_mean: torch.Tensor,
        whitening: torch.Tensor,
        eigenvalues: torch.Tensor,
    ) -> None:
        super().__init__()
        self.network = network
        self.register_buffer("output_mean", output_mean)
        self.register_buffer("whitening", whitening)
        self.register_buffer("eigenvalues", eigenvalues)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        network_dtype = self.output_mean.dtype
        points = points.to(dtype=network_dtype, device=self.output_mean.device)
        nonconstant = (self.network(points) - self.output_mean) @ self.whitening
        constant = torch.ones(
            (*points.shape[:-1], 1),
            dtype=nonconstant.dtype,
            device=nonconstant.device,
        )
        return torch.cat((constant, nonconstant), dim=-1)


@dataclass(frozen=True)
class SpINEigenfunctionFit:
    system: SpINEigenfunctionSystem
    mass_matrix: torch.Tensor
    stiffness_matrix: torch.Tensor
    residuals: torch.Tensor
    losses: tuple[float, ...]
    selected_jitter: float
    condition_number: float
    posthoc_eigenvalues: torch.Tensor
    posthoc_rotation: torch.Tensor
    posthoc_offdiagonal_ratio: float
    fitting_time: float

    @property
    def eigenvalues(self) -> torch.Tensor:
        return self.system.eigenvalues


def _regularized_cholesky(
    mass: torch.Tensor, relative_jitter: float, max_attempts: int = 8
) -> tuple[torch.Tensor, float, torch.Tensor]:
    dimension = len(mass)
    scale = (torch.trace(mass.detach()).abs() / dimension).clamp_min(
        torch.finfo(mass.dtype).eps
    )
    jitter = relative_jitter * scale
    identity = torch.eye(dimension, dtype=mass.dtype, device=mass.device)
    for _ in range(max_attempts):
        regularized = mass + jitter * identity
        chol, info = torch.linalg.cholesky_ex(regularized.detach())
        if int(info.max().cpu()) == 0:
            # Recompute from the differentiable matrix for training.
            return torch.linalg.cholesky(regularized), float(jitter.cpu()), regularized
        jitter = 10.0 * jitter
    raise RuntimeError("SpIN mass matrix remained singular after adaptive jitter")


def _center_and_mass(
    outputs: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.sum(weights.unsqueeze(1) * outputs, dim=0)
    centered = outputs - mean
    mass = centered.T @ (weights.unsqueeze(1) * centered)
    return mean, centered, 0.5 * (mass + mass.T)


def _stiffness_from_outputs(
    outputs: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    gradients = []
    for output in range(outputs.shape[1]):
        gradients.append(
            torch.autograd.grad(
                outputs[:, output].sum(),
                points,
                create_graph=create_graph,
                retain_graph=True,
            )[0]
        )
    jacobian = torch.stack(gradients, dim=1)
    stiffness = torch.einsum("n,nid,njd->ij", weights, jacobian, jacobian)
    return 0.5 * (stiffness + stiffness.T)


def _reference_moments(
    network: SharedEigenfunctionNetwork,
    points: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    evaluation_points = points.detach().requires_grad_(True)
    outputs = network(evaluation_points)
    mean, _, mass = _center_and_mass(outputs, weights)
    stiffness = _stiffness_from_outputs(
        outputs, evaluation_points, weights, create_graph=False
    )
    return mean.detach(), mass.detach(), stiffness.detach()


def fit_spin_eigenfunctions(
    reference_points: torch.Tensor,
    probability_weights: torch.Tensor,
    config: SpINConfig,
    *,
    validation_points: torch.Tensor,
    validation_probability_weights: torch.Tensor,
    verbose: bool = True,
) -> SpINEigenfunctionFit:
    """Fit ordered nonconstant modes with the SpIN masked gradient."""

    if reference_points.ndim != 2 or not reference_points.dtype.is_floating_point:
        raise ValueError("reference_points must be a floating tensor of shape (N,d)")
    # Moment matrices and their Cholesky/eigendecompositions are deliberately
    # assembled in float64. This also avoids losing the masked signal when the
    # learned outputs become nearly collinear.
    reference_points = reference_points.to(dtype=torch.float64)
    validation_points = torch.as_tensor(
        validation_points,
        dtype=torch.float64,
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

    center = 0.5 * (reference_points.amin(dim=0) + reference_points.amax(dim=0))
    scale = 0.5 * (reference_points.amax(dim=0) - reference_points.amin(dim=0))
    if bool((scale <= 0).any()):
        raise ValueError("every coordinate must have nonzero range")
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
    generator = torch.Generator(device=reference_points.device)
    generator.manual_seed(config.seed + 1)
    losses: list[float] = []
    start = time.perf_counter()

    for step in range(1, config.steps + 1):
        optimizer.zero_grad()
        reference_outputs = network(reference_points)
        _, _, mass = _center_and_mass(reference_outputs, weights)
        _, jitter, regularized_mass = _regularized_cholesky(
            mass, config.relative_mass_jitter
        )

        sample_indices = torch.multinomial(
            weights,
            config.batch_size,
            replacement=True,
            generator=generator,
        )
        energy_points = reference_points[sample_indices].detach().requires_grad_(True)
        energy_outputs = network(energy_points)
        batch_weights = torch.full(
            (config.batch_size,),
            1.0 / config.batch_size,
            dtype=reference_points.dtype,
            device=reference_points.device,
        )
        stiffness = _stiffness_from_outputs(
            energy_outputs, energy_points, batch_weights, create_graph=True
        )
        loss, estimated_eigenvalues = masked_spin_objective(
            regularized_mass, stiffness
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), config.gradient_clip)
        optimizer.step()

        should_record = step == 1 or step % config.log_every == 0 or step == config.steps
        if should_record:
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            if verbose:
                values = ", ".join(
                    f"{value:.3f}" for value in estimated_eigenvalues.detach().cpu()
                )
                print(
                    f"  SpIN step {step:4d}/{config.steps}: loss={loss_value:.5f}, "
                    f"jitter={jitter:.2e}, diag=[{values}]"
                )

    output_mean, mass, stiffness = _reference_moments(
        network, validation_points, validation_weights
    )
    chol, jitter, regularized_mass = _regularized_cholesky(
        mass, config.relative_mass_jitter
    )
    identity = torch.eye(len(mass), dtype=mass.dtype, device=mass.device)
    whitening = torch.linalg.solve_triangular(chol.T, identity, upper=True)
    whitened_stiffness = whitening.T @ stiffness @ whitening
    whitened_stiffness = 0.5 * (whitened_stiffness + whitened_stiffness.T)
    direct_eigenvalues = torch.diagonal(whitened_stiffness).clamp_min(0.0)
    # The mask is intended to learn this ordering.  A finite optimization can
    # leave output labels inverted, so apply only a permutation by their
    # held-out energies.  This does not mix modes and is not the diagnostic
    # generalized-eigenvector rotation below.
    order = torch.argsort(direct_eigenvalues)
    direct_eigenvalues = direct_eigenvalues[order]
    whitening = whitening[:, order]
    whitened_stiffness = whitened_stiffness[order][:, order]
    residual_matrix = whitened_stiffness - torch.diag(direct_eigenvalues)
    residuals = torch.linalg.vector_norm(residual_matrix, dim=0) / (
        torch.linalg.matrix_norm(whitened_stiffness)
        + direct_eigenvalues.abs()
    ).clamp_min(torch.finfo(stiffness.dtype).eps)
    offdiagonal_ratio = float(
        (
            torch.linalg.matrix_norm(residual_matrix)
            / torch.linalg.matrix_norm(whitened_stiffness).clamp_min(
                torch.finfo(stiffness.dtype).eps
            )
        ).cpu()
    )
    posthoc = solve_generalized_eigenproblem(
        stiffness,
        mass,
        num_modes=config.num_nonconstant_modes,
        relative_jitter=config.relative_mass_jitter,
    )
    mass_spectrum = torch.linalg.eigvalsh(regularized_mass)
    condition_number = float((mass_spectrum[-1] / mass_spectrum[0]).cpu())
    eigenvalues = torch.cat(
        (
            torch.zeros(1, dtype=mass.dtype, device=mass.device),
            direct_eigenvalues,
        )
    )
    system = SpINEigenfunctionSystem(
        network=network,
        output_mean=output_mean,
        whitening=whitening,
        eigenvalues=eigenvalues,
    )
    return SpINEigenfunctionFit(
        system=system,
        mass_matrix=mass,
        stiffness_matrix=stiffness,
        residuals=residuals,
        losses=tuple(losses),
        selected_jitter=jitter,
        condition_number=condition_number,
        posthoc_eigenvalues=posthoc.eigenvalues,
        posthoc_rotation=posthoc.eigenvectors,
        posthoc_offdiagonal_ratio=offdiagonal_ratio,
        fitting_time=time.perf_counter() - start,
    )


__all__ = [
    "SpINConfig",
    "SpINEigenfunctionFit",
    "SpINEigenfunctionSystem",
    "fit_spin_eigenfunctions",
    "masked_spin_objective",
]
