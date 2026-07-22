"""Compare FEM, KLAP, and SpIN on the weighted spiral Laplacian.

The primary experiment learns KLAP and SpIN eigenfunctions from identical iid
draws from the spiral density.  A weighted-grid regime is also provided as a
controlled quadrature comparison.  The earlier augmented-Lagrangian neural
subspace solver remains available with ``--include-subspace``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.eigensystems import (
    EigensystemComparison,
    compare_eigensystems,
    normalize_eigenfunction_columns,
)
from src.fem import weighted_laplacian_eigendecomposition_2d
from src.gp import gaussian_nll, marginal_nll, predict_exact_gp
from src.grid import make_grid, nearest_grid_index
from src.klap_eigenfunctions import KlapConfig, fit_klap_eigenfunctions
from src.metrics import evaluate_predictions
from src.neural_eigenfunctions import (
    NeuralEigenfunctionConfig,
    fit_neural_eigenfunctions,
)
from src.spin_eigenfunctions import SpINConfig, fit_spin_eigenfunctions
from src.training import fit_density_gp, fit_euclidean_gp


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
GRID_SIZE = 61
DOMAIN_LIMIT = 2.05
N_MODES = 7
N_TRAIN = 34
N_TEST = 120
NOISE_SD = 0.035

SPIRAL_TURNS = 3.45
R_OUTER = 1.85
R_INNER = 0.24
TUBE_WIDTH = 0.05
DENSITY_FLOOR = 0.001
TUBE_THRESHOLD = 0.20


@dataclass
class LearnedEigensystem:
    name: str
    eigenvalues: torch.Tensor
    eigenfunctions: torch.Tensor
    comparison: EigensystemComparison
    condition_number: float
    maximum_residual: float
    fitting_time: float
    extra: str = ""


def make_spiral_curve(n_curve: int = 2400):
    t_max = 2.0 * math.pi * SPIRAL_TURNS
    t = np.linspace(0.0, t_max, n_curve)
    radius = R_OUTER - (R_OUTER - R_INNER) * t / t_max
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    arclength = np.zeros_like(t)
    arclength[1:] = np.cumsum(np.hypot(np.diff(x), np.diff(y)))
    return t, x, y, arclength / arclength[-1]


def nearest_spiral_coordinates(points, curve_x, curve_y, curve_s, chunk_size=500):
    curve = np.column_stack((curve_x, curve_y))
    best_dist2 = np.full(len(points), np.inf)
    best_idx = np.zeros(len(points), dtype=int)
    for start in range(0, len(curve), chunk_size):
        curve_chunk = curve[start : start + chunk_size]
        dist2 = np.sum((points[:, None, :] - curve_chunk[None, :, :]) ** 2, axis=2)
        local_idx = np.argmin(dist2, axis=1)
        local_dist2 = dist2[np.arange(len(points)), local_idx]
        update = local_dist2 < best_dist2
        best_dist2[update] = local_dist2[update]
        best_idx[update] = start + local_idx[update]
    return np.sqrt(best_dist2), best_idx, curve_s[best_idx]


def spiral_density(distance_to_spiral):
    return DENSITY_FLOOR + np.exp(-0.5 * (distance_to_spiral / TUBE_WIDTH) ** 2)


def spiral_target(normalized_arclength):
    return (
        2.0 * normalized_arclength
        - 1.0
        + 0.12 * np.sin(2.0 * math.pi * normalized_arclength)
    )


def sample_spiral_density(n, curve_x, curve_y, curve_s, *, seed):
    """Draw continuous iid samples by rejection from the bounded density."""

    rng = np.random.default_rng(seed)
    accepted = []
    while sum(len(chunk) for chunk in accepted) < n:
        remaining = n - sum(len(chunk) for chunk in accepted)
        proposal_count = max(4096, 20 * remaining)
        proposals = rng.uniform(
            -DOMAIN_LIMIT, DOMAIN_LIMIT, size=(proposal_count, 2)
        )
        distance, _, _ = nearest_spiral_coordinates(
            proposals, curve_x, curve_y, curve_s
        )
        probability = spiral_density(distance) / (1.0 + DENSITY_FLOOR)
        chosen = proposals[rng.random(proposal_count) < probability]
        if len(chosen):
            accepted.append(chosen[:remaining])
    return np.concatenate(accepted, axis=0)[:n]


def choose_training_indices(points, curve_x, curve_y, curve_s, n_train=N_TRAIN):
    indices = []
    for target_s in np.linspace(0.035, 0.965, n_train):
        curve_idx = int(np.argmin(np.abs(curve_s - target_s)))
        indices.append(
            nearest_grid_index(points, (curve_x[curve_idx], curve_y[curve_idx]))
        )
    return torch.tensor(
        list(dict.fromkeys(indices)), dtype=torch.long, device=DEVICE
    )


def choose_test_indices(points, curve_x, curve_y, curve_s, idx_train, n_test=N_TEST):
    train_set = set(idx_train.detach().cpu().tolist())
    indices = []
    for target_s in np.linspace(0.015, 0.985, n_test * 2):
        curve_idx = int(np.argmin(np.abs(curve_s - target_s)))
        grid_idx = nearest_grid_index(
            points, (curve_x[curve_idx], curve_y[curve_idx])
        )
        if grid_idx not in train_set and grid_idx not in indices:
            indices.append(grid_idx)
        if len(indices) == n_test:
            break
    return torch.tensor(indices, dtype=torch.long, device=DEVICE)


def _method_configs(seed: int, quick: bool):
    if quick:
        klap = KlapConfig(
            num_nonconstant_modes=N_MODES - 1,
            num_centers=32,
            bandwidth_candidates=(0.25, 0.40),
            seed=seed,
        )
        spin = SpINConfig(
            num_nonconstant_modes=N_MODES - 1,
            hidden_width=24,
            hidden_layers=1,
            fourier_frequencies=2,
            steps=20,
            batch_size=64,
            learning_rate=1e-3,
            gradient_clip=10.0,
            seed=seed,
            log_every=10,
        )
    else:
        klap = KlapConfig(seed=seed)
        spin = SpINConfig(seed=seed)
    return klap, spin


def fit_regime(
    name,
    reference_points,
    reference_weights,
    validation_points,
    validation_weights,
    evaluation_points,
    evaluation_weights,
    fem_eigenvalues,
    fem_eigenfunctions,
    *,
    seed,
    quick,
    include_subspace,
):
    klap_config, spin_config = _method_configs(seed, quick)
    print(f"\n[{name}] fitting KLAP")
    klap_fit = fit_klap_eigenfunctions(
        reference_points.to(dtype=torch.float64),
        reference_weights.to(dtype=torch.float64),
        klap_config,
        validation_points=validation_points.to(dtype=torch.float64),
        validation_probability_weights=validation_weights.to(dtype=torch.float64),
        verbose=True,
    )
    with torch.no_grad():
        klap_functions = klap_fit.system(
            evaluation_points.to(dtype=torch.float64)
        ).to(DTYPE)
    klap_values = klap_fit.eigenvalues.to(DTYPE)
    klap_comparison = compare_eigensystems(
        fem_eigenvalues,
        fem_eigenfunctions,
        klap_values,
        klap_functions,
        evaluation_weights,
    )
    systems = {
        "klap": LearnedEigensystem(
            "KLAP",
            klap_values,
            klap_functions,
            klap_comparison,
            klap_fit.condition_number,
            float(klap_fit.residuals.max().cpu()),
            klap_fit.fitting_time,
            f"sigma={klap_fit.bandwidth:.3f}, validation={klap_fit.validation_score:.3f}",
        )
    }

    print(f"\n[{name}] fitting SpIN")
    spin_fit = fit_spin_eigenfunctions(
        reference_points,
        reference_weights,
        spin_config,
        validation_points=validation_points,
        validation_probability_weights=validation_weights,
        verbose=True,
    )
    with torch.no_grad():
        spin_functions = spin_fit.system(evaluation_points).to(DTYPE)
    spin_values = spin_fit.eigenvalues.to(DTYPE)
    spin_comparison = compare_eigensystems(
        fem_eigenvalues,
        fem_eigenfunctions,
        spin_values,
        spin_functions,
        evaluation_weights,
    )
    systems["spin"] = LearnedEigensystem(
        "SpIN",
        spin_values,
        spin_functions,
        spin_comparison,
        spin_fit.condition_number,
        float(spin_fit.residuals.max().cpu()),
        spin_fit.fitting_time,
        f"posthoc offdiag={spin_fit.posthoc_offdiagonal_ratio:.3e}",
    )

    if include_subspace:
        steps = 20 if quick else 2_500
        subspace_config = NeuralEigenfunctionConfig(
            num_nonconstant_modes=N_MODES - 1,
            hidden_width=24 if quick else 96,
            hidden_layers=1 if quick else 3,
            fourier_frequencies=2 if quick else 4,
            steps=steps,
            batch_size=64 if quick else 1_024,
            learning_rate=1e-3,
            constraint_strength=50.0,
            multiplier_update_every=max(10, steps // 5),
            gradient_clip=10.0,
            seed=seed,
            log_every=max(10, steps // 5),
        )
        print(f"\n[{name}] fitting optional augmented-Lagrangian subspace")
        subspace_fit = fit_neural_eigenfunctions(
            reference_points.float(),
            reference_weights.float(),
            subspace_config,
            validation_points=validation_points.float(),
            validation_probability_weights=validation_weights.float(),
            verbose=True,
        )
        with torch.no_grad():
            functions = subspace_fit.system(evaluation_points.float()).to(DTYPE)
        values = subspace_fit.eigenvalues.to(DTYPE)
        comparison = compare_eigensystems(
            fem_eigenvalues,
            fem_eigenfunctions,
            values,
            functions,
            evaluation_weights,
        )
        systems["subspace"] = LearnedEigensystem(
            "subspace",
            values,
            functions,
            comparison,
            float(torch.linalg.cond(subspace_fit.mass_matrix).cpu()),
            float("nan"),
            float("nan"),
            "secondary augmented-Lagrangian baseline",
        )
    return systems


def print_eigensolver_table(regime_name, systems, fem_values):
    print(f"\n{regime_name.capitalize()} eigensolver comparison")
    header = (
        f"{'method':<10} {'mean eig err':>12} {'min cosine':>12} "
        f"{'proj err':>10} {'orth err':>10} {'heat err':>10} "
        f"{'condition':>11} {'residual':>10} {'time(s)':>9}"
    )
    print(header)
    print("-" * len(header))
    for system in systems.values():
        c = system.comparison
        print(
            f"{system.name:<10} {c.mean_eigenvalue_relative_error:>12.4f} "
            f"{c.minimum_principal_cosine:>12.4f} {c.projector_error:>10.4f} "
            f"{c.mass_orthogonality_error:>10.2e} "
            f"{c.heat_kernel_relative_error:>10.4f} "
            f"{system.condition_number:>11.2e} {system.maximum_residual:>10.2e} "
            f"{system.fitting_time:>9.2f}"
        )
        print(f"  {system.extra}")
        for mode in range(1, N_MODES):
            print(
                f"    mode {mode}: lambda FEM={float(fem_values[mode]):.4f}, "
                f"learned={float(system.eigenvalues[mode]):.4f}, "
                f"rel.err={float(c.eigenvalue_relative_errors[mode - 1]):.4f}, "
                f"|corr|={float(c.mode_correlations[mode - 1]):.4f}"
            )
    print(
        "  diagnostic targets (informational): mean eig err < 0.25, "
        "min cosine > 0.80, heat err < 0.30, orth err < 1e-4"
    )


def _fit_gp_models(y_train, idx_train, points, systems, fem_values, fem_functions, quick):
    spectral_starts = (-2.0, 0.0, 2.0) if quick else (
        -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0
    )
    sigma_starts = (0.0,) if quick else (-1.0, 0.0, 1.0)
    length_starts = (-2.0, 0.0, 2.0) if quick else (
        -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0
    )
    steps = 25 if quick else 300
    eigenpairs = {
        "fem": (fem_values, fem_functions),
        "klap": (systems["klap"].eigenvalues, systems["klap"].eigenfunctions),
        "spin": (systems["spin"].eigenvalues, systems["spin"].eigenfunctions),
    }
    fits = {}
    for name, (values, functions) in eigenpairs.items():
        fits[name] = fit_density_gp(
            y_train=y_train,
            train_grid_indices=idx_train,
            noise_var=NOISE_SD**2,
            eigenvalues=values,
            eigenvectors=functions,
            kind="heat",
            spectral_raw_inits=spectral_starts,
            sigma_raw_inits=sigma_starts,
            steps=steps,
            lr=0.05,
        )
    fits["rbf"] = fit_euclidean_gp(
        y_train=y_train,
        x_train=points[idx_train],
        noise_var=NOISE_SD**2,
        kind="rbf",
        lengthscale_raw_inits=length_starts,
        sigma_raw_inits=sigma_starts,
        steps=steps,
        lr=0.05,
    )
    return fits


def evaluate_gp_models(fits, points, idx_test, y_true, masks):
    grid_indices = torch.arange(
        len(points), dtype=DTYPE, device=points.device
    ).unsqueeze(1)
    predictions = {}
    test_predictions = {}
    for name in ("fem", "klap", "spin"):
        predictions[name] = predict_exact_gp(
            fits[name].model, grid_indices, NOISE_SD**2
        )
        test_predictions[name] = predict_exact_gp(
            fits[name].model,
            grid_indices[idx_test],
            NOISE_SD**2,
            full_covariance=True,
        )
    predictions["rbf"] = predict_exact_gp(
        fits["rbf"].model, points, NOISE_SD**2
    )
    test_predictions["rbf"] = predict_exact_gp(
        fits["rbf"].model,
        points[idx_test],
        NOISE_SD**2,
        full_covariance=True,
    )

    diagnostics = {}
    y_test = y_true[idx_test]
    for name in ("fem", "klap", "spin", "rbf"):
        prediction = predictions[name]
        test_prediction = test_predictions[name]
        joint = gaussian_nll(
            y_test,
            test_prediction.latent_mean,
            test_prediction.latent_covariance,
            NOISE_SD**2,
        )
        marginal = marginal_nll(
            y_test,
            test_prediction.latent_mean,
            test_prediction.latent_covariance,
            NOISE_SD**2,
        )
        diagnostics[name] = {
            "regions": evaluate_predictions(
                y_true,
                prediction.latent_mean,
                prediction.observed_variance,
                masks,
            ),
            "joint_nll_per_point": float((joint / len(idx_test)).cpu()),
            "marginal_nll": float(marginal.cpu()),
            "fit_loss": fits[name].total_negative_mll,
        }
        with torch.no_grad():
            if name == "rbf":
                diagnostics[name]["scale"] = float(
                    fits[name].model.covar_module.base_kernel.lengthscale.cpu()
                )
            else:
                diagnostics[name]["scale"] = float(
                    fits[name].model.covar_module.base_kernel.tau.cpu()
                )
            diagnostics[name]["sigma"] = float(
                fits[name].model.covar_module.outputscale.sqrt().cpu()
            )
    fem_mean = predictions["fem"].latent_mean
    for name in ("klap", "spin", "rbf"):
        diagnostics[name]["fem_mean_rmse"] = float(
            torch.sqrt(
                torch.mean(
                    (predictions[name].latent_mean[masks["tube"]] - fem_mean[masks["tube"]]).square()
                )
            ).cpu()
        )
    return predictions, diagnostics


def print_gp_table(diagnostics, n_test, regime_name):
    print(f"\n{regime_name.capitalize()}-regime GP comparison")
    header = (
        f"{'model':<8} {'RMSE':>8} {'NLPD':>8} {'coverage':>9} "
        f"{'joint NLL':>10} {'marg NLL':>10} {'scale':>9} {'sigma':>9} "
        f"{'fit loss':>10} {'vs FEM':>9}"
    )
    print(header)
    print("-" * len(header))
    for name in ("fem", "klap", "spin", "rbf"):
        data = diagnostics[name]
        tube = data["regions"]["tube"]
        disagreement = data.get("fem_mean_rmse", 0.0)
        print(
            f"{name:<8} {tube['rmse']:>8.4f} {tube['nlpd']:>8.4f} "
            f"{tube['coverage']:>9.3f} {data['joint_nll_per_point']:>10.4f} "
            f"{data['marginal_nll']:>10.4f} {data['scale']:>9.4f} "
            f"{data['sigma']:>9.4f} {data['fit_loss']:>10.3f} "
            f"{disagreement:>9.4f}"
        )
    print(f"  explicit held-out test points: {n_test}")


def _masked(values, mask):
    result = values.detach().cpu().numpy().copy()
    result[~mask] = np.nan
    return result.reshape(GRID_SIZE, GRID_SIZE)


def plot_eigenfunctions(
    path,
    x_axis,
    y_axis,
    mask,
    weights,
    fem_values,
    fem_functions,
    systems,
):
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    normalized_weights = weights / weights.sum()
    fig, axes = plt.subplots(3, N_MODES - 1, figsize=(3.0 * (N_MODES - 1), 8.5))
    rows = (
        ("FEM", fem_values, fem_functions),
        ("KLAP", systems["klap"].eigenvalues, systems["klap"].eigenfunctions),
        ("SpIN", systems["spin"].eigenvalues, systems["spin"].eigenfunctions),
    )
    for column, mode in enumerate(range(1, N_MODES)):
        reference = fem_functions[:, mode]
        aligned = []
        for name, values, functions in rows:
            function = functions[:, mode]
            if float(torch.sum(normalized_weights * reference * function)) < 0:
                function = -function
            aligned.append((name, values, function))
        scale = max(float(item[2].abs().max().cpu()) for item in aligned)
        for row, (name, values, function) in enumerate(aligned):
            axes[row, column].imshow(
                _masked(function, mask),
                extent=extent,
                origin="lower",
                cmap="coolwarm",
                vmin=-scale,
                vmax=scale,
                aspect="equal",
            )
            axes[row, column].set_title(
                f"{name} mode {mode}\n$\\lambda$={float(values[mode]):.3f}"
            )
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.suptitle("Weighted-Laplacian eigenfunctions", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_gp_results(
    path,
    x_axis,
    y_axis,
    p_vals,
    mask,
    points_np,
    idx_train,
    y_true,
    curve_x,
    curve_y,
    curve_s,
    predictions,
    diagnostics,
    regime_name,
):
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    vmin = float(y_true[torch.as_tensor(mask, device=y_true.device)].min().cpu())
    vmax = float(y_true[torch.as_tensor(mask, device=y_true.device)].max().cpu())
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    panels = [
        (axes[0, 0], p_vals.reshape(GRID_SIZE, GRID_SIZE), "spiral density", "viridis", None, None),
        (axes[0, 1], _masked(y_true, mask), "true function", "coolwarm", vmin, vmax),
        (axes[0, 2], _masked(predictions["fem"].latent_mean, mask), "FEM GP", "coolwarm", vmin, vmax),
        (axes[0, 3], _masked(predictions["klap"].latent_mean, mask), "KLAP GP", "coolwarm", vmin, vmax),
        (axes[1, 0], _masked(predictions["spin"].latent_mean, mask), "SpIN GP", "coolwarm", vmin, vmax),
        (axes[1, 1], _masked(predictions["rbf"].latent_mean, mask), "Euclidean RBF GP", "coolwarm", vmin, vmax),
    ]
    train_np = idx_train.detach().cpu().numpy()
    for ax, values, title, cmap, panel_min, panel_max in panels:
        image = ax.imshow(
            values,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=panel_min,
            vmax=panel_max,
            aspect="equal",
        )
        ax.plot(curve_x, curve_y, "k-", lw=0.6, alpha=0.45)
        ax.scatter(
            points_np[train_np, 0],
            points_np[train_np, 1],
            facecolors="none",
            edgecolors="black",
            s=30,
        )
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    curve_slice = slice(None, None, 4)
    curve_indices = torch.as_tensor(
        [
            nearest_grid_index(points_np, (x, y))
            for x, y in zip(curve_x[curve_slice], curve_y[curve_slice])
        ],
        dtype=torch.long,
        device=y_true.device,
    )
    ax = axes[1, 2]
    ax.plot(curve_s[curve_slice], y_true[curve_indices].cpu(), "k:", lw=2, label="truth")
    styles = {"fem": "b-", "klap": "C1-", "spin": "C2--", "rbf": "r-"}
    for name in ("fem", "klap", "spin", "rbf"):
        ax.plot(
            curve_s[curve_slice],
            predictions[name].latent_mean[curve_indices].detach().cpu(),
            styles[name],
            lw=1.6,
            label=name.upper(),
        )
    ax.set_title("posterior means along arclength")
    ax.set_xlabel("normalized arclength")
    ax.set_ylabel("f")
    ax.legend(fontsize=8)

    axes[1, 3].axis("off")
    summary = [f"{regime_name.capitalize()}-regime tube diagnostics"]
    for name in ("fem", "klap", "spin", "rbf"):
        tube = diagnostics[name]["regions"]["tube"]
        summary.append(
            f"{name.upper():4s}: RMSE {tube['rmse']:.3f}, "
            f"NLPD {tube['nlpd']:.3f}, coverage {tube['coverage']:.2f}"
        )
    axes[1, 3].text(0.02, 0.95, "\n".join(summary), va="top", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--regime", choices=("quadrature", "sampled", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument(
        "--include-subspace",
        action="store_true",
        help="also run the earlier augmented-Lagrangian neural baseline",
    )
    parser.add_argument(
        "--quick", action="store_true", help="use reduced smoke-test settings"
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x_axis, y_axis, _, _, points_np = make_grid(GRID_SIZE, DOMAIN_LIMIT)
    spacing = x_axis[1] - x_axis[0]
    _, curve_x, curve_y, curve_s = make_spiral_curve()
    distance, _, nearest_s = nearest_spiral_coordinates(
        points_np, curve_x, curve_y, curve_s
    )
    p_vals = spiral_density(distance)
    tube_mask_np = p_vals > TUBE_THRESHOLD
    points = torch.as_tensor(points_np, dtype=DTYPE, device=DEVICE)
    grid_weights = torch.as_tensor(p_vals, dtype=DTYPE, device=DEVICE)

    print("Computing the fixed 61-by-61 FEM reference...")
    fem_values, fem_functions = weighted_laplacian_eigendecomposition_2d(
        p_vals.reshape(GRID_SIZE, GRID_SIZE),
        spacing,
        n_modes=N_MODES,
        dtype=DTYPE,
        device=DEVICE,
    )
    # SciPy normalizes against the unnormalized grid mass (including cell
    # area). Convert to the probability convention shared by KLAP and SpIN.
    fem_functions = normalize_eigenfunction_columns(
        fem_functions, grid_weights, constant_mode=True
    )

    validation_x = 0.5 * (x_axis[:-1] + x_axis[1:])
    validation_y = 0.5 * (y_axis[:-1] + y_axis[1:])
    validation_xx, validation_yy = np.meshgrid(validation_x, validation_y, indexing="xy")
    grid_validation_np = np.column_stack((validation_xx.ravel(), validation_yy.ravel()))
    validation_distance, _, _ = nearest_spiral_coordinates(
        grid_validation_np, curve_x, curve_y, curve_s
    )
    grid_validation_weights = torch.as_tensor(
        spiral_density(validation_distance), dtype=DTYPE, device=DEVICE
    )
    grid_validation = torch.as_tensor(
        grid_validation_np, dtype=DTYPE, device=DEVICE
    )

    regimes = {}
    if args.regime in ("quadrature", "both"):
        regimes["quadrature"] = fit_regime(
            "quadrature",
            points,
            grid_weights,
            grid_validation,
            grid_validation_weights,
            points,
            grid_weights,
            fem_values,
            fem_functions,
            seed=args.seed,
            quick=args.quick,
            include_subspace=args.include_subspace,
        )
        print_eigensolver_table("quadrature", regimes["quadrature"], fem_values)

    if args.regime in ("sampled", "both"):
        sample_count = 256 if args.quick else 4_096
        reference_np = sample_spiral_density(
            sample_count, curve_x, curve_y, curve_s, seed=args.seed
        )
        validation_np = sample_spiral_density(
            sample_count, curve_x, curve_y, curve_s, seed=args.seed + 1
        )
        reference = torch.as_tensor(reference_np, dtype=DTYPE, device=DEVICE)
        validation = torch.as_tensor(validation_np, dtype=DTYPE, device=DEVICE)
        uniform = torch.ones(sample_count, dtype=DTYPE, device=DEVICE)
        regimes["sampled"] = fit_regime(
            "sampled",
            reference,
            uniform,
            validation,
            uniform,
            points,
            grid_weights,
            fem_values,
            fem_functions,
            seed=args.seed,
            quick=args.quick,
            include_subspace=args.include_subspace,
        )
        print_eigensolver_table("sampled", regimes["sampled"], fem_values)

    primary_name = "sampled" if "sampled" in regimes else "quadrature"
    primary = regimes[primary_name]
    y_true = torch.as_tensor(spiral_target(nearest_s), dtype=DTYPE, device=DEVICE)
    tube_mask = torch.as_tensor(tube_mask_np, dtype=torch.bool, device=DEVICE)
    idx_train = choose_training_indices(points_np, curve_x, curve_y, curve_s)
    idx_test = choose_test_indices(points_np, curve_x, curve_y, curve_s, idx_train)
    train_mask = torch.zeros(len(points), dtype=torch.bool, device=DEVICE)
    train_mask[idx_train] = True
    masks = {
        "tube": tube_mask & ~train_mask,
        "outer": tube_mask
        & torch.as_tensor(nearest_s < 0.38, device=DEVICE)
        & ~train_mask,
        "inner": tube_mask
        & torch.as_tensor(nearest_s > 0.62, device=DEVICE)
        & ~train_mask,
    }
    noise_generator = torch.Generator(device=DEVICE).manual_seed(args.seed)
    y_train = y_true[idx_train] + NOISE_SD * torch.randn(
        len(idx_train), dtype=DTYPE, device=DEVICE, generator=noise_generator
    )
    print("\nFitting matched seven-mode GPs...")
    gp_fits = _fit_gp_models(
        y_train, idx_train, points, primary, fem_values, fem_functions, args.quick
    )
    predictions, gp_diagnostics = evaluate_gp_models(
        gp_fits, points, idx_test, y_true, masks
    )
    print_gp_table(gp_diagnostics, len(idx_test), primary_name)

    figures = Path(__file__).resolve().parent / "figs"
    eigen_path = figures / "example_spiral_eigenpairs.png"
    gp_path = figures / "example_spiral.png"
    plot_eigenfunctions(
        eigen_path,
        x_axis,
        y_axis,
        tube_mask_np,
        grid_weights,
        fem_values,
        fem_functions,
        primary,
    )
    plot_gp_results(
        gp_path,
        x_axis,
        y_axis,
        p_vals,
        tube_mask_np,
        points_np,
        idx_train,
        y_true,
        curve_x,
        curve_y,
        curve_s,
        predictions,
        gp_diagnostics,
        primary_name,
    )
    print(f"\nSaved primary GP figure: {gp_path}")
    print(f"Saved eigenfunction figure: {eigen_path}")


if __name__ == "__main__":
    main()
