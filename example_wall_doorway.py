import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.fem import weighted_laplacian_eigendecomposition_2d
from src.gp import dense_prior_covariance, predict_exact_gp
from src.grid import gradient_norm, image, make_grid, nearest_grid_index
from src.kernels import kernel_to_correlation
from src.metrics import average_pair_correlation, evaluate_predictions
from src.training import fit_density_gp, fit_euclidean_gp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.manual_seed(4)

GRID_SIZE = 41
DOMAIN_LIMIT = 2.0
N_MODES = 120
MATERN_ALPHA = 1.5
DOOR_Y = 1.0


def doorway_density(xx, yy):
    wall_width = 0.13
    door_width = 0.28
    floor = 0.035
    wall_depth = 0.97

    wall = np.exp(-0.5 * (xx / wall_width) ** 2)
    door = np.exp(-0.5 * ((yy - DOOR_Y) / door_width) ** 2)
    barrier = wall * (1.0 - door)

    return floor + 1.0 - wall_depth * barrier


def doorway_target(xx, yy):
    door = np.exp(-0.5 * ((yy - DOOR_Y) / 0.35) ** 2)
    transition_width = 0.055 + 0.42 * door
    step = -np.tanh(xx / transition_width)

    left_room_variation = 0.10 * np.sin(np.pi * (yy + DOMAIN_LIMIT) / (2.0 * DOMAIN_LIMIT))
    right_room_variation = -0.10 * np.cos(np.pi * (yy + DOMAIN_LIMIT) / (2.0 * DOMAIN_LIMIT))

    return step + np.where(xx < 0.0, left_room_variation, right_room_variation)


def training_indices(points):
    label_coords = []
    for x_coord in (-1.55, -1.10, -0.65, -0.32):
        for y_coord in (-1.55, -0.65, 0.25, DOOR_Y, 1.65):
            label_coords.append((x_coord, y_coord))
    for x_coord in (0.32, 0.65, 1.10, 1.55):
        for y_coord in (-1.55, -0.65, 0.25, DOOR_Y, 1.65):
            label_coords.append((x_coord, y_coord))

    return torch.tensor(
        [nearest_grid_index(points, coord) for coord in label_coords],
        dtype=torch.long,
        device=device,
    )


def diagnostic_pairs(points):
    pair_half_width = 0.45
    wall_pairs = [
        (nearest_grid_index(points, (-pair_half_width, y)), nearest_grid_index(points, (pair_half_width, y)))
        for y in (-1.60, -1.20, -0.80, -0.40, 0.00, 0.40, 1.60)
    ]
    doorway_pairs = [
        (nearest_grid_index(points, (-pair_half_width, y)), nearest_grid_index(points, (pair_half_width, y)))
        for y in (0.80, 1.00, 1.20)
    ]
    within_room_pairs = [
        (nearest_grid_index(points, (-1.40, y)), nearest_grid_index(points, (-1.40 + 2.0 * pair_half_width, y)))
        for y in (-1.60, -0.80, 0.00, 0.80, 1.60)
    ]
    return wall_pairs, doorway_pairs, within_room_pairs


def region_masks(points, idx_train):
    train_mask = torch.zeros(len(points), dtype=torch.bool, device=device)
    train_mask[idx_train] = True
    test_mask = ~train_mask

    x_coord = torch.as_tensor(points[:, 0], dtype=dtype, device=device)
    y_coord = torch.as_tensor(points[:, 1], dtype=dtype, device=device)

    wall_mask = (torch.abs(x_coord) < 0.25) & (torch.abs(y_coord - DOOR_Y) > 0.45)
    doorway_mask = (torch.abs(x_coord) < 0.35) & (torch.abs(y_coord - DOOR_Y) < 0.28)
    room_mask = torch.abs(x_coord) > 0.45

    return {
        "overall": test_mask,
        "rooms": test_mask & room_mask,
        "wall": test_mask & wall_mask,
        "doorway": test_mask & doorway_mask,
    }


def plot_results(
    *,
    output_path,
    x_axis,
    y_axis,
    grid_spacing,
    p_vals,
    f_true,
    points,
    idx_train,
    mean_density,
    var_density,
    mean_rbf,
    var_rbf,
    corr_density,
    corr_rbf,
    diagnostics,
):
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    train_points = points[idx_train.detach().cpu().numpy()]
    vmin = min(float(f_true.min().cpu()), float(mean_density.min().cpu()), float(mean_rbf.min().cpu()))
    vmax = max(float(f_true.max().cpu()), float(mean_density.max().cpu()), float(mean_rbf.max().cpu()))
    err_diff = torch.abs(mean_rbf - f_true) - torch.abs(mean_density - f_true)
    max_abs_err_diff = float(torch.abs(err_diff).max().cpu())
    grad_diff = gradient_norm(mean_density, grid_spacing, GRID_SIZE) - gradient_norm(mean_rbf, grid_spacing, GRID_SIZE)
    max_abs_grad_diff = float(torch.abs(grad_diff).max().cpu())

    fig, axs = plt.subplots(2, 4, figsize=(21, 10))

    panels = [
        (axs[0, 0], image(torch.as_tensor(p_vals).cpu().numpy(), GRID_SIZE), "ambient density p(x,y)", "viridis", None, None),
        (axs[0, 1], image(f_true.detach().cpu().numpy(), GRID_SIZE), "true function", "coolwarm", vmin, vmax),
        (axs[0, 2], image(mean_rbf.detach().cpu().numpy(), GRID_SIZE), "RBF GP mean", "coolwarm", vmin, vmax),
        (axs[0, 3], image(mean_density.detach().cpu().numpy(), GRID_SIZE), "density GP mean", "coolwarm", vmin, vmax),
        (
            axs[1, 0],
            image(torch.sqrt(torch.clamp(var_rbf, min=0.0)).detach().cpu().numpy(), GRID_SIZE),
            "RBF posterior std",
            "magma",
            None,
            None,
        ),
        (
            axs[1, 1],
            image(torch.sqrt(torch.clamp(var_density, min=0.0)).detach().cpu().numpy(), GRID_SIZE),
            "density GP posterior std",
            "magma",
            None,
            None,
        ),
        (
            axs[1, 2],
            image(err_diff.detach().cpu().numpy(), GRID_SIZE),
            "|error RBF| - |error dGP|",
            "coolwarm",
            -max_abs_err_diff,
            max_abs_err_diff,
        ),
        (
            axs[1, 3],
            image(grad_diff.detach().cpu().numpy(), GRID_SIZE),
            "|grad dGP mean| - |grad RBF mean|",
            "coolwarm",
            -max_abs_grad_diff,
            max_abs_grad_diff,
        ),
    ]

    for ax, values, title, cmap, panel_vmin, panel_vmax in panels:
        im = ax.imshow(
            values,
            extent=extent,
            origin="lower",
            cmap=cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.scatter(
            train_points[:, 0],
            train_points[:, 1],
            s=28,
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            label="labels",
        )
        ax.axvline(0.0, color="white" if "density" in title else "gray", ls="--", lw=0.8, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    summary = (
        f"overall RMSE: RBF={diagnostics['rbf']['overall']['rmse']:.3f}, dGP={diagnostics['density']['overall']['rmse']:.3f}\n"
        f"wall RMSE: RBF={diagnostics['rbf']['wall']['rmse']:.3f}, dGP={diagnostics['density']['wall']['rmse']:.3f}\n"
        f"rooms RMSE: RBF={diagnostics['rbf']['rooms']['rmse']:.3f}, dGP={diagnostics['density']['rooms']['rmse']:.3f}\n"
        f"mean |grad| wall: RBF={diagnostics['rbf_grad_wall']:.2f}, dGP={diagnostics['density_grad_wall']:.2f}\n"
        f"mean |grad| rooms: RBF={diagnostics['rbf_grad_rooms']:.2f}, dGP={diagnostics['density_grad_rooms']:.2f}"
    )
    axs[1, 2].text(
        0.03,
        0.97,
        summary,
        transform=axs[1, 2].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def print_diagnostics(diagnostics):
    print("\nHyperparameters")
    print(
        f"  dGP kappa={diagnostics['density_kappa']:.4f}, "
        f"alpha={diagnostics['density_alpha']:.2f}, "
        f"sigma={diagnostics['density_sigma']:.4f}, loss={diagnostics['density_loss']:.4f}"
    )
    print(
        f"  RBF lengthscale={diagnostics['rbf_lengthscale']:.4f}, "
        f"sigma={diagnostics['rbf_sigma']:.4f}, loss={diagnostics['rbf_loss']:.4f}"
    )

    print("\nDensity and correlation diagnostics")
    print(f"  min density={diagnostics['density_min']:.4f}, max density={diagnostics['density_max']:.4f}")
    print(f"  across-wall corr outside doorway: RBF={diagnostics['rbf_wall_corr']:.4f}, dGP={diagnostics['density_wall_corr']:.4f}")
    print(f"  across-doorway corr:             RBF={diagnostics['rbf_door_corr']:.4f}, dGP={diagnostics['density_door_corr']:.4f}")
    print(f"  within-room corr:                RBF={diagnostics['rbf_within_corr']:.4f}, dGP={diagnostics['density_within_corr']:.4f}")
    print(
        f"  mean |grad mean| in wall:        "
        f"RBF={diagnostics['rbf_grad_wall']:.4f}, dGP={diagnostics['density_grad_wall']:.4f}, "
        f"true={diagnostics['true_grad_wall']:.4f}"
    )
    print(
        f"  mean |grad mean| in rooms:       "
        f"RBF={diagnostics['rbf_grad_rooms']:.4f}, dGP={diagnostics['density_grad_rooms']:.4f}, "
        f"true={diagnostics['true_grad_rooms']:.4f}"
    )

    print("\nDiagnostics on held-out grid points")
    header = f"{'region':<10} {'model':<8} {'RMSE':>9} {'NLPD':>9} {'95% cov':>9}"
    print(header)
    print("-" * len(header))
    for region in ("overall", "rooms", "wall", "doorway"):
        for model in ("rbf", "density"):
            vals = diagnostics[model][region]
            print(f"{region:<10} {model:<8} {vals['rmse']:>9.4f} {vals['nlpd']:>9.4f} {vals['coverage']:>9.3f}")


def main():
    x_axis, y_axis, xx, yy, points_np = make_grid(GRID_SIZE, DOMAIN_LIMIT)
    grid_spacing = x_axis[1] - x_axis[0]
    p_vals = doorway_density(xx, yy)
    f_vals = doorway_target(xx, yy)

    eigenvalues, eigenvectors = weighted_laplacian_eigendecomposition_2d(
        p_vals,
        grid_spacing,
        n_modes=N_MODES,
        dtype=dtype,
        device=device,
    )
    points = torch.as_tensor(points_np, dtype=dtype, device=device)
    f_true = torch.as_tensor(f_vals.ravel(), dtype=dtype, device=device)

    idx_train = training_indices(points_np)
    noise_var = 0.04**2
    y_train = f_true[idx_train] + math.sqrt(noise_var) * torch.randn(len(idx_train), dtype=dtype, device=device)

    grid_indices = torch.arange(len(points), dtype=dtype, device=device).unsqueeze(-1)
    density_fit = fit_density_gp(
        y_train=y_train,
        train_grid_indices=idx_train,
        noise_var=noise_var,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kind="matern",
        spectral_raw_inits=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        sigma_raw_inits=(-1.0, 0.0, 1.0),
        steps=350,
        lr=0.05,
        alpha=MATERN_ALPHA,
    )
    rbf_fit = fit_euclidean_gp(
        y_train=y_train,
        x_train=points[idx_train],
        noise_var=noise_var,
        kind="rbf",
        lengthscale_raw_inits=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0),
        sigma_raw_inits=(-1.0, 0.0, 1.0),
        steps=350,
        lr=0.05,
    )

    density_model = density_fit.model
    rbf_model = rbf_fit.model
    density_prediction = predict_exact_gp(density_model, grid_indices, noise_var)
    rbf_prediction = predict_exact_gp(rbf_model, points, noise_var)
    kernel_density = dense_prior_covariance(density_model, grid_indices)
    kernel_rbf = dense_prior_covariance(rbf_model, points)

    with torch.no_grad():
        mean_density = density_prediction.latent_mean
        var_density = density_prediction.latent_variance
        mean_rbf = rbf_prediction.latent_mean
        var_rbf = rbf_prediction.latent_variance

        corr_density = kernel_to_correlation(kernel_density)
        corr_rbf = kernel_to_correlation(kernel_rbf)

        wall_pairs, doorway_pairs, within_room_pairs = diagnostic_pairs(points_np)
        regions = region_masks(points_np, idx_train)
        grad_true = gradient_norm(f_true, grid_spacing, GRID_SIZE)
        grad_density = gradient_norm(mean_density, grid_spacing, GRID_SIZE)
        grad_rbf = gradient_norm(mean_rbf, grid_spacing, GRID_SIZE)

        diagnostics = {
            "density_alpha": MATERN_ALPHA,
            "density_kappa": float(density_model.covar_module.base_kernel.kappa.cpu()),
            "density_sigma": float(density_model.covar_module.outputscale.sqrt().cpu()),
            "density_loss": density_fit.total_negative_mll,
            "rbf_lengthscale": float(rbf_model.covar_module.base_kernel.lengthscale.cpu()),
            "rbf_sigma": float(rbf_model.covar_module.outputscale.sqrt().cpu()),
            "rbf_loss": rbf_fit.total_negative_mll,
            "density_min": float(p_vals.min()),
            "density_max": float(p_vals.max()),
            "density_wall_corr": float(average_pair_correlation(corr_density, wall_pairs).cpu()),
            "rbf_wall_corr": float(average_pair_correlation(corr_rbf, wall_pairs).cpu()),
            "density_door_corr": float(average_pair_correlation(corr_density, doorway_pairs).cpu()),
            "rbf_door_corr": float(average_pair_correlation(corr_rbf, doorway_pairs).cpu()),
            "density_within_corr": float(average_pair_correlation(corr_density, within_room_pairs).cpu()),
            "rbf_within_corr": float(average_pair_correlation(corr_rbf, within_room_pairs).cpu()),
            "density_grad_wall": float(grad_density[regions["wall"]].mean().cpu()),
            "rbf_grad_wall": float(grad_rbf[regions["wall"]].mean().cpu()),
            "true_grad_wall": float(grad_true[regions["wall"]].mean().cpu()),
            "density_grad_rooms": float(grad_density[regions["rooms"]].mean().cpu()),
            "rbf_grad_rooms": float(grad_rbf[regions["rooms"]].mean().cpu()),
            "true_grad_rooms": float(grad_true[regions["rooms"]].mean().cpu()),
            "density": evaluate_predictions(
                f_true,
                mean_density,
                density_prediction.observed_variance,
                regions,
            ),
            "rbf": evaluate_predictions(
                f_true,
                mean_rbf,
                rbf_prediction.observed_variance,
                regions,
            ),
        }

    output_path = Path(__file__).resolve().parent / "figs" / "example_wall_doorway.png"
    plot_results(
        output_path=output_path,
        x_axis=x_axis,
        y_axis=y_axis,
        grid_spacing=grid_spacing,
        p_vals=p_vals,
        f_true=f_true,
        points=points_np,
        idx_train=idx_train,
        mean_density=mean_density,
        var_density=var_density,
        mean_rbf=mean_rbf,
        var_rbf=var_rbf,
        corr_density=corr_density,
        corr_rbf=corr_rbf,
        diagnostics=diagnostics,
    )
    print_diagnostics(diagnostics)
    print(f"\nSaved figure: {output_path}")


if __name__ == "__main__":
    main()
