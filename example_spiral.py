import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.fem import weighted_laplacian_eigendecomposition_2d
from src.gp import (
    dense_prior_covariance,
    gaussian_nll,
    marginal_nll,
    predict_exact_gp,
)
from src.grid import image, make_grid, nearest_grid_index
from src.kernels import kernel_to_correlation
from src.metrics import (
    average_pair_correlation,
    average_pair_distance,
    average_pair_label_difference,
    evaluate_predictions,
)
from src.training import fit_density_gp, fit_euclidean_gp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.manual_seed(12)

GRID_SIZE = 61
DOMAIN_LIMIT = 2.05
N_MODES = 180
N_TRAIN = 34
N_TEST = 120
NOISE_SD = 0.035

SPIRAL_TURNS = 3.45
R_OUTER = 1.85
R_INNER = 0.24
TUBE_WIDTH = 0.05
DENSITY_FLOOR = 0.001
TUBE_THRESHOLD = 0.20


def make_spiral_curve(n_curve=2400):
    t_max = 2.0 * math.pi * SPIRAL_TURNS
    t = np.linspace(0.0, t_max, n_curve)
    radius = R_OUTER - (R_OUTER - R_INNER) * t / t_max
    x = radius * np.cos(t)
    y = radius * np.sin(t)

    arclength = np.zeros_like(t)
    arclength[1:] = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))
    arclength = arclength / arclength[-1]
    return t, x, y, arclength


def nearest_spiral_coordinates(points, curve_x, curve_y, curve_s, chunk_size=500):
    curve = np.column_stack([curve_x, curve_y])
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
    # Smooth and strictly increasing along the spiral.
    return 2.0 * normalized_arclength - 1.0 + 0.12 * np.sin(2.0 * math.pi * normalized_arclength)


def choose_training_indices(points, curve_x, curve_y, curve_s, n_train=N_TRAIN):
    indices = []
    for target_s in np.linspace(0.035, 0.965, n_train):
        curve_idx = int(np.argmin(np.abs(curve_s - target_s)))
        indices.append(nearest_grid_index(points, (curve_x[curve_idx], curve_y[curve_idx])))

    # Multiple curve points can snap to the same grid point; preserve order and
    # drop duplicates.
    unique_indices = list(dict.fromkeys(indices))
    return torch.tensor(unique_indices, dtype=torch.long, device=device)


def choose_test_indices(points, curve_x, curve_y, curve_s, idx_train, n_test=N_TEST):
    train_set = set(idx_train.detach().cpu().tolist())
    indices = []

    for target_s in np.linspace(0.015, 0.985, n_test * 2):
        curve_idx = int(np.argmin(np.abs(curve_s - target_s)))
        grid_idx = nearest_grid_index(points, (curve_x[curve_idx], curve_y[curve_idx]))
        if grid_idx not in train_set and grid_idx not in indices:
            indices.append(grid_idx)
        if len(indices) == n_test:
            break

    return torch.tensor(indices, dtype=torch.long, device=device)


def diagnostic_pairs(points, curve_t, curve_x, curve_y, curve_s):
    t_max = curve_t[-1]

    adjacent_turn_pairs = []
    for t_ref in np.linspace(0.7 * math.pi, t_max - 2.7 * math.pi, 9):
        idx_a = int(np.argmin(np.abs(curve_t - t_ref)))
        idx_b = int(np.argmin(np.abs(curve_t - (t_ref + 2.0 * math.pi))))
        adjacent_turn_pairs.append(
            (
                nearest_grid_index(points, (curve_x[idx_a], curve_y[idx_a])),
                nearest_grid_index(points, (curve_x[idx_b], curve_y[idx_b])),
            )
        )

    along_spiral_pairs = []
    for s_ref in np.linspace(0.10, 0.87, 9):
        idx_a = int(np.argmin(np.abs(curve_s - s_ref)))
        idx_b = int(np.argmin(np.abs(curve_s - (s_ref + 0.025))))
        along_spiral_pairs.append(
            (
                nearest_grid_index(points, (curve_x[idx_a], curve_y[idx_a])),
                nearest_grid_index(points, (curve_x[idx_b], curve_y[idx_b])),
            )
        )

    return adjacent_turn_pairs, along_spiral_pairs


def masked_image(values, tube_mask, n=GRID_SIZE):
    arr = values.detach().cpu().numpy().copy()
    arr[~tube_mask] = np.nan
    return arr.reshape(n, n)


def plot_results(
    *,
    output_path,
    x_axis,
    y_axis,
    p_vals,
    tube_mask_np,
    curve_x,
    curve_y,
    points,
    idx_train,
    y_true,
    curve_s_plot,
    curve_grid_indices,
    nearest_s,
    mean_density,
    var_density,
    mean_rbf,
    var_rbf,
    corr_density,
    corr_rbf,
    ref_idx,
    diagnostics,
):
    extent = [x_axis.min(), x_axis.max(), y_axis.min(), y_axis.max()]
    train_np = idx_train.detach().cpu().numpy()
    train_points = points[train_np]

    cmap_density = plt.get_cmap("viridis").copy()
    cmap_coolwarm = plt.get_cmap("coolwarm").copy()
    cmap_magma = plt.get_cmap("magma").copy()
    for cmap in (cmap_density, cmap_coolwarm, cmap_magma):
        cmap.set_bad("white")

    vmin = float(y_true[tube_mask_np].min().cpu())
    vmax = float(y_true[tube_mask_np].max().cpu())
    err_diff = torch.abs(mean_rbf - y_true) - torch.abs(mean_density - y_true)
    max_err = float(torch.nan_to_num(torch.abs(err_diff)).max().cpu())

    fig, axs = plt.subplots(2, 4, figsize=(21, 10))
    image_panels = [
        (axs[0, 0], image(p_vals, GRID_SIZE), "ambient spiral density", cmap_density, None, None),
        (axs[0, 1], masked_image(y_true, tube_mask_np), "true function on spiral tube", cmap_coolwarm, vmin, vmax),
        (axs[0, 2], masked_image(mean_rbf, tube_mask_np), "RBF GP mean", cmap_coolwarm, vmin, vmax),
        (axs[0, 3], masked_image(mean_density, tube_mask_np), "density GP mean", cmap_coolwarm, vmin, vmax),
        (
            axs[1, 0],
            masked_image(corr_rbf[ref_idx], tube_mask_np),
            "RBF corr from reference",
            cmap_magma,
            0.0,
            1.0,
        ),
        (
            axs[1, 1],
            masked_image(corr_density[ref_idx], tube_mask_np),
            "dGP corr from reference",
            cmap_magma,
            0.0,
            1.0,
        ),
        (
            axs[1, 2],
            masked_image(err_diff, tube_mask_np),
            "|error RBF| - |error dGP|",
            cmap_coolwarm,
            -max_err,
            max_err,
        ),
    ]

    for ax, values, title, cmap, panel_vmin, panel_vmax in image_panels:
        im = ax.imshow(
            values,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=panel_vmin,
            vmax=panel_vmax,
            aspect="equal",
        )
        ax.plot(curve_x, curve_y, color="black", lw=0.7, alpha=0.45)
        ax.scatter(
            train_points[:, 0],
            train_points[:, 1],
            facecolors="none",
            edgecolors="black",
            linewidths=1.1,
            s=38,
        )
        ax.scatter(
            points[ref_idx, 0],
            points[ref_idx, 1],
            marker="x",
            s=90,
            color="yellow",
            linewidths=2.0,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axs[1, 3]
    curve_idx_torch = torch.as_tensor(curve_grid_indices, dtype=torch.long, device=y_true.device)
    ax.plot(curve_s_plot, y_true[curve_idx_torch].detach().cpu().numpy(), "k:", lw=2, label="true")
    ax.plot(curve_s_plot, mean_rbf[curve_idx_torch].detach().cpu().numpy(), "r-", lw=1.8, label="RBF")
    ax.plot(curve_s_plot, mean_density[curve_idx_torch].detach().cpu().numpy(), "b-", lw=1.8, label="dGP")
    ax.scatter(
        nearest_s[train_np],
        y_true[idx_train].detach().cpu().numpy(),
        facecolors="none",
        edgecolors="black",
        s=32,
        zorder=5,
        label="labels",
    )
    ax.set_title("posterior along spiral arclength")
    ax.set_xlabel("normalized arclength")
    ax.set_ylabel("f")
    ax.legend(loc="best", fontsize=9)

    summary = (
        f"tube RMSE: RBF={diagnostics['rbf']['tube']['rmse']:.3f}, "
        f"dGP={diagnostics['density']['tube']['rmse']:.3f}\n"
        f"inner RMSE: RBF={diagnostics['rbf']['inner']['rmse']:.3f}, "
        f"dGP={diagnostics['density']['inner']['rmse']:.3f}\n"
        f"outer RMSE: RBF={diagnostics['rbf']['outer']['rmse']:.3f}, "
        f"dGP={diagnostics['density']['outer']['rmse']:.3f}\n"
        f"corr along/cross: RBF={diagnostics['rbf_corr_ratio']:.3f}, "
        f"dGP={diagnostics['density_corr_ratio']:.3f}"
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
        f"  dGP heat tau={diagnostics['density_tau']:.4f}, "
        f"sigma={diagnostics['density_sigma']:.4f}, loss={diagnostics['density_loss']:.4f}"
    )
    print(
        f"  RBF lengthscale={diagnostics['rbf_lengthscale']:.4f}, "
        f"sigma={diagnostics['rbf_sigma']:.4f}, loss={diagnostics['rbf_loss']:.4f}"
    )

    print("\nGeometry diagnostics")
    print(f"  density min={diagnostics['density_min']:.4f}, max={diagnostics['density_max']:.4f}")
    print(f"  held-out tube points={diagnostics['n_tube_test']}")
    print(
        f"  adjacent-turn pairs: mean Euclidean distance={diagnostics['cross_turn_distance']:.4f}, "
        f"mean label difference={diagnostics['cross_turn_label_diff']:.4f}"
    )
    print(
        f"  along-spiral pairs:  mean Euclidean distance={diagnostics['along_distance']:.4f}, "
        f"mean label difference={diagnostics['along_label_diff']:.4f}"
    )
    print(
        f"  adjacent-turn corr: RBF={diagnostics['rbf_cross_turn_corr']:.4f}, "
        f"dGP={diagnostics['density_cross_turn_corr']:.4f}"
    )
    print(
        f"  along-spiral corr:  RBF={diagnostics['rbf_along_corr']:.4f}, "
        f"dGP={diagnostics['density_along_corr']:.4f}"
    )
    print(
        f"  along/cross corr ratio: RBF={diagnostics['rbf_corr_ratio']:.4f}, "
        f"dGP={diagnostics['density_corr_ratio']:.4f}"
    )
    print("\nHeld-out test-set predictive NLL")
    print(f"  n_test={diagnostics['n_explicit_test']}")
    print(
        f"  joint NLL:      RBF={diagnostics['rbf_test_joint_nll']:.4f}, "
        f"dGP={diagnostics['density_test_joint_nll']:.4f}"
    )
    print(
        f"  joint NLL / pt: RBF={diagnostics['rbf_test_joint_nll_per_point']:.4f}, "
        f"dGP={diagnostics['density_test_joint_nll_per_point']:.4f}"
    )
    print(
        f"  marginal NLL / pt: RBF={diagnostics['rbf_test_marginal_nll']:.4f}, "
        f"dGP={diagnostics['density_test_marginal_nll']:.4f}"
    )

    print("\nDiagnostics on held-out spiral-tube grid points")
    header = f"{'region':<10} {'model':<8} {'RMSE':>9} {'NLPD':>9} {'95% cov':>9}"
    print(header)
    print("-" * len(header))
    for region in ("tube", "outer", "inner"):
        for model in ("rbf", "density"):
            vals = diagnostics[model][region]
            print(f"{region:<10} {model:<8} {vals['rmse']:>9.4f} {vals['nlpd']:>9.4f} {vals['coverage']:>9.3f}")


def main():
    x_axis, y_axis, xx, yy, points_np = make_grid(GRID_SIZE, DOMAIN_LIMIT)
    grid_spacing = x_axis[1] - x_axis[0]
    curve_t, curve_x, curve_y, curve_s = make_spiral_curve()

    distance_to_spiral, _, nearest_s = nearest_spiral_coordinates(points_np, curve_x, curve_y, curve_s)
    p_vals = spiral_density(distance_to_spiral)
    target_vals = spiral_target(nearest_s)
    tube_mask_np = p_vals > TUBE_THRESHOLD

    eigenvalues, eigenvectors = weighted_laplacian_eigendecomposition_2d(
        p_vals.reshape(GRID_SIZE, GRID_SIZE),
        grid_spacing,
        n_modes=N_MODES,
        dtype=dtype,
        device=device,
    )

    points = torch.as_tensor(points_np, dtype=dtype, device=device)
    y_true = torch.as_tensor(target_vals, dtype=dtype, device=device)
    tube_mask = torch.as_tensor(tube_mask_np, dtype=torch.bool, device=device)
    inner_mask = torch.as_tensor(tube_mask_np & (nearest_s > 0.62), dtype=torch.bool, device=device)
    outer_mask = torch.as_tensor(tube_mask_np & (nearest_s < 0.38), dtype=torch.bool, device=device)

    idx_train = choose_training_indices(points_np, curve_x, curve_y, curve_s)
    idx_test = choose_test_indices(points_np, curve_x, curve_y, curve_s, idx_train)
    train_mask = torch.zeros(len(points_np), dtype=torch.bool, device=device)
    train_mask[idx_train] = True
    test_tube_mask = tube_mask & (~train_mask)

    noise_var = NOISE_SD**2
    y_train = y_true[idx_train] + NOISE_SD * torch.randn(len(idx_train), dtype=dtype, device=device)

    grid_indices = torch.arange(len(points), dtype=dtype, device=device).unsqueeze(-1)
    density_fit = fit_density_gp(
        y_train=y_train,
        train_grid_indices=idx_train,
        noise_var=noise_var,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        kind="heat",
        spectral_raw_inits=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
        sigma_raw_inits=(-1.0, 0.0, 1.0),
        steps=300,
        lr=0.05,
    )
    rbf_fit = fit_euclidean_gp(
        y_train=y_train,
        x_train=points[idx_train],
        noise_var=noise_var,
        kind="rbf",
        lengthscale_raw_inits=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0),
        sigma_raw_inits=(-1.0, 0.0, 1.0),
        steps=300,
        lr=0.05,
    )

    density_model = density_fit.model
    rbf_model = rbf_fit.model
    density_prediction = predict_exact_gp(density_model, grid_indices, noise_var)
    rbf_prediction = predict_exact_gp(rbf_model, points, noise_var)
    density_test_prediction = predict_exact_gp(
        density_model,
        grid_indices[idx_test],
        noise_var,
        full_covariance=True,
    )
    rbf_test_prediction = predict_exact_gp(
        rbf_model,
        points[idx_test],
        noise_var,
        full_covariance=True,
    )
    kernel_density = dense_prior_covariance(density_model, grid_indices)
    kernel_rbf = dense_prior_covariance(rbf_model, points)

    with torch.no_grad():
        mean_density = density_prediction.latent_mean
        var_density = density_prediction.latent_variance
        mean_rbf = rbf_prediction.latent_mean
        var_rbf = rbf_prediction.latent_variance
        test_mean_density = density_test_prediction.latent_mean
        test_cov_density = density_test_prediction.latent_covariance
        test_mean_rbf = rbf_test_prediction.latent_mean
        test_cov_rbf = rbf_test_prediction.latent_covariance
        y_test = y_true[idx_test]
        density_test_joint_nll = gaussian_nll(y_test, test_mean_density, test_cov_density, noise_var)
        rbf_test_joint_nll = gaussian_nll(y_test, test_mean_rbf, test_cov_rbf, noise_var)
        density_test_marginal_nll = marginal_nll(y_test, test_mean_density, test_cov_density, noise_var)
        rbf_test_marginal_nll = marginal_nll(y_test, test_mean_rbf, test_cov_rbf, noise_var)

        corr_density = kernel_to_correlation(kernel_density)
        corr_rbf = kernel_to_correlation(kernel_rbf)

        adjacent_turn_pairs, along_spiral_pairs = diagnostic_pairs(points_np, curve_t, curve_x, curve_y, curve_s)
        density_cross_corr = float(average_pair_correlation(corr_density, adjacent_turn_pairs).cpu())
        rbf_cross_corr = float(average_pair_correlation(corr_rbf, adjacent_turn_pairs).cpu())
        density_along_corr = float(average_pair_correlation(corr_density, along_spiral_pairs).cpu())
        rbf_along_corr = float(average_pair_correlation(corr_rbf, along_spiral_pairs).cpu())

        masks = {
            "tube": test_tube_mask,
            "outer": outer_mask & (~train_mask),
            "inner": inner_mask & (~train_mask),
        }
        ref_curve_idx = int(np.argmin(np.abs(curve_s - 0.53)))
        ref_idx = nearest_grid_index(points_np, (curve_x[ref_curve_idx], curve_y[ref_curve_idx]))
        curve_plot_slice = slice(None, None, 4)
        curve_s_plot = curve_s[curve_plot_slice]
        curve_grid_indices = np.array(
            [
                nearest_grid_index(points_np, (x_coord, y_coord))
                for x_coord, y_coord in zip(curve_x[curve_plot_slice], curve_y[curve_plot_slice])
            ]
        )

        diagnostics = {
            "density_tau": float(density_model.covar_module.base_kernel.tau.cpu()),
            "density_sigma": float(density_model.covar_module.outputscale.sqrt().cpu()),
            "density_loss": density_fit.total_negative_mll,
            "rbf_lengthscale": float(rbf_model.covar_module.base_kernel.lengthscale.cpu()),
            "rbf_sigma": float(rbf_model.covar_module.outputscale.sqrt().cpu()),
            "rbf_loss": rbf_fit.total_negative_mll,
            "density_min": float(p_vals.min()),
            "density_max": float(p_vals.max()),
            "n_tube_test": int(test_tube_mask.sum().cpu()),
            "cross_turn_distance": average_pair_distance(points_np, adjacent_turn_pairs),
            "along_distance": average_pair_distance(points_np, along_spiral_pairs),
            "cross_turn_label_diff": average_pair_label_difference(y_true, adjacent_turn_pairs),
            "along_label_diff": average_pair_label_difference(y_true, along_spiral_pairs),
            "density_cross_turn_corr": density_cross_corr,
            "rbf_cross_turn_corr": rbf_cross_corr,
            "density_along_corr": density_along_corr,
            "rbf_along_corr": rbf_along_corr,
            "density_corr_ratio": density_along_corr / density_cross_corr,
            "rbf_corr_ratio": rbf_along_corr / rbf_cross_corr,
            "n_explicit_test": int(len(idx_test)),
            "density_test_joint_nll": float(density_test_joint_nll.cpu()),
            "rbf_test_joint_nll": float(rbf_test_joint_nll.cpu()),
            "density_test_joint_nll_per_point": float((density_test_joint_nll / len(idx_test)).cpu()),
            "rbf_test_joint_nll_per_point": float((rbf_test_joint_nll / len(idx_test)).cpu()),
            "density_test_marginal_nll": float(density_test_marginal_nll.cpu()),
            "rbf_test_marginal_nll": float(rbf_test_marginal_nll.cpu()),
            "density": evaluate_predictions(
                y_true,
                mean_density,
                density_prediction.observed_variance,
                masks,
            ),
            "rbf": evaluate_predictions(
                y_true,
                mean_rbf,
                rbf_prediction.observed_variance,
                masks,
            ),
        }

    output_path = Path(__file__).resolve().parent / "figs" / "example_spiral.png"
    plot_results(
        output_path=output_path,
        x_axis=x_axis,
        y_axis=y_axis,
        p_vals=p_vals,
        tube_mask_np=tube_mask_np,
        curve_x=curve_x,
        curve_y=curve_y,
        points=points_np,
        idx_train=idx_train,
        y_true=y_true,
        curve_s_plot=curve_s_plot,
        curve_grid_indices=curve_grid_indices,
        nearest_s=nearest_s,
        mean_density=mean_density,
        var_density=var_density,
        mean_rbf=mean_rbf,
        var_rbf=var_rbf,
        corr_density=corr_density,
        corr_rbf=corr_rbf,
        ref_idx=ref_idx,
        diagnostics=diagnostics,
    )
    print_diagnostics(diagnostics)
    print(f"\nSaved figure: {output_path}")


if __name__ == "__main__":
    main()
