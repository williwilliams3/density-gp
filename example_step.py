import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.densities import build_notched_gaussian_density, sample_notched_gaussian_density
from src.fem import weighted_laplacian_eigendecomposition
from src.gp import gp_posterior
from src.kernels import (
    density_amplitude,
    density_heat_kernel,
    heat_kernel_weights,
    kernel_to_correlation,
    rbf_kernel,
)
from src.training import fit_density_heat_kernel, fit_rbf_kernel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.manual_seed(7)

USE_DENSITY_AMPLITUDE = False


def two_regime_target(x):
    left = 1.0 + 0.15 * torch.sin(5.0 * (x + 1.0))
    right = -1.0 + 0.15 * torch.sin(5.0 * (x - 1.0))
    return torch.where(x < 0.0, left, right)


def plot_example_step(
    *,
    output_path,
    x_grid,
    data,
    p_vals,
    f_true,
    x_train,
    y_train,
    corr_density,
    corr_rbf,
    idx_left_ref,
    idx_right_ref,
    mu_density,
    var_density,
    mu_rbf,
    var_rbf,
):
    xg = x_grid.detach().cpu().numpy()
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].hist(
        data.detach().cpu().numpy(),
        bins=50,
        density=True,
        alpha=0.3,
        color="gray",
        label="Samples from density",
    )
    axs[0, 0].plot(xg, p_vals.detach().cpu().numpy(), "k-", lw=2, label="density p(x)")
    axs[0, 0].scatter(
        x_train.detach().cpu().numpy(),
        torch.zeros_like(x_train).detach().cpu().numpy(),
        color="black",
        s=40,
        zorder=5,
        label="label locations",
    )
    axs[0, 0].axvspan(-0.35, 0.35, color="orange", alpha=0.12, label="low-density valley")
    axs[0, 0].set_title("Two Dense Modes Separated by a Low-Density Valley")
    axs[0, 0].legend()

    axs[0, 1].plot(xg, f_true.detach().cpu().numpy(), "k:", lw=2, label="true function")
    axs[0, 1].scatter(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        color="black",
        zorder=5,
        label="training labels",
    )
    axs[0, 1].axvline(0.0, color="gray", ls="--", lw=1)
    axs[0, 1].set_title("Different Smooth Behavior on Each Mode")
    axs[0, 1].legend()

    axs[1, 0].plot(
        xg,
        corr_density[idx_left_ref].detach().cpu().numpy(),
        "b-",
        label="dGP from left mode",
    )
    axs[1, 0].plot(
        xg,
        corr_rbf[idx_left_ref].detach().cpu().numpy(),
        "b--",
        alpha=0.6,
        label="RBF from left mode",
    )
    axs[1, 0].plot(
        xg,
        corr_density[idx_right_ref].detach().cpu().numpy(),
        "r-",
        label="dGP from right mode",
    )
    axs[1, 0].plot(
        xg,
        corr_rbf[idx_right_ref].detach().cpu().numpy(),
        "r--",
        alpha=0.6,
        label="RBF from right mode",
    )
    axs[1, 0].axvline(0.0, color="gray", ls="--", lw=1)
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].set_title("Kernel Correlations Drop Across the Valley")
    axs[1, 0].legend()

    std_density = torch.sqrt(torch.clamp(var_density, min=0.0))
    std_rbf = torch.sqrt(torch.clamp(var_rbf, min=0.0))

    axs[1, 1].plot(xg, f_true.detach().cpu().numpy(), "k:", lw=2, label="true function")
    axs[1, 1].plot(xg, mu_density.detach().cpu().numpy(), "b-", label="density GP mean")
    axs[1, 1].fill_between(
        xg,
        (mu_density - 2.0 * std_density).detach().cpu().numpy(),
        (mu_density + 2.0 * std_density).detach().cpu().numpy(),
        color="blue",
        alpha=0.15,
    )
    axs[1, 1].plot(xg, mu_rbf.detach().cpu().numpy(), "r-", label="RBF GP mean")
    axs[1, 1].fill_between(
        xg,
        (mu_rbf - 2.0 * std_rbf).detach().cpu().numpy(),
        (mu_rbf + 2.0 * std_rbf).detach().cpu().numpy(),
        color="red",
        alpha=0.15,
    )
    axs[1, 1].scatter(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        color="black",
        zorder=5,
        label="training labels",
    )
    axs[1, 1].axvline(0.0, color="gray", ls="--", lw=1)
    axs[1, 1].set_title("Posterior: dGP Can Decouple the Two Modes")
    axs[1, 1].legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    x_grid = torch.linspace(-3, 3, 301, dtype=dtype, device=device)
    p_true, p_true_numpy = build_notched_gaussian_density()
    data = sample_notched_gaussian_density(n=1200, device=device, dtype=dtype)

    eigenvalues, eigenvectors = weighted_laplacian_eigendecomposition(x_grid, p_true_numpy)
    p_vals = p_true(x_grid)
    amp_vals = density_amplitude(p_vals) if USE_DENSITY_AMPLITUDE else torch.ones_like(p_vals)

    f_true = two_regime_target(x_grid)
    x_labels = torch.tensor(
        [-2.6, -2.0, -1.45, -0.85, -0.55, 0.55, 0.85, 1.45, 2.0, 2.6],
        dtype=dtype,
        device=device,
    )
    idx_train = torch.argmin(torch.abs(x_grid.unsqueeze(1) - x_labels.unsqueeze(0)), dim=0)
    x_train = x_grid[idx_train]

    noise_var = 0.06**2
    y_train = f_true[idx_train] + math.sqrt(noise_var) * torch.randn_like(x_train)

    with torch.no_grad():
        initial_tau_raw = torch.tensor(0.0, dtype=dtype, device=device)
        weights = heat_kernel_weights(initial_tau_raw, eigenvalues)
        print("first heat-kernel weights:", weights[:10].cpu().numpy())

    tau_raw, sigma_density_raw = fit_density_heat_kernel(
        y_train=y_train,
        eigenvalues=eigenvalues,
        train_eigenvectors=eigenvectors[idx_train],
        train_amp=amp_vals[idx_train],
        noise_var=noise_var,
        steps=300,
        lr=0.1,
    )
    lengthscale_raw, sigma_rbf_raw = fit_rbf_kernel(
        y_train=y_train,
        x_train=x_train,
        noise_var=noise_var,
        steps=300,
        lr=0.1,
    )

    with torch.no_grad():
        kernel_density = density_heat_kernel(
            tau_raw,
            sigma_density_raw,
            eigenvalues,
            eigenvectors,
            amp_vals,
        )
        kernel_rbf = rbf_kernel(lengthscale_raw, sigma_rbf_raw, x_grid)

        corr_density = kernel_to_correlation(kernel_density)
        corr_rbf = kernel_to_correlation(kernel_rbf)

        mu_density, var_density = gp_posterior(kernel_density, idx_train, y_train, noise_var)
        mu_rbf, var_rbf = gp_posterior(kernel_rbf, idx_train, y_train, noise_var)

        idx_left_ref = torch.argmin(torch.abs(x_grid + 0.85))
        idx_right_ref = torch.argmin(torch.abs(x_grid - 0.85))

        print(
            "density GP tau:",
            float(F.softplus(tau_raw).cpu()),
            "sigma:",
            float(F.softplus(sigma_density_raw).cpu()),
        )
        print(
            "RBF lengthscale:",
            float(F.softplus(lengthscale_raw).cpu()),
            "sigma:",
            float(F.softplus(sigma_rbf_raw).cpu()),
        )

    output_path = Path(__file__).resolve().parent / "figs" / "example_step.png"
    plot_example_step(
        output_path=output_path,
        x_grid=x_grid,
        data=data,
        p_vals=p_vals,
        f_true=f_true,
        x_train=x_train,
        y_train=y_train,
        corr_density=corr_density,
        corr_rbf=corr_rbf,
        idx_left_ref=idx_left_ref,
        idx_right_ref=idx_right_ref,
        mu_density=mu_density,
        var_density=var_density,
        mu_rbf=mu_rbf,
        var_rbf=var_rbf,
    )


if __name__ == "__main__":
    main()
