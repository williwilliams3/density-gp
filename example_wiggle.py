import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.densities import build_notched_gaussian_density, sample_notched_gaussian_density
from src.fem import weighted_laplacian_eigendecomposition
from src.gp import gp_posterior, neg_mll_loss
from src.kernels import (
    density_amplitude,
    kernel_to_correlation,
    rbf_kernel,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float64
torch.manual_seed(23)

USE_DENSITY_AMPLITUDE = False
VALLEY_WIDTH = 0.35
VALLEY_REGION = 0.65


def low_density_oscillation_target(x):
    smooth_background = 0.65 * torch.cos(0.95 * x) - 0.05 * x
    valley_gate = torch.exp(-0.5 * (x / 0.42) ** 2)
    valley_oscillation = 0.42 * valley_gate * torch.sin(32.0 * x)
    return smooth_background + valley_oscillation


def rmse(pred, truth, mask):
    return torch.sqrt(torch.mean((pred[mask] - truth[mask]) ** 2))


def density_matern_kernel(kappa_raw, sigma_f_raw, eigenvalues, eigenvectors, amp=None, alpha=1.5):
    kappa = F.softplus(kappa_raw)
    sigma_f = F.softplus(sigma_f_raw)
    weights = ((kappa**2) / (kappa**2 + eigenvalues)).pow(alpha)

    if amp is not None:
        eigenvectors = amp.unsqueeze(1) * eigenvectors

    return (sigma_f**2) * (eigenvectors * weights.unsqueeze(0)) @ eigenvectors.T


def fit_density_matern_kernel(
    y_train,
    eigenvalues,
    train_eigenvectors,
    train_amp,
    noise_var,
    alpha=1.5,
    kappa_inits=(-2.0, 0.0),
    steps=500,
    lr=0.05,
):
    best = None

    for kappa_init in kappa_inits:
        kappa_raw = torch.tensor(
            kappa_init,
            requires_grad=True,
            dtype=y_train.dtype,
            device=y_train.device,
        )
        sigma_raw = torch.tensor(
            0.0,
            requires_grad=True,
            dtype=y_train.dtype,
            device=y_train.device,
        )
        opt = torch.optim.Adam([kappa_raw, sigma_raw], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            kernel = density_matern_kernel(
                kappa_raw,
                sigma_raw,
                eigenvalues,
                train_eigenvectors,
                train_amp,
                alpha=alpha,
            )
            loss = neg_mll_loss(y_train, kernel, noise_var)
            loss.backward()
            opt.step()

        with torch.no_grad():
            kernel = density_matern_kernel(
                kappa_raw,
                sigma_raw,
                eigenvalues,
                train_eigenvectors,
                train_amp,
                alpha=alpha,
            )
            final_loss = float(neg_mll_loss(y_train, kernel, noise_var).cpu())

        if best is None or final_loss < best[0]:
            best = (
                final_loss,
                kappa_raw.detach().clone(),
                sigma_raw.detach().clone(),
            )

    return best[1], best[2], best[0]


def fit_rbf_kernel_multistart(
    y_train,
    x_train,
    noise_var,
    lengthscale_inits=(-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0),
    steps=500,
    lr=0.05,
):
    best = None

    for lengthscale_init in lengthscale_inits:
        lengthscale_raw = torch.tensor(
            lengthscale_init,
            requires_grad=True,
            dtype=y_train.dtype,
            device=y_train.device,
        )
        sigma_raw = torch.tensor(
            0.0,
            requires_grad=True,
            dtype=y_train.dtype,
            device=y_train.device,
        )
        opt = torch.optim.Adam([lengthscale_raw, sigma_raw], lr=lr)

        for _ in range(steps):
            opt.zero_grad()
            kernel = rbf_kernel(lengthscale_raw, sigma_raw, x_train)
            loss = neg_mll_loss(y_train, kernel, noise_var)
            loss.backward()
            opt.step()

        with torch.no_grad():
            kernel = rbf_kernel(lengthscale_raw, sigma_raw, x_train)
            final_loss = float(neg_mll_loss(y_train, kernel, noise_var).cpu())

        if best is None or final_loss < best[0]:
            best = (
                final_loss,
                lengthscale_raw.detach().clone(),
                sigma_raw.detach().clone(),
            )

    return best[1], best[2], best[0]


def plot_example_4(
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
    idx_high_ref,
    idx_low_ref,
    mu_density,
    var_density,
    mu_rbf,
    var_rbf,
    metrics,
):
    xg = x_grid.detach().cpu().numpy()
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].hist(
        data.detach().cpu().numpy(),
        bins=60,
        density=True,
        alpha=0.3,
        color="gray",
        label="samples from density",
    )
    axs[0, 0].plot(xg, p_vals.detach().cpu().numpy(), "k-", lw=2, label="density p(x)")
    axs[0, 0].scatter(
        x_train.detach().cpu().numpy(),
        torch.zeros_like(x_train).detach().cpu().numpy(),
        color="black",
        s=28,
        zorder=5,
        label="label locations",
    )
    axs[0, 0].axvspan(
        -VALLEY_REGION,
        VALLEY_REGION,
        color="orange",
        alpha=0.12,
        label="low-density oscillatory region",
    )
    axs[0, 0].set_title("Labels Include the Low-Density Region")
    axs[0, 0].legend()

    axs[0, 1].plot(xg, f_true.detach().cpu().numpy(), "k:", lw=2, label="true function")
    axs[0, 1].scatter(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        color="black",
        zorder=5,
        label="training labels",
    )
    axs[0, 1].axvspan(-VALLEY_REGION, VALLEY_REGION, color="orange", alpha=0.12)
    axs[0, 1].text(-2.7, 0.2, "smooth high-density mode", fontsize=10)
    axs[0, 1].text(-0.58, 1.12, "oscillatory\nlow density", fontsize=10)
    axs[0, 1].text(1.0, 0.0, "smooth high-density mode", fontsize=10)
    axs[0, 1].set_title("Smooth in Modes, Oscillatory in the Valley")
    axs[0, 1].legend()

    axs[1, 0].plot(
        xg,
        corr_density[idx_high_ref].detach().cpu().numpy(),
        "b-",
        label="dGP from high density",
    )
    axs[1, 0].plot(
        xg,
        corr_rbf[idx_high_ref].detach().cpu().numpy(),
        "b--",
        alpha=0.6,
        label="RBF from high density",
    )
    axs[1, 0].plot(
        xg,
        corr_density[idx_low_ref].detach().cpu().numpy(),
        "r-",
        label="dGP from low density",
    )
    axs[1, 0].plot(
        xg,
        corr_rbf[idx_low_ref].detach().cpu().numpy(),
        "r--",
        alpha=0.6,
        label="RBF from low density",
    )
    axs[1, 0].axvspan(-VALLEY_REGION, VALLEY_REGION, color="orange", alpha=0.08)
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].set_title("Kernel Correlation Cross-Sections")
    axs[1, 0].legend()

    std_density = torch.sqrt(torch.clamp(var_density, min=0.0))
    std_rbf = torch.sqrt(torch.clamp(var_rbf, min=0.0))

    axs[1, 1].plot(xg, f_true.detach().cpu().numpy(), "k:", lw=2, label="true function")
    axs[1, 1].plot(xg, mu_density.detach().cpu().numpy(), "b-", label="density Matérn GP mean")
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
    axs[1, 1].axvspan(-VALLEY_REGION, VALLEY_REGION, color="orange", alpha=0.08)
    axs[1, 1].set_title("Posterior Comparison")
    axs[1, 1].legend()

    metric_text = (
        f"RMSE high-density: dGP={metrics['density_high']:.3f}, RBF={metrics['rbf_high']:.3f}\n"
        f"RMSE low-density:  dGP={metrics['density_low']:.3f}, RBF={metrics['rbf_low']:.3f}"
    )
    axs[1, 1].text(
        0.02,
        0.98,
        metric_text,
        transform=axs[1, 1].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    x_grid = torch.linspace(-3, 3, 601, dtype=dtype, device=device)
    p_true, p_true_numpy = build_notched_gaussian_density(valley_width=VALLEY_WIDTH)
    data = sample_notched_gaussian_density(
        n=1500,
        valley_width=VALLEY_WIDTH,
        device=device,
        dtype=dtype,
    )

    eigenvalues, eigenvectors = weighted_laplacian_eigendecomposition(x_grid, p_true_numpy)
    p_vals = p_true(x_grid)
    amp_vals = density_amplitude(p_vals) if USE_DENSITY_AMPLITUDE else torch.ones_like(p_vals)

    f_true = low_density_oscillation_target(x_grid)

    high_density_labels = torch.tensor(
        [0.85, 1.15, 1.45, 1.8, 2.2, 2.65],
        dtype=dtype,
        device=device,
    )
    low_density_labels = torch.linspace(
        -0.60,
        0.60,
        27,
        dtype=dtype,
        device=device,
    )
    x_labels = torch.cat(
        [
            -torch.flip(high_density_labels, dims=(0,)),
            low_density_labels,
            high_density_labels,
        ]
    )
    idx_train = torch.argmin(torch.abs(x_grid.unsqueeze(1) - x_labels.unsqueeze(0)), dim=0)
    x_train = x_grid[idx_train]

    noise_var = 0.04**2
    y_train = f_true[idx_train] + math.sqrt(noise_var) * torch.randn_like(x_train)

    kappa_raw, sigma_density_raw, density_loss = fit_density_matern_kernel(
        y_train=y_train,
        eigenvalues=eigenvalues,
        train_eigenvectors=eigenvectors[idx_train],
        train_amp=amp_vals[idx_train],
        noise_var=noise_var,
    )
    lengthscale_raw, sigma_rbf_raw, rbf_loss = fit_rbf_kernel_multistart(
        y_train=y_train,
        x_train=x_train,
        noise_var=noise_var,
    )

    with torch.no_grad():
        kernel_density = density_matern_kernel(
            kappa_raw,
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

        high_density_mask = torch.abs(x_grid) >= 0.85
        low_density_mask = torch.abs(x_grid) <= VALLEY_REGION
        metrics = {
            "density_high": float(rmse(mu_density, f_true, high_density_mask).cpu()),
            "rbf_high": float(rmse(mu_rbf, f_true, high_density_mask).cpu()),
            "density_low": float(rmse(mu_density, f_true, low_density_mask).cpu()),
            "rbf_low": float(rmse(mu_rbf, f_true, low_density_mask).cpu()),
        }

        idx_high_ref = torch.argmin(torch.abs(x_grid - 1.45))
        idx_low_ref = torch.argmin(torch.abs(x_grid - 0.20))

        print(
            "density Matérn GP kappa:",
            float(F.softplus(kappa_raw).cpu()),
            "sigma:",
            float(F.softplus(sigma_density_raw).cpu()),
            "loss:",
            density_loss,
        )
        print(
            "RBF lengthscale:",
            float(F.softplus(lengthscale_raw).cpu()),
            "sigma:",
            float(F.softplus(sigma_rbf_raw).cpu()),
            "loss:",
            rbf_loss,
        )
        print("RMSE metrics:", metrics)

    output_path = Path(__file__).resolve().parent / "figs" / "example_wiggle.png"
    plot_example_4(
        output_path=output_path,
        x_grid=x_grid,
        data=data,
        p_vals=p_vals,
        f_true=f_true,
        x_train=x_train,
        y_train=y_train,
        corr_density=corr_density,
        corr_rbf=corr_rbf,
        idx_high_ref=idx_high_ref,
        idx_low_ref=idx_low_ref,
        mu_density=mu_density,
        var_density=var_density,
        mu_rbf=mu_rbf,
        var_rbf=var_rbf,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
