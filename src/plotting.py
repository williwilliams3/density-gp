from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_1d_density_gp_comparison(
    *,
    output_path,
    x_grid,
    data,
    p_vals,
    eigenvectors,
    corr_density,
    corr_rbf,
    idx_high,
    idx_low,
    f_true,
    x_train,
    y_train,
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
        label="Data Hist",
    )
    axs[0, 0].plot(xg, p_vals.detach().cpu().numpy(), "k-", lw=2, label="True density p(x)")
    axs[0, 0].set_title("Samples & True Density with a Low-Density Valley")
    axs[0, 0].legend()

    for i in range(1, 5):
        axs[0, 1].plot(xg, eigenvectors[:, i].detach().cpu().numpy(), label=f"Eigenvec {i}")
    axs[0, 1].set_title("First 4 Non-Constant FEM Eigenfunctions of -L_p")
    axs[0, 1].legend()

    axs[1, 0].plot(
        xg,
        corr_density[idx_high].detach().cpu().numpy(),
        "b-",
        label="Dense (High Density)",
    )
    axs[1, 0].plot(
        xg,
        corr_rbf[idx_high].detach().cpu().numpy(),
        "b--",
        alpha=0.5,
        label="RBF (High Density)",
    )
    axs[1, 0].plot(
        xg,
        corr_density[idx_low].detach().cpu().numpy(),
        "r-",
        label="Dense (Low Density)",
    )
    axs[1, 0].plot(
        xg,
        corr_rbf[idx_low].detach().cpu().numpy(),
        "r--",
        alpha=0.5,
        label="RBF (Low Density)",
    )
    axs[1, 0].set_title("Kernel Correlation Cross-Sections")
    axs[1, 0].set_ylim(-0.05, 1.05)
    axs[1, 0].legend()

    std_density = torch.sqrt(torch.clamp(var_density, min=0.0))
    std_rbf = torch.sqrt(torch.clamp(var_rbf, min=0.0))

    axs[1, 1].plot(xg, f_true.detach().cpu().numpy(), "k:", lw=2, label="True Func")
    axs[1, 1].plot(xg, mu_density.detach().cpu().numpy(), "b-", label="Density GP Mean")
    axs[1, 1].fill_between(
        xg,
        (mu_density - 2 * std_density).detach().cpu().numpy(),
        (mu_density + 2 * std_density).detach().cpu().numpy(),
        color="blue",
        alpha=0.15,
    )
    axs[1, 1].plot(xg, mu_rbf.detach().cpu().numpy(), "r-", label="RBF GP Mean")
    axs[1, 1].fill_between(
        xg,
        (mu_rbf - 2 * std_rbf).detach().cpu().numpy(),
        (mu_rbf + 2 * std_rbf).detach().cpu().numpy(),
        color="red",
        alpha=0.15,
    )
    axs[1, 1].scatter(
        x_train.detach().cpu().numpy(),
        y_train.detach().cpu().numpy(),
        color="black",
        zorder=5,
        label="Train Pts",
    )
    axs[1, 1].set_title("GP Posteriors")
    axs[1, 1].legend()

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path
