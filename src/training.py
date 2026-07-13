import torch

from .gp import neg_mll_loss
from .kernels import density_heat_kernel, density_matern_kernel, rbf_kernel, rbf_kernel_2d


def fit_density_heat_kernel(
    y_train,
    eigenvalues,
    train_eigenvectors,
    train_amp,
    noise_var,
    steps=300,
    lr=0.1,
):
    tau_raw = torch.tensor(0.0, requires_grad=True, dtype=y_train.dtype, device=y_train.device)
    sigma_raw = torch.tensor(0.0, requires_grad=True, dtype=y_train.dtype, device=y_train.device)
    opt = torch.optim.Adam([tau_raw, sigma_raw], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        kernel = density_heat_kernel(
            tau_raw,
            sigma_raw,
            eigenvalues,
            train_eigenvectors,
            train_amp,
        )
        loss = neg_mll_loss(y_train, kernel, noise_var)
        loss.backward()
        opt.step()

    return tau_raw, sigma_raw


def fit_density_heat_kernel_multistart(
    y_train,
    eigenvalues,
    train_eigenvectors,
    noise_var,
    train_amp=None,
    tau_inits=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    sigma_inits=(-1.0, 0.0, 1.0),
    steps=350,
    lr=0.05,
):
    best = None

    for tau_init in tau_inits:
        for sigma_init in sigma_inits:
            tau_raw = torch.tensor(
                tau_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
            )
            sigma_raw = torch.tensor(
                sigma_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
            )
            opt = torch.optim.Adam([tau_raw, sigma_raw], lr=lr)

            for _ in range(steps):
                opt.zero_grad()
                kernel = density_heat_kernel(
                    tau_raw,
                    sigma_raw,
                    eigenvalues,
                    train_eigenvectors,
                    train_amp,
                )
                loss = neg_mll_loss(y_train, kernel, noise_var)
                loss.backward()
                opt.step()

            with torch.no_grad():
                kernel = density_heat_kernel(
                    tau_raw,
                    sigma_raw,
                    eigenvalues,
                    train_eigenvectors,
                    train_amp,
                )
                final_loss = float(neg_mll_loss(y_train, kernel, noise_var).cpu())

            if best is None or final_loss < best[0]:
                best = (final_loss, tau_raw.detach().clone(), sigma_raw.detach().clone())

    return best[1], best[2], best[0]


def fit_rbf_kernel(y_train, x_train, noise_var, steps=300, lr=0.1):
    lengthscale_raw = torch.tensor(
        0.0,
        requires_grad=True,
        dtype=y_train.dtype,
        device=y_train.device,
    )
    sigma_raw = torch.tensor(0.0, requires_grad=True, dtype=y_train.dtype, device=y_train.device)
    opt = torch.optim.Adam([lengthscale_raw, sigma_raw], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        kernel = rbf_kernel(lengthscale_raw, sigma_raw, x_train)
        loss = neg_mll_loss(y_train, kernel, noise_var)
        loss.backward()
        opt.step()

    return lengthscale_raw, sigma_raw


def fit_rbf_kernel_multistart(
    y_train,
    x_train,
    noise_var,
    lengthscale_inits=(-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0),
    sigma_inits=(0.0,),
    steps=500,
    lr=0.05,
):
    best = None

    for lengthscale_init in lengthscale_inits:
        for sigma_init in sigma_inits:
            lengthscale_raw = torch.tensor(
                lengthscale_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
            )
            sigma_raw = torch.tensor(
                sigma_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
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
                best = (final_loss, lengthscale_raw.detach().clone(), sigma_raw.detach().clone())

    return best[1], best[2], best[0]


def fit_density_matern_kernel(
    y_train,
    eigenvalues,
    train_eigenvectors,
    noise_var,
    train_amp=None,
    alpha=1.5,
    kappa_inits=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0),
    sigma_inits=(-1.0, 0.0, 1.0),
    steps=350,
    lr=0.05,
    normalize=True,
):
    best = None

    for kappa_init in kappa_inits:
        for sigma_init in sigma_inits:
            kappa_raw = torch.tensor(
                kappa_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
            )
            sigma_raw = torch.tensor(
                sigma_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
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
                    normalize=normalize,
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
                    normalize=normalize,
                )
                final_loss = float(neg_mll_loss(y_train, kernel, noise_var).cpu())

            if best is None or final_loss < best[0]:
                best = (final_loss, kappa_raw.detach().clone(), sigma_raw.detach().clone())

    return best[1], best[2], best[0]


def fit_rbf_kernel_2d(
    y_train,
    x_train,
    noise_var,
    lengthscale_inits=(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0),
    sigma_inits=(-1.0, 0.0, 1.0),
    steps=350,
    lr=0.05,
):
    best = None

    for lengthscale_init in lengthscale_inits:
        for sigma_init in sigma_inits:
            lengthscale_raw = torch.tensor(
                lengthscale_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
            )
            sigma_raw = torch.tensor(
                sigma_init,
                dtype=y_train.dtype,
                device=y_train.device,
                requires_grad=True,
            )
            opt = torch.optim.Adam([lengthscale_raw, sigma_raw], lr=lr)

            for _ in range(steps):
                opt.zero_grad()
                kernel = rbf_kernel_2d(lengthscale_raw, sigma_raw, x_train)
                loss = neg_mll_loss(y_train, kernel, noise_var)
                loss.backward()
                opt.step()

            with torch.no_grad():
                kernel = rbf_kernel_2d(lengthscale_raw, sigma_raw, x_train)
                final_loss = float(neg_mll_loss(y_train, kernel, noise_var).cpu())

            if best is None or final_loss < best[0]:
                best = (final_loss, lengthscale_raw.detach().clone(), sigma_raw.detach().clone())

    return best[1], best[2], best[0]
