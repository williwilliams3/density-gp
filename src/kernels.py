import gpytorch
import torch
import torch.nn.functional as F
from linear_operator.operators import MatmulLinearOperator, RootLinearOperator


def density_amplitude(p_vals, beta=0.5, eps=1e-3, max_amp=3.0):
    q = p_vals / p_vals.max()
    amp = (q + eps).pow(-0.5 * beta)
    return torch.clamp(amp, max=max_amp)


def heat_kernel_weights(tau_raw, eigenvalues):
    tau = F.softplus(tau_raw)
    return torch.exp(-tau * eigenvalues)


def density_heat_kernel(tau_raw, sigma_f_raw, eigenvalues, eigenvectors, amp=None):
    sigma_f = F.softplus(sigma_f_raw)
    weights = heat_kernel_weights(tau_raw, eigenvalues)

    if amp is not None:
        eigenvectors = amp.unsqueeze(1) * eigenvectors

    return (sigma_f**2) * (eigenvectors * weights.unsqueeze(0)) @ eigenvectors.T


def density_matern_kernel(
    kappa_raw,
    sigma_f_raw,
    eigenvalues,
    eigenvectors,
    amp=None,
    alpha=1.5,
):
    kappa = F.softplus(kappa_raw)
    sigma_f = F.softplus(sigma_f_raw)
    weights = ((kappa**2) / (kappa**2 + eigenvalues)).pow(alpha)

    if amp is not None:
        eigenvectors = amp.unsqueeze(1) * eigenvectors

    return (sigma_f**2) * (eigenvectors * weights.unsqueeze(0)) @ eigenvectors.T


def kernel_to_correlation(kernel, eps=1e-12):
    std = torch.sqrt(torch.clamp(torch.diag(kernel), min=eps))
    return kernel / (std.unsqueeze(1) * std.unsqueeze(0))


class GridEigenfunctionEvaluator(torch.nn.Module):
    """Evaluate precomputed eigenfunctions by stable grid index.

    The spectral kernels depend only on this module's ``forward`` contract:
    inputs of shape ``(..., n, input_dims)`` map to eigenfunction features of
    shape ``(..., n, n_modes)``. A coordinate-based neural evaluator can
    therefore replace this lookup module without changing the GP model.
    """

    def __init__(self, eigenvectors, amplitude=None):
        super().__init__()
        if eigenvectors.ndim != 2:
            raise ValueError("eigenvectors must have shape (n_grid_points, n_modes)")
        if amplitude is not None and amplitude.shape != eigenvectors.shape[:1]:
            raise ValueError("amplitude must have shape (n_grid_points,)")
        self.register_buffer("eigenvectors", eigenvectors)
        self.register_buffer("amplitude", amplitude)

    @property
    def num_grid_points(self):
        return self.eigenvectors.shape[0]

    @property
    def num_modes(self):
        return self.eigenvectors.shape[1]

    def forward(self, x):
        if x.ndim < 2 or x.shape[-1] != 1:
            raise ValueError("grid-index inputs must have shape (..., n_points, 1)")
        index_values = x.squeeze(-1)
        indices = index_values.round().long()
        if not torch.equal(index_values, indices.to(dtype=index_values.dtype)):
            raise ValueError("grid-index inputs must contain integer-valued indices")
        if torch.any(indices < 0) or torch.any(indices >= self.num_grid_points):
            raise IndexError("grid index is outside the eigenfunction table")

        features = self.eigenvectors[indices]
        if self.amplitude is not None:
            features = self.amplitude[indices].unsqueeze(-1) * features
        return features


class DensitySpectralKernel(gpytorch.kernels.Kernel):
    """Base class for low-rank kernels built from Laplacian eigenfeatures."""

    has_lengthscale = False

    def __init__(self, eigenvalues, eigenfunction_evaluator, **kwargs):
        super().__init__(**kwargs)
        if eigenvalues.ndim != 1:
            raise ValueError("eigenvalues must have shape (n_modes,)")
        self.register_buffer("eigenvalues", eigenvalues)
        self.eigenfunction_evaluator = eigenfunction_evaluator

    def spectral_weights(self):
        raise NotImplementedError

    def spectral_feature_scales(self):
        return self.spectral_weights().sqrt()

    def features(self, x):
        eigenfunctions = self.eigenfunction_evaluator(x)
        if eigenfunctions.shape[-1] != len(self.eigenvalues):
            raise ValueError("eigenfunction evaluator output does not match eigenvalue count")
        return eigenfunctions * self.spectral_feature_scales()

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **kwargs):
        if last_dim_is_batch:
            raise NotImplementedError("last_dim_is_batch is not supported for spectral feature inputs")
        features_1 = self.features(x1)
        features_2 = features_1 if torch.equal(x1, x2) else self.features(x2)

        if diag:
            if features_1.shape != features_2.shape:
                raise ValueError("diag=True requires matching input shapes")
            return (features_1 * features_2).sum(dim=-1)
        if torch.equal(x1, x2):
            return RootLinearOperator(features_1)
        return MatmulLinearOperator(features_1, features_2.transpose(-1, -2))


class DensityHeatKernel(DensitySpectralKernel):
    def __init__(self, eigenvalues, eigenfunction_evaluator, tau_constraint=None, **kwargs):
        super().__init__(eigenvalues, eigenfunction_evaluator, **kwargs)
        tau_constraint = tau_constraint or gpytorch.constraints.Positive()
        self.register_parameter(
            "raw_tau",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_tau", tau_constraint)

    @property
    def tau(self):
        return self.raw_tau_constraint.transform(self.raw_tau)

    @tau.setter
    def tau(self, value):
        value = torch.as_tensor(value, dtype=self.raw_tau.dtype, device=self.raw_tau.device)
        self.initialize(raw_tau=self.raw_tau_constraint.inverse_transform(value))

    def spectral_weights(self):
        return torch.exp(-self.tau * self.eigenvalues)

    def spectral_feature_scales(self):
        # Computing exp(-tau * lambda / 2) directly avoids the infinite
        # derivative of sqrt at heat weights that underflow to exactly zero.
        return torch.exp(-0.5 * self.tau * self.eigenvalues)


class DensityMaternKernel(DensitySpectralKernel):
    def __init__(
        self,
        eigenvalues,
        eigenfunction_evaluator,
        *,
        alpha=1.5,
        kappa_constraint=None,
        **kwargs,
    ):
        super().__init__(eigenvalues, eigenfunction_evaluator, **kwargs)
        self.alpha = alpha
        kappa_constraint = kappa_constraint or gpytorch.constraints.Positive()
        self.register_parameter(
            "raw_kappa",
            torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)),
        )
        self.register_constraint("raw_kappa", kappa_constraint)

    @property
    def kappa(self):
        return self.raw_kappa_constraint.transform(self.raw_kappa)

    @kappa.setter
    def kappa(self, value):
        value = torch.as_tensor(value, dtype=self.raw_kappa.dtype, device=self.raw_kappa.device)
        self.initialize(raw_kappa=self.raw_kappa_constraint.inverse_transform(value))

    def spectral_weights(self):
        kappa_squared = self.kappa.square()
        return (kappa_squared / (kappa_squared + self.eigenvalues)).pow(self.alpha)

    def spectral_feature_scales(self):
        kappa_squared = self.kappa.square()
        ratio = kappa_squared / (kappa_squared + self.eigenvalues)
        return ratio.pow(0.5 * self.alpha)


def build_grid_density_kernel(kind, eigenvalues, eigenvectors, *, amplitude=None, alpha=1.5):
    """Build a scaled density kernel backed by precomputed grid eigenvectors."""
    evaluator = GridEigenfunctionEvaluator(eigenvectors, amplitude=amplitude)
    kind = kind.lower()
    if kind == "heat":
        base_kernel = DensityHeatKernel(eigenvalues, evaluator)
    elif kind == "matern":
        base_kernel = DensityMaternKernel(eigenvalues, evaluator, alpha=alpha)
    else:
        raise ValueError(f"Unsupported density spectral kernel: {kind!r}")
    return gpytorch.kernels.ScaleKernel(base_kernel)
