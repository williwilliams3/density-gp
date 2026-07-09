import numpy as np
import torch
from scipy.linalg import eigh
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from skfem import Basis, BilinearForm, ElementLineP1, MeshLine, asm
from skfem.helpers import dot, grad


def weighted_laplacian_eigendecomposition(x_grid, density_numpy, intorder=4):
    x_np = x_grid.detach().cpu().numpy()
    mesh = MeshLine(x_np.reshape(1, -1))
    basis = Basis(mesh, ElementLineP1(), intorder=intorder)

    @BilinearForm
    def stiffness(u, v, w):
        return density_numpy(w.x[0]) * dot(grad(u), grad(v))

    @BilinearForm
    def mass(u, v, w):
        return density_numpy(w.x[0]) * u * v

    stiffness_matrix = asm(stiffness, basis).toarray()
    mass_matrix = asm(mass, basis).toarray()
    eigenvalues, eigenvectors = eigh(stiffness_matrix, mass_matrix)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    dof_x = basis.doflocs[0]
    grid_order = np.array([np.argmin(np.abs(dof_x - x)) for x in x_np])
    eigenvectors = eigenvectors[grid_order]

    return [
        torch.as_tensor(eigenvalues, dtype=x_grid.dtype, device=x_grid.device),
        torch.as_tensor(eigenvectors, dtype=x_grid.dtype, device=x_grid.device),
    ]


def weighted_laplacian_eigendecomposition_2d(
    density_grid,
    grid_spacing,
    n_modes,
    dtype=torch.float64,
    device=None,
):
    """Compute low-frequency eigenpairs of the 2D weighted Laplacian on a grid.

    This solves the generalized eigenproblem associated with

        ∫ p(x, y) ∇u · ∇v dx dy = lambda ∫ p(x, y) u v dx dy.

    The sparse stiffness matrix uses nearest-neighbor grid edges with density
    averaged across each edge. This is an ambient-grid discretization, not a
    graph Laplacian built from data points.
    """
    device = device or torch.device("cpu")
    n_y, n_x = density_grid.shape
    n_total = n_x * n_y
    rows = []
    cols = []
    vals = []

    def idx(row, col):
        return row * n_x + col

    for row in range(n_y):
        for col in range(n_x):
            center = idx(row, col)
            diag = 0.0

            for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nrow = row + drow
                ncol = col + dcol
                if 0 <= nrow < n_y and 0 <= ncol < n_x:
                    neighbor = idx(nrow, ncol)
                    conductance = 0.5 * (density_grid[row, col] + density_grid[nrow, ncol])
                    diag += conductance
                    rows.append(center)
                    cols.append(neighbor)
                    vals.append(-conductance)

            rows.append(center)
            cols.append(center)
            vals.append(diag)

    stiffness = coo_matrix((vals, (rows, cols)), shape=(n_total, n_total)).tocsr()
    mass = diags(density_grid.ravel() * grid_spacing**2)

    eigenvalues, eigenvectors = eigsh(
        stiffness,
        k=n_modes,
        M=mass,
        which="SM",
        tol=1e-7,
        maxiter=5000,
    )
    order = np.argsort(eigenvalues)
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    return [
        torch.as_tensor(eigenvalues, dtype=dtype, device=device),
        torch.as_tensor(eigenvectors, dtype=dtype, device=device),
    ]
