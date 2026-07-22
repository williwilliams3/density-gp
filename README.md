# density-gp

Small experiments for density-based Gaussian process regression. The main idea is to build GP kernels from eigenpairs of a density-weighted Laplacian, then compare them against standard Euclidean RBF GP regression.

The weighted Laplacian uses the energy

```math
\mathcal E(f)=\int p(x)\|\nabla f(x)\|^2 dx,
```

so functions are encouraged to be smooth in high-density regions and allowed to change faster in low-density regions.

## Environment

The current Python version is 3.11.15. Install the required packages with:

```bash
pip install -r requirements.txt
```

The GP implementation is tested with GPyTorch 1.15.x and
`linear-operator` 0.6.x.

## Local modules

- `src/densities.py`: 1D toy densities and samplers.
- `src/fem.py`: 1D FEM and 2D grid weighted-Laplacian eigendecompositions.
- `src/gp.py`: shared GPyTorch exact-GP model, Euclidean kernel factory,
  fixed-noise likelihood, and prediction utilities.
- `src/grid.py`: shared 2D grid helpers.
- `src/kernels.py`: low-rank density spectral kernels, grid eigenfunction
  evaluation, dense spectral references, amplitude scaling, and correlation utilities.
- `src/metrics.py`: RMSE, NLPD, coverage, and pairwise diagnostic helpers.
- `src/neural_eigenfunctions.py`: shared-network weighted Rayleigh--Ritz
  eigensolver with an augmented-Lagrangian mass constraint and ordered-mode
  recovery through a small generalized eigendecomposition.
- `src/eigensystems.py`: Torch generalized eigendecomposition, weighted
  moments, and eigensystem/heat-kernel diagnostics shared by all solvers.
- `src/klap_eigenfunctions.py`: Torch Gaussian Nyström Galerkin solver adapted
  from the MIT-licensed KLAP implementation.
- `src/spin_eigenfunctions.py`: Torch SpIN solver with the masked Cholesky
  gradient translated from DeepMind's Apache-2.0 implementation.
- `src/plotting.py`: shared 1D plotting utilities.
- `src/rotated_mnist.py`: torchvision MNIST loading, deterministic SciPy
  rotations, tensor split metadata, source selection, and caching for SR-MNIST
  and MR-MNIST.
- `src/training.py`: shared multistart exact-GP fitting for Euclidean and density kernels.

The density kernels use grid indices as GP inputs and evaluate only the
requested rows of the precomputed eigenvector matrix. Their covariance is
represented by low-rank spectral features rather than a dense full-grid matrix
during training. The eigenfunction evaluator is a separate `torch.nn.Module`,
so it can later be replaced by a coordinate-based evaluator without changing
the GP model or spectral weighting code.

Run the numerical equivalence tests with:

```bash
python -m unittest discover -s tests -v
```

## Rotated MNIST data

`RotatedMNISTConfig("sr_mnist")` implements the paper profile: it uses the ten
fixed MNIST training indices `[1, 8, 5, 7, 2, 0, 18, 15, 17, 4]`, generates
1,000 random train rotations and 100 random test rotations per image, and does
not add an extra original image. The exact split sizes are therefore 10,000
and 1,000.

`RotatedMNISTConfig("mr_mnist")` randomly selects 100 source images without
replacement from each MNIST split, generates 1,000 train rotations and 100
test rotations per source, and also excludes extra originals. Its exact split
sizes are 100,000 and 10,000. This deliberately follows the paper rather than
the repository's smaller first-100-samples MR-MNIST dataset.

```python
from src.rotated_mnist import RotatedMNISTConfig, load_rotated_mnist

config = RotatedMNISTConfig(
    "sr_mnist",
    root="data",
    cache_dir="data/rotated_mnist_cache",
    rotation_seed=1337,
    shuffle=True,
    shuffle_seed=1337,
)
data = load_rotated_mnist(config)

train_images = data.train.images              # (10000, 1, 28, 28)
train_angles = data.train.angles_deg           # (10000,)
train_inputs = data.train.flattened_images()   # (10000, 784)
```

Rotation uses `scipy.ndimage.rotate(..., reshape=False)` with SciPy's default
interpolation settings to preserve the reference implementation's pixels.
With `scaling=True`, the legacy `(pixel - 127.5) / 255` transform is applied.
Enabling `include_original` or overriding rotation/source counts creates a
custom profile; the resulting counts are available through
`config.sample_counts`.


### `example_step.py`

1D step/two-regime example. The target changes sharply across a low-density valley. The density kernel can decouple the two high-density modes more effectively than a stationary RBF kernel.

```bash
python example_step.py
```

Output:

```text
figs/example_step.png
```

### `example_wiggle.py`

1D different-smoothness example. The target is smooth in high-density regions and more oscillatory in a low-density region. The density Matérn kernel can fit the low-density oscillations while remaining smooth in the dense modes.

```bash
python example_wiggle.py
```

Output:

```text
figs/example_wiggle.png
```

### `example_wall_doorway.py`

2D ambient-grid example. The density forms two high-density rooms separated by a low-density wall with a high-density doorway. The weighted-Laplacian Matérn kernel is computed directly from the grid density; no graph Laplacian is used.

```bash
python example_wall_doorway.py
```

Output:

```text
figs/example_wall_doorway.png
```

### `example_spiral.py`

2D spiral-geometry example. The density is concentrated on a spiral tube, and labels are smooth along spiral arclength. The RBF GP only sees Euclidean distance, while the density GP better follows the high-density spiral geometry.

The primary experiment compares native PyTorch implementations of KLAP and
SpIN with a matched seven-mode FEM reference. It runs a controlled weighted-grid
quadrature regime and a paper-faithful regime in which both learners receive
the same iid samples from the density. The sampled eigenpairs are then used in
matched FEM, KLAP, and SpIN heat-kernel GPs and compared with a Euclidean RBF
GP. The older augmented-Lagrangian neural subspace solver is retained as an
optional secondary baseline.

```bash
python example_spiral.py
```

Useful alternatives are:

```bash
python example_spiral.py --regime sampled
python example_spiral.py --regime quadrature --include-subspace
python example_spiral.py --regime both --quick
```

Output:

```text
figs/example_spiral.png
figs/example_spiral_eigenpairs.png
```

The script prints per-regime eigenvalue, subspace, orthogonality, residual,
conditioning, heat-kernel, and timing diagnostics. Its GP table reports RMSE,
NLPD, coverage, joint and marginal NLL, fitted hyperparameters, and posterior
mean disagreement with FEM.
