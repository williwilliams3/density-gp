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
- `src/plotting.py`: shared 1D plotting utilities.
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

```bash
python example_spiral.py
```

Output:

```text
figs/example_spiral.png
```

This script also prints RMSE and predictive NLL diagnostics on held-out spiral test points.
