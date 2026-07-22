"""Reusable pieces for density-weighted GP examples."""

from .neural_eigenfunctions import (
    EigensystemComparison,
    NeuralEigenfunctionConfig,
    NeuralEigenfunctionFit,
    NeuralEigenfunctionSystem,
    SharedEigenfunctionNetwork,
    compare_eigensystems,
    fit_neural_eigenfunctions,
)
from .klap_eigenfunctions import (
    KlapConfig,
    KlapEigenfunctionFit,
    KlapEigenfunctionSystem,
    fit_klap_eigenfunctions,
)
from .spin_eigenfunctions import (
    SpINConfig,
    SpINEigenfunctionFit,
    SpINEigenfunctionSystem,
    fit_spin_eigenfunctions,
)
from .eigensystems import (
    GeneralizedEigenResult,
    normalize_eigenfunction_columns,
    solve_generalized_eigenproblem,
    weighted_center,
    weighted_mass,
)
from .rotated_mnist import (
    SR_MNIST_SOURCE_INDICES,
    RotatedMNISTConfig,
    RotatedMNISTData,
    RotatedMNISTMetadata,
    RotatedMNISTSplit,
    generate_rotated_mnist,
    generate_rotated_split,
    load_mnist_tensors,
    load_rotated_mnist,
    select_source_indices,
)

__all__ = [
    "EigensystemComparison",
    "KlapConfig",
    "KlapEigenfunctionFit",
    "KlapEigenfunctionSystem",
    "GeneralizedEigenResult",
    "NeuralEigenfunctionConfig",
    "NeuralEigenfunctionFit",
    "NeuralEigenfunctionSystem",
    "SR_MNIST_SOURCE_INDICES",
    "SharedEigenfunctionNetwork",
    "SpINConfig",
    "SpINEigenfunctionFit",
    "SpINEigenfunctionSystem",
    "RotatedMNISTConfig",
    "RotatedMNISTData",
    "RotatedMNISTMetadata",
    "RotatedMNISTSplit",
    "compare_eigensystems",
    "normalize_eigenfunction_columns",
    "solve_generalized_eigenproblem",
    "weighted_center",
    "weighted_mass",
    "fit_neural_eigenfunctions",
    "fit_klap_eigenfunctions",
    "fit_spin_eigenfunctions",
    "generate_rotated_mnist",
    "generate_rotated_split",
    "load_mnist_tensors",
    "load_rotated_mnist",
    "select_source_indices",
]
