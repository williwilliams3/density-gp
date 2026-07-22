"""Tensor-native loading and generation utilities for paper-sized rotated MNIST.

The only NumPy boundary in this module is the call to
``scipy.ndimage.rotate``.  MNIST is loaded with torchvision and all public
dataset objects contain PyTorch tensors.

The paper profiles intentionally differ from the original repository code:

* SR-MNIST uses the repository's ten fixed training-image indices and creates
  1,000 train rotations and 100 test rotations for each source, without an
  additional zero-degree image (10,000 train and 1,000 test observations).
* MR-MNIST randomly selects 100 sources from each MNIST split and creates
  1,000 train rotations and 100 test rotations for each source, again without
  additional originals (100,000 train and 10,000 test observations).

Overriding these profile values is supported, except that SR-MNIST always has
exactly the ten fixed sources.  Enabling ``include_original`` adds one sample
per source and therefore defines a custom, non-paper dataset.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from scipy import ndimage


Variant = Literal["sr_mnist", "mr_mnist"]
MNISTFactory = Callable[..., Any]

SR_MNIST_SOURCE_INDICES = torch.tensor(
    [1, 8, 5, 7, 2, 0, 18, 15, 17, 4], dtype=torch.long
)

_GENERATOR_VERSION = "paper-v1"
_PROFILE_DEFAULTS: dict[Variant, dict[str, int | bool]] = {
    "sr_mnist": {
        "train_rotations_per_source": 1_000,
        "test_rotations_per_source": 100,
        "include_original": False,
        "num_source_images": 10,
    },
    "mr_mnist": {
        "train_rotations_per_source": 1_000,
        "test_rotations_per_source": 100,
        "include_original": False,
        "num_source_images": 100,
    },
}


@dataclass(frozen=True)
class RotatedMNISTConfig:
    """Configuration separate from generated tensor data.

    ``None`` for a profile-specific field means to use the paper default from
    ``_PROFILE_DEFAULTS``.  Call :meth:`resolved` to obtain a configuration in
    which those fields and ``cache_dir`` are concrete.

    With ``scaling=True``, pixels use the legacy transformation
    ``(pixel - 127.5) / 255``.  With ``scaling=False``, they retain the MNIST
    intensity scale, usually ``[0, 255]``.
    """

    variant: Variant
    root: str | Path = Path("data")
    cache_dir: str | Path | None = None
    download: bool = True
    angle_min: float = -45.0
    angle_max: float = 45.0
    train_rotations_per_source: int | None = None
    test_rotations_per_source: int | None = None
    include_original: bool | None = None
    num_source_images: int | None = None
    source_selection_seed: int | None = 0
    rotation_seed: int | None = 0
    shuffle_seed: int | None = 0
    shuffle: bool = False
    scaling: bool = True
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.variant not in _PROFILE_DEFAULTS:
            raise ValueError(
                f"variant must be 'sr_mnist' or 'mr_mnist', got {self.variant!r}"
            )
        object.__setattr__(self, "root", Path(self.root))
        if self.cache_dir is not None:
            object.__setattr__(self, "cache_dir", Path(self.cache_dir))

    def resolved(self) -> "RotatedMNISTConfig":
        """Return a validated config with all profile defaults filled in."""

        defaults = _PROFILE_DEFAULTS[self.variant]
        config = replace(
            self,
            cache_dir=(
                Path(self.cache_dir)
                if self.cache_dir is not None
                else Path(self.root) / "rotated_mnist_cache"
            ),
            train_rotations_per_source=(
                self.train_rotations_per_source
                if self.train_rotations_per_source is not None
                else int(defaults["train_rotations_per_source"])
            ),
            test_rotations_per_source=(
                self.test_rotations_per_source
                if self.test_rotations_per_source is not None
                else int(defaults["test_rotations_per_source"])
            ),
            include_original=(
                self.include_original
                if self.include_original is not None
                else bool(defaults["include_original"])
            ),
            num_source_images=(
                self.num_source_images
                if self.num_source_images is not None
                else int(defaults["num_source_images"])
            ),
        )
        config._validate_resolved()
        return config

    def _validate_resolved(self) -> None:
        if any(
            value is None
            for value in (
                self.cache_dir,
                self.train_rotations_per_source,
                self.test_rotations_per_source,
                self.include_original,
                self.num_source_images,
            )
        ):
            raise ValueError("configuration must be resolved before validation")
        if not self.angle_min < self.angle_max:
            raise ValueError("angle_min must be strictly less than angle_max")
        if self.train_rotations_per_source < 0:
            raise ValueError("train_rotations_per_source must be non-negative")
        if self.test_rotations_per_source < 0:
            raise ValueError("test_rotations_per_source must be non-negative")
        if self.num_source_images <= 0:
            raise ValueError("num_source_images must be positive")
        if self.variant == "sr_mnist" and self.num_source_images != 10:
            raise ValueError(
                "sr_mnist uses exactly the ten fixed paper source indices; "
                "num_source_images must be 10"
            )
        if not self.include_original and (
            self.train_rotations_per_source == 0
            or self.test_rotations_per_source == 0
        ):
            raise ValueError(
                "each split needs at least one rotation when include_original=False"
            )
        if not isinstance(self.dtype, torch.dtype) or not self.dtype.is_floating_point:
            raise TypeError("dtype must be a floating-point torch.dtype")

    @property
    def sample_counts(self) -> tuple[int, int]:
        """Return ``(train_count, test_count)`` for this configuration."""

        config = self.resolved()
        extra = int(config.include_original)
        return (
            config.num_source_images
            * (config.train_rotations_per_source + extra),
            config.num_source_images
            * (config.test_rotations_per_source + extra),
        )

    @property
    def is_paper_profile(self) -> bool:
        config = self.resolved()
        defaults = _PROFILE_DEFAULTS[config.variant]
        return (
            all(getattr(config, key) == value for key, value in defaults.items())
            and config.angle_min == -45.0
            and config.angle_max == 45.0
            and config.scaling
        )


@dataclass(frozen=True)
class RotatedMNISTSplit:
    """One aligned rotated-MNIST tensor split."""

    images: torch.Tensor
    angles_deg: torch.Tensor
    digit_labels: torch.Tensor
    source_ids: torch.Tensor
    source_mnist_indices: torch.Tensor
    is_original: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    def __len__(self) -> int:
        return self.images.shape[0]

    def validate(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1:] != (1, 28, 28):
            raise ValueError(
                "images must have shape (N, 1, 28, 28), got "
                f"{tuple(self.images.shape)}"
            )
        if not self.images.dtype.is_floating_point:
            raise TypeError("images must have a floating-point dtype")
        n = self.images.shape[0]
        vectors = {
            "angles_deg": self.angles_deg,
            "digit_labels": self.digit_labels,
            "source_ids": self.source_ids,
            "source_mnist_indices": self.source_mnist_indices,
            "is_original": self.is_original,
        }
        for name, tensor in vectors.items():
            if tensor.ndim != 1 or tensor.shape[0] != n:
                raise ValueError(f"{name} must have shape ({n},)")
        if not self.angles_deg.dtype.is_floating_point:
            raise TypeError("angles_deg must have a floating-point dtype")
        if self.angles_deg.dtype != self.images.dtype:
            raise TypeError("angles_deg and images must have the same dtype")
        for name in ("digit_labels", "source_ids", "source_mnist_indices"):
            if vectors[name].dtype != torch.long:
                raise TypeError(f"{name} must have dtype torch.long")
        if self.is_original.dtype != torch.bool:
            raise TypeError("is_original must have dtype torch.bool")

    def subset(self, indices: Any) -> "RotatedMNISTSplit":
        """Return an aligned subset without modifying this split."""

        if isinstance(indices, int):
            indices = [indices]
        return RotatedMNISTSplit(
            images=self.images[indices],
            angles_deg=self.angles_deg[indices],
            digit_labels=self.digit_labels[indices],
            source_ids=self.source_ids[indices],
            source_mnist_indices=self.source_mnist_indices[indices],
            is_original=self.is_original[indices],
        )

    def flattened_images(self) -> torch.Tensor:
        return self.images.flatten(start_dim=1)

    def to(self, device: torch.device | str) -> "RotatedMNISTSplit":
        """Return a split whose tensors are on ``device``."""

        return RotatedMNISTSplit(
            images=self.images.to(device),
            angles_deg=self.angles_deg.to(device),
            digit_labels=self.digit_labels.to(device),
            source_ids=self.source_ids.to(device),
            source_mnist_indices=self.source_mnist_indices.to(device),
            is_original=self.is_original.to(device),
        )


@dataclass(frozen=True)
class RotatedMNISTMetadata:
    generator_version: str
    source: str
    rotation_backend: str
    source_selection: str
    paper_profile: bool
    cache_hit: bool
    cache_path: Path | None


@dataclass(frozen=True)
class RotatedMNISTData:
    train: RotatedMNISTSplit
    test: RotatedMNISTSplit
    config: RotatedMNISTConfig
    train_source_indices: torch.Tensor
    test_source_indices: torch.Tensor
    metadata: RotatedMNISTMetadata

    def __post_init__(self) -> None:
        if self.train_source_indices.dtype != torch.long:
            raise TypeError("train_source_indices must have dtype torch.long")
        if self.test_source_indices.dtype != torch.long:
            raise TypeError("test_source_indices must have dtype torch.long")
        if self.train_source_indices.ndim != 1 or self.test_source_indices.ndim != 1:
            raise ValueError("selected source indices must be one-dimensional")
        if len(self.train_source_indices) != self.config.resolved().num_source_images:
            raise ValueError("train_source_indices count does not match configuration")
        if len(self.test_source_indices) != self.config.resolved().num_source_images:
            raise ValueError("test_source_indices count does not match configuration")
        expected_train, expected_test = self.config.sample_counts
        if len(self.train) != expected_train or len(self.test) != expected_test:
            raise ValueError(
                "split sizes do not match configuration: expected "
                f"({expected_train}, {expected_test}), got "
                f"({len(self.train)}, {len(self.test)})"
            )


def select_source_indices(
    num_available: int,
    num_sources: int,
    *,
    generator: torch.Generator | None = None,
    fixed_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select valid source indices, either fixed or uniformly without replacement."""

    if num_sources <= 0:
        raise ValueError("num_sources must be positive")
    if num_sources > num_available:
        raise ValueError(
            f"cannot select {num_sources} sources from {num_available} samples"
        )
    if fixed_indices is not None:
        indices = torch.as_tensor(fixed_indices, dtype=torch.long).clone()
        if indices.ndim != 1 or len(indices) != num_sources:
            raise ValueError("fixed_indices must contain exactly num_sources entries")
        if len(torch.unique(indices)) != len(indices):
            raise ValueError("fixed_indices must not contain duplicates")
        if bool((indices < 0).any()) or bool((indices >= num_available).any()):
            raise ValueError("fixed source index is outside the available dataset")
        return indices
    return torch.randperm(num_available, generator=generator)[:num_sources]


def generate_rotated_split(
    source_images: torch.Tensor,
    digit_labels: torch.Tensor,
    source_mnist_indices: torch.Tensor,
    *,
    rotations_per_source: int,
    angle_min: float = -45.0,
    angle_max: float = 45.0,
    include_original: bool = False,
    scaling: bool = True,
    dtype: torch.dtype = torch.float32,
    rng: np.random.Generator | None = None,
    shuffle: bool = False,
    shuffle_generator: torch.Generator | None = None,
) -> RotatedMNISTSplit:
    """Generate an aligned split from already-selected source tensors.

    SciPy is deliberately called with ``reshape=False`` and all other
    interpolation arguments left at its defaults, matching the reference
    implementation.  Source-major ordering is used unless ``shuffle=True``.
    """

    images = torch.as_tensor(source_images).detach().cpu()
    if images.ndim == 4 and images.shape[1] == 1:
        images = images[:, 0]
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError("source_images must have shape (S, 28, 28) or (S, 1, 28, 28)")
    labels = torch.as_tensor(digit_labels, dtype=torch.long).detach().cpu()
    mnist_indices = torch.as_tensor(
        source_mnist_indices, dtype=torch.long
    ).detach().cpu()
    num_sources = images.shape[0]
    if labels.shape != (num_sources,) or mnist_indices.shape != (num_sources,):
        raise ValueError("labels and source_mnist_indices must have shape (S,)")
    if rotations_per_source < 0:
        raise ValueError("rotations_per_source must be non-negative")
    if not include_original and rotations_per_source == 0:
        raise ValueError("at least one rotation is required when originals are excluded")
    if not angle_min < angle_max:
        raise ValueError("angle_min must be strictly less than angle_max")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating-point torch.dtype")

    rng = np.random.default_rng() if rng is None else rng
    random_angles = rng.uniform(
        low=angle_min,
        high=angle_max,
        size=(num_sources, rotations_per_source),
    )
    per_source = rotations_per_source + int(include_original)
    total = num_sources * per_source
    output_images = torch.empty((total, 1, 28, 28), dtype=dtype)
    output_angles = torch.empty(total, dtype=dtype)
    output_is_original = torch.zeros(total, dtype=torch.bool)

    source_arrays = images.contiguous().numpy()
    for source_id in range(num_sources):
        offset = source_id * per_source
        rotation_offset = offset
        if include_original:
            output_images[offset, 0].copy_(images[source_id].to(dtype=dtype))
            output_angles[offset] = 0.0
            output_is_original[offset] = True
            rotation_offset += 1
        for rotation_id, angle in enumerate(random_angles[source_id]):
            rotated = ndimage.rotate(
                source_arrays[source_id], float(angle), reshape=False
            )
            output_images[rotation_offset + rotation_id, 0].copy_(
                torch.as_tensor(rotated).to(dtype=dtype)
            )
            output_angles[rotation_offset + rotation_id] = float(angle)

    if scaling:
        output_images.sub_(255.0 / 2.0).div_(255.0)

    output_labels = labels.repeat_interleave(per_source)
    output_source_ids = torch.arange(num_sources, dtype=torch.long).repeat_interleave(
        per_source
    )
    output_mnist_indices = mnist_indices.repeat_interleave(per_source)
    split = RotatedMNISTSplit(
        images=output_images,
        angles_deg=output_angles,
        digit_labels=output_labels,
        source_ids=output_source_ids,
        source_mnist_indices=output_mnist_indices,
        is_original=output_is_original,
    )
    if shuffle:
        order = torch.randperm(total, generator=shuffle_generator)
        split = split.subset(order)
    return split


def load_mnist_tensors(
    root: str | Path,
    *,
    download: bool,
    dataset_factory: MNISTFactory | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load torchvision MNIST train/test tensors without TensorFlow."""

    if dataset_factory is None:
        try:
            from torchvision.datasets import MNIST
        except ImportError as error:
            raise ImportError(
                "torchvision is required to load MNIST; install project requirements"
            ) from error
        dataset_factory = MNIST

    train_dataset = dataset_factory(root=str(root), train=True, download=download)
    test_dataset = dataset_factory(root=str(root), train=False, download=download)
    train_images, train_labels = _extract_mnist_tensors(train_dataset, "train")
    test_images, test_labels = _extract_mnist_tensors(test_dataset, "test")
    return train_images, train_labels, test_images, test_labels


def generate_rotated_mnist(
    config: RotatedMNISTConfig,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    test_images: torch.Tensor,
    test_labels: torch.Tensor,
) -> RotatedMNISTData:
    """Generate rotated MNIST from in-memory MNIST tensors, without caching."""

    config = config.resolved()
    train_images, train_labels = _validate_mnist_tensors(
        train_images, train_labels, "train"
    )
    test_images, test_labels = _validate_mnist_tensors(
        test_images, test_labels, "test"
    )

    if config.variant == "sr_mnist":
        train_source_indices = select_source_indices(
            len(train_images),
            config.num_source_images,
            fixed_indices=SR_MNIST_SOURCE_INDICES,
        )
        # The SR paper profile, like the repository, rotates the same ten source
        # images for both splits rather than selecting MNIST test images.
        test_source_indices = train_source_indices.clone()
        train_source_images = train_images[train_source_indices]
        train_source_labels = train_labels[train_source_indices]
        test_source_images = train_source_images
        test_source_labels = train_source_labels
        source_selection = "fixed ten MNIST training indices for both splits"
    else:
        source_generator = _torch_generator(config.source_selection_seed)
        train_source_indices = select_source_indices(
            len(train_images), config.num_source_images, generator=source_generator
        )
        test_source_indices = select_source_indices(
            len(test_images), config.num_source_images, generator=source_generator
        )
        train_source_images = train_images[train_source_indices]
        train_source_labels = train_labels[train_source_indices]
        test_source_images = test_images[test_source_indices]
        test_source_labels = test_labels[test_source_indices]
        source_selection = "random without replacement from each MNIST split"

    rotation_sequences = np.random.SeedSequence(config.rotation_seed).spawn(2)
    train_rotation_rng = np.random.default_rng(rotation_sequences[0])
    test_rotation_rng = np.random.default_rng(rotation_sequences[1])
    shuffle_generator = _torch_generator(config.shuffle_seed)
    train = generate_rotated_split(
        train_source_images,
        train_source_labels,
        train_source_indices,
        rotations_per_source=config.train_rotations_per_source,
        angle_min=config.angle_min,
        angle_max=config.angle_max,
        include_original=config.include_original,
        scaling=config.scaling,
        dtype=config.dtype,
        rng=train_rotation_rng,
        shuffle=config.shuffle,
        shuffle_generator=shuffle_generator,
    )
    test = generate_rotated_split(
        test_source_images,
        test_source_labels,
        test_source_indices,
        rotations_per_source=config.test_rotations_per_source,
        angle_min=config.angle_min,
        angle_max=config.angle_max,
        include_original=config.include_original,
        scaling=config.scaling,
        dtype=config.dtype,
        rng=test_rotation_rng,
        shuffle=config.shuffle,
        shuffle_generator=shuffle_generator,
    )
    metadata = RotatedMNISTMetadata(
        generator_version=_GENERATOR_VERSION,
        source="torchvision.datasets.MNIST",
        rotation_backend="scipy.ndimage.rotate(reshape=False; default interpolation)",
        source_selection=source_selection,
        paper_profile=config.is_paper_profile,
        cache_hit=False,
        cache_path=None,
    )
    return RotatedMNISTData(
        train=train,
        test=test,
        config=config,
        train_source_indices=train_source_indices,
        test_source_indices=test_source_indices,
        metadata=metadata,
    )


def load_rotated_mnist(
    config: RotatedMNISTConfig,
    *,
    regenerate: bool = False,
    use_cache: bool = True,
    dataset_factory: MNISTFactory | None = None,
) -> RotatedMNISTData:
    """Load a cached dataset or generate it from torchvision MNIST.

    ``dataset_factory`` is an injection point for tests; normal callers should
    leave it unset.  Cache files contain only tensors and primitive metadata.
    """

    config = config.resolved()
    cache_path = _cache_path(config)
    signature = _cache_signature(config)
    if use_cache and cache_path.is_file() and not regenerate:
        data = _load_cache(cache_path, config, signature)
        return replace(
            data,
            metadata=replace(
                data.metadata,
                cache_hit=True,
                cache_path=cache_path,
            ),
        )

    mnist_tensors = load_mnist_tensors(
        config.root, download=config.download, dataset_factory=dataset_factory
    )
    data = generate_rotated_mnist(config, *mnist_tensors)
    data = replace(
        data,
        metadata=replace(
            data.metadata,
            cache_hit=False,
            cache_path=cache_path if use_cache else None,
        ),
    )
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _save_cache(cache_path, signature, data)
    return data


def _extract_mnist_tensors(
    dataset: Any, split_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(dataset, "data") or not hasattr(dataset, "targets"):
        raise TypeError(
            f"{split_name} MNIST dataset must expose .data and .targets tensors"
        )
    return _validate_mnist_tensors(dataset.data, dataset.targets, split_name)


def _validate_mnist_tensors(
    images: torch.Tensor, labels: torch.Tensor, split_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.as_tensor(images).detach().cpu()
    labels = torch.as_tensor(labels, dtype=torch.long).detach().cpu()
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(
            f"{split_name} MNIST images must have shape (N, 28, 28)"
        )
    if labels.shape != (len(images),):
        raise ValueError(f"{split_name} MNIST labels must have shape (N,)")
    return images, labels


def _torch_generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def _config_payload(config: RotatedMNISTConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload.pop("root")
    payload.pop("cache_dir")
    payload.pop("download")
    payload["dtype"] = str(config.dtype)
    payload["generator_version"] = _GENERATOR_VERSION
    return payload


def _cache_signature(config: RotatedMNISTConfig) -> str:
    serialized = json.dumps(
        _config_payload(config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_path(config: RotatedMNISTConfig) -> Path:
    signature = _cache_signature(config)
    return Path(config.cache_dir) / f"{config.variant}-{signature[:16]}.pt"


def _split_state(split: RotatedMNISTSplit) -> dict[str, torch.Tensor]:
    return {field: getattr(split, field) for field in RotatedMNISTSplit.__dataclass_fields__}


def _save_cache(
    cache_path: Path, signature: str, data: RotatedMNISTData
) -> None:
    payload = {
        "signature": signature,
        "train": _split_state(data.train),
        "test": _split_state(data.test),
        "train_source_indices": data.train_source_indices,
        "test_source_indices": data.test_source_indices,
        "source_selection": data.metadata.source_selection,
        "paper_profile": data.metadata.paper_profile,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_path.parent, prefix=f".{cache_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_cache(
    cache_path: Path, config: RotatedMNISTConfig, signature: str
) -> RotatedMNISTData:
    try:
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch before weights_only was introduced.
        payload = torch.load(cache_path, map_location="cpu")
    if payload.get("signature") != signature:
        raise ValueError(
            f"cache signature mismatch for {cache_path}; regenerate the dataset"
        )
    train = RotatedMNISTSplit(**payload["train"])
    test = RotatedMNISTSplit(**payload["test"])
    metadata = RotatedMNISTMetadata(
        generator_version=_GENERATOR_VERSION,
        source="torchvision.datasets.MNIST",
        rotation_backend="scipy.ndimage.rotate(reshape=False; default interpolation)",
        source_selection=payload["source_selection"],
        paper_profile=bool(payload["paper_profile"]),
        cache_hit=True,
        cache_path=cache_path,
    )
    return RotatedMNISTData(
        train=train,
        test=test,
        config=config,
        train_source_indices=payload["train_source_indices"],
        test_source_indices=payload["test_source_indices"],
        metadata=metadata,
    )


__all__ = [
    "SR_MNIST_SOURCE_INDICES",
    "RotatedMNISTConfig",
    "RotatedMNISTData",
    "RotatedMNISTMetadata",
    "RotatedMNISTSplit",
    "generate_rotated_mnist",
    "generate_rotated_split",
    "load_mnist_tensors",
    "load_rotated_mnist",
    "select_source_indices",
]
