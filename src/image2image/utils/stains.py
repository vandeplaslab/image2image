"""Fast optical-density deconvolution, profile estimation, and reference reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numba
import numpy as np

from image2image.models.stains import StainEstimation, StainProfile


def stain_matrix(profile: StainProfile) -> np.ndarray:
    """Return row-wise unit stain vectors, completing two stains with a signed residual."""
    matrix = np.asarray(profile.vectors, dtype=np.float64)
    if len(matrix) == 2:
        residual = np.cross(matrix[0], matrix[1])
        matrix = np.vstack((matrix, residual / np.linalg.norm(residual)))
    return matrix


def _validate_rgb(image: np.ndarray) -> None:
    if image.ndim < 2 or image.shape[-1] != 3 or image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("Stain processing requires uint8 or uint16 pixels with a final RGB axis.")
    if image.size == 0:
        raise ValueError("The selected image region is empty.")


def optical_density(image: np.ndarray, profile: StainProfile) -> np.ndarray:
    """Convert integer RGB to natural-log OD, clamping zero to one native intensity unit."""
    _validate_rgb(image)
    white = np.asarray(profile.white) * np.iinfo(image.dtype).max
    return -np.log(np.minimum(np.maximum(image.astype(np.float64), 1) / white, 1))


@lru_cache(maxsize=32)
def _transform(vectors: tuple, white: tuple, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    maximum = np.iinfo(np.dtype(dtype)).max
    pixels = np.maximum(np.arange(maximum + 1, dtype=np.float64), 1)
    lookup = -np.log(np.minimum(pixels[:, None] / (np.asarray(white) * maximum), 1)).astype(np.float32)
    rows = np.asarray(vectors)
    if len(rows) == 2:
        residual = np.cross(rows[0], rows[1])
        rows = np.vstack((rows, residual / np.linalg.norm(residual)))
    return lookup, np.linalg.inv(rows).astype(np.float32)


@numba.njit(cache=True, nogil=True)
def _separate(pixels: np.ndarray, lookup: np.ndarray, inverse: np.ndarray) -> np.ndarray:
    result = np.empty((len(pixels), 3), dtype=np.float32)
    for i in range(len(pixels)):
        red = lookup[pixels[i, 0], 0]
        green = lookup[pixels[i, 1], 1]
        blue = lookup[pixels[i, 2], 2]
        for j in range(3):
            result[i, j] = red * inverse[0, j] + green * inverse[1, j] + blue * inverse[2, j]
    return result


def separate_stains(image: np.ndarray, profile: StainProfile, *, compiled: bool = True) -> np.ndarray:
    """Return three signed float32 OD coefficients without display scaling or intensity matching."""
    image = np.asarray(image)
    _validate_rgb(image)
    if not compiled:
        return (optical_density(image, profile) @ np.linalg.inv(stain_matrix(profile))).astype(np.float32)
    lookup, inverse = _transform(profile.vectors, profile.white, image.dtype.name)
    return _separate(np.ascontiguousarray(image).reshape(-1, 3), lookup, inverse).reshape(image.shape)


@numba.njit(cache=True, nogil=True)
def _reconstruct(pixels: np.ndarray, matrix: np.ndarray, white: np.ndarray, maximum: int) -> np.ndarray:
    result = np.empty((len(pixels), 3), dtype=np.float32)
    for i in range(len(pixels)):
        for j in range(3):
            density = pixels[i, 0] * matrix[0, j] + pixels[i, 1] * matrix[1, j] + pixels[i, 2] * matrix[2, j]
            # Clip OD before exp to avoid overflow from large, signed residual coefficients.
            value = white[j] * np.exp(-max(-80.0, min(80.0, density)))
            result[i, j] = np.rint(max(0.0, min(1.0, value)) * maximum)
    return result


def reconstruct_stains(
    coefficients: np.ndarray,
    profile: StainProfile,
    *,
    dtype: str | np.dtype = "uint8",
    residual_profile: StainProfile | None = None,
    scales: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Reconstruct RGB, optionally retaining the original residual and scaling real stains."""
    dtype = np.dtype(dtype)
    values = np.asarray(coefficients, dtype=np.float32)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")) or values.ndim < 2 or values.shape[-1] != 3:
        raise ValueError("Reconstruction needs three coefficients and uint8 or uint16 output.")
    if not np.isfinite(values).all() or len(scales) != 3 or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("Coefficients and positive reconstruction scales must be finite.")
    matrix = stain_matrix(profile)
    if residual_profile is not None and len(profile.names) == 2:
        matrix[2] = stain_matrix(residual_profile)[2]
    matrix *= np.asarray(scales)[:, None]
    return (
        _reconstruct(
            np.ascontiguousarray(values).reshape(-1, 3),
            matrix.astype(np.float32),
            np.asarray(profile.white, dtype=np.float32),
            np.iinfo(dtype).max,
        )
        .reshape(values.shape)
        .astype(dtype)
    )


def tissue_mask(image: np.ndarray, profile: StainProfile, settings: StainEstimation) -> np.ndarray:
    """Select informative tissue while excluding white background and dense or clipped pixels."""
    density = optical_density(image, profile)
    return (
        (np.linalg.norm(density, axis=-1) >= settings.min_od)
        & (density.max(axis=-1) <= settings.max_od)
        & np.all(image > 1, axis=-1)
    )


def estimate_stain_profile(
    samples: np.ndarray, template: StainProfile, settings: StainEstimation | None = None
) -> StainProfile:
    """Fit two stain directions with angular percentiles and preserve template stain ordering."""
    settings = settings or StainEstimation()
    if len(template.names) != 2:
        raise ValueError("Automatic estimation supports two stains. Edit or sample three-stain vectors manually.")
    samples = np.asarray(samples)
    _validate_rgb(samples)
    samples = samples.reshape(-1, 3)
    if len(samples) > settings.max_samples:
        indices = np.random.default_rng(settings.seed).choice(len(samples), settings.max_samples, replace=False)
        samples = samples[indices]
    density = optical_density(samples, template)[tissue_mask(samples, template, settings)]
    if len(density) < 64:
        raise ValueError("Too little usable tissue. Select a representative region containing both stains.")
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(density, rowvar=False))
    if eigenvalues[-1] <= 1e-10 or eigenvalues[-2] / eigenvalues[-1] < 0.001:
        raise ValueError("Two distinct stains could not be identified. Sample both stains or use manual vectors.")
    plane = eigenvectors[:, [-1, -2]]
    if np.dot(density.mean(axis=0), plane[:, 0]) < 0:
        plane[:, 0] *= -1
    projected = density @ plane
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    ends = np.percentile(angles, [settings.angular_percentile, 100 - settings.angular_percentile])
    vectors = (plane @ np.vstack((np.cos(ends), np.sin(ends)))).T
    if vectors.min() < -0.05:
        raise ValueError("Unexpected tissue colors prevent estimation. Select a cleaner region or edit the vectors.")
    vectors = np.maximum(vectors, 0)
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    similarities = vectors @ np.asarray(template.vectors).T
    direct, swapped = np.trace(similarities), np.trace(similarities[::-1])
    if abs(direct - swapped) < 0.01:
        raise ValueError("Stain ordering is ambiguous. Refine the starting vectors or sample individual stains.")
    if swapped > direct:
        vectors = vectors[::-1]
    return StainProfile(
        name=template.name, names=template.names, vectors=tuple(map(tuple, vectors)), white=template.white
    )


def sample_stain_vector(samples: np.ndarray, profile: StainProfile, index: int) -> StainProfile:
    """Replace one vector with the mean OD direction of a region containing a single stain."""
    if not 0 <= index < len(profile.names):
        raise ValueError("Select a real stain, not the residual, to sample.")
    density = optical_density(samples, profile).reshape(-1, 3)
    density = density[np.linalg.norm(density, axis=1) >= 0.15]
    if len(density) < 16:
        raise ValueError("Select a region containing a clear example of this stain.")
    vectors = list(profile.vectors)
    vectors[index] = tuple(density.mean(axis=0))
    return StainProfile(name=profile.name, names=profile.names, vectors=tuple(vectors), white=profile.white)


def sample_background(samples: np.ndarray, profile: StainProfile) -> StainProfile:
    """Set white RGB from the median of a user-selected blank slide region."""
    _validate_rgb(samples)
    white = tuple(np.median(samples.reshape(-1, 3), axis=0) / np.iinfo(samples.dtype).max)
    return StainProfile(name=profile.name, names=profile.names, vectors=profile.vectors, white=white)


@dataclass(frozen=True)
class StainFit:
    """A fitted profile and source-wide positive coefficient percentiles for reproducible scaling."""

    profile: StainProfile
    percentiles: tuple[float, float, float]
    sample_count: int


def fit_stains(
    samples: np.ndarray, template: StainProfile, settings: StainEstimation, *, estimate: bool = True
) -> StainFit:
    """Fit a profile and concentration statistics from the same bounded tissue sample."""
    samples = np.asarray(samples)
    _validate_rgb(samples)
    samples = samples.reshape(-1, 3)
    if len(samples) > settings.max_samples:
        indices = np.random.default_rng(settings.seed).choice(len(samples), settings.max_samples, replace=False)
        samples = samples[indices]
    profile = estimate_stain_profile(samples, template, settings) if estimate else template
    mask = tissue_mask(samples, profile, settings)
    coefficients = separate_stains(samples[mask], profile) if mask.any() else np.zeros((1, 3), dtype=np.float32)
    percentiles = np.percentile(np.maximum(coefficients, 0), settings.concentration_percentile, axis=0)
    return StainFit(profile, tuple(float(value) for value in percentiles), int(mask.sum()))


def normalization_scales(source: StainFit, reference: StainFit, match_intensity: bool = False) -> tuple[float, ...]:
    """Return per-stain reference ratios; color-only normalization leaves coefficients unchanged."""
    if source.profile.names != reference.profile.names:
        raise ValueError("Source and reference stain names/order must match.")
    scales = np.ones(3)
    if match_intensity:
        n = len(source.profile.names)
        left, right = np.asarray(source.percentiles[:n]), np.asarray(reference.percentiles[:n])
        if np.any(left <= 1e-6) or np.any(right <= 1e-6):
            raise ValueError("Intensity matching requires measurable amounts of every stain in source and reference.")
        scales[:n] = right / left
    return tuple(float(value) for value in scales)


def registration_channels(coefficients: np.ndarray, fit: StainFit) -> np.ndarray:
    """Scale positive coefficients to uint16 using frozen slide-wide percentiles; clip negatives."""
    scale = np.maximum(np.asarray(fit.percentiles, dtype=np.float32), 1e-6)
    return np.rint(np.clip(coefficients / scale, 0, 1) * 65535).astype(np.uint16)
