"""Image-similarity evidence for registration validation: assisted, dense and automatic correspondences.

All methods compare the fixed image with the moving image resampled *through* the registration under test,
so only a local translation remains to be estimated. Offsets are found by normalized cross-correlation and
reported with the same sign convention as independent landmarks: residual = predicted - observed.
This evidence is image based, not independent ground truth, and fails on low-texture or cross-modal images.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from threading import Event
from typing import Any

import numpy as np

from image2image.models.validation import ImageGeometry
from image2image.utils.validation import ValidationTransform, check_cancel, distance_statistics, sample_pixels

UNAVAILABLE_REASON = "Unavailable: too few reliable matches (low texture or cross-modal images)."


@dataclass
class TemplateMatch:
    """Local correspondence for one fixed point; offsets are physical XY in the fixed frame."""

    fixed_xy: tuple[float, float]
    offset_xy_um: tuple[float, float] | None = None
    moving_xy: tuple[float, float] | None = None
    score: float | None = None
    reason: str = ""

    @property
    def ok(self) -> bool:
        """Return whether a correspondence was found, regardless of its score."""
        return self.moving_xy is not None


def _window(center_xy: np.ndarray, half: int, spacing: float) -> ImageGeometry:
    """Square physical window of ``2 * half + 1`` samples centered on a physical point."""
    return ImageGeometry(
        shape=(2 * half + 1, 2 * half + 1),
        spacing=(spacing, spacing),
        origin=tuple(float(v) for v in np.asarray(center_xy) - half * spacing),
    )


def _window_xy(window: ImageGeometry) -> np.ndarray:
    yy, xx = np.indices(window.shape, dtype=np.float64)
    return window.to_physical(np.stack((xx, yy), axis=-1))


def _subpixel_peak(surface: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return the YX peak with parabolic refinement, and whether it lies on the surface border."""
    peak = np.array(np.unravel_index(int(np.nanargmax(surface)), surface.shape), dtype=float)
    border = bool(np.any(peak == 0) or np.any(peak == np.asarray(surface.shape) - 1))
    if not border:
        for axis in range(2):
            index = peak.astype(int)
            before, after = index.copy(), index.copy()
            before[axis] -= 1
            after[axis] += 1
            low, mid, high = surface[tuple(before)], surface[tuple(index)], surface[tuple(after)]
            denominator = low - 2 * mid + high
            if denominator < 0:
                peak[axis] += float(np.clip(0.5 * (low - high) / denominator, -0.5, 0.5))
    return peak, border


def match_through_transform(
    fixed_array: Any,
    fixed_geometry: ImageGeometry,
    moving_array: Any,
    moving_geometry: ImageGeometry,
    transform: ValidationTransform,
    fixed_xy: tuple[float, float] | np.ndarray,
    template_um: float,
    search_um: float,
    *,
    initial_offset_um: tuple[float, float] = (0.0, 0.0),
    max_template_px: int = 128,
    min_coverage: float = 0.9,
) -> TemplateMatch:
    """Locate the moving correspondence of a fixed point by NCC against the registered moving image.

    A fixed template around ``fixed_xy`` is compared with the moving image sampled through ``transform``
    on a fixed-frame search window. The best offset ``d`` means the fixed content at ``fixed_xy`` appears
    in the registered image at ``fixed_xy + d``, so the observed moving point is ``G(fixed_xy + d)``.
    """
    from skimage.feature import match_template

    fixed_xy = np.asarray(fixed_xy, dtype=float)
    result = TemplateMatch(fixed_xy=(float(fixed_xy[0]), float(fixed_xy[1])))
    spacing = max(float(min(fixed_geometry.spacing)), template_um / max_template_px)
    half = max(4, round(template_um / (2 * spacing)))
    radius = max(1, math.ceil(search_um / spacing))
    template_geometry = _window(fixed_xy, half, spacing)
    template, covered = sample_pixels(fixed_array, fixed_geometry, _window_xy(template_geometry))
    if covered.mean() < min_coverage:
        result.reason = "Template lies outside the fixed image"
        return result
    if float(np.std(template)) <= 1e-6 * max(1.0, float(np.max(np.abs(template)))):
        result.reason = "Template has no texture"
        return result
    center = fixed_xy + np.asarray(initial_offset_um, dtype=float)
    search_geometry = _window(center, half + radius, spacing)
    mapped = transform.map_grid(search_geometry)
    search, covered = sample_pixels(moving_array, moving_geometry, mapped)
    if covered.mean() < min_coverage:
        result.reason = "Search window lies outside the moving image"
        return result
    surface = match_template(search.astype(np.float32), template.astype(np.float32))
    if not np.isfinite(surface).any():
        result.reason = "Correlation is undefined"
        return result
    peak, border = _subpixel_peak(np.nan_to_num(surface, nan=-1.0))
    offset = np.asarray(initial_offset_um, dtype=float) + (peak[::-1] - radius) * spacing
    observed = transform.map_points((fixed_xy + offset)[None])[0]
    if not np.isfinite(observed).all():
        result.reason = "Transform produced non-finite coordinates"
        return result
    result.offset_xy_um = (float(offset[0]), float(offset[1]))
    result.moving_xy = (float(observed[0]), float(observed[1]))
    result.score = float(surface[tuple(np.round(peak).astype(int))])
    if border:
        result.reason = "Best match is on the search boundary; increase the search radius"
    return result


def residual_shift_map(
    fixed_image: np.ndarray,
    registered_image: np.ndarray,
    coverage: np.ndarray,
    grid: ImageGeometry,
    tile_um: float,
    *,
    min_confidence: float = 0.5,
    upsample_factor: int = 10,
    cancel: Event | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Estimate local residual misalignment on overlapping tiles of the common fixed grid.

    Shifts are physical XY residuals (predicted - observed) in the fixed frame. Tiles are rejected for
    incomplete coverage, low texture, low post-shift correlation, or shifts beyond a quarter tile.
    """
    from scipy import ndimage
    from skimage.registration import phase_cross_correlation

    spacing = np.asarray(grid.spacing, dtype=float)
    tile = max(16, round(tile_um / float(spacing.max())))
    step = max(1, tile // 2)
    ny = (grid.shape[0] - tile) // step + 1 if grid.shape[0] >= tile else 0
    nx = (grid.shape[1] - tile) // step + 1 if grid.shape[1] >= tile else 0
    shift = np.full((max(ny, 0), max(nx, 0), 2), np.nan, dtype=np.float32)
    confidence = np.full((max(ny, 0), max(nx, 0)), np.nan, dtype=np.float32)
    valid = np.zeros((max(ny, 0), max(nx, 0)), dtype=bool)
    finite = fixed_image[np.isfinite(fixed_image)]
    texture = 1e-3 * (float(np.ptp(finite)) if finite.size else 0.0)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32)
    limit = tile / 4
    for i in range(ny):
        check_cancel(cancel)
        for j in range(nx):
            y, x = i * step, j * step
            if coverage[y : y + tile, x : x + tile].mean() < 0.95:
                continue
            f = fixed_image[y : y + tile, x : x + tile].astype(np.float32)
            m = registered_image[y : y + tile, x : x + tile].astype(np.float32)
            if f.std() <= texture or m.std() <= texture:
                continue
            f_w = (f - f.mean()) * window
            estimate = np.zeros(2)
            # Windowing biases a single estimate toward zero; re-estimate on the shifted tile to remove it.
            for _ in range(3):
                current = ndimage.shift(m, estimate, order=3, mode="nearest") if estimate.any() else m
                update, _, _ = phase_cross_correlation(
                    f_w, (current - current.mean()) * window, upsample_factor=upsample_factor
                )
                estimate = estimate + update
                if np.max(np.abs(update)) < 0.5 / upsample_factor:
                    break
            if np.max(np.abs(estimate)) > limit:
                continue
            # The registered content at fixed position p appears in m at p - estimate.
            k = math.ceil(np.max(np.abs(estimate))) + 1
            dy, dx = (round(v) for v in estimate)
            a = f[k : tile - k, k : tile - k]
            b = m[k - dy : tile - k - dy, k - dx : tile - k - dx]
            if a.size < 16 or a.std() == 0 or b.std() == 0:
                continue
            score = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
            confidence[i, j] = score
            # Fixed content at p appears in the registered image at p - estimate, so the registration
            # predicts p where the content is observed: predicted - observed = estimate (XY, um).
            shift[i, j] = estimate[::-1] * spacing
            valid[i, j] = score >= min_confidence
    magnitude = np.linalg.norm(shift, axis=-1).astype(np.float32)
    magnitude[~valid] = np.nan
    summary: dict[str, Any] = distance_statistics(magnitude[valid])
    summary.update(
        tiles=int(valid.size),
        valid_tiles=int(valid.sum()),
        valid_fraction=float(valid.mean()) if valid.size else 0.0,
        tile_um=float(tile * spacing.max()),
        resolution_um=float(spacing.max() / upsample_factor),
        frame="fixed",
        units="um",
        reason="" if valid.sum() >= 3 else UNAVAILABLE_REASON,
    )
    center = (tile - 1) / 2
    maps = {
        "shift_x_um": shift[..., 0],
        "shift_y_um": shift[..., 1],
        "shift_um": magnitude,
        "shift_confidence": confidence,
        "shift_valid": valid,
        "shift_origin_xy_um": np.asarray(grid.origin) + center * spacing,
        "shift_spacing_xy_um": step * spacing,
    }
    return maps, summary


def _normalize(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, (1, 99))
    return np.clip((np.nan_to_num(image) - low) / max(float(high - low), 1e-9), 0, 1).astype(np.float32)


def _outliers(errors: np.ndarray) -> np.ndarray:
    """Flag errors more than 3.5 robust standard deviations above the median."""
    if len(errors) < 5:
        return np.zeros(len(errors), dtype=bool)
    median = np.median(errors)
    mad = 1.4826 * np.median(np.abs(errors - median))
    return errors > median + 3.5 * max(mad, 1e-12)


def automatic_landmarks(
    fixed_image: np.ndarray,
    registered_image: np.ndarray,
    coverage: np.ndarray,
    grid: ImageGeometry,
    transform: ValidationTransform,
    fixed_array: Any,
    fixed_geometry: ImageGeometry,
    moving_array: Any,
    moving_geometry: ImageGeometry,
    template_um: float,
    search_um: float,
    *,
    max_points: int = 300,
    min_score: float = 0.6,
    max_detection_px: int = 1500,
    cancel: Event | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Detect SIFT correspondences between fixed and registered images, refine them natively, and score residuals.

    Rows match the independent landmark schema (moving frame, um) with ``score`` and ``outlier`` added.
    """
    from skimage.feature import SIFT, match_descriptors

    summary: dict[str, Any] = distance_statistics(np.array([]))
    summary.update(detected=0, matched=0, refined=0, outliers=0, frame="moving", units="um", reason="")
    factor = max(1, math.ceil(max(grid.shape) / max_detection_px))
    images = []
    for image in (fixed_image, registered_image):
        reduced = image[::factor, ::factor]
        images.append(_normalize(np.where(coverage[::factor, ::factor], reduced, np.nan)))
    keypoints, descriptors = [], []
    for image in images:
        check_cancel(cancel)
        detector = SIFT()
        try:
            detector.detect_and_extract(image)
        except (RuntimeError, ValueError):
            summary["reason"] = UNAVAILABLE_REASON
            return [], summary
        keypoints.append(detector.keypoints)
        descriptors.append(detector.descriptors)
    summary["detected"] = len(keypoints[0])
    matches = match_descriptors(descriptors[0], descriptors[1], cross_check=True, max_ratio=0.8)
    detection = ImageGeometry(
        shape=images[0].shape, spacing=tuple(v * factor for v in grid.spacing), origin=grid.origin
    )
    fixed_xy = detection.to_physical(keypoints[0][matches[:, 0]][:, ::-1].astype(float))
    registered_xy = detection.to_physical(keypoints[1][matches[:, 1]][:, ::-1].astype(float))
    offsets = registered_xy - fixed_xy
    keep = np.linalg.norm(offsets, axis=1) <= search_um
    fixed_xy, offsets = fixed_xy[keep], offsets[keep]
    summary["matched"] = len(fixed_xy)
    if len(fixed_xy) > max_points:
        chosen = np.sort(np.random.default_rng(0).choice(len(fixed_xy), max_points, replace=False))
        fixed_xy, offsets = fixed_xy[chosen], offsets[chosen]
    tolerance = 2 * float(max(detection.spacing))
    rows = []
    for point, offset in zip(fixed_xy, offsets, strict=True):
        check_cancel(cancel)
        match = match_through_transform(
            fixed_array,
            fixed_geometry,
            moving_array,
            moving_geometry,
            transform,
            point,
            template_um,
            tolerance,
            initial_offset_um=tuple(offset),
        )
        if not match.ok or match.reason or match.score is None or match.score < min_score:
            continue
        if np.linalg.norm(np.asarray(match.offset_xy_um) - offset) > tolerance:
            continue
        predicted = transform.map_points(point[None])[0]
        residual = predicted - np.asarray(match.moving_xy)
        rows.append(
            {
                "id": f"auto-{len(rows) + 1}",
                "fixed_x": float(point[0]),
                "fixed_y": float(point[1]),
                "moving_x": match.moving_xy[0],
                "moving_y": match.moving_xy[1],
                "dx_um": float(residual[0]),
                "dy_um": float(residual[1]),
                "error_um": float(np.linalg.norm(residual)),
                "score": match.score,
                "outlier": False,
            }
        )
    errors = np.array([row["error_um"] for row in rows], dtype=float)
    for row, flag in zip(rows, _outliers(errors), strict=True):
        row["outlier"] = bool(flag)
    summary.update(distance_statistics(errors))
    summary.update(
        refined=len(rows),
        outliers=int(sum(row["outlier"] for row in rows)),
        reason="" if len(rows) >= 3 else UNAVAILABLE_REASON,
    )
    return rows, summary
