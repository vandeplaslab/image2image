"""Automated image-similarity evidence and supporting validation helpers."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import gaussian_filter, shift

from image2image.models.validation import (
    ImageGeometry,
    ValidationCandidate,
    ValidationLandmark,
    ValidationSession,
)
from image2image.utils.validation import (
    ValidationTransform,
    analyze_validation_session,
    export_validation_landmarks,
    import_validation_landmarks,
    inspect_validation_image,
    landmark_errors,
    landmark_warnings,
    merge_landmarks,
    validation_image_pyramid,
)
from image2image.utils.validation_auto import (
    UNAVAILABLE_REASON,
    automatic_landmarks,
    detect_landmarks,
    match_through_transform,
    residual_shift_map,
)
from image2image.utils.validation_report import export_validation_report

ALIGNED = ValidationCandidate(id="aligned", name="aligned", kind="aligned", path="")


def textured(shape: tuple[int, int] = (256, 256), seed: int = 0) -> np.ndarray:
    """Smooth random texture with distinctive blobs at several scales."""
    rng = np.random.default_rng(seed)
    image = gaussian_filter(rng.random(shape), 3) + 0.5 * gaussian_filter(rng.random(shape), 8)
    return ((image - image.min()) / np.ptp(image) * 1000).astype(np.float32)


def translated(image: np.ndarray, dx_px: float, dy_px: float) -> np.ndarray:
    """Content at p in ``image`` appears at p + (dx, dy) in the result."""
    return shift(image, (dy_px, dx_px), order=3, mode="reflect").astype(np.float32)


def test_match_through_identity_recovers_translation() -> None:
    spacing = 0.5
    fixed = textured()
    moving = translated(fixed, 3.4, -2.1)
    geometry = ImageGeometry(shape=fixed.shape, spacing=(spacing, spacing))
    match = match_through_transform(
        fixed, geometry, moving, geometry, ValidationTransform(ALIGNED, matrix=np.eye(3)), (64.0, 64.0), 24, 4
    )
    assert match.ok and not match.reason and match.score > 0.9
    np.testing.assert_allclose(match.offset_xy_um, (3.4 * spacing, -2.1 * spacing), atol=0.2 * spacing)
    np.testing.assert_allclose(match.moving_xy, (64 + 3.4 * spacing, 64 - 2.1 * spacing), atol=0.2 * spacing)


def test_match_through_rotation_and_scale_recovers_local_offset() -> None:
    """The registered image removes rotation/scale; only the deliberate local offset remains."""
    fixed = textured((300, 300), seed=3)
    angle, scale = np.deg2rad(20), 1.2
    rotation = scale * np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = np.array([150.0, 150.0])
    offset = np.array([2.0, -1.5])  # moving content is displaced by this in the moving frame
    # moving(q) = fixed(R^-1 (q - offset - center) + center)
    yy, xx = np.indices(fixed.shape, dtype=float)
    q = np.stack((xx, yy), axis=-1) - offset - center
    source = q @ np.linalg.inv(rotation).T + center
    from scipy.ndimage import map_coordinates

    moving = map_coordinates(fixed, [source[..., 1], source[..., 0]], order=3, mode="reflect").astype(np.float32)
    matrix = np.eye(3)
    matrix[:2, :2] = rotation
    matrix[:2, 2] = center - rotation @ center
    transform = ValidationTransform(ALIGNED, matrix=matrix)
    geometry = ImageGeometry(shape=fixed.shape)
    point = np.array([140.0, 160.0])
    match = match_through_transform(fixed, geometry, moving, geometry, transform, point, 40, 6)
    assert match.ok and match.score > 0.9
    expected_moving = transform.map_points(point[None])[0] + offset
    np.testing.assert_allclose(match.moving_xy, expected_moving, atol=0.3)


def test_match_rejects_flat_templates_and_outside_points() -> None:
    flat = np.ones((64, 64), dtype=np.float32)
    geometry = ImageGeometry(shape=flat.shape)
    transform = ValidationTransform(ALIGNED, matrix=np.eye(3))
    assert not match_through_transform(flat, geometry, flat, geometry, transform, (32, 32), 16, 4).ok
    assert "outside" in match_through_transform(flat, geometry, flat, geometry, transform, (500, 5), 16, 4).reason


def test_residual_shift_map_recovers_translation_and_flags_flat_images() -> None:
    spacing = 2.0
    fixed = textured((256, 256), seed=1)
    registered = translated(fixed, 3.4, -2.1)
    grid = ImageGeometry(shape=fixed.shape, spacing=(spacing, spacing), origin=(10.0, 20.0))
    coverage = np.ones(fixed.shape, dtype=bool)
    maps, summary = residual_shift_map(fixed, registered, coverage, grid, 64 * spacing)
    assert summary["valid_tiles"] >= 9 and not summary["reason"]
    valid = maps["shift_valid"]
    # Content observed at p + d means the registration predicted p: predicted - observed = -d.
    assert np.median(maps["shift_x_um"][valid]) == pytest.approx(-3.4 * spacing, abs=0.2 * spacing)
    assert np.median(maps["shift_y_um"][valid]) == pytest.approx(2.1 * spacing, abs=0.2 * spacing)
    assert summary["median"] == pytest.approx(np.hypot(3.4, 2.1) * spacing, abs=0.3 * spacing)
    assert summary["frame"] == "fixed"
    flat = np.zeros_like(fixed)
    _, summary = residual_shift_map(flat, flat, coverage, grid, 64 * spacing)
    assert summary["valid_tiles"] == 0 and summary["median"] is None and summary["reason"] == UNAVAILABLE_REASON


def test_automatic_landmarks_recover_translation_and_unavailable_on_flat() -> None:
    fixed = textured((400, 400), seed=2)
    moving = translated(fixed, 3.4, -2.1)
    geometry = ImageGeometry(shape=fixed.shape)
    transform = ValidationTransform(ALIGNED, matrix=np.eye(3))
    coverage = np.ones(fixed.shape, dtype=bool)
    rows, summary = automatic_landmarks(
        fixed, moving, coverage, geometry, transform, fixed, geometry, moving, geometry, 24, 10
    )
    assert summary["refined"] >= 10 and not summary["reason"] and summary["frame"] == "moving"
    errors = np.array([row["error_um"] for row in rows])
    assert np.median(errors) == pytest.approx(np.hypot(3.4, 2.1), abs=0.2)
    assert np.median([row["dx_um"] for row in rows]) == pytest.approx(-3.4, abs=0.2)
    flat = np.zeros_like(fixed)
    rows, summary = automatic_landmarks(
        flat, flat, coverage, geometry, transform, flat, geometry, flat, geometry, 24, 10
    )
    assert rows == [] and summary["reason"] == UNAVAILABLE_REASON and summary["median"] is None


def test_inverse_points_matrix_and_newton() -> None:
    matrix = np.array([[1.1, 0.2, 5.0], [-0.1, 0.9, -3.0], [0, 0, 1]])
    exact = ValidationTransform(ALIGNED, matrix=matrix)
    points = np.array([[10.0, 20.0], [-5.0, 7.5]])
    np.testing.assert_allclose(exact.inverse_points(exact.map_points(points)), points, atol=1e-9)

    class Warp:
        def TransformPoint(self, xy: tuple[float, float]) -> tuple[float, float]:
            x, y = xy
            return x + 3 * np.sin(y / 20), y + 2 * np.cos(x / 25)

    warped = ValidationTransform(ALIGNED, itk_transform=Warp())
    np.testing.assert_allclose(warped.inverse_points(warped.map_points(points)), points, atol=1e-3)
    np.testing.assert_allclose(warped.jacobian_at(points)[0], [[1, 3 * np.cos(1) / 20], [0, 1]], atol=0.05)


def test_proximity_warning_and_rows(tmp_path: Path) -> None:
    geometry = ImageGeometry(shape=(100, 100), spacing=(2.0, 2.0))
    images = {}
    for name in ("fixed", "moving"):
        tifffile.imwrite(tmp_path / f"{name}.tiff", np.zeros((100, 100), dtype=np.uint8))
        images[name] = inspect_validation_image(tmp_path / f"{name}.tiff", (2.0, 2.0))
    session = ValidationSession(
        fixed=images["fixed"],
        moving=images["moving"],
        landmarks=[
            ValidationLandmark(id="near", fixed=(53.0, 50.0), moving=(80, 80)),
            ValidationLandmark(id="far", fixed=(150.0, 150.0), moving=(150, 150)),
        ],
    )
    transform = ValidationTransform(ALIGNED, matrix=np.eye(3))
    transform.fitting = [ValidationLandmark(id="fit-1", fixed=(50.0, 50.0), moving=(50.0, 50.0))]
    warnings = landmark_warnings(session, [transform])
    assert set(warnings) == {"near"} and geometry.spacing == (2.0, 2.0)
    rows, _ = landmark_errors(transform, session.landmarks, {}, warnings)
    assert rows[0]["warning"] and not rows[1]["warning"] and rows[0]["source"] == "manual"


def test_csv_optional_columns_and_merge(tmp_path: Path) -> None:
    for name in ("fixed", "moving"):
        tifffile.imwrite(tmp_path / f"{name}.tiff", np.zeros((50, 50), dtype=np.uint8))
    session = ValidationSession(
        fixed=inspect_validation_image(tmp_path / "fixed.tiff", (1.0, 1.0)),
        moving=inspect_validation_image(tmp_path / "moving.tiff", (1.0, 1.0)),
        landmarks=[
            ValidationLandmark(id="a", fixed=(1, 2), moving=(3, 4), enabled=False, note="blurred"),
            ValidationLandmark(id="b", fixed=(5, 6), moving=(7, 8), source="assisted", score=0.9),
        ],
    )
    path = tmp_path / "landmarks.csv"
    export_validation_landmarks(path, session)
    imported = import_validation_landmarks(path, session, "um")
    assert [(p.enabled, p.note, p.source) for p in imported] == [(False, "blurred", "manual"), (True, "", "assisted")]
    merged, renames = merge_landmarks(session.landmarks, imported)
    assert renames == {"a": "a-2", "b": "b-2"} and len({p.id for p in merged}) == 4
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows([["id", "fixed_x", "fixed_y", "moving_x", "moving_y"], ["c", 1, 1, 1, 1]])
    assert import_validation_landmarks(path, session, "um")[0].enabled


def test_label_pyramid_does_not_create_labels() -> None:
    class Reader:
        n_in_pyramid = 1

        @staticmethod
        def get_channel(_channel: int, _level: int, split_rgb: bool = True) -> np.ndarray:
            labels = np.zeros((4096, 4096), dtype=np.uint16)
            labels[:, 1::2] = 7
            return labels

    levels = validation_image_pyramid(Reader(), 0, labels=True)
    assert len(levels) > 1
    assert set(np.unique(np.asarray(levels[-1]))) <= {0, 7}


def test_detected_landmarks_are_reviewable_and_reported_separately(tmp_path: Path) -> None:
    fixed = textured((300, 300), seed=4)
    moving = translated(fixed, 3.0, -2.0)
    for name, array in (("fixed", fixed), ("moving", moving)):
        tifffile.imwrite(tmp_path / f"{name}.tiff", array.astype(np.uint16))
    session = ValidationSession(
        fixed=inspect_validation_image(tmp_path / "fixed.tiff", (1.0, 1.0)),
        moving=inspect_validation_image(tmp_path / "moving.tiff", (1.0, 1.0)),
        landmarks=[ValidationLandmark(id="manual", fixed=(150, 150), moving=(152, 149))],
        auto_residual_map=True,
        auto_tile_um=48,
    )
    identity = ValidationTransform(ALIGNED, matrix=np.eye(3))
    detected, _ = detect_landmarks(session, identity, max_points=30, template_um=24, search_um=10)
    assert 10 <= len(detected) <= 30
    assert all(point.source == "automatic" and point.score > 0.6 for point in detected)
    assert detected[0].id == "auto-1" and len({point.id for point in detected}) == len(detected)
    # Points are spread over the image rather than clustered, and avoid the existing manual landmark.
    xy = np.array([point.fixed for point in detected])
    assert np.ptp(xy[:, 0]) > 150 and np.ptp(xy[:, 1]) > 150
    assert np.linalg.norm(xy - (150, 150), axis=1).min() >= 12
    errors = [np.hypot(*np.subtract(point.fixed, point.moving)) for point in detected if point.enabled]
    assert np.median(errors) == pytest.approx(np.hypot(3, 2), abs=0.2)
    session.landmarks = [*session.landmarks, *detected]
    again, _ = detect_landmarks(session, identity, max_points=30, template_um=24, search_um=10)
    assert not {point.id for point in again} & {point.id for point in session.landmarks}
    assert all(np.linalg.norm(xy - point.fixed, axis=1).min() >= 12 for point in again)

    analysis = analyze_validation_session(session, tmp_path / "out")
    result = analysis.candidates[0]
    # Headline accuracy uses the manual landmark only; automatic landmarks have their own statistics.
    assert result.validation["count"] == result.validation["total"] == 1
    assert result.validation["median"] == pytest.approx(np.hypot(2, 1))
    assert result.automatic["count"] == sum(point.enabled for point in detected)
    assert result.automatic["median"] == pytest.approx(np.hypot(3, 2), abs=0.2)
    assert result.residual_map["median"] == pytest.approx(np.hypot(3, 2), abs=0.3)
    with np.load(result.map_path) as arrays:
        assert {"shift_um", "shift_valid", "shift_origin_xy_um"} <= set(arrays)
    report = export_validation_report(analysis, tmp_path / "report")
    text = report.read_text()
    assert "Automatic evidence (image similarity" in text and "Automatic landmark residuals" in text
    assert "automatic" in (report.parent / "landmarks.csv").read_text()
    assert (report.parent / "report.pdf").stat().st_size > 0
