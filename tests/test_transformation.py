"""Transformation model tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from image2image.models.data import DataModel
from image2image.models.transformation import Transformation


def _write_i2r(path: Path, fixed_path: Path, moving_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write a minimal image2image registration file."""
    moving_points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    fixed_points = np.array([[3.0, 4.0], [5.0, 4.0], [3.0, 6.0]])
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.4",
                "tool": "register",
                "time_created": "2026-08-25T00:00:00.000000Z",
                "transformation_type": "affine",
                "fixed_paths": [{"path": str(fixed_path), "pixel_size_um": 2.0, "reader_kws": {}}],
                "moving_paths": [{"path": str(moving_path), "pixel_size_um": 3.0, "reader_kws": {}}],
                "fixed_points_yx_px": fixed_points.tolist(),
                "moving_points_yx_px": moving_points.tolist(),
            }
        )
    )
    return fixed_points, moving_points


def test_transformation_from_i2r_is_ready_to_use(tmp_path: Path) -> None:
    """Load points, image metadata, and a computed transform from an i2r file."""
    fixed_path = tmp_path / "fixed.tiff"
    moving_path = tmp_path / "moving.tiff"
    fixed_path.touch()
    moving_path.touch()
    project_path = tmp_path / "project.i2r.json"
    fixed_points, moving_points = _write_i2r(project_path, fixed_path, moving_path)

    transformation = Transformation.from_i2r(project_path)

    assert transformation.transformation_type == "affine"
    assert transformation.fixed_model is not None
    assert transformation.fixed_model.paths == [fixed_path]
    assert transformation.fixed_model.resolution == 2.0
    assert transformation.fixed_model.is_fixed
    assert transformation.moving_model is not None
    assert transformation.moving_model.paths == [moving_path]
    assert transformation.moving_model.resolution == 3.0
    assert not transformation.moving_model.is_fixed
    np.testing.assert_array_equal(transformation.fixed_points, fixed_points)
    np.testing.assert_array_equal(transformation.moving_points, moving_points)
    assert transformation.is_valid()
    np.testing.assert_allclose(transformation(moving_points), fixed_points)


def test_transformation_from_i2r_without_path_validation(tmp_path: Path) -> None:
    """Compute a transform without requiring the referenced images to exist."""
    project_path = tmp_path / "project.i2r.json"
    fixed_points, moving_points = _write_i2r(
        project_path,
        tmp_path / "missing-fixed.tiff",
        tmp_path / "missing-moving.tiff",
    )

    transformation = Transformation.from_i2r(project_path, validate_paths=False)

    assert transformation.fixed_model is not None
    assert transformation.fixed_model.paths == []
    assert transformation.fixed_model.resolution == 2.0
    assert transformation.moving_model is not None
    assert transformation.moving_model.paths == []
    assert transformation.moving_model.resolution == 3.0
    assert transformation.is_valid()
    np.testing.assert_allclose(transformation(moving_points), fixed_points)


@pytest.mark.parametrize("initial_translation", [(961, 384), (385, 385)])
def test_initial_display_affine_is_removed_before_fitting_and_export(
    initial_translation: tuple[int, int],
) -> None:
    """Recover native landmarks for corrected and legacy previews with unequal pixel sizes."""
    native_points = np.array([[10.0, 20.0], [400.0, 30.0], [35.0, 300.0], [800.0, 200.0]])
    expected_matrix = np.array([[1.2, 0.1, 5], [-0.2, 0.9, 7], [0, 0, 1]])
    fixed_points = native_points @ expected_matrix[:2, :2].T + expected_matrix[:2, 2]
    initial_matrix = np.array(
        [[-1, 0, initial_translation[0]], [0, -1, initial_translation[1]], [0, 0, 1]], dtype=float
    )
    preview_points = native_points @ initial_matrix[:2, :2].T + initial_matrix[:2, 2]
    reference = Transformation(
        fixed_model=DataModel(resolution=2.0, is_fixed=True),
        moving_model=DataModel(resolution=3.0),
        fixed_points=fixed_points,
        moving_points=native_points,
        time_created=datetime(2026, 10, 30, tzinfo=UTC),
    )
    transformation = reference.model_copy(
        update={"moving_points": preview_points.copy(), "moving_initial_affine": initial_matrix.copy()}
    )

    np.testing.assert_allclose(transformation.apply_moving_initial_transform(preview_points), native_points, atol=1e-10)
    np.testing.assert_allclose(transformation.compute().params, expected_matrix, atol=1e-10)
    exported = transformation.to_dict()
    expected_export = reference.to_dict()

    np.testing.assert_allclose(exported["moving_points_yx_px"], native_points, atol=1e-10)
    np.testing.assert_allclose(exported["moving_points_yx_um"], native_points * 3, atol=1e-10)
    np.testing.assert_allclose(exported["fixed_points_yx_um"], fixed_points * 2, atol=1e-10)
    for key, value in expected_export.items():
        if key.startswith("matrix_"):
            np.testing.assert_allclose(exported[key], value, atol=1e-10)
    assert exported["schema_version"] == "1.4"
    assert exported["initial_matrix_yx_um"] == initial_matrix.tolist()
    assert "initial_matrix_yx_px" not in exported
    np.testing.assert_array_equal(transformation.moving_points, preview_points)
    np.testing.assert_array_equal(transformation.moving_initial_affine, initial_matrix)
