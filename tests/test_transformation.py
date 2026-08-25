"""Transformation model tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

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
