"""Tests for image transforms about pixel centres."""

from __future__ import annotations

import numpy as np
import pytest

from image2image.utils.transform import (
    centered_horizontal_flip,
    centered_rotation_transform,
    centered_vertical_flip,
)


@pytest.mark.parametrize("shape", [(962, 385), (7, 10), (9, 5), (1, 1)])
@pytest.mark.parametrize("spacing", [(1.0, 1.0), (2.0, 3.0)])
@pytest.mark.parametrize("angle", [0, 37, 90, 180])
def test_rotation_preserves_pixel_centre(shape: tuple[int, int], spacing: tuple[float, float], angle: float) -> None:
    """Keep the pixel centre fixed and map opposite corners onto each other at 180 degrees."""
    extent = np.array([(shape[0] - 1) * spacing[0], (shape[1] - 1) * spacing[1]])
    centre = np.append(extent / 2, 1)

    matrix = centered_rotation_transform(shape, spacing, angle)

    np.testing.assert_allclose(matrix @ centre, centre, atol=1e-10)
    if angle == 180:
        corners = np.array([[0, 0], [0, extent[1]], [extent[0], 0], extent])
        transformed = corners @ matrix[:2, :2].T + matrix[:2, 2]
        np.testing.assert_allclose(transformed, extent - corners, atol=1e-10)


@pytest.mark.parametrize("shape", [(962, 385), (7, 10), (9, 5), (1, 1)])
@pytest.mark.parametrize("spacing", [(1.0, 1.0), (2.0, 3.0)])
@pytest.mark.parametrize("vertical", [False, True])
def test_flip_preserves_pixel_centre(shape: tuple[int, int], spacing: tuple[float, float], vertical: bool) -> None:
    """Reflect corners within the pixel extent and preserve the centre in either direction."""
    extent = np.array([(shape[0] - 1) * spacing[0], (shape[1] - 1) * spacing[1]])
    centre = np.append(extent / 2, 1)
    corners = np.array([[0, 0], [0, extent[1]], [extent[0], 0], extent])
    expected = corners.copy()
    axis = 0 if vertical else 1
    expected[:, axis] = extent[axis] - corners[:, axis]

    matrix = centered_vertical_flip(shape, spacing) if vertical else centered_horizontal_flip(shape, spacing)

    np.testing.assert_allclose(matrix @ centre, centre, atol=1e-10)
    np.testing.assert_allclose(corners @ matrix[:2, :2].T + matrix[:2, 2], expected, atol=1e-10)
    np.testing.assert_allclose(matrix @ matrix, np.eye(3), atol=1e-10)
