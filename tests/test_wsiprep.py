"""Tests for WSI preprocessing transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from image2image.models.wsiprep import RegistrationImage


@pytest.mark.parametrize(
    "rotation, flip, expected",
    [
        (180, False, [[-1, 0, 1929], [0, -1, 1141], [0, 0, 1]]),
        (0, True, [[1, 0, 7], [0, -1, 1141], [0, 0, 1]]),
        (180, True, [[-1, 0, 1929], [0, 1, -11], [0, 0, 1]]),
    ],
)
def test_wsi_affine_uses_pixel_centres(tmp_path: Path, rotation: int, flip: bool, expected: list[list[int]]) -> None:
    """Rotate and flip WSI images about pixel centres before applying translation."""
    image = RegistrationImage(
        path=tmp_path / "moving.tiff",
        rotate=rotation,
        flip_lr=flip,
        scale=(2.0, 3.0),
        translate_y=7,
        translate_x=-11,
    )

    np.testing.assert_allclose(image.affine((962, 385)), expected, atol=1e-10)
    np.testing.assert_allclose(
        image.affine((962, 385), only_translate=True), [[1, 0, 7], [0, 1, -11], [0, 0, 1]], atol=1e-10
    )
