"""Display compositing, connected region growth, zoom, and bounded image reads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from shapely.geometry import Point, box

from image2image.models.annotation import ImagePatch
from image2image.utils.annotation.images import PatchReader
from image2image.utils.annotation.wand import WandLayer, WandSource, display_patch, wand_region

GRAY = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32)


def test_wand_connected_filled_region_and_color_contrast() -> None:
    image = np.full((128, 160, 3), 0.9, dtype=np.float32)
    image[20:90, 10:70] = [0.2, 0.4, 0.6]
    image[40:50, 30:40] = 0.9
    image[20:90, 95:150] = [0.2, 0.4, 0.6]
    patch = ImagePatch(image, (0, 0), (1, 1), (0, 1, 2))
    geometry = wand_region(patch, (25.5, 35.5), 120, 0.15)
    assert geometry.is_valid and geometry.covers(Point(25.5, 35.5))
    assert geometry.covers(box(32, 42, 38, 48))
    assert not geometry.intersects(box(100, 30, 140, 80))
    assert not geometry.intersects(box(72, 20, 90, 90))
    assert not geometry.interiors


def test_wand_sensitivity_does_not_creep_along_gradient() -> None:
    image = np.broadcast_to(np.linspace(0, 1, 101)[None, :, None], (31, 101, 3)).copy()
    patch = ImagePatch(image, (0, 0), (1, 1), (0, 1, 2))
    narrow = wand_region(patch, (10.5, 15.5), 150, 0.05)
    broad = wand_region(patch, (10.5, 15.5), 150, 0.3)
    assert narrow.bounds[2] < 20
    assert 35 < broad.bounds[2] < 45
    assert broad.area > narrow.area


def test_wand_coordinates_radius_and_missing_pixels() -> None:
    image = np.ones((51, 61, 3), dtype=np.float32)
    patch = ImagePatch(image, (100, 200), (2, 3), (0, 1, 2))
    geometry = wand_region(patch, (151, 276.5), 15, 0.1)
    assert geometry.is_valid and geometry.covers(Point(151, 276.5))
    assert box(135, 260, 167, 293).covers(geometry)
    assert geometry.area == pytest.approx(np.pi * 15**2, rel=0.1)
    assert wand_region(patch, (99, 200), 10, 0.1).is_empty
    image[25, 25] = np.nan
    assert wand_region(patch, (151, 276.5), 10, 0.1).is_empty


def test_display_channels_gamma_color_opacity_and_order() -> None:
    data = np.array([[[50, 200]]], dtype=np.uint16)
    red = WandLayer((0,), (0, 100), 2, np.array([[0, 0, 0, 1], [1, 0, 0, 1]]))
    green = WandLayer((1,), (0, 200), 1, np.array([[0, 0, 0, 1], [0, 1, 0, 1]]), 0.5)
    np.testing.assert_allclose(display_patch(data, (red, green), (0, 1)), [[[0.25, 0.5, 0]]])
    translucent = replace(green, blending="translucent")
    np.testing.assert_allclose(display_patch(data, (red, translucent), (0, 1)), [[[0.125, 0.5, 0]]])
    np.testing.assert_allclose(display_patch(data, (translucent, red), (0, 1)), [[[0.25, 0.5, 0]]])
    rgb = np.array([[[0, 128, 255, 128]]], dtype=np.uint8)
    layer = WandLayer((0, 1, 2, 3), (0, 255), 1, opacity=0.5, blending="translucent")
    np.testing.assert_allclose(
        display_patch(rgb, (layer,), (0, 1, 2, 3)), rgb[..., :3] / 255 * 128 / 255 * 0.5, atol=1e-7
    )
    np.testing.assert_allclose(
        display_patch(data, (replace(red, opacity=0.5, blending="translucent"),), (0, 1), background=(1, 1, 1)),
        [[[0.625, 0.5, 0.5]]],
    )
    np.testing.assert_allclose(display_patch(data, (replace(red, blending="minimum"),), (0, 1)), [[[0.25, 0, 0]]])


@pytest.mark.parametrize("pyramid", [False, True])
def test_source_zoom_cache_edges_and_work_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pyramid: bool) -> None:
    path = tmp_path / "slide.ome.tiff"
    image = np.zeros((1537, 2049), dtype=np.uint16)
    image[500:1000, 700:1200] = 40000
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(image, tile=(256, 256), metadata={"axes": "YX"}, subifds=1 if pyramid else None)
        if pyramid:
            writer.write(image[::4, ::4], tile=(256, 256), subfiletype=1)
    calls = []
    shapes = []
    from image2image.utils.annotation.wand import wand_region as grow

    def bounded(patch, *args):
        shapes.append(patch.data.shape)
        return grow(patch, *args)

    monkeypatch.setattr("image2image.utils.annotation.wand.wand_region", bounded)
    with PatchReader(path) as reader:
        read = reader.read

        def track(*args, **kwargs):
            result = read(*args, **kwargs)
            calls.append(result.data.shape)
            return result

        monkeypatch.setattr(reader, "read", track)
        source = WandSource(reader, (WandLayer((0,), (0, 65535), 1, GRAY),))
        small = source.region((900.5, 750.5), 75, 1, 0.15)
        first_calls = len(calls)
        assert source.region((900.5, 750.5), 75, 1, 0.15).equals(small)
        assert len(calls) == first_calls
        large = source.region((900.5, 750.5), 75, 0.1, 0.15)
        assert large.area > small.area * 5
        assert large.area == pytest.approx(500 * 500, rel=0.12)
        assert box(0, 0, 2049, 1537).covers(source.region((2048.5, 1536.5), 500, 0.013, 0.15))
        source.MAX_BYTES = source.TILE * source.TILE * 3 * 4
        source.region((10, 10), 500, 1, 0.15)
        assert source.cache_bytes <= source.MAX_BYTES
        assert all(h <= 256 and w <= 256 for h, w, _ in calls)
        assert all(h <= 512 and w <= 512 for h, w, _ in shapes)
