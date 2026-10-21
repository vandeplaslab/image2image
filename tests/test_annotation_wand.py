"""Display compositing, connected region growth, zoom, and bounded image reads."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from scipy.ndimage import label
from shapely.geometry import Point, box

from image2image.models.annotation import ImagePatch
from image2image.utils.annotation.images import PatchReader
from image2image.utils.annotation.wand import WandLayer, WandSource, _connected_mask, display_patch, wand_region

GRAY = np.array([[0, 0, 0, 1], [1, 1, 1, 1]], dtype=np.float32)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("size", [(1, 1), (7, 9), (154, 154), (512, 512)])
def test_compiled_component_matches_reference(dtype: type, size: tuple[int, int]) -> None:
    rng = np.random.default_rng(2026)
    smooth = rng.choice([0.0, 0.1, 0.15, 0.2, 0.9], (*size, 3)).astype(dtype)
    valid = rng.random(size) > 0.05
    row, col = size[0] // 2, size[1] // 2
    valid[row, col] = True
    smooth[row, col] = 0
    x, y, sx, sy, radius = col + 0.2, row + 0.9, 2.0, 3.0, min(size) * 0.8
    threshold = dtype(0.15)
    yy, xx = np.ogrid[: size[0], : size[1]]
    expected_disk = ((xx + 0.5 - x) * sx) ** 2 + ((yy + 0.5 - y) * sy) ** 2 <= radius**2
    expected_disk[row, col] = True
    allowed = (np.max(np.abs(smooth - smooth[row, col]), axis=-1) <= threshold) & expected_disk & valid
    labels, _ = label(allowed)
    expected = labels == labels[row, col]
    actual, disk = _connected_mask(smooth, valid, row, col, x, y, sx, sy, radius, threshold)
    np.testing.assert_array_equal(disk, expected_disk)
    np.testing.assert_array_equal(actual, expected)
    # Exercise the largest possible queue as well as fragmented/isolated regions.
    smooth[:] = 0
    actual, disk = _connected_mask(smooth, valid, row, col, x, y, sx, sy, radius, threshold)
    labels, _ = label(disk & valid)
    np.testing.assert_array_equal(actual, labels == labels[row, col])
    valid[:] = True
    actual, _ = _connected_mask(smooth, valid, row, col, x, y, sx, sy, 1e9, threshold)
    assert actual.all()


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
    adjacent = wand_region(patch, (153, 276.5), 10, 0.1)
    assert adjacent.is_valid and adjacent.covers(Point(153, 276.5))
    assert not adjacent.covers(Point(151, 276.5))


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


@pytest.mark.parametrize("setting", ["channels", "contrast", "gamma", "opacity", "blending", "colors", "background"])
def test_display_cache_invalidates_only_when_settings_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    path = tmp_path / "image.ome.tiff"
    image = np.full((2, 64, 64), 40, dtype=np.uint8)
    image[1] = 180
    tifffile.imwrite(path, image, photometric="minisblack", metadata={"axes": "CYX"})
    layer = WandLayer((0,), (0, 255), 1, GRAY.copy())
    with PatchReader(path) as reader:
        source = WandSource(reader, (layer,))
        tile = source._tile(0, 0, 1)
        read = reader.read
        calls = []

        def track(*args, **kwargs):
            calls.append(1)
            return read(*args, **kwargs)

        monkeypatch.setattr(reader, "read", track)
        source.set_display((replace(layer, colors=GRAY.copy()),), (0, 0, 0))
        assert source._tile(0, 0, 1) is tile and not calls
        settings = {
            "channels": {"channels": (1,)},
            "contrast": {"contrast": (0, 100)},
            "gamma": {"gamma": 2},
            "opacity": {"opacity": 0.5},
            "blending": {"blending": "minimum"},
            "colors": {"colors": np.array([[0, 0, 0, 1], [0, 1, 0, 1]], dtype=np.float32)},
            "background": {},
        }
        changed = replace(layer, **settings[setting])
        source.set_display((changed,), (1, 1, 1) if setting == "background" else (0, 0, 0))
        assert source.cache_bytes == 0
        assert source._tile(0, 0, 1) is not tile and calls == [1]
        source.set_display((), (0, 0, 0))
        assert source.cache_bytes == 0 and not source._tiles
