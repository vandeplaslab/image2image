"""Volume assembly choices, source selection, and transactional pixel export."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
import tifffile
from image2image_io.volume import open_volume

from image2image.models.volume import VolumeSlide, VolumeSource
from image2image.utils.volume import (
    build_volume_plan,
    detect_slide_number,
    export_volume,
    inspect_volume_file,
    reconcile_channels,
    reorder_slides,
    resolve_volume_selection,
    source_channel,
)


def make_slide(path: Path, number: int, channels: tuple[str, ...] = ("DAPI", "FITC")) -> VolumeSlide:
    """Build a metadata-only slide for selection tests."""
    source = VolumeSource(path, 0, (12, 14), 0.5, channels, "uint16")
    return VolumeSlide(path, (source,), 0, number, number)


def write_slide(path: Path, channels: tuple[str, ...], pixel_size: float, data: np.ndarray) -> VolumeSlide:
    """Write a calibrated synthetic OME-TIFF and inspect its metadata."""
    tifffile.imwrite(
        path,
        data,
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "Channel": {"Name": list(channels)},
            "PhysicalSizeX": pixel_size,
            "PhysicalSizeY": pixel_size,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )
    return inspect_volume_file(path)


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("sample_slide_001.ome.tif", 1),
        ("sample_section12.czi", 12),
        ("SEC-003.tif", 3),
        ("slide0.tif", 0),
        ("sample_2026.tif", None),
        ("slide1_section2.tif", None),
    ],
)
def test_detect_number(filename: str, expected: int | None) -> None:
    number, error = detect_slide_number(filename)
    assert number == expected
    assert bool(error) == (expected is None)


def test_custom_detection() -> None:
    assert detect_slide_number("sample_04_cycle12.tif", r"sample_(?P<section>\d+)_")[0] == 4
    assert detect_slide_number("sample_04.tif", r"sample_(\d+)")[0] == 4
    for pattern in ("[", "no_group"):
        with pytest.raises(ValueError):
            detect_slide_number("slide1", pattern)


def test_matching_policies_and_overrides(tmp_path: Path) -> None:
    slides = [make_slide(tmp_path / "a", 1), make_slide(tmp_path / "b", 3, (" dapi ", "Alias"))]
    channels = reconcile_channels(slides, [])
    assert [channel.selected for channel in channels] == [True, False, False]
    channels[1].selected = True
    selection = resolve_volume_selection(slides, channels, "channels")
    assert selection.slide_keys == tuple(slide.key for slide in slides)
    assert selection.channel_keys == ("dapi",)
    assert channels[1].selected
    selection = resolve_volume_selection(slides, channels, "slides")
    assert selection.excluded_slides == (slides[1].key,)
    channels[1].overrides[slides[1].key] = 1
    channels[1].name = "Renamed FITC"
    assert source_channel(slides[1], channels[1]) == (1, "")
    selection = resolve_volume_selection(slides, channels, "slides")
    assert not selection.excluded_slides
    assert reconcile_channels(slides + [make_slide(tmp_path / "c", 8)], channels)[1] is channels[1]
    slides[1].sources = (replace(slides[1].source, channel_names=("DAPI", " dapi ")),)
    assert "Ambiguous" in source_channel(slides[1], channels[0])[1]
    channels[0].overrides[slides[1].key] = 0
    assert source_channel(slides[1], channels[0]) == (0, "")


def test_reorder_preserves_gaps_and_endpoints(tmp_path: Path) -> None:
    slides = [make_slide(tmp_path / str(index), number) for index, number in enumerate((2, 4, 7))]
    first = slides[0]
    slides = reorder_slides(slides, [slides[2].key, slides[1].key, slides[0].key])
    assert [slide.position for slide in slides] == [2, 4, 7]
    assert slides[-1] is first
    slides[0].included = slides[-1].included = False
    plan = build_volume_plan(slides, reconcile_channels(slides, []), z_spacing=5)
    assert plan.first_position == 2
    assert plan.present == (False, False, True, False, False, False)
    assert plan.shape == (2, 6, 12, 14)
    with pytest.raises(FrozenInstanceError):
        plan.tile_size = 256
    slides[1].position = 9
    assert plan.slides[1].position == 4


def test_validation(tmp_path: Path) -> None:
    slides = [make_slide(tmp_path / "a", 1), make_slide(tmp_path / "b", 1)]
    channels = reconcile_channels(slides, [])
    with pytest.raises(ValueError, match="Duplicate"):
        build_volume_plan(slides, channels, z_spacing=5)
    slides[1].position = 3
    for spacing in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="spacing"):
            build_volume_plan(slides, channels, z_spacing=spacing)
    channels[1].name = " dapi "
    with pytest.raises(ValueError, match="unique"):
        build_volume_plan(slides, channels, z_spacing=5)
    channels[1].name = ""
    with pytest.raises(ValueError, match="nonempty"):
        build_volume_plan(slides, channels, z_spacing=5)
    channels[1].name = "FITC"
    slides[1].sources = (replace(slides[1].source, dtype="float32"),)
    with pytest.raises(ValueError, match="dtype"):
        build_volume_plan(slides, channels, z_spacing=5)


@pytest.mark.parametrize("sharding", [False, True])
@pytest.mark.parametrize("dot_separator", [False, True])
def test_export_pixels_mapping_geometry_and_metadata(tmp_path: Path, sharding: bool, dot_separator: bool) -> None:
    slides = [
        write_slide(
            tmp_path / "slide1.ome.tif",
            ("DAPI", "FITC"),
            1.0,
            np.stack([np.full((12, 14), 11), np.full((12, 14), 22)]).astype("uint16"),
        ),
        write_slide(
            tmp_path / "slide3.ome.tif",
            ("FITC", " dapi "),
            0.5,
            np.stack([np.full((12, 14), 33), np.full((12, 14), 44)]).astype("uint16"),
        ),
    ]
    channels = reconcile_channels(slides, [])[::-1]
    channels[0].name = "Green"
    plan = build_volume_plan(
        slides, channels, z_spacing=7.5, tile_size=4, sharding=sharding, dot_separator=dot_separator
    )
    path = export_volume(plan, tmp_path / "result.ome.zarr")
    volume = open_volume(path)
    assert volume.channel_names == ("Green", "DAPI")
    assert volume.spacing == (7.5, 1.0, 1.0)
    assert volume.origin == (0, 0, 0)
    assert volume.present == (True, False, True)
    pixels = volume.levels[0].compute()
    np.testing.assert_array_equal(pixels[:, 1], 0)
    np.testing.assert_array_equal(pixels[0, 0], 22)
    np.testing.assert_array_equal(pixels[1, 0], 11)
    np.testing.assert_array_equal(pixels[0, 2, :6, :7], 33)
    np.testing.assert_array_equal(pixels[1, 2, :6, :7], 44)
    np.testing.assert_array_equal(pixels[:, 2, 6:], 0)
    np.testing.assert_array_equal(pixels[:, 2, :, 7:], 0)
    assert [level.shape for level in volume.levels] == [(2, 3, 12, 14), (2, 3, 6, 7), (2, 3, 3, 4)]
    for index in range(len(volume.levels)):
        level_path = path / str(index)
        assert (level_path / ("c.0.0.0.0" if dot_separator else "c/0/0/0/0")).is_file()
        if dot_separator:
            assert all(child.is_file() for child in level_path.iterdir())
    assert volume.provenance["slides"][1]["channel_indices"] == [0, 1]
    assert volume.provenance["slides"][0]["channel_indices"] == [1, 0]
    assert volume.provenance["spacing_um"] == [7.5, 1.0, 1.0]
    with pytest.raises(FileExistsError):
        export_volume(plan, path)


def test_select_tiff_scene(tmp_path: Path) -> None:
    path = tmp_path / "section4.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        for size, value in ((20, 10), (12, 70)):
            writer.write(
                np.full((size, size), value, dtype="uint16"), photometric="minisblack", metadata={"axes": "YX"}
            )
    slide = inspect_volume_file(path)
    assert slide.scene_index is None
    assert [source.shape for source in slide.sources] == [(20, 20), (12, 12)]
    slide.scene_index = 1
    plan = build_volume_plan([slide], reconcile_channels([slide], []), z_spacing=2)
    result = open_volume(export_volume(plan, tmp_path / "scene.ome.zarr"))
    np.testing.assert_array_equal(result.levels[0].compute(), 70)
    assert result.levels[0].shape == (1, 1, 12, 12)
    assert result.provenance["slides"][0]["scene_index"] == 1


def test_linear_resampling_and_snapshot(tmp_path: Path) -> None:
    yy, xx = np.mgrid[:20, :22]
    slide = write_slide(tmp_path / "slide1.ome.tif", ("DAPI",), 0.5, (3 * yy + xx).astype("float32")[None])
    channels = reconcile_channels([slide], [])
    plan = build_volume_plan([slide], channels, z_spacing=2, xy_spacing=1.25, tile_size=3)
    channels[0].name = "Changed after snapshot"
    channels[0].overrides[slide.key] = None
    actual = open_volume(export_volume(plan, tmp_path / "interpolated.ome.zarr"))
    expected_y, expected_x = np.mgrid[:7, :8]
    np.testing.assert_allclose(actual.levels[0][0, 0, :7, :8].compute(), (3 * expected_y + expected_x) * 2.5)
    assert actual.channel_names == ("DAPI",)


def test_rgb_source_and_export(tmp_path: Path) -> None:
    path = tmp_path / "slide1.tif"
    data = np.zeros((12, 14, 3), dtype="uint8")
    data[:] = [17, 31, 92]
    tifffile.imwrite(path, data, photometric="rgb")
    slide = inspect_volume_file(path)
    assert slide.source.channel_names == ("R", "G", "B")
    channels = reconcile_channels([slide], [])[::-1]
    plan = build_volume_plan([slide], channels, z_spacing=1)
    volume = open_volume(export_volume(plan, tmp_path / "rgb.ome.zarr"))
    assert volume.channel_names == ("B", "G", "R")
    np.testing.assert_array_equal(volume.levels[0][:, 0, 0, 0].compute(), [92, 31, 17])


def test_reject_z_stack_and_extreme_number_range(tmp_path: Path) -> None:
    path = tmp_path / "slide1.ome.tif"
    tifffile.imwrite(
        path, np.ones((4, 12, 14), dtype="uint16"), ome=True, photometric="minisblack", metadata={"axes": "ZYX"}
    )
    with pytest.raises(ValueError, match="Z/T"):
        inspect_volume_file(path)
    slides = [make_slide(tmp_path / "a", 1), make_slide(tmp_path / "b", 10**12)]
    with pytest.raises(ValueError, match="million"):
        build_volume_plan(slides, reconcile_channels(slides, []), z_spacing=1)


@pytest.mark.parametrize("failure", ["cancel", "read"])
def test_export_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    slide = write_slide(tmp_path / "slide1.ome.tif", ("DAPI",), 1, np.ones((1, 12, 14), dtype="uint16"))
    plan = build_volume_plan([slide], reconcile_channels([slide], []), z_spacing=1, tile_size=4)
    from image2image.utils import volume as module

    original = module.open_volume_source
    readers = []

    def open_reader(*args: object) -> object:
        reader = original(*args)
        readers.append(reader)
        return reader

    monkeypatch.setattr(module, "open_volume_source", open_reader)
    count = 0

    def read(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal count
        count += 1
        if failure == "read":
            raise RuntimeError("bad tile")
        return np.ones(kwargs["region"][2:], dtype="uint16")

    monkeypatch.setattr("image2image_reg.elastix.tiles.resample_region", read)
    output = tmp_path / "failed.ome.zarr"
    with pytest.raises((RuntimeError, InterruptedError)):
        export_volume(plan, output, cancelled=lambda: count > 0)
    assert not output.exists()
    assert not list(tmp_path.glob("*.partial"))
    assert readers and all(reader.fh is None for reader in readers)
