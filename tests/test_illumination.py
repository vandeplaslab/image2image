"""Numerical, lazy-processing, and export contracts for illumination correction."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest
import tifffile
import zarr

from image2image.models.illumination import CorrectionRecipe, IlluminationSource
from image2image.utils.illumination import (
    CorrectionProcessor,
    convert_intensity,
    estimate_background,
    export_correction,
    export_correction_batch,
    inspect_illumination_file,
    load_correction_recipe,
    preview_correction,
    save_correction_recipe,
    validate_recipe,
)


def make_source(
    path: Path, image: np.ndarray, channels: tuple[str, ...] = ("A",), spacing: tuple[float, float] | None = (0.5, 0.7)
) -> IlluminationSource:
    metadata = {"axes": "YX" if image.ndim == 2 else "CYX", "Channel": {"Name": list(channels)}}
    if spacing:
        metadata.update(
            PhysicalSizeY=spacing[0], PhysicalSizeX=spacing[1], PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm"
        )
    tifffile.imwrite(path, image, photometric="minisblack", metadata=metadata)
    return inspect_illumination_file(path)[0]


def test_dark_flat_native_units_and_whole_frame_normalization(tmp_path: Path) -> None:
    yy, xx = np.indices((40, 70))
    gain = (1 + xx / 70).astype(np.float32)
    signal = yy.astype(np.float32) - 5
    source = make_source(tmp_path / "source.ome.tiff", signal * gain + 20)
    dark = make_source(tmp_path / "dark.ome.tiff", np.full(gain.shape, 20, dtype=np.uint16))
    flat = make_source(tmp_path / "flat.ome.tiff", gain * 100 + 20)
    recipe = CorrectionRecipe(channels=source.channels, dark=dark, flat=flat, dark_channels=(0,), flat_channels=(0,))
    preview = preview_correction(source, recipe, region=(4, 11, 19, 21))
    np.testing.assert_allclose(preview.corrected, (signal * gain.mean())[4:23, 11:32], atol=1e-5)
    assert preview.statistics["negative"] > 0
    assert preview.statistics["flat_means"][0] == pytest.approx(100 * gain.mean())
    already = make_source(tmp_path / "corrected-flat.ome.tiff", gain * 100)
    other = recipe.model_copy(update={"flat": already, "flat_dark_corrected": True})
    np.testing.assert_allclose(preview_correction(source, other).corrected, signal * gain.mean(), atol=1e-5)


@pytest.mark.parametrize("method", ["none", "gaussian", "morphology", "rolling_ball"])
@pytest.mark.parametrize("polarity", ["bright", "dark"])
def test_native_tiling_and_region_match_whole_image(tmp_path: Path, method: str, polarity: str) -> None:
    rng = np.random.default_rng(2026)
    image = rng.normal(120, 8, (65, 79)).astype(np.float32)
    source = make_source(tmp_path / "image.ome.tiff", image)
    recipe = CorrectionRecipe(
        channels=source.channels, background=method, polarity=polarity, sigma=2.3, radius=4, height=17
    )
    expected = image.astype(float) - estimate_background(image, recipe)
    actual = np.empty(image.shape)
    with CorrectionProcessor(source, recipe) as processor:
        for y in range(0, 65, 16):
            for x in range(0, 79, 16):
                region = (y, x, min(16, 65 - y), min(16, 79 - x))
                _, result, _ = processor.region(0, region)
                actual[y : y + region[2], x : x + region[3]] = result
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("bad", [0, -1, 1e-10])
def test_invalid_flat_is_counted_and_blocks_output(tmp_path: Path, bad: float) -> None:
    image = np.ones((17, 23), dtype=np.float32)
    source = make_source(tmp_path / "image.ome.tiff", image)
    image[:2, :3] = bad
    flat = make_source(tmp_path / "flat.ome.tiff", image)
    recipe = CorrectionRecipe(channels=source.channels, flat=flat, flat_channels=(0,))
    output = tmp_path / "corrected.ome.tiff"
    with pytest.raises(ValueError, match="6 invalid pixels"):
        export_correction(source, recipe, output)
    assert not output.exists()
    assert not output.with_name(output.name + ".json").exists()
    assert not list(tmp_path.glob(".illumination-*"))


def test_channel_mapping_geometry_and_recipe_serialization(tmp_path: Path) -> None:
    source = make_source(tmp_path / "image.ome.tiff", np.full((2, 21, 25), 100, dtype=np.uint16), ("A", "B"))
    dark = make_source(
        tmp_path / "dark.ome.tiff", np.stack([np.full((21, 25), 3), np.full((21, 25), 8)]).astype(np.uint16), ("B", "A")
    )
    recipe = CorrectionRecipe(channels=source.channels, dark=dark, dark_channels=(1, 0))
    assert np.all(preview_correction(source, recipe, 0).corrected == 92)
    assert np.all(preview_correction(source, recipe, 1).corrected == 97)
    with pytest.raises(ValueError, match="explicitly"):
        validate_recipe(source, recipe.model_copy(update={"dark_channels": (-1, 0)}))
    with pytest.raises(ValueError, match="dimensions"):
        validate_recipe(source.model_copy(update={"shape": (20, 25)}), recipe)
    with pytest.raises(ValueError, match="pixel sizes"):
        validate_recipe(source.model_copy(update={"spacing": (2.0, 2.0)}), recipe)
    path = tmp_path / "recipe.json"
    save_correction_recipe(path, recipe)
    assert load_correction_recipe(path) == recipe
    path.write_text('{"schema_version": 2}')
    with pytest.raises(ValueError):
        load_correction_recipe(path)


def test_intensity_conversion_never_wraps_or_rescales() -> None:
    image = np.array([-10, -0.5, 0, 0.5, 1.5, 2.5, 65535, 70000], dtype=float)
    np.testing.assert_array_equal(convert_intensity(image, "uint16"), [0, 0, 0, 0, 2, 2, 65535, 65535])
    np.testing.assert_array_equal(convert_intensity(image, "float32"), image.astype(np.float32))
    np.testing.assert_array_equal(convert_intensity(image, "int16"), [-10, 0, 0, 0, 2, 2, 32767, 32767])
    with pytest.raises(ValueError, match="non-finite"):
        convert_intensity(np.array([np.nan]), "float32")
    with pytest.raises(ValueError, match="finite range"):
        convert_intensity(np.array([1e50]), "float32")


@pytest.mark.parametrize("extension", ["ome.tiff", "ome.zarr"])
@pytest.mark.parametrize("output_dtype", ["float32", "uint8", "source"])
def test_export_round_trip_pixels_metadata_and_provenance(tmp_path: Path, extension: str, output_dtype: str) -> None:
    image = np.stack([np.arange(35 * 39).reshape(35, 39), np.full((35, 39), 12)]).astype(np.uint16)
    source = make_source(tmp_path / "image.ome.tiff", image, ("A", "B"))
    dark = make_source(tmp_path / "dark.ome.tiff", np.full(image.shape, 100, dtype=np.uint16), ("A", "B"))
    recipe = CorrectionRecipe(channels=source.channels, dark=dark, dark_channels=(0, 1), output_dtype=output_dtype)
    output = tmp_path / f"corrected.{extension}"
    export_correction(source, recipe, output)
    dtype = source.dtype if output_dtype == "source" else output_dtype
    expected = convert_intensity(image.astype(float) - 100, dtype)
    if extension == "ome.tiff":
        with tifffile.TiffFile(output) as handle:
            np.testing.assert_array_equal(handle.asarray(), expected)
            assert "intensity_handling" in handle.ome_metadata
    else:
        root = zarr.open_group(output, mode="r")
        np.testing.assert_array_equal(root["0"][:], expected)
        assert root.attrs["image2image"]["recipe"]["output_dtype"] == output_dtype
    recovered = inspect_illumination_file(output)[0]
    assert recovered.channels == source.channels
    assert recovered.spacing == pytest.approx(source.spacing)
    record = json.loads(output.with_name(output.name + ".json").read_text())
    assert record["recipe"]["dark_channels"] == [0, 1]
    assert record["statistics"][1]["negative"] == image[1].size
    assert record["output_dtype"] == dtype
    with pytest.raises(FileExistsError):
        export_correction(source, recipe, output)
    assert not list(tmp_path.glob(".illumination-*"))


@pytest.mark.parametrize("extension", ["ome.tiff", "ome.zarr"])
def test_pyramids_odd_edges_and_unknown_spacing(tmp_path: Path, extension: str) -> None:
    image = np.arange(517 * 523, dtype=np.float32).reshape(517, 523) - 300
    source = make_source(tmp_path / "image.ome.tiff", image, spacing=None)
    output = tmp_path / f"corrected.{extension}"
    export_correction(source, CorrectionRecipe(channels=source.channels), output)
    padded = np.pad(image.astype(float), ((0, 1), (0, 1)), mode="edge")
    reduced = padded.reshape(259, 2, 262, 2).mean(axis=(1, 3)).astype(np.float32)
    if extension == "ome.tiff":
        with tifffile.TiffFile(output) as handle:
            assert len(handle.series[0].levels) == 2
            np.testing.assert_array_equal(handle.series[0].levels[1].asarray(), reduced)
    else:
        root = zarr.open_group(output, mode="r")
        np.testing.assert_array_equal(root["1"][0], reduced)
    assert inspect_illumination_file(output)[0].spacing is None


def test_batch_continues_failures_and_preserves_inputs(tmp_path: Path) -> None:
    image = np.ones((20, 21), dtype=np.float32)
    a = make_source(tmp_path / "a.ome.tiff", image)
    b = make_source(tmp_path / "b.ome.tiff", image, ("B",))
    c = make_source(tmp_path / "c.ome.tiff", image * 3)
    results = export_correction_batch([a, b, c], CorrectionRecipe(channels=a.channels), tmp_path / "results")
    assert [result.output is not None for result in results] == [True, False, True]
    assert "channel names" in results[1].error
    np.testing.assert_array_equal(tifffile.imread(a.path), image)
    with pytest.raises(ValueError, match="inside an input"):
        export_correction(a, CorrectionRecipe(channels=a.channels), a.path)


def test_cancellation_and_publication_failure_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = make_source(tmp_path / "image.ome.tiff", np.ones((517, 523), dtype=np.float32))
    recipe = CorrectionRecipe(channels=source.channels)
    output = tmp_path / "corrected.ome.tiff"
    cancelled = Event()
    with pytest.raises(InterruptedError):
        export_correction(source, recipe, output, cancelled=cancelled.is_set, progress=lambda _message: cancelled.set())
    assert not output.exists()
    assert not list(tmp_path.glob(".illumination-*"))
    import os

    original_link = os.link

    def fail_publication(src: Path, dest: Path) -> None:
        if str(dest).endswith(".ome.tiff"):
            raise OSError("Simulated publication failure")
        original_link(src, dest)

    monkeypatch.setattr("image2image.utils.illumination.os.link", fail_publication)
    with pytest.raises(OSError, match="publication failure"):
        export_correction(source, recipe, output)
    assert not output.exists()
    assert not output.with_name(output.name + ".json").exists()


def test_changed_nonfinite_and_stacked_inputs_rejected(tmp_path: Path) -> None:
    image = np.ones((21, 25), dtype=np.float32)
    source = make_source(tmp_path / "image.ome.tiff", image)
    recipe = CorrectionRecipe(channels=source.channels)
    make_source(source.path, image * 2)
    with pytest.raises(ValueError, match="changed after loading"):
        preview_correction(source, recipe)
    image[3, 5] = np.nan
    source = make_source(source.path, image)
    with pytest.raises(ValueError, match="non-finite"):
        preview_correction(source, recipe)
    path = tmp_path / "stack.ome.tiff"
    tifffile.imwrite(path, np.ones((3, 21, 25), dtype=np.uint16), metadata={"axes": "ZYX"}, photometric="minisblack")
    with pytest.raises(ValueError, match="stacks"):
        inspect_illumination_file(path)


@pytest.mark.parametrize("extension", ["ome.tiff", "ome.zarr"])
def test_rgb_is_exported_as_native_numeric_channels(tmp_path: Path, extension: str) -> None:
    image = np.empty((29, 31, 3), dtype=np.uint8)
    image[:] = [5, 60, 210]
    path = tmp_path / "rgb.tiff"
    tifffile.imwrite(path, image, photometric="rgb")
    source = inspect_illumination_file(path)[0]
    assert source.rgb and source.channels == ("R", "G", "B")
    output = tmp_path / f"corrected.{extension}"
    export_correction(source, CorrectionRecipe(channels=source.channels), output)
    recovered = inspect_illumination_file(output)[0]
    for i, value in enumerate((5, 60, 210)):
        preview = preview_correction(recovered, CorrectionRecipe(channels=recovered.channels), i)
        np.testing.assert_array_equal(preview.corrected, np.full((29, 31), value))


def test_multiscene_selection_and_zarr_stack_rejection(tmp_path: Path) -> None:
    path = tmp_path / "scenes.ome.tiff"
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(np.ones((21, 29), dtype=np.uint16), metadata={"axes": "YX"})
        writer.write(np.full((23, 31), 2, dtype=np.uint16), metadata={"axes": "YX"})
    scenes = inspect_illumination_file(path)
    assert len(scenes) == 2
    assert scenes[1].scene == 1 and scenes[1].shape == (23, 31)
    np.testing.assert_array_equal(
        preview_correction(scenes[1], CorrectionRecipe(channels=scenes[1].channels)).corrected, np.full((23, 31), 2)
    )
    root = zarr.open_group(tmp_path / "stack.ome.zarr", mode="w", zarr_format=2)
    root.create_array("0", shape=(2, 21, 29), dtype="uint16")
    root.attrs["multiscales"] = [{"version": "0.4", "axes": ["z", "y", "x"], "datasets": [{"path": "0"}]}]
    with pytest.raises(ValueError, match="stacks"):
        inspect_illumination_file(tmp_path / "stack.ome.zarr")


def test_lazy_source_only_materializes_requested_neighborhood(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from contextlib import contextmanager

    requested = []

    class LazyPlane:
        def __getitem__(self, slices: tuple[slice, slice]) -> np.ndarray:
            requested.append(slices)
            height, width = (len(range(s.start, s.stop, s.step or 1)) for s in slices)
            return np.ones((height, width))

    @contextmanager
    def pixels(_source: IlluminationSource):
        yield [LazyPlane()]

    source = IlluminationSource(
        path=tmp_path / "huge.tiff", shape=(100000, 100000), channels=("A",), dtype="uint16", size=0, modified_ns=0
    )
    monkeypatch.setattr("image2image.utils.illumination._pixels", pixels)
    recipe = CorrectionRecipe(channels=source.channels, background="gaussian", sigma=3)
    preview = preview_correction(source, recipe, region=(1000, 2000, 128, 128))
    assert preview.corrected.shape == (128, 128)
    assert requested == [(slice(988, 1140, 1), slice(1988, 2140, 1))]
    with pytest.raises(ValueError, match="memory limit"):
        preview_correction(source, recipe.model_copy(update={"sigma": 4096}), region=(1000, 2000, 128, 128))


def test_overview_does_not_scale_rolling_ball_intensity_height(tmp_path: Path) -> None:
    yy, xx = np.indices((75, 95))
    image = (100 + yy * 0.1 + 80 * np.exp(-((xx - 40) ** 2 + (yy - 30) ** 2) / 50)).astype(np.float32)
    source = make_source(tmp_path / "image.ome.tiff", image)
    recipe = CorrectionRecipe(channels=source.channels, background="rolling_ball", radius=9, height=20)
    overview = preview_correction(source, recipe, max_size=32)
    assert overview.scale == 3
    expected = estimate_background(image[::3, ::3], recipe.model_copy(update={"radius": 3}))
    np.testing.assert_array_equal(overview.background, expected)
