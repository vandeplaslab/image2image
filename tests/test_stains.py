"""Numerical, sampling, metadata, and batch contracts for the stain mini app."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
import zarr

from image2image.models.stains import (
    StainEstimation,
    StainOverride,
    StainProfile,
    StainRecipe,
    StainSource,
    stain_preset,
)
from image2image.utils.stain_io import (
    export_stain_batch,
    export_stains,
    fit_source,
    inspect_stain_file,
    load_stain_overview,
    load_stain_recipe,
    preview_stains,
    read_rgb_region,
    rgb_levels,
    sample_rgb,
    save_stain_recipe,
)
from image2image.utils.stains import (
    StainFit,
    estimate_stain_profile,
    fit_stains,
    normalization_scales,
    optical_density,
    reconstruct_stains,
    registration_channels,
    sample_background,
    sample_stain_vector,
    separate_stains,
    stain_matrix,
)


def mixture(shape: tuple[int, int] = (83, 97), dtype: str = "uint8", profile: StainProfile | None = None) -> np.ndarray:
    profile = profile or stain_preset()
    rng = np.random.default_rng(2026)
    coefficients = np.zeros((*shape, 3), dtype=np.float32)
    coefficients[..., : len(profile.names)] = np.clip(rng.gamma(0.5, 0.6, (*shape, len(profile.names))), 0, 2.5)
    # Independent Beer-Lambert construction, without using the implementation's reconstruction kernel.
    density = coefficients @ stain_matrix(profile)
    return np.rint(np.exp(-density) * np.iinfo(np.dtype(dtype)).max).astype(dtype)


def make_stain_source(path: Path, pixels: np.ndarray | None = None) -> StainSource:
    pixels = mixture() if pixels is None else pixels
    tifffile.imwrite(
        path,
        pixels,
        photometric="rgb",
        tile=(32, 32),
        compression="deflate",
        metadata={
            "axes": "YXS",
            "PhysicalSizeY": 0.6,
            "PhysicalSizeX": 0.4,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeYUnit": "µm",
        },
    )
    return inspect_stain_file(path)[0]


def make_stain_pyramid(path: Path, pixels: np.ndarray, *, planar: bool = False) -> StainSource:
    """Write a tiled OME-TIFF with two reduced SubIFD levels."""
    metadata = {"axes": "CYX", "Channel": {"Name": ["R", "G", "B"]}} if planar else {"axes": "YXS"}
    with tifffile.TiffWriter(path, ome=True) as writer:
        for index, step in enumerate((1, 2, 4)):
            level = pixels[::step, ::step]
            writer.write(
                np.moveaxis(level, -1, 0) if planar else level,
                photometric="minisblack" if planar else "rgb",
                tile=(128, 128),
                compression="deflate",
                subifds=2 if index == 0 else None,
                subfiletype=0 if index == 0 else 1,
                metadata=metadata if index == 0 else None,
            )
    return inspect_stain_file(path)[0]


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("planar", [False, True])
def test_pyramidal_tiff_levels_preview_and_export(tmp_path: Path, dtype: str, planar: bool) -> None:
    pixels = mixture((1025, 1101), dtype)
    source = make_stain_pyramid(tmp_path / "pyramid.ome.tiff", pixels, planar=planar)
    with rgb_levels(source) as levels:
        assert len(levels) == 3
        for index, level in enumerate(levels):
            expected = pixels[:: 2**index, :: 2**index]
            assert level.shape == expected.shape
            np.testing.assert_array_equal(read_rgb_region(level, (17, 19, 12, 13)), expected[17:29, 19:32])
    overview, scale, origin = load_stain_overview(source)
    np.testing.assert_array_equal(overview, pixels[::2, ::2])
    assert scale == (1025 / 513, 1101 / 551)
    assert origin == (0, 0)
    recipe = StainRecipe(vector_mode="shared", export_channels=(0, 1, 2))
    preview = preview_stains(source, recipe, region=(503, 499, 14, 32))
    np.testing.assert_array_equal(preview.original, pixels[503:517, 499:531])
    outputs = export_stains(source, recipe, tmp_path / "out")
    actual = np.moveaxis(tifffile.imread(outputs[0]), 0, -1)
    np.testing.assert_array_equal(actual, separate_stains(pixels, recipe.profile))


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("preset", ["H&E", "H-DAB", "Custom (3 stains)"])
def test_numba_reference_and_reconstruction(dtype: str, preset: str) -> None:
    profile = stain_preset(preset)
    pixels = mixture(dtype=dtype, profile=profile)
    actual = separate_stains(pixels, profile)
    expected = (optical_density(pixels, profile) @ np.linalg.inv(stain_matrix(profile))).astype(np.float32)
    np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_allclose(actual, separate_stains(pixels, profile, compiled=False), atol=2e-6)
    reconstructed = reconstruct_stains(actual, profile, dtype=dtype)
    assert np.abs(reconstructed.astype(float) - pixels).max() <= 1
    # Strided input and odd edges use the same transform.
    np.testing.assert_array_equal(separate_stains(pixels[::2, ::3], profile), actual[::2, ::3])


def test_known_signed_coefficients_and_residual() -> None:
    profile = StainProfile(vectors=((1, 0, 0), (0, 1, 0)))
    pixels = np.array([[[100, 70, 200], [255, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    np.testing.assert_allclose(
        separate_stains(pixels, profile), -np.log(np.maximum(pixels.astype(float), 1) / 255), atol=1e-6
    )
    values = separate_stains(np.array([[[20, 250, 240]]], dtype=np.uint8), stain_preset())
    assert (values < 0).any()
    assert np.isfinite(values).all()


@pytest.mark.parametrize("preset", ["H&E", "H-DAB"])
def test_automatic_estimation_recovers_ordered_vectors(preset: str) -> None:
    template = stain_preset(preset)
    actual = estimate_stain_profile(mixture((256, 256), "uint16", template), template)
    np.testing.assert_allclose(actual.vectors, template.vectors, atol=0.025)
    fit = fit_stains(mixture(), template, StainEstimation())
    assert fit.sample_count > 64
    assert all(value > 0 for value in fit.percentiles[:2])


def test_blank_and_single_stain_estimation_fail_actionably() -> None:
    with pytest.raises(ValueError, match="Too little"):
        estimate_stain_profile(np.full((30, 40, 3), 255, dtype=np.uint8), stain_preset())
    concentrations = np.zeros((256, 128, 3), dtype=np.float32)
    concentrations[..., 0] = np.linspace(0.2, 2, 256)[:, None]
    with pytest.raises(ValueError, match="distinct stains"):
        estimate_stain_profile(reconstruct_stains(concentrations, stain_preset(), dtype="uint16"), stain_preset())
    with pytest.raises(ValueError, match="two stains"):
        estimate_stain_profile(mixture(), stain_preset("Custom (3 stains)"))


def test_invalid_profiles_inputs_and_sample_regions() -> None:
    for vectors in (((1, 1, 1), (2, 2, 2)), ((0, 0, 0), (0, 1, 0)), ((-1, 0, 0), (0, 1, 0))):
        with pytest.raises(ValueError):
            StainProfile(vectors=vectors)
    with pytest.raises(ValueError):
        StainProfile(white=(0, 1, 1))
    with pytest.raises(ValueError):
        StainProfile(names=("Same", "Same"))
    with pytest.raises(ValueError):
        StainEstimation(min_od=1, max_od=0.5)
    with pytest.raises(ValueError, match="three stains"):
        StainRecipe(profile=stain_preset("Custom (3 stains)"))
    with pytest.raises(ValueError, match="uint8 or uint16"):
        separate_stains(np.zeros((20, 20, 3), dtype=np.float32), stain_preset())
    with pytest.raises(ValueError, match="empty"):
        separate_stains(np.zeros((0, 20, 3), dtype=np.uint8), stain_preset())
    profile = stain_preset()
    samples = np.full((20, 20, 3), [100, 70, 140], dtype=np.uint8)
    updated = sample_stain_vector(samples, profile, 0)
    direction = -np.log(np.array([100, 70, 140]) / 255)
    np.testing.assert_allclose(updated.vectors[0], direction / np.linalg.norm(direction))
    assert updated.vectors[1] == profile.vectors[1]
    blank = np.full((10, 10, 3), [245, 250, 253], dtype=np.uint8)
    np.testing.assert_allclose(sample_background(blank, profile).white, np.array([245, 250, 253]) / 255)


def test_normalization_changes_only_requested_stain_scales() -> None:
    profile = stain_preset()
    target = StainProfile(vectors=((0.7, 0.6, 0.3), (0.15, 0.9, 0.3)))
    source_fit, reference_fit = StainFit(profile, (1, 2, 0.1), 100), StainFit(target, (2, 1, 9), 100)
    assert normalization_scales(source_fit, reference_fit) == (1, 1, 1)
    assert normalization_scales(source_fit, reference_fit, True) == (2, 0.5, 1)
    coefficients = np.array([[[0.2, 0.4, -0.02]]], dtype=np.float32)
    actual = reconstruct_stains(coefficients, target, residual_profile=profile, dtype="uint16")
    basis = stain_matrix(target)
    basis[2] = stain_matrix(profile)[2]
    expected = np.rint(np.clip(np.exp(-coefficients @ basis), 0, 1) * 65535).astype(np.uint16)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(registration_channels(np.array([[[-1, 1, 0.2]]]), source_fit), [[[0, 32768, 65535]]])
    with pytest.raises(ValueError, match="measurable"):
        normalization_scales(StainFit(profile, (1, 0, 0), 100), reference_fit, True)


def test_recipe_roundtrip_and_override_precedence(tmp_path: Path) -> None:
    source = make_stain_source(tmp_path / "source.ome.tiff")
    manual = StainProfile(vectors=((0.7, 0.6, 0.3), (0.1, 0.9, 0.3)))
    recipe = StainRecipe(
        reference=source, reference_profile=manual, overrides=(StainOverride(source=source, profile=manual),)
    )
    path = tmp_path / "recipe.json"
    save_stain_recipe(path, recipe)
    assert load_stain_recipe(path) == recipe
    cache = {}
    fit = fit_source(source, recipe, cache=cache)
    assert fit.profile == manual
    assert fit_source(source, recipe, cache=cache) is fit
    assert fit_source(source, recipe, reference=True).profile == manual
    with pytest.raises(ValueError, match="names/order"):
        StainRecipe(reference_profile=stain_preset("H-DAB"))
    with pytest.raises(ValueError, match="reference"):
        StainRecipe(export_normalized=True)


@pytest.mark.parametrize("extension", ["ome.tiff", "ome.zarr"])
@pytest.mark.parametrize("dtype", ["float32", "uint16"])
def test_tiled_export_pixels_metadata_and_normalized_rgb(tmp_path: Path, extension: str, dtype: str) -> None:
    pixels = mixture((517, 531))
    source = make_stain_source(tmp_path / "source.ome.tiff", pixels)
    recipe = StainRecipe(
        vector_mode="shared",
        reference=source,
        reference_profile=stain_preset(),
        export_normalized=True,
        export_channels=(0, 1, 2),
        channel_dtype=dtype,
    )
    outputs = export_stains(source, recipe, tmp_path / "out", output_format=extension)
    preview = preview_stains(source, recipe, region=(503, 499, 14, 32))
    record = json.loads(outputs[0].with_name(outputs[0].name + ".json").read_text())
    expected = separate_stains(pixels, recipe.profile)
    if dtype == "uint16":
        expected = registration_channels(expected, fit_source(source, recipe))
    if extension == "ome.tiff":
        with tifffile.TiffFile(outputs[0]) as handle:
            actual = np.moveaxis(handle.series[0].asarray(), 0, -1)
            assert len(handle.series[0].levels) == 2
            assert "Hematoxylin" in handle.ome_metadata and "Residual" in handle.ome_metadata
            assert 'PhysicalSizeX="0.4"' in handle.ome_metadata
            assert 'PhysicalSizeY="0.6"' in handle.ome_metadata
        normalized = tifffile.imread(outputs[1])
        with tifffile.TiffFile(outputs[1]) as handle:
            assert handle.series[0].axes == "YXS"
            assert handle.pages[0].photometric.name == "RGB"
    else:
        root = zarr.open_group(str(outputs[0]), mode="r")
        actual = np.moveaxis(root["0"][:], 0, -1)
        assert root["1"].shape == (3, 259, 266)
        assert root.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"] == [1, 0.6, 0.4]
        normalized = np.moveaxis(zarr.open_group(str(outputs[1]), mode="r")["0"][:], 0, -1)
    np.testing.assert_array_equal(actual, expected)
    assert np.abs(normalized.astype(float) - pixels).max() <= 1
    np.testing.assert_array_equal(preview.channels, separate_stains(pixels[503:517, 499:531], recipe.profile))
    assert record["source_fit"]["profile"]["names"] == ["Hematoxylin", "Eosin"]
    assert record["normalization_scales"] == [1, 1, 1]
    assert record["output_dtype"] == dtype
    with pytest.raises(FileExistsError):
        export_stains(source, recipe, tmp_path / "out", output_format=extension)


def test_rgb_mapping_changed_inputs_and_stacks(tmp_path: Path) -> None:
    pixels = mixture()
    path = tmp_path / "planar.ome.tiff"
    tifffile.imwrite(
        path,
        np.moveaxis(pixels, -1, 0),
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["A", "B", "C"]}},
    )
    source = inspect_stain_file(path)[0]
    assert source.rgb_channels is None
    with pytest.raises(ValueError, match="Map"):
        sample_rgb(source, StainRecipe())
    source = source.model_copy(update={"rgb_channels": (2, 1, 0)})
    with rgb_levels(source) as levels:
        np.testing.assert_array_equal(read_rgb_region(levels[0], (0, 0, *source.shape)), pixels[..., ::-1])
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        fit_source(source, StainRecipe())
    stack = tmp_path / "stack.ome.tiff"
    tifffile.imwrite(stack, np.ones((2, 20, 30, 3), dtype=np.uint8), photometric="rgb", metadata={"axes": "ZYXS"})
    with pytest.raises(ValueError, match="stacks"):
        inspect_stain_file(stack)


def test_zarr_mapping_unknown_spacing_and_roundtrip(tmp_path: Path) -> None:
    pixels = mixture()
    path = tmp_path / "input.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.create_array("0", data=np.moveaxis(pixels, -1, 0), chunks=(3, 32, 32))
    root.attrs["multiscales"] = [{"axes": ["c", "y", "x"], "datasets": [{"path": "0"}]}]
    root.attrs["omero"] = {"channels": [{"label": name} for name in ["R", "G", "B"]]}
    source = inspect_stain_file(path)[0]
    assert source.rgb_channels == (0, 1, 2)
    assert source.spacing is None
    with rgb_levels(source) as levels:
        np.testing.assert_array_equal(read_rgb_region(levels[0], (0, 0, *source.shape)), pixels)
    outputs = export_stains(source, StainRecipe(vector_mode="shared", export_channels=(0,)), tmp_path / "out")
    with tifffile.TiffFile(outputs[0]) as handle:
        assert "PhysicalSizeX" not in handle.ome_metadata
        assert handle.series[0].shape == pixels.shape[:2]


def test_reference_crop_coordinates_and_intensity_normalization(tmp_path: Path) -> None:
    source = make_stain_source(tmp_path / "source.ome.tiff")
    reference = make_stain_source(tmp_path / "reference.ome.tiff", mixture((1100, 1300), "uint16"))
    profile = StainProfile(vectors=((0.72, 0.64, 0.25), (0.15, 0.97, 0.15)))
    recipe = StainRecipe(reference=reference, reference_profile=profile, vector_mode="shared")
    colors = preview_stains(source, recipe)
    both = preview_stains(source, recipe.model_copy(update={"normalization": "colors_intensity"}))
    np.testing.assert_array_equal(colors.channels, both.channels)
    assert colors.reference_scale == (1, 1)
    assert colors.reference_origin == (38, 138)
    assert colors.normalized.dtype == np.uint8
    assert np.any(colors.normalized != both.normalized)


def test_sampling_and_overview_have_bounded_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import image2image.utils.stain_io as io

    source = make_stain_source(tmp_path / "source.ome.tiff").model_copy(update={"shape": (100_000, 120_000)})
    requests = []

    class LazyRGB:
        shape = (*source.shape, 3)

        def __getitem__(self, slices: tuple[slice, ...]) -> np.ndarray:
            y, x, _ = slices
            requests.append((y.stop - y.start, x.stop - x.start))
            assert max(requests[-1]) <= 1024
            return np.full((*requests[-1], 3), [150, 70, 120], dtype=np.uint8)

    @contextmanager
    def levels(_source: StainSource):
        yield [LazyRGB()]

    monkeypatch.setattr(io, "rgb_levels", levels)
    first = sample_rgb(source, StainRecipe())
    second = sample_rgb(source, StainRecipe())
    np.testing.assert_array_equal(first, second)
    assert len(first) <= 100_000
    assert len(requests) == 128
    assert all(height <= 256 and width <= 256 for height, width in requests)
    overview, scale, origin = load_stain_overview(source)
    assert overview.shape == (1024, 1024, 3)
    assert scale == (1, 1)
    assert origin == ((100_000 - 1024) // 2, (120_000 - 1024) // 2)


def test_batch_failures_cancellation_and_publication_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import image2image.utils.stain_io as io

    source = make_stain_source(tmp_path / "source.ome.tiff", mixture((519, 537)))
    blank = make_stain_source(tmp_path / "blank.ome.tiff", np.full((60, 70, 3), 255, dtype=np.uint8))
    results = export_stain_batch((blank, source), StainRecipe(), tmp_path / "batch")
    assert "Too little" in results[0].error
    assert results[1].outputs[0].exists()
    cancelled = Event()
    with pytest.raises(InterruptedError):
        export_stains(
            source,
            StainRecipe(),
            tmp_path / "cancel",
            cancelled=cancelled.is_set,
            progress=lambda _text: cancelled.set(),
        )
    assert list((tmp_path / "cancel").iterdir()) == []
    link = io.os.link
    calls = []

    def failing_link(source_path: Path, target: Path) -> None:
        calls.append(target)
        if len(calls) == 3:
            raise OSError("Simulated publication failure")
        link(source_path, target)

    monkeypatch.setattr(io.os, "link", failing_link)
    with pytest.raises(OSError, match="publication"):
        export_stains(source, StainRecipe(reference=source, export_normalized=True), tmp_path / "failed")
    assert list((tmp_path / "failed").iterdir()) == []


@pytest.mark.parametrize("planar", [False, True])
def test_czi_only_decodes_intersecting_native_subblocks(planar: bool) -> None:
    from image2image.utils._stain_czi import CziStainArray

    reads = []
    entries = []
    for index, x in enumerate((0, 64, 128)):
        channels = range(3) if planar else (0,)
        for channel in channels:
            shape = (1, 1, 64, 64, 1 if planar else 3)
            values = np.full(shape, channel * 50 + index + 10, dtype=np.uint8)
            if not planar:
                values[..., 1] += 50
                values[..., 2] += 100

            def data(*, resize: bool, _values=values, _index=index, _channel=channel):
                assert not resize
                reads.append((_index, _channel))
                return _values

            segment = SimpleNamespace(data=data)
            entries.append(
                SimpleNamespace(
                    shape=shape,
                    stored_shape=shape,
                    start=(0, channel, 0, x, 0),
                    data_segment=lambda _segment=segment: _segment,
                )
            )
    handle = SimpleNamespace(
        axes="SCYX0", dtype=np.uint8, start=(0, 0, 0, 0, 0), is_rgb=not planar, filtered_subblock_directory=entries
    )
    array = CziStainArray(handle, (64, 192), (2, 1, 0))
    assert not reads
    actual = read_rgb_region(array, (10, 70, 20, 30))
    np.testing.assert_array_equal(actual[0, 0], [111, 61, 11])
    assert len(reads) == (3 if planar else 1)
    assert {index for index, _ in reads} == {1}
    np.testing.assert_array_equal(read_rgb_region(array, (10, 70, 20, 30)), actual)
    assert len(reads) == (3 if planar else 1)
