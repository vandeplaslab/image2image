"""Native-intensity, tiled illumination correction and lossless microscopy export."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Self

import numpy as np
from scipy import ndimage as ndi

from image2image.models.illumination import CorrectionOutcome, CorrectionRecipe, IlluminationSource

TILE_SIZE = 512
MAX_REGION_PIXELS = 8_000_000
Cancel = Callable[[], bool] | None
Progress = Callable[[str], None] | None


def _check_cancel(cancelled: Cancel) -> None:
    if cancelled and cancelled():
        raise InterruptedError("Correction cancelled; partial output removed.")


def _identity(path: Path) -> tuple[int, int]:
    # Directory mtimes do not reflect edits to existing Zarr chunks.
    if path.is_dir():
        stats = [item.stat() for item in path.rglob("*") if item.is_file()]
        return sum(item.st_size for item in stats), max((item.st_mtime_ns for item in stats), default=0)
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


@contextmanager
def _reader(path: Path, scene: int) -> Iterator[Any]:
    from image2image_io.config import CONFIG
    from image2image_io.readers import (
        TIFF_EXTENSIONS,
        BaseReader,
        CziImageReader,
        CziSceneImageReader,
        TiffImageReader,
        get_czi_metadata,
        get_simple_reader,
    )

    reader: BaseReader
    if path.suffix.lower() in TIFF_EXTENSIONS:
        reader = TiffImageReader(path, series_index=scene, init_pyramid=False, auto_pyramid=False)
    elif path.suffix.lower() == ".czi":
        if len(get_czi_metadata(path)) > 1:
            reader = CziSceneImageReader(path, scene_index=scene, init_pyramid=False, auto_pyramid=False)
        else:
            reader = CziImageReader(path, init_pyramid=False, auto_pyramid=False)
    else:
        settings = CONFIG.init_pyramid, CONFIG.auto_pyramid, CONFIG.quiet
        try:
            reader = get_simple_reader(path, scene_index=scene, init_pyramid=False, auto_pyramid=False)
        finally:
            CONFIG.init_pyramid, CONFIG.auto_pyramid, CONFIG.quiet = settings
    try:
        yield reader
    finally:
        reader.close()


def _zarr_info(path: Path) -> tuple[Any, list[str], dict[str, Any], dict[str, Any]]:
    import zarr

    root = zarr.open_group(str(path), mode="r")
    attrs: dict[str, Any] = dict(root.attrs)
    attrs = attrs.get("ome", attrs)
    scales = attrs.get("multiscales", [])
    if len(scales) != 1:
        raise ValueError("Select an OME-Zarr image group containing exactly one multiscale image.")
    scale = scales[0]
    axes = [a["name"] if isinstance(a, dict) else a for a in scale["axes"]]
    array = root[scale["datasets"][0]["path"]]
    if not isinstance(array, zarr.Array):
        raise ValueError("The first OME-Zarr dataset must be an array.")  # noqa: TRY004
    if len(axes) != array.ndim or len(set(axes)) != len(axes):
        raise ValueError("Invalid OME-Zarr axis metadata.")
    for axis, size in zip(axes, array.shape, strict=True):
        if axis not in ("c", "y", "x") and size != 1:
            raise ValueError("Z/time stacks are not supported; select a 2D image.")
    if "y" not in axes or "x" not in axes:
        raise ValueError("The image must have X and Y axes.")
    return array, axes, scale, attrs.get("omero", scale.get("omero", {}))


def _spacing_zarr(scale: dict[str, Any]) -> tuple[float, float] | None:
    axes = scale["axes"]
    names = [a["name"] if isinstance(a, dict) else a for a in axes]
    factors = {"micrometer": 1.0, "nanometer": 0.001, "millimeter": 1000.0, "meter": 1e6}
    sizes = np.ones(len(axes), dtype=float)
    for transform in [
        *scale.get("coordinateTransformations", []),
        *scale["datasets"][0].get("coordinateTransformations", []),
    ]:
        if transform["type"] == "scale":
            sizes *= np.asarray(transform["scale"])
    result = []
    for name in ("y", "x"):
        i = names.index(name)
        unit = axes[i].get("unit") if isinstance(axes[i], dict) else None
        if unit not in factors:
            return None
        result.append(float(sizes[i] * factors[unit]))
    return tuple(result)  # type: ignore[return-value]


def _spacing_tiff(handle: Any, scene: int) -> tuple[float, float] | None:
    if handle.ome_metadata:
        from ome_types import from_xml
        from pint import UnitRegistry

        pixels = from_xml(handle.ome_metadata).images[scene].pixels
        if pixels.physical_size_x is None or pixels.physical_size_y is None:
            return None
        units: Any = UnitRegistry()
        return tuple(
            float((value * units(unit.value)).to("micrometer").magnitude)
            for value, unit in (
                (pixels.physical_size_y, pixels.physical_size_y_unit),
                (pixels.physical_size_x, pixels.physical_size_x_unit),
            )
        )  # type: ignore[return-value]
    tags = handle.series[scene].pages[0].tags
    unit = tags.get("ResolutionUnit")
    if unit is None or unit.value not in (2, 3):
        return None
    factor = 25400.0 if unit.value == 2 else 10000.0
    result = []
    for axis in ("YResolution", "XResolution"):
        if axis not in tags:
            return None
        numerator, denominator = tags[axis].value
        result.append(factor * denominator / numerator)
    return tuple(result)  # type: ignore[return-value]


def inspect_illumination_file(path: str | Path) -> tuple[IlluminationSource, ...]:
    """Discover 2D scenes without guessing a scene or collapsing Z/time dimensions."""
    from image2image_io.readers import TIFF_EXTENSIONS, get_czi_metadata

    path = Path(path).resolve()
    size, modified = _identity(path)
    common: dict[str, Any] = {"path": path, "size": size, "modified_ns": modified}
    if path.is_dir():
        array, axes, scale, omero = _zarr_info(path)
        count = array.shape[axes.index("c")] if "c" in axes else 1
        labels = omero.get("channels", [])
        names = tuple(labels[i].get("label", f"C{i}") if i < len(labels) else f"C{i}" for i in range(count))
        source = IlluminationSource(
            **common,
            shape=(array.shape[axes.index("y")], array.shape[axes.index("x")]),
            channels=names,
            dtype=np.dtype(array.dtype).name,
            spacing=_spacing_zarr(scale),
        )
        _validate_source(source)
        return (source,)
    scenes = [0]
    spacings: dict[int, tuple[float, float] | None] = {}
    if path.suffix.lower() in TIFF_EXTENSIONS:
        from tifffile import TiffFile

        with TiffFile(path) as handle:
            scenes = list(range(len(handle.series)))
            for i, series in enumerate(handle.series):
                if any(axis not in "CYXS" and n > 1 for axis, n in zip(series.axes, series.shape, strict=True)):
                    raise ValueError(f"Scene {i}: Z/time or unidentified stacks are not supported.")
                spacings[i] = _spacing_tiff(handle, i)
    elif path.suffix.lower() == ".czi":
        scenes = [item["scene_index"] or 0 for item in get_czi_metadata(path).values()]
    result = []
    for scene in scenes:
        with _reader(path, scene) as reader:
            if reader.reader_type != "image" or len(reader.image_shape) != 2:
                raise ValueError("Supply a 2D microscopy image.")
            if path.suffix.lower() == ".czi":
                if any(a in "ZT" and n > 1 for a, n in zip(reader.fh.axes, reader.fh.shape, strict=True)):
                    raise ValueError("Z/time stacks are not supported; select a 2D image.")
                distances = reader.xml_metadata()["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
                if isinstance(distances, dict):
                    distances = [distances]
                values = {item.get("Id", item.get("@Id")): float(item["Value"]) * 1e6 for item in distances}
                spacings[scene] = (values["Y"], values["X"]) if "Y" in values and "X" in values else None
            channels = tuple(reader.channel_names)
            if reader.is_rgb:
                if reader.n_channels != 3:
                    raise ValueError("Only three-component RGB images are supported; remove alpha explicitly.")
                channels = ("R", "G", "B")
            source = IlluminationSource(
                **common,
                scene=scene,
                shape=reader.image_shape,
                channels=channels,
                dtype=np.dtype(reader.dtype).name,
                spacing=spacings.get(scene),
                rgb=reader.is_rgb,
            )
            _validate_source(source)
            result.append(source)
    return tuple(result)


def _validate_source(source: IlluminationSource) -> None:
    dtype = np.dtype(source.dtype)
    if dtype.kind not in "uif" or (dtype.kind in "ui" and dtype.itemsize > 4):
        raise ValueError("Supported intensities are 8/16/32-bit integers and 32/64-bit floats.")
    if min(source.shape) <= 0 or not source.channels:
        raise ValueError("Image dimensions and channels must be nonempty.")
    if source.spacing is not None and (not np.isfinite(source.spacing).all() or min(source.spacing) <= 0):
        raise ValueError("Spatial calibration must be finite and positive.")


def validate_recipe(source: IlluminationSource, recipe: CorrectionRecipe) -> None:
    """Check batch compatibility and explicit reference mappings before reading pixels."""
    _validate_source(source)
    if source.channels != recipe.channels:
        raise ValueError("Image channel names/order do not match the shared recipe.")
    for reference, mapping in ((recipe.dark, recipe.dark_channels), (recipe.flat, recipe.flat_channels)):
        if reference is None:
            continue
        if reference.shape != source.shape:
            raise ValueError(f"Calibration dimensions differ: {reference.path.name}; references are never resized.")
        if reference.spacing and source.spacing and not np.allclose(reference.spacing, source.spacing, rtol=1e-6):
            raise ValueError("Calibration and source pixel sizes differ.")
        if len(mapping) != len(source.channels) or any(i < 0 or i >= len(reference.channels) for i in mapping):
            raise ValueError("Select a calibration channel explicitly for every source channel.")


def save_correction_recipe(path: str | Path, recipe: CorrectionRecipe) -> None:
    """Save a versioned recipe containing explicit reference scene and channel selections."""
    Path(path).write_text(recipe.model_dump_json(indent=2), encoding="utf-8")


def load_correction_recipe(path: str | Path) -> CorrectionRecipe:
    """Load and validate a versioned recipe without changing its parameters."""
    return CorrectionRecipe.model_validate_json(Path(path).read_text(encoding="utf-8"))


def estimate_background(image: np.ndarray, recipe: CorrectionRecipe, *, scale: float = 1.0) -> np.ndarray:
    """Estimate a background in native intensity units; scale only spatial parameters for overviews."""
    from skimage.restoration import ellipsoid_kernel, rolling_ball

    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Background estimation requires finite 2D pixels.")
    if recipe.background == "none":
        return np.zeros_like(image)
    if recipe.background == "gaussian":
        sigma = recipe.sigma / scale
        return ndi.gaussian_filter(image, sigma, radius=math.ceil(4 * sigma), mode="reflect")
    radius = max(1, math.ceil(recipe.radius / scale))
    signed = image if recipe.polarity == "bright" else -image
    if recipe.background == "morphology":
        # A square footprint permits separable, bounded-memory opening on large slides.
        background = ndi.grey_opening(signed, size=(2 * radius + 1,) * 2, mode="reflect")
    else:
        kernel = ellipsoid_kernel((2 * radius, 2 * radius), recipe.height)
        background = rolling_ball(signed, kernel=kernel)
    return background if recipe.polarity == "bright" else -background


def correction_halo(recipe: CorrectionRecipe) -> int:
    """Return the complete spatial support needed for a seam-free native region."""
    if recipe.background == "gaussian":
        return math.ceil(4 * recipe.sigma)
    if recipe.background == "morphology":
        return 2 * recipe.radius
    return recipe.radius if recipe.background == "rolling_ball" else 0


def convert_intensity(image: np.ndarray, dtype: str | np.dtype) -> np.ndarray:
    """Cast without rescaling; integers clip then round to nearest even, floats must remain finite."""
    dtype = np.dtype(dtype)
    if not np.isfinite(image).all():
        raise ValueError("Correction produced non-finite pixels.")
    if dtype.kind in "ui":
        if dtype.itemsize > 4:
            raise ValueError("Integer export is limited to at most 32 bits.")
        limits = np.iinfo(dtype)
        return np.rint(np.clip(image, limits.min, limits.max)).astype(dtype)
    if dtype.kind != "f":
        raise ValueError("Output must use a real numeric dtype.")
    if np.any(np.abs(image) > np.finfo(dtype).max):
        raise ValueError(f"Corrected intensities exceed the finite range of {dtype.name}.")
    return image.astype(dtype)


@contextmanager
def _pixels(source: IlluminationSource) -> Iterator[list[Any]]:
    if _identity(source.path) != (source.size, source.modified_ns):
        raise ValueError(f"Input changed after loading: {source.path}. Reload it before continuing.")
    if source.path.is_dir():
        import dask.array as da

        array, axes, _, _ = _zarr_info(source.path)
        pixels = da.from_zarr(array)
        for i in reversed(range(len(axes))):
            if axes[i] not in ("c", "y", "x"):
                pixels = pixels.squeeze(axis=i)
                axes.pop(i)
        if "c" not in axes:
            pixels = pixels[None]
            axes.insert(0, "c")
        pixels = pixels.transpose(*(axes.index(a) for a in ("c", "y", "x")))
        yield [pixels[i] for i in range(pixels.shape[0])]
    else:
        with _reader(source.path, source.scene) as reader:
            yield [reader.get_channel(i, split_rgb=True) for i in range(len(source.channels))]


def _read(array: Any, region: tuple[int, int, int, int], step: int = 1) -> np.ndarray:
    y, x, height, width = region
    part = array[y : y + height : step, x : x + width : step]
    if hasattr(part, "compute"):
        part = part.compute()
    result = np.asarray(part, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"Input contains {np.count_nonzero(~np.isfinite(result))} non-finite pixels in this region.")
    return result


def _regions(shape: tuple[int, int], tile: int = TILE_SIZE) -> Iterator[tuple[int, int, int, int]]:
    for y in range(0, shape[0], tile):
        for x in range(0, shape[1], tile):
            yield y, x, min(tile, shape[0] - y), min(tile, shape[1] - x)


class CorrectionProcessor:
    """Own independent lazy readers and whole-reference normalization for one worker."""

    def __init__(
        self,
        source: IlluminationSource,
        recipe: CorrectionRecipe,
        *,
        cancelled: Cancel = None,
        calibration_cache: dict[str, tuple[float, ...]] | None = None,
    ) -> None:
        validate_recipe(source, recipe)
        self.source, self.recipe, self.cancelled = source, recipe, cancelled
        self.stack = ExitStack()
        self.cache = calibration_cache if calibration_cache is not None else {}
        self.means: tuple[float, ...] = ()

    def __enter__(self) -> Self:
        """Open independent readers and validate complete calibration frames."""
        try:
            self.images = self.stack.enter_context(_pixels(self.source))
            self.dark = self.stack.enter_context(_pixels(self.recipe.dark)) if self.recipe.dark else None
            self.flat = self.stack.enter_context(_pixels(self.recipe.flat)) if self.recipe.flat else None
            key = self.recipe.model_dump_json(
                include={"dark", "flat", "dark_channels", "flat_channels", "flat_dark_corrected", "minimum_response"}
            )
            if self.flat is not None:
                if key not in self.cache:
                    self.cache[key] = self._flat_means()
                self.means = self.cache[key]
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *_args: object) -> None:
        """Release all owned readers."""
        self.stack.close()

    def _flat_response(self, channel: int, region: tuple[int, int, int, int], step: int = 1) -> np.ndarray:
        if self.flat is None:
            raise ValueError("A flat reference is required.")
        response = _read(self.flat[self.recipe.flat_channels[channel]], region, step)
        if self.dark is not None and not self.recipe.flat_dark_corrected:
            response -= _read(self.dark[self.recipe.dark_channels[channel]], region, step)
        return response

    def _flat_means(self) -> tuple[float, ...]:
        means = []
        for channel in range(len(self.source.channels)):
            total, count, minimum = 0.0, 0, math.inf
            for region in _regions(self.source.shape):
                _check_cancel(self.cancelled)
                response = self._flat_response(channel, region)
                total += float(response.sum())
                count += response.size
                minimum = min(minimum, float(response.min()))
            mean = total / count
            if not np.isfinite(mean) or mean <= 0 or minimum <= mean * self.recipe.minimum_response:
                invalid = 0
                for region in _regions(self.source.shape):
                    _check_cancel(self.cancelled)
                    response = self._flat_response(channel, region)
                    invalid += int(np.count_nonzero(response <= max(0, mean * self.recipe.minimum_response)))
                raise ValueError(
                    f"Flat channel {channel}: {invalid} invalid pixels; mean response {mean:g}. Export blocked."
                )
            means.append(mean)
        return tuple(means)

    def calibrated(
        self, channel: int, region: tuple[int, int, int, int], step: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read and calibrate one region, retaining the original pixels for comparison."""
        _check_cancel(self.cancelled)
        original = _read(self.images[channel], region, step)
        calibrated = original.copy()
        if self.dark is not None:
            calibrated -= _read(self.dark[self.recipe.dark_channels[channel]], region, step)
        if self.flat is not None:
            calibrated *= self.means[channel] / self._flat_response(channel, region, step)
        if not np.isfinite(calibrated).all():
            raise ValueError("Calibration produced non-finite intensities.")
        return original, calibrated

    def region(self, channel: int, region: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Correct a native region using its complete neighborhood, identical to export."""
        y, x, height, width = region
        if (
            min(y, x) < 0
            or min(height, width) <= 0
            or y + height > self.source.shape[0]
            or x + width > self.source.shape[1]
        ):
            raise ValueError("Preview region lies outside the source image.")
        halo = correction_halo(self.recipe)
        top, left = max(0, y - halo), max(0, x - halo)
        bottom, right = min(self.source.shape[0], y + height + halo), min(self.source.shape[1], x + width + halo)
        if (bottom - top) * (right - left) > MAX_REGION_PIXELS:
            raise ValueError("Filter neighborhood exceeds the memory limit; reduce the radius or preview size.")
        original, calibrated = self.calibrated(channel, (top, left, bottom - top, right - left))
        background = estimate_background(calibrated, self.recipe)
        _check_cancel(self.cancelled)
        crop = np.s_[y - top : y - top + height, x - left : x - left + width]
        return original[crop], (calibrated - background)[crop], background[crop]


@dataclass(frozen=True)
class CorrectionPreview:
    """One display-only comparison, including its native-pixel placement and intensity statistics."""

    original: np.ndarray
    corrected: np.ndarray
    background: np.ndarray
    scale: int
    origin: tuple[int, int]
    statistics: dict[str, Any]


def _stats(image: np.ndarray, dtype: np.dtype) -> dict[str, Any]:
    lower, upper = (np.iinfo(dtype).min, np.iinfo(dtype).max) if dtype.kind in "ui" else (-math.inf, math.inf)
    return {
        "minimum": float(image.min()),
        "maximum": float(image.max()),
        "pixels": image.size,
        "negative": int(np.count_nonzero(image < 0)),
        "clipped_low": int(np.count_nonzero(image < lower)),
        "clipped_high": int(np.count_nonzero(image > upper)),
    }


def preview_correction(
    source: IlluminationSource,
    recipe: CorrectionRecipe,
    channel: int = 0,
    *,
    region: tuple[int, int, int, int] | None = None,
    max_size: int = 512,
    cancelled: Cancel = None,
    calibration_cache: dict[str, tuple[float, ...]] | None = None,
) -> CorrectionPreview:
    """Compute an approximate overview or exact native region without display normalization."""
    if not 0 <= channel < len(source.channels) or max_size <= 0:
        raise ValueError("Invalid preview channel or size.")
    with CorrectionProcessor(source, recipe, cancelled=cancelled, calibration_cache=calibration_cache) as processor:
        if region is None:
            step = max(1, math.ceil(max(source.shape) / max_size))
            original, calibrated = processor.calibrated(channel, (0, 0, *source.shape), step)
            background = estimate_background(calibrated, recipe, scale=step)
            corrected = calibrated - background
            origin = (0, 0)
        else:
            original, corrected, background = processor.region(channel, region)
            step, origin = 1, region[:2]
        dtype = np.dtype(source.dtype if recipe.output_dtype == "source" else recipe.output_dtype)
        convert_intensity(corrected, dtype)  # Validate the selected export range even for a float preview.
        statistics = _stats(corrected, dtype)
        statistics.update(
            original_minimum=float(original.min()),
            original_maximum=float(original.max()),
            flat_means=processor.means,
            output_dtype=dtype.name,
        )
        return CorrectionPreview(original, corrected, background, step, origin, statistics)


def _merge_stats(total: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        if key == "minimum":
            total[key] = min(total.get(key, math.inf), value)
        elif key == "maximum":
            total[key] = max(total.get(key, -math.inf), value)
        else:
            total[key] = total.get(key, 0) + value


def _provenance(source: IlluminationSource, recipe: CorrectionRecipe) -> dict[str, Any]:
    from importlib.metadata import version

    from image2image import __version__

    return {
        "schema_version": 1,
        "app": "illumination",
        "image2image_version": __version__,
        "libraries": {name: version(name) for name in ("numpy", "scipy", "scikit-image", "tifffile", "zarr")},
        "source": source.model_dump(mode="json"),
        "recipe": recipe.model_dump(mode="json"),
        "intensity_handling": "No rescaling. Float output retains negatives; integer output clips then rounds to nearest even.",
        "pyramid": "2x2 means, edge padding for odd dimensions; integer levels round to nearest even.",
        "morphology_footprint": "square",
        "statistics": [],
    }


def _build_store(
    path: Path,
    processor: CorrectionProcessor,
    provenance: dict[str, Any],
    progress: Progress,
) -> list[Any]:
    import zarr

    source, recipe = processor.source, processor.recipe
    dtype = np.dtype(source.dtype if recipe.output_dtype == "source" else recipe.output_dtype)
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    shape = (len(source.channels), *source.shape)
    current = root.create_array("0", shape=shape, chunks=(1, TILE_SIZE, TILE_SIZE), dtype=dtype)
    provenance.update(output_dtype=dtype.name, flat_normalization_means=list(processor.means))
    for channel, name in enumerate(source.channels):
        total: dict[str, Any] = {}
        # Smaller rolling-ball tiles bound the time between cancellation checks.
        tile = 128 if recipe.background == "rolling_ball" else TILE_SIZE
        for y, x, height, width in _regions(source.shape, tile):
            _, corrected, _ = processor.region(channel, (y, x, height, width))
            _merge_stats(total, _stats(corrected, dtype))
            current[channel, y : y + height, x : x + width] = convert_intensity(corrected, dtype)
            if progress:
                progress(f"{source.path.name}: {name}, row {y + height}/{source.shape[0]}")
        provenance["statistics"].append({"channel": name, **total})
    levels = [current]
    while max(current.shape[1:]) > TILE_SIZE:
        size = ((current.shape[1] + 1) // 2, (current.shape[2] + 1) // 2)
        following = root.create_array(
            str(len(levels)), shape=(shape[0], *size), chunks=(1, TILE_SIZE, TILE_SIZE), dtype=dtype
        )
        for channel in range(shape[0]):
            for y, x, height, width in _regions(size):
                _check_cancel(processor.cancelled)
                block = np.asarray(
                    current[channel, y * 2 : (y + height) * 2, x * 2 : (x + width) * 2], dtype=np.float64
                )
                block = np.pad(block, ((0, block.shape[0] % 2), (0, block.shape[1] % 2)), mode="edge")
                reduced = block.reshape(height, 2, width, 2).mean(axis=(1, 3))
                following[channel, y : y + height, x : x + width] = convert_intensity(reduced, dtype)
        levels.append(following)
        current = following
    spacing = source.spacing or (1.0, 1.0)
    spatial = {"type": "space", **({"unit": "micrometer"} if source.spacing else {})}
    root.attrs["multiscales"] = [
        {
            "version": "0.4",
            "name": source.path.name,
            "axes": [{"name": "c", "type": "channel"}, {"name": "y", **spatial}, {"name": "x", **spatial}],
            "datasets": [
                {
                    "path": str(i),
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, spacing[0] * 2**i, spacing[1] * 2**i]}
                    ],
                }
                for i in range(len(levels))
            ],
        }
    ]
    root.attrs["omero"] = {
        "channels": [
            {"label": name, "color": color} for name, color in zip(source.channels, _colors(source), strict=True)
        ]
    }
    root.attrs["image2image"] = provenance
    return levels


def _colors(source: IlluminationSource) -> list[str]:
    return ["FF0000", "00FF00", "0000FF"] if source.channels == ("R", "G", "B") else ["FFFFFF"] * len(source.channels)


def _write_tiff(
    path: Path, levels: list[Any], source: IlluminationSource, provenance: dict[str, Any], cancelled: Cancel
) -> None:
    from tifffile import TiffWriter

    def tiles(array: Any) -> Iterator[np.ndarray]:
        for channel in range(array.shape[0]):
            for y, x, height, width in _regions(array.shape[1:]):
                _check_cancel(cancelled)
                tile = np.zeros((TILE_SIZE, TILE_SIZE), dtype=array.dtype)
                tile[:height, :width] = array[channel, y : y + height, x : x + width]
                yield tile

    metadata: dict[str, Any] = {
        "axes": "CYX",
        "Channel": {"Name": list(source.channels)},
        "Description": json.dumps(provenance),
    }
    if source.spacing:
        metadata.update(
            PhysicalSizeY=source.spacing[0],
            PhysicalSizeX=source.spacing[1],
            PhysicalSizeYUnit="µm",
            PhysicalSizeXUnit="µm",
        )
    with TiffWriter(path, ome=True, bigtiff=True) as writer:
        for index, array in enumerate(levels):
            writer.write(
                tiles(array),
                shape=array.shape,
                dtype=array.dtype,
                tile=(TILE_SIZE, TILE_SIZE),
                photometric="minisblack",
                compression="deflate",
                metadata=metadata if index == 0 else None,
                subifds=len(levels) - 1 if index == 0 else None,
                subfiletype=0 if index == 0 else 1,
            )


def export_correction(
    source: IlluminationSource,
    recipe: CorrectionRecipe,
    output: str | Path,
    *,
    cancelled: Cancel = None,
    progress: Progress = None,
    calibration_cache: dict[str, tuple[float, ...]] | None = None,
) -> Path:
    """Write a new lossless OME image and JSON record; clean up incomplete output on failure."""
    output = Path(output).resolve()
    is_zarr = output.name.endswith(".ome.zarr")
    if not is_zarr and not output.name.endswith((".ome.tiff", ".ome.tif")):
        raise ValueError("Output must end with .ome.tiff, .ome.tif, or .ome.zarr.")
    sidecar = output.with_name(output.name + ".json")
    inputs = [s.path.resolve() for s in (source, recipe.dark, recipe.flat) if s]
    if any(output == path or path in output.parents or sidecar == path for path in inputs):
        raise ValueError("Output must not replace or be inside an input image.")
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Output or parameter record already exists: {output.name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    published_record = False
    try:
        with TemporaryDirectory(prefix=".illumination-", dir=output.parent) as directory:
            temporary = Path(directory)
            store = temporary / "corrected.ome.zarr"
            provenance = _provenance(source, recipe)
            with CorrectionProcessor(
                source, recipe, cancelled=cancelled, calibration_cache=calibration_cache
            ) as processor:
                levels = _build_store(store, processor, provenance, progress)
                staged = store
                if not is_zarr:
                    staged = temporary / "corrected.ome.tiff"
                    _write_tiff(staged, levels, source, provenance, cancelled)
                _check_cancel(cancelled)
            record = temporary / "parameters.json"
            record.write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
            if output.exists():
                raise FileExistsError(f"Output already exists: {output}")  # noqa: TRY301
            os.link(record, sidecar)  # Exclusive publication: never overwrite an existing sidecar.
            published_record = True
            if is_zarr:
                if output.exists():
                    raise FileExistsError(f"Output already exists: {output}")  # noqa: TRY301
                staged.rename(output)
            else:
                os.link(staged, output)
    except BaseException:
        if published_record:
            sidecar.unlink(missing_ok=True)
        raise
    return output


def export_correction_batch(
    sources: Sequence[IlluminationSource],
    recipe: CorrectionRecipe,
    directory: str | Path,
    *,
    output_format: str = "ome.tiff",
    cancelled: Cancel = None,
    progress: Progress = None,
    completed: Callable[[CorrectionOutcome], None] | None = None,
) -> list[CorrectionOutcome]:
    """Apply a frozen recipe sequentially, retaining per-image failures and stopping on cancellation."""
    if output_format not in ("ome.tiff", "ome.zarr"):
        raise ValueError("Select OME-TIFF or OME-Zarr output.")
    results = []
    cache: dict[str, tuple[float, ...]] = {}
    for source in sources:
        _check_cancel(cancelled)
        stem = source.path.name
        for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr", source.path.suffix):
            if suffix and stem.endswith(suffix):
                stem = stem.removesuffix(suffix)
                break
        output = Path(directory) / f"{stem}.scene-{source.scene}.corrected.{output_format}"
        try:
            export_correction(source, recipe, output, cancelled=cancelled, progress=progress, calibration_cache=cache)
            results.append(CorrectionOutcome(source=source, output=output))
        except InterruptedError:
            raise
        except (OSError, ValueError, RuntimeError, MemoryError) as exc:
            results.append(CorrectionOutcome(source=source, error=str(exc)))
        if completed:
            completed(results[-1])
        if progress:
            progress(f"{source.path.name}: {results[-1].error or 'Completed'}")
    return results
