"""Bounded RGB reading, asynchronous-preview data, and staged stain batch exports."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np

from image2image.models.stains import StainOutcome, StainRecipe, StainSource
from image2image.utils.illumination import _identity, _reader, inspect_illumination_file
from image2image.utils.stains import (
    StainFit,
    fit_stains,
    normalization_scales,
    reconstruct_stains,
    registration_channels,
    separate_stains,
)

TILE_SIZE = 512
Cancel = Callable[[], bool] | None
Progress = Callable[[str], None] | None
FitCache = dict[tuple[str, ...], StainFit]
Region = tuple[int, int, int, int]


def _check_cancel(cancelled: Cancel) -> None:
    if cancelled and cancelled():
        raise InterruptedError("Stain processing cancelled; incomplete outputs removed.")


def inspect_stain_file(path: str | Path) -> tuple[StainSource, ...]:
    """Inspect supported 2D scenes, inferring RGB only from RGB metadata or unambiguous names."""
    sources = []
    for info in inspect_illumination_file(path):
        if info.dtype not in ("uint8", "uint16") or len(info.channels) != 3:
            raise ValueError("Supply a 2D uint8/uint16 brightfield image containing exactly three RGB components.")
        names = tuple(name.lower() for name in info.channels)
        mapping = (0, 1, 2) if info.rgb else None
        for labels in (("r", "g", "b"), ("red", "green", "blue")):
            if set(names) == set(labels):
                mapping = tuple(names.index(label) for label in labels)
        sources.append(StainSource(**info.model_dump(), rgb_channels=mapping))
    return tuple(sources)


def _validate_source(source: StainSource) -> None:
    if source.rgb_channels is None or sorted(source.rgb_channels) != [0, 1, 2]:
        raise ValueError("Map the source channels to red, green, and blue explicitly.")
    if source.dtype not in ("uint8", "uint16") or len(source.channels) != 3:
        raise ValueError("Only three-component uint8/uint16 brightfield sources are supported.")
    if min(source.shape) <= 0:
        raise ValueError("Image dimensions must be positive.")
    if _identity(source.path) != (source.size, source.modified_ns):
        raise ValueError(f"Input changed after loading: {source.path.name}. Reload the image.")


def _as_rgb(array: Any, axes: str, mapping: tuple[int, int, int]) -> Any:
    axes = axes.lower().replace("s", "c")
    for index in reversed(range(len(axes))):
        if axes[index] not in "cyx":
            if array.shape[index] != 1:
                raise ValueError("Z/time stacks are not supported; select a 2D image.")
            array = array.squeeze(axis=index)
            axes = axes[:index] + axes[index + 1 :]
    if sorted(axes) != ["c", "x", "y"]:
        raise ValueError("The RGB image axes must identify X, Y, and three components.")
    array = array.transpose(*(axes.index(axis) for axis in "yxc"))
    return array if mapping == (0, 1, 2) else array[..., list(mapping)]


@contextmanager
def rgb_levels(source: StainSource) -> Iterator[list[Any]]:
    """Own lazy RGB pyramids without splitting compressed tiles into separate channel reads."""
    import dask.array as da
    import zarr
    from image2image_io.readers import TIFF_EXTENSIONS

    _validate_source(source)
    mapping = cast(tuple[int, int, int], source.rgb_channels)
    if source.path.is_dir():
        root = zarr.open_group(str(source.path), mode="r")
        attrs = dict(root.attrs)
        attrs = attrs.get("ome", attrs)
        scale = attrs["multiscales"][0]
        axes = "".join(axis["name"] if isinstance(axis, dict) else axis for axis in scale["axes"])
        yield [_as_rgb(da.from_zarr(root[item["path"]]), axes, mapping) for item in scale["datasets"]]
    elif source.path.suffix.lower() in TIFF_EXTENSIONS:
        from image2image_io._zarr import ensure_zarr_store
        from tifffile import TiffFile

        stores = []
        try:
            with TiffFile(source.path) as handle:
                levels = []
                series = handle.series[source.scene]
                for index, level in enumerate(series.levels):
                    # Without an explicit level, a pyramid opens as a group, not a pixel array.
                    store = series.aszarr(level=index)
                    stores.append(store)
                    array = zarr.open(ensure_zarr_store(store), mode="r")
                    levels.append(_as_rgb(da.from_zarr(array), level.axes, mapping))
                yield levels
        finally:
            for store in stores:
                store.close()
    else:
        with _reader(source.path, source.scene) as reader:
            if source.path.suffix.lower() == ".czi":
                from image2image.utils._stain_czi import CziStainArray

                yield [CziStainArray(reader.fh, source.shape, mapping)]
                return
            levels = []
            for array in reader.pyramid:
                channel_axis, _ = reader.get_channel_axis_and_n_channels(array.shape)
                axes = {0: "cyx", 1: "ycx", 2: "yxc"}.get(channel_axis)
                if axes is None:
                    raise ValueError("Cannot determine the source RGB axis.")
                levels.append(_as_rgb(array, axes, mapping))
            yield levels


def read_rgb_region(array: Any, region: Region, cancelled: Cancel = None) -> np.ndarray:
    """Materialize one bounded native region, retaining all three color components together."""
    _check_cancel(cancelled)
    y, x, height, width = region
    if min(y, x) < 0 or min(height, width) <= 0 or y + height > array.shape[0] or x + width > array.shape[1]:
        raise ValueError("Selected region lies outside the image.")
    if height * width > 2048**2:
        raise ValueError("Preview region exceeds 2048 x 2048 pixels.")
    part = array[y : y + height, x : x + width, :]
    if hasattr(part, "compute"):
        part = part.compute(scheduler="synchronous")
    _check_cancel(cancelled)
    return np.asarray(part)


def sample_rgb(
    source: StainSource, recipe: StainRecipe, region: Region | None = None, cancelled: Cancel = None
) -> np.ndarray:
    """Sample at most 100,000 native pixels from at most 64 stratified 256-pixel patches."""
    _check_cancel(cancelled)
    rng = np.random.default_rng(recipe.estimation.seed)
    with rgb_levels(source) as levels:
        array = levels[0]
        top, left, height, width = region or (0, 0, *source.shape)
        if (
            min(top, left) < 0
            or min(height, width) <= 0
            or top + height > source.shape[0]
            or left + width > source.shape[1]
        ):
            raise ValueError("Sampling region lies outside the image.")
        ys = np.unique(np.linspace(top, top + max(0, height - 256), min(8, math.ceil(height / 256))).astype(int))
        xs = np.unique(np.linspace(left, left + max(0, width - 256), min(8, math.ceil(width / 256))).astype(int))
        per_patch = max(1, recipe.estimation.max_samples // (len(ys) * len(xs)))
        parts = []
        for y in ys:
            for x in xs:
                pixels = read_rgb_region(array, (int(y), int(x), min(height, 256), min(width, 256)), cancelled).reshape(
                    -1, 3
                )
                if len(pixels) > per_patch:
                    pixels = pixels[rng.choice(len(pixels), per_patch, replace=False)]
                parts.append(pixels)
        return np.concatenate(parts)


def fit_source(
    source: StainSource,
    recipe: StainRecipe,
    *,
    reference: bool = False,
    cache: FitCache | None = None,
    cancelled: Cancel = None,
) -> StainFit:
    """Resolve manual overrides before shared/automatic profiles and cache reproducible fits."""
    _validate_source(source)
    profile = recipe.profile
    estimate = recipe.vector_mode == "auto" and len(profile.names) == 2
    if reference:
        profile = recipe.reference_profile or profile
        estimate = recipe.reference_profile is None and len(profile.names) == 2
    else:
        override = next((item for item in recipe.overrides if item.source.key == source.key), None)
        if override is not None:
            if override.source != source:
                raise ValueError("The saved override belongs to a different file identity or RGB mapping; reset it.")
            profile, estimate = override.profile, False
    key = (source.model_dump_json(), profile.model_dump_json(), recipe.estimation.model_dump_json(), str(estimate))
    if cache is not None and key in cache:
        return cache[key]
    samples = sample_rgb(source, recipe, cancelled=cancelled)
    _check_cancel(cancelled)
    fit = fit_stains(samples, profile, recipe.estimation, estimate=estimate)
    if cache is not None:
        if len(cache) >= 128:
            cache.pop(next(iter(cache)))
        cache[key] = fit
    return fit


def load_stain_overview(
    source: StainSource, cancelled: Cancel = None
) -> tuple[np.ndarray, tuple[float, float], tuple[int, int]]:
    """Read a bounded pyramid overview, or a center crop when no suitable pyramid exists."""
    with rgb_levels(source) as levels:
        # Prefer the most detailed existing level that fits the preview budget.
        candidates = [level for level in levels if max(level.shape[:2]) <= 1024]
        array = candidates[0] if candidates else levels[-1]
        height, width = array.shape[:2]
        scale = (source.shape[0] / height, source.shape[1] / width)
        crop_h, crop_w = min(1024, height), min(1024, width)
        y, x = (height - crop_h) // 2, (width - crop_w) // 2
        image = read_rgb_region(array, (y, x, crop_h, crop_w), cancelled)
        return image, scale, (round(y * scale[0]), round(x * scale[1]))


@dataclass(frozen=True)
class StainPreview:
    """Comparison pixels and resolved fits; native regions use exactly the export transform."""

    original: np.ndarray
    channels: np.ndarray
    normalized: np.ndarray | None
    reference_image: np.ndarray | None
    source_fit: StainFit
    reference_fit: StainFit | None
    scale: tuple[float, float]
    origin: tuple[int, int]
    reference_scale: tuple[float, float] = (1.0, 1.0)
    reference_origin: tuple[int, int] = (0, 0)


def preview_stains(
    source: StainSource,
    recipe: StainRecipe,
    *,
    region: Region | None = None,
    cache: FitCache | None = None,
    cancelled: Cancel = None,
) -> StainPreview:
    """Compute a pyramid overview/center crop or an exact native-region comparison."""
    fit = fit_source(source, recipe, cache=cache, cancelled=cancelled)
    if region is None:
        original, scale, origin = load_stain_overview(source, cancelled)
    else:
        with rgb_levels(source) as levels:
            original = read_rgb_region(levels[0], region, cancelled)
        scale, origin = (1.0, 1.0), region[:2]
    channels = separate_stains(original, fit.profile)
    reference_fit = reference_image = normalized = None
    reference_scale, reference_origin = (1.0, 1.0), (0, 0)
    if recipe.reference is not None:
        reference_fit = fit_source(recipe.reference, recipe, reference=True, cache=cache, cancelled=cancelled)
        reference_image, reference_scale, reference_origin = load_stain_overview(recipe.reference, cancelled)
        normalized = reconstruct_stains(
            channels,
            reference_fit.profile,
            dtype=source.dtype,
            residual_profile=fit.profile,
            scales=normalization_scales(fit, reference_fit, recipe.normalization == "colors_intensity"),
        )
    _check_cancel(cancelled)
    return StainPreview(
        original,
        channels,
        normalized,
        reference_image,
        fit,
        reference_fit,
        scale,
        origin,
        reference_scale,
        reference_origin,
    )


def save_stain_recipe(path: str | Path, recipe: StainRecipe) -> None:
    """Save a versioned recipe including reference identity and per-slide overrides."""
    Path(path).write_text(recipe.model_dump_json(indent=2), encoding="utf-8")


def load_stain_recipe(path: str | Path) -> StainRecipe:
    """Read and validate a saved recipe without silently substituting settings."""
    return StainRecipe.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _regions(shape: tuple[int, ...]) -> Iterator[Region]:
    for y in range(0, shape[0], TILE_SIZE):
        for x in range(0, shape[1], TILE_SIZE):
            yield y, x, min(TILE_SIZE, shape[0] - y), min(TILE_SIZE, shape[1] - x)


def _pyramid(root: Any, cancelled: Cancel) -> list[Any]:
    current = root["0"]
    levels = [current]
    while max(current.shape[1:]) > TILE_SIZE:
        shape = (current.shape[0], (current.shape[1] + 1) // 2, (current.shape[2] + 1) // 2)
        following = root.create_array(
            str(len(levels)), shape=shape, chunks=(1, TILE_SIZE, TILE_SIZE), dtype=current.dtype
        )
        for y, x, height, width in _regions(shape[1:]):
            _check_cancel(cancelled)
            block = np.asarray(current[:, y * 2 : (y + height) * 2, x * 2 : (x + width) * 2], dtype=np.float32)
            block = np.pad(block, ((0, 0), (0, block.shape[1] % 2), (0, block.shape[2] % 2)), mode="edge")
            reduced = block.reshape(shape[0], height, 2, width, 2).mean(axis=(2, 4))
            if current.dtype.kind == "u":
                reduced = np.rint(reduced)
            following[:, y : y + height, x : x + width] = reduced.astype(current.dtype)
        levels.append(following)
        current = following
    return levels


def _metadata(
    root: Any, source: StainSource, names: tuple[str, ...], levels: list[Any], record: dict[str, Any]
) -> None:
    spatial = {"type": "space", **({"unit": "micrometer"} if source.spacing else {})}
    spacing = source.spacing or (1.0, 1.0)
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
    root.attrs["omero"] = {"channels": [{"label": name, "color": "FFFFFF"} for name in names]}
    root.attrs["image2image"] = record


def _write_tiff(
    path: Path,
    levels: list[Any],
    source: StainSource,
    names: tuple[str, ...],
    record: dict[str, Any],
    rgb: bool,
    cancelled: Cancel,
) -> None:
    from tifffile import TiffWriter

    def tiles(array: Any) -> Iterator[np.ndarray]:
        for channel in range(1 if rgb else array.shape[0]):
            for y, x, height, width in _regions(array.shape[1:]):
                _check_cancel(cancelled)
                shape = (TILE_SIZE, TILE_SIZE, 3) if rgb else (TILE_SIZE, TILE_SIZE)
                tile = np.zeros(shape, dtype=array.dtype)
                block = (
                    np.moveaxis(array[:, y : y + height, x : x + width], 0, -1)
                    if rgb
                    else array[channel, y : y + height, x : x + width]
                )
                tile[:height, :width] = block
                yield tile

    metadata: dict[str, Any] = {"axes": "YXS" if rgb else "CYX", "Description": json.dumps(record)}
    if not rgb:
        metadata["Channel"] = {"Name": list(names)}
    if source.spacing:
        metadata.update(
            PhysicalSizeY=source.spacing[0],
            PhysicalSizeX=source.spacing[1],
            PhysicalSizeYUnit="\u00b5m",
            PhysicalSizeXUnit="\u00b5m",
        )
    with TiffWriter(path, ome=True, bigtiff=True) as writer:
        for index, array in enumerate(levels):
            shape = (*array.shape[1:], 3) if rgb else array.shape
            writer.write(
                tiles(array),
                shape=shape,
                dtype=array.dtype,
                tile=(TILE_SIZE, TILE_SIZE),
                photometric="rgb" if rgb else "minisblack",
                compression="deflate",
                metadata=metadata if index == 0 else None,
                subifds=len(levels) - 1 if index == 0 else None,
                subfiletype=0 if index == 0 else 1,
            )


def _fit_record(fit: StainFit | None) -> dict[str, Any] | None:
    if fit is None:
        return None
    return {
        "profile": fit.profile.model_dump(mode="json"),
        "percentiles": fit.percentiles,
        "sample_count": fit.sample_count,
    }


def export_stains(
    source: StainSource,
    recipe: StainRecipe,
    directory: str | Path,
    *,
    output_format: str = "ome.tiff",
    cache: FitCache | None = None,
    cancelled: Cancel = None,
    progress: Progress = None,
) -> tuple[Path, ...]:
    """Write selected components and normalized RGB as staged pyramidal images plus provenance."""
    import zarr

    from image2image import __version__

    if output_format not in ("ome.tiff", "ome.zarr"):
        raise ValueError("Select OME-TIFF or OME-Zarr output.")
    directory = Path(directory).resolve()
    for item in (source, recipe.reference):
        if item is not None and (directory == item.path or item.path in directory.parents):
            raise ValueError("Output directory cannot be inside an input image.")
    stem = source.path.name
    for suffix in (".ome.tiff", ".ome.tif", ".ome.zarr", source.path.suffix):
        if suffix and stem.endswith(suffix):
            stem = stem.removesuffix(suffix)
            break
    kinds = (["stains"] if recipe.export_channels else []) + (["normalized"] if recipe.export_normalized else [])
    outputs = tuple(directory / f"{stem}.scene-{source.scene}.{kind}.{output_format}" for kind in kinds)
    sidecars = tuple(path.with_name(path.name + ".json") for path in outputs)
    if any(path.exists() for path in (*outputs, *sidecars)):
        raise FileExistsError("An output or parameter record already exists; choose a different output directory.")
    _check_cancel(cancelled)
    cache = cache if cache is not None else {}
    fit = fit_source(source, recipe, cache=cache, cancelled=cancelled)
    reference = (
        fit_source(recipe.reference, recipe, reference=True, cache=cache, cancelled=cancelled)
        if recipe.export_normalized and recipe.reference
        else None
    )
    scales = (
        normalization_scales(fit, reference, recipe.normalization == "colors_intensity")
        if reference
        else (1.0, 1.0, 1.0)
    )
    record = {
        "schema_version": 1,
        "app": "stains",
        "image2image_version": __version__,
        "source": source.model_dump(mode="json"),
        "recipe": recipe.model_dump(mode="json"),
        "source_fit": _fit_record(fit),
        "reference_fit": _fit_record(reference),
        "normalization_scales": scales,
        "optical_density": "-ln(min(max(I, 1) / white_native, 1)); signed unit-vector coefficients",
        "channel_scaling": "Unscaled signed float32"
        if recipe.channel_dtype == "float32"
        else "clip(C / max(slide_percentile, 1e-6), 0, 1) * 65535; rounded",
        "pyramid": "2x2 mean with edge padding; integers rounded",
        "negative_counts": [0, 0, 0],
    }
    directory.mkdir(parents=True, exist_ok=True)
    published: list[Path] = []
    try:
        with TemporaryDirectory(prefix=".stains-", dir=directory) as temporary:
            staging = Path(temporary)
            stores = []
            for kind in kinds:
                names = (
                    tuple(fit.profile.channel_names[i] for i in recipe.export_channels)
                    if kind == "stains"
                    else ("R", "G", "B")
                )
                dtype = recipe.channel_dtype if kind == "stains" else source.dtype
                path = staging / f"{kind}.ome.zarr"
                root = zarr.open_group(str(path), mode="w", zarr_format=2)
                root.create_array(
                    "0", shape=(len(names), *source.shape), chunks=(len(names), TILE_SIZE, TILE_SIZE), dtype=dtype
                )
                stores.append((kind, path, root, names))
            with rgb_levels(source) as levels:
                total = math.ceil(source.shape[0] / TILE_SIZE) * math.ceil(source.shape[1] / TILE_SIZE)
                for index, (y, x, height, width) in enumerate(_regions(source.shape)):
                    original = read_rgb_region(levels[0], (y, x, height, width), cancelled)
                    channels = separate_stains(original, fit.profile)
                    record["negative_counts"] = (
                        np.asarray(record["negative_counts"]) + np.count_nonzero(channels < 0, axis=(0, 1))
                    ).tolist()
                    for kind, _, root, _ in stores:
                        if kind == "stains":
                            data = (
                                registration_channels(channels, fit) if recipe.channel_dtype == "uint16" else channels
                            )
                            data = data[..., list(recipe.export_channels)]
                        else:
                            data = reconstruct_stains(
                                channels,
                                cast(StainFit, reference).profile,
                                dtype=source.dtype,
                                residual_profile=fit.profile,
                                scales=scales,
                            )
                        root["0"][:, y : y + height, x : x + width] = np.moveaxis(data, -1, 0)
                    if progress:
                        progress(f"{source.path.name}: tile {index + 1}/{total}")
            staged = []
            for kind, path, root, names in stores:
                if progress:
                    progress(f"{source.path.name}: building {kind} pyramid")
                pyramid = _pyramid(root, cancelled)
                product_record = {
                    **record,
                    "product": kind,
                    "channel_names": names,
                    "output_dtype": str(root["0"].dtype),
                }
                _metadata(root, source, names, pyramid, product_record)
                if output_format == "ome.tiff":
                    path = staging / f"{kind}.ome.tiff"
                    _write_tiff(path, pyramid, source, names, product_record, kind == "normalized", cancelled)
                record_path = staging / f"{kind}.json"
                record_path.write_text(json.dumps(product_record, indent=2, allow_nan=False), encoding="utf-8")
                staged.append((path, record_path))
            _check_cancel(cancelled)
            for (path, record_path), output, sidecar in zip(staged, outputs, sidecars, strict=True):
                os.link(record_path, sidecar)
                published.append(sidecar)
                if path.is_dir():
                    output.mkdir()  # Exclusive claim; never replace an existing image group.
                    published.append(output)
                    for child in path.iterdir():
                        child.rename(output / child.name)
                else:
                    os.link(path, output)
                    published.append(output)
    except BaseException:
        for path in reversed(published):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        raise
    return outputs


def export_stain_batch(
    sources: Sequence[StainSource],
    recipe: StainRecipe,
    directory: str | Path,
    *,
    output_format: str = "ome.tiff",
    cancelled: Cancel = None,
    progress: Progress = None,
    completed: Callable[[StainOutcome], None] | None = None,
    cache: FitCache | None = None,
) -> list[StainOutcome]:
    """Export a frozen batch sequentially, sharing reference fits and retaining individual failures."""
    cache = cache if cache is not None else {}
    results = []
    for source in sources:
        _check_cancel(cancelled)
        try:
            outputs = export_stains(
                source,
                recipe,
                directory,
                output_format=output_format,
                cache=cache,
                cancelled=cancelled,
                progress=progress,
            )
            outcome = StainOutcome(source=source, outputs=outputs)
        except InterruptedError:
            raise
        except (OSError, ValueError, RuntimeError, MemoryError) as exc:
            outcome = StainOutcome(source=source, error=str(exc))
        results.append(outcome)
        if completed:
            completed(outcome)
    return results
