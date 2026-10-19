"""Lazy source images and calibrated patches, independent of display coordinates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Self

import numpy as np

from image2image.models.annotation import AnnotationSlide, ImagePatch, SlideGeometry


def fingerprint(path: Path) -> str:
    """Fingerprint source metadata without scanning all WSI pixels or Zarr chunks."""
    paths = (
        [path]
        if path.is_file()
        else [path / name for name in ("zarr.json", ".zattrs", ".zgroup") if (path / name).exists()]
    )
    return hashlib.sha256(
        json.dumps([(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]).encode()
    ).hexdigest()


def _microns(value: float | None, unit: Any = "um") -> float | None:
    if value is None:
        return None
    import pint

    unit = getattr(unit, "value", unit) or "um"
    try:
        result = float(pint.UnitRegistry().Quantity(float(value), str(unit)).to("um").magnitude)
    except (ValueError, pint.errors.PintError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _tiff_spacing(reader: Any, series: int) -> tuple[float, float] | None:
    tif = reader.fh
    if tif.ome_metadata:
        from ome_types import from_xml

        pixels = from_xml(tif.ome_metadata).images[series].pixels
        x = _microns(pixels.physical_size_x, pixels.physical_size_x_unit)
        y = _microns(pixels.physical_size_y, pixels.physical_size_y_unit)
        return (x, y) if x is not None and y is not None else None
    page = tif.series[series].pages[0]
    if Path(reader.path).suffix.lower() == ".svs":
        for part in (page.description or "").split("|"):
            if part.strip().startswith("MPP =") or part.strip().startswith("MPP="):
                spacing = float(part.split("=", 1)[1])
                return (spacing, spacing) if spacing > 0 else None
    tags = page.tags
    unit = tags.get("ResolutionUnit")
    factor = {2: 25400.0, 3: 10000.0}.get(int(unit.value)) if unit else None
    if factor and "XResolution" in tags and "YResolution" in tags:
        values = []
        for name in ("XResolution", "YResolution"):
            numerator, denominator = tags[name].value
            if numerator <= 0 or denominator <= 0:
                return None
            values.append(factor * denominator / numerator)
        return tuple(values)
    return None


class PatchReader:
    """Read only requested pyramid regions, returning HWC patches with exact transforms."""

    def __init__(self, path: str | Path, *, series: int = 0) -> None:
        self.path = Path(path).resolve()
        if not self.path.exists():
            raise FileNotFoundError(f"Image missing; relocate this slide: {self.path}")
        self.reader: Any = None
        self.origin = (0.0, 0.0)
        self.spacing: tuple[float, float] | None = None
        if self.path.is_dir():
            import dask.array as da
            import zarr

            root = zarr.open(str(self.path), mode="r")
            if isinstance(root, zarr.Array):
                self.levels = [da.from_zarr(root)]  # type: ignore[no-untyped-call]
                self.axes = "yx" if root.ndim == 2 else "yxc" if root.ndim == 3 and root.shape[-1] in (3, 4) else "cyx"
            else:
                attrs: dict[str, Any] = root.attrs.asdict()
                attrs = attrs.get("ome", attrs)
                multi = attrs.get("multiscales", [{}])[0]
                datasets = multi.get("datasets", [{"path": "0"}])
                self.levels = [da.from_zarr(root[item["path"]]) for item in datasets]  # type: ignore[no-untyped-call]
                axes = multi.get("axes", [])
                self.axes = "".join(axis["name"] if isinstance(axis, dict) else axis for axis in axes) or "yx"
                self._normalize_zarr_axes()
                transforms = [
                    *multi.get("coordinateTransformations", []),
                    *datasets[0].get("coordinateTransformations", []),
                ]
                scale = np.ones(len(axes))
                origin = np.zeros(len(axes))
                for transform in transforms:
                    if transform["type"] == "scale":
                        factor = np.asarray(transform["scale"])
                        origin *= factor
                        scale *= factor
                    elif transform["type"] == "translation":
                        origin += transform["translation"]
                names = [axis["name"] if isinstance(axis, dict) else axis for axis in axes]
                if "x" in names and "y" in names:
                    xy, origins = [], []
                    for name in ("x", "y"):
                        index = names.index(name)
                        unit = axes[index].get("unit") if isinstance(axes[index], dict) else None
                        value = _microns(scale[index], unit) if unit else None
                        xy.append(value)
                        origins.append(origin[index] * value / scale[index] if value else 0)
                    if all(value is not None for value in xy):
                        self.spacing, self.origin = tuple(xy), tuple(origins)
            self.channel_names = [f"C{i + 1}" for i in range(self.n_channels)]
        else:
            from image2image_io.readers import get_simple_reader
            from image2image_io.readers.tiff_reader import TiffImageReader

            suffix = self.path.suffix.lower()
            if suffix in (".tif", ".tiff", ".svs", ".ndpi", ".scn", ".qptiff"):
                self.reader = TiffImageReader(self.path, init_pyramid=False, auto_pyramid=False, series_index=series)
                self.spacing = _tiff_spacing(self.reader, series)
            else:
                self.reader = get_simple_reader(
                    self.path, init_pyramid=False, auto_pyramid=False, quick=True, scene_index=series
                )
                if suffix == ".czi":
                    # CZI XML distances are meters. Never infer physical units from a fallback resolution=1.
                    import xml.etree.ElementTree as ET

                    xml_root = ET.fromstring(self.reader.fh.metadata())  # noqa: S314
                    sizes = [xml_root.find(f'.//Scaling/Items/Distance[@Id="{axis}"]/Value') for axis in ("X", "Y")]
                    values = tuple(float(item.text) * 1_000_000 for item in sizes if item is not None and item.text)
                    if len(values) == 2 and min(values) > 0:
                        self.spacing = values
            self.levels = self.reader.pyramid
            array = self.levels[0]
            if array.ndim == 2:
                self.axes = "yx"
            elif array.ndim == 3:
                axis, _ = self.reader.get_channel_axis_and_n_channels()
                self.axes = {0: "cyx", 1: "ycx", 2: "yxc"}.get(axis, "yxc")
            else:
                self.close()
                raise ValueError("Annotate supports 2D images with channels, not volumes or time series.")
            self.channel_names = self.reader.get_channel_names(split_rgb=True)
        if len(self.axes) != self.levels[0].ndim or set(self.axes) - set("cyx"):
            self.close()
            raise ValueError("Select a two-dimensional image series.")
        # Napari must have a small lazy level even when the input has no pyramid.
        # Strided source indexing bounds the displayed array without allocating a WSI.
        self.display_levels = list(self.levels)
        while max(self.display_levels[-1].shape[self.axes.index(axis)] for axis in "yx") > 1024:
            selection = [slice(None)] * self.display_levels[-1].ndim
            for axis in "yx":
                selection[self.axes.index(axis)] = slice(None, None, 2)
            self.display_levels.append(self.display_levels[-1][tuple(selection)])

    def _normalize_zarr_axes(self) -> None:
        axes = list(self.axes)
        for index in reversed(range(len(axes))):
            if axes[index] not in "cyx":
                if any(level.shape[index] != 1 for level in self.levels):
                    raise ValueError("Annotate supports 2D image planes only.")
                self.levels = [level.squeeze(axis=index) for level in self.levels]
                axes.pop(index)
        self.axes = "".join(axes)

    @property
    def shape(self) -> tuple[int, int]:
        """Return full-resolution height and width."""
        array = self.levels[0]
        return array.shape[self.axes.index("y")], array.shape[self.axes.index("x")]

    @property
    def n_channels(self) -> int:
        """Return the available number of channels."""
        return self.levels[0].shape[self.axes.index("c")] if "c" in self.axes else 1

    def close(self) -> None:
        """Release image handles after all lazy computations finish."""
        if self.reader is not None:
            self.reader.close()

    def __enter__(self) -> Self:
        """Use this reader within a bounded file-handle lifetime."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Close image handles on context exit."""
        self.close()

    def read(
        self,
        bounds: tuple[float, float, float, float],
        *,
        downsample: float = 1,
        channels: tuple[int, ...] = (0,),
        max_pixels: int = 16_777_216,
    ) -> ImagePatch:
        """Sample a bounded native-pixel rectangle, using the nearest suitable pyramid level."""
        if not channels or min(channels) < 0 or max(channels) >= self.n_channels:
            raise ValueError(f"Choose channel indices between 0 and {self.n_channels - 1}.")
        x0, y0, x1, y1 = bounds
        height, width = self.shape
        x0, y0, x1, y1 = (
            max(0, math.floor(x0)),
            max(0, math.floor(y0)),
            min(width, math.ceil(x1)),
            min(height, math.ceil(y1)),
        )
        if x1 <= x0 or y1 <= y0 or not math.isfinite(downsample) or downsample < 1:
            raise ValueError("The patch must intersect the slide and have downsample >= 1.")
        out_shape = (max(1, math.ceil((y1 - y0) / downsample)), max(1, math.ceil((x1 - x0) / downsample)))
        if out_shape[0] * out_shape[1] * len(channels) > max_pixels:
            raise ValueError("Patch exceeds the memory limit; use a smaller ROI or a larger downsample.")
        level_index = 0
        for index, level in enumerate(self.levels):
            factor = max(height / level.shape[self.axes.index("y")], width / level.shape[self.axes.index("x")])
            if factor <= downsample:
                level_index = index
        level = self.levels[level_index]
        sy, sx = height / level.shape[self.axes.index("y")], width / level.shape[self.axes.index("x")]
        slices = [slice(None)] * level.ndim
        # Subsample before compute when a source lacks a suitable pyramid. This keeps
        # overview reads bounded even for unpyramided whole-slide images.
        stride_x, stride_y = max(1, int(downsample / sx)), max(1, int(downsample / sy))
        slices[self.axes.index("x")] = slice(math.floor(x0 / sx), math.ceil(x1 / sx), stride_x)
        slices[self.axes.index("y")] = slice(math.floor(y0 / sy), math.ceil(y1 / sy), stride_y)
        data = level[tuple(slices)]
        if "c" in self.axes:
            selection: list[Any] = [slice(None)] * data.ndim
            selection[self.axes.index("c")] = list(channels)
            data = data[tuple(selection)]
            data = data.transpose(tuple(self.axes.index(name) for name in "yxc"))
        else:
            data = data[..., None]
        if hasattr(data, "compute"):
            data = data.compute()
        data = np.asarray(data)
        # Align samples to native pixel centers, accounting for non-power-of-two pyramids.
        from scipy.ndimage import map_coordinates

        output_sy, output_sx = (y1 - y0) / out_shape[0], (x1 - x0) / out_shape[1]
        yy = ((y0 + (np.arange(out_shape[0]) + 0.5) * output_sy) / sy - 0.5 - math.floor(y0 / sy)) / stride_y
        xx = ((x0 + (np.arange(out_shape[1]) + 0.5) * output_sx) / sx - 0.5 - math.floor(x0 / sx)) / stride_x
        gy, gx = np.meshgrid(yy, xx, indexing="ij")
        if data.shape[:2] == out_shape and sx == sy == 1 and downsample == 1:
            result = data
        else:
            result = np.stack(
                [
                    map_coordinates(data[..., c].astype(np.float32), (gy, gx), order=1, mode="nearest")
                    for c in range(len(channels))
                ],
                axis=-1,
            )
        if np.issubdtype(data.dtype, np.integer):
            result = result.astype(data.dtype)
        return ImagePatch(result, (x0, y0), (output_sx, output_sy), channels)


def inspect_slide(path: str | Path, *, series: int = 0) -> AnnotationSlide:
    """Inspect image metadata without reading a full-resolution slide into memory."""
    path = Path(path).resolve()
    with PatchReader(path, series=series) as reader:
        return AnnotationSlide(
            name=path.name,
            path=str(path),
            series=series,
            fingerprint=fingerprint(path),
            geometry=SlideGeometry(shape=reader.shape, spacing=reader.spacing, origin=reader.origin),
            channel_names=reader.channel_names,
        )
