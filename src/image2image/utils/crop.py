"""Crop functions."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from image2image_io.crop import export_crop_regions as export_crop_regions_for_path
from image2image_io.mask import export_mask_regions as export_mask_regions_for_path

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


def export_crop_regions(
    data_model: DataModel,
    output_dir: Path | None,
    regions: list[tuple[int, int, int, int] | np.ndarray],
    tile_size: int = 1024,
    as_uint8: bool = True,
) -> ty.Generator[tuple[Path, int, int], None, None]:
    """Crop images."""
    n = data_model.n_paths
    for current, (path, _reader) in enumerate(data_model.wrapper.path_reader_iter(), start=1):
        for _ in export_crop_regions_for_path(path, output_dir, regions, tile_size, as_uint8):
            pass
        yield path, current, n


def preview_crop_regions(
    data_model: DataModel, regions: list[tuple[int, int, int, int] | np.ndarray]
) -> ty.Generator[tuple[str, str, np.ndarray, float, int, int], None, None]:
    """Preview images."""
    n = len(regions)
    for current, polygon_or_bbox in enumerate(regions, start=1):
        for path, reader, _channel_index, channel_name, cropped, (left, right, top, bottom) in _crop_regions_iter(
            data_model, polygon_or_bbox
        ):
            name = f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
            yield name, channel_name, cropped, reader.resolution, current, n


def _crop_regions_iter(
    data_model: DataModel, polygon_or_bbox: list[tuple[int, int, int, int] | np.ndarray], skip_empty: bool = True
) -> ty.Generator[tuple[Path, ty.Any, int, str, np.ndarray, tuple[int, int, int, int]], None, None]:
    """Crop regions."""
    if isinstance(polygon_or_bbox, tuple):
        left, right, top, bottom = polygon_or_bbox
        for path, reader, channel_index, channel_name, cropped_channel, (
            crop_left,
            crop_right,
            crop_top,
            crop_bottom,
        ) in data_model.crop_bbox_iter(left, right, top, bottom):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (crop_left, crop_right, crop_top, crop_bottom),
            )

    else:
        if not isinstance(polygon_or_bbox, np.ndarray):
            raise TypeError(f"Invalid type: {type(polygon_or_bbox)}")
        for path, reader, channel_index, channel_name, cropped_channel, (
            crop_left,
            crop_right,
            crop_top,
            crop_bottom,
        ) in data_model.crop_polygon_iter(polygon_or_bbox):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (crop_left, crop_right, crop_top, crop_bottom),
            )


def export_mask_regions(
    data_model: DataModel,
    output_dir: Path | None,
    regions: list[tuple[int, int, int, int] | np.ndarray],
    tile_size: int = 1024,
    as_uint8: bool = True,
) -> ty.Generator[tuple[Path, int, int], None, None]:
    """Export mask images."""
    n = data_model.n_paths
    for current, (path, _reader) in enumerate(data_model.wrapper.path_reader_iter(), start=1):
        for _ in export_mask_regions_for_path(path, output_dir, regions, tile_size, as_uint8):
            pass
        yield path, current, n


def preview_mask_regions(
    data_model: DataModel, regions: list[tuple[int, int, int, int] | np.ndarray]
) -> ty.Generator[tuple[str, str, np.ndarray, float, int, int], None, None]:
    """Preview images."""
    n = len(regions)
    for current, polygon_or_bbox in enumerate(regions, start=1):
        for path, reader, _channel_index, channel_name, cropped, (left, right, top, bottom) in _mask_regions_iter(
            data_model, polygon_or_bbox
        ):
            name = f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
            yield name, channel_name, cropped, reader.resolution, current, n


def _mask_regions_iter(
    data_model: DataModel, polygon_or_bbox: list[tuple[int, int, int, int] | np.ndarray], skip_empty: bool = True
) -> ty.Generator[tuple[Path, ty.Any, int, str, np.ndarray, tuple[int, int, int, int]], None, None]:
    """Crop regions."""
    if isinstance(polygon_or_bbox, tuple):
        left, right, top, bottom = polygon_or_bbox
        for path, reader, channel_index, channel_name, cropped_channel, (
            crop_left,
            crop_right,
            crop_top,
            crop_bottom,
        ) in data_model.mask_bbox_iter(left, right, top, bottom):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (crop_left, crop_right, crop_top, crop_bottom),
            )
    else:
        if not isinstance(polygon_or_bbox, np.ndarray):
            raise TypeError(f"Invalid type: {type(polygon_or_bbox)}")
        for path, reader, channel_index, channel_name, cropped_channel, (
            crop_left,
            crop_right,
            crop_top,
            crop_bottom,
        ) in data_model.mask_polygon_iter(polygon_or_bbox):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (crop_left, crop_right, crop_top, crop_bottom),
            )
