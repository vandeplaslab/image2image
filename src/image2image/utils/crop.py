"""Crop functions."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from loguru import logger

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader

    from image2image.models.data import DataModel


def _get_new_image_shape(reader: BaseReader, left: int, right: int, top: int, bottom: int) -> tuple[int, ...]:
    """Get new image shape."""
    x_size = right - left
    y_size = bottom - top
    channel_axis, n_channels = reader.get_channel_axis_and_n_channels()
    if reader.is_rgb:
        return y_size, x_size, n_channels
    if channel_axis is None:
        return y_size, x_size
    if channel_axis == 0:
        return n_channels, y_size, x_size
    if channel_axis == 1:
        return y_size, n_channels, x_size
    return y_size, x_size, n_channels


def _crop_regions_iter(
    data_model: DataModel, polygon_or_bbox: list[tuple[int, int, int, int] | np.ndarray], skip_empty: bool = True
) -> ty.Generator[tuple[Path, ty.Any, int, str, np.ndarray, tuple[int, int, int, int]], None, None]:
    """Crop regions."""
    if isinstance(polygon_or_bbox, tuple):
        left, right, top, bottom = polygon_or_bbox
        for path, reader, channel_index, channel_name, cropped_channel, (
            left,
            right,
            top,
            bottom,
        ) in data_model.crop_bbox_iter(left, right, top, bottom):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (left, right, top, bottom),
            )
    else:
        assert isinstance(polygon_or_bbox, np.ndarray), f"Invalid type: {type(polygon_or_bbox)}"
        for path, reader, channel_index, channel_name, cropped_channel, (
            left,
            right,
            top,
            bottom,
        ) in data_model.crop_polygon_iter(polygon_or_bbox):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (left, right, top, bottom),
            )


def export_crop_regions(
    data_model: DataModel,
    output_dir: Path,
    regions: list[tuple[int, int, int, int] | np.ndarray],
    tile_size: int = 512,
    as_uint8: bool = True,
) -> ty.Generator[tuple[Path, int, int], None, None]:
    """Crop images."""
    from image2image_io.writers import OmeTiffWrapper

    n = len(regions)
    for current, polygon_or_bbox in enumerate(regions, start=1):
        for path, reader in data_model.wrapper.path_reader_iter():
            output_path, dtype, shape, rgb = None, None, None, []
            wrapper = OmeTiffWrapper()
            for channel, (left, right, top, bottom) in reader.crop_region_iter(polygon_or_bbox):
                output_path = output_dir / f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
                dtype = reader.dtype
                shape = _get_new_image_shape(reader, left, right, top, bottom)
                break

            if dtype is None or shape is None or output_path is None:
                logger.warning(f"Skipping {path} as it has no data.")
                continue
            if output_path.with_suffix(".ome.tiff").exists():
                logger.warning(f"Skipping {output_path} as it already exists.")
                yield output_path, current, n
            else:
                with wrapper.write(
                    channel_names=reader.channel_names,
                    resolution=reader.resolution,
                    dtype=dtype,
                    shape=shape,
                    name=output_path.name,
                    output_dir=output_dir,
                    tile_size=tile_size,
                    as_uint8=as_uint8,
                ):
                    channel_index = 0
                    for channel, _ in reader.crop_region_iter(polygon_or_bbox):
                        if channel is None:
                            continue
                        if reader.is_rgb:
                            rgb.append(channel)
                        else:
                            wrapper.add_channel(channel_index, reader.channel_names[channel_index], channel)
                            channel_index += 1
                    if rgb:
                        wrapper.add_channel([0, 1, 2], ["R", "G", "B"], np.dstack(rgb))
                yield wrapper.path, current, n


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


def _mask_regions_iter(
    data_model: DataModel, polygon_or_bbox: list[tuple[int, int, int, int] | np.ndarray], skip_empty: bool = True
) -> ty.Generator[tuple[Path, ty.Any, int, str, np.ndarray, tuple[int, int, int, int]], None, None]:
    """Crop regions."""
    if isinstance(polygon_or_bbox, tuple):
        left, right, top, bottom = polygon_or_bbox
        for path, reader, channel_index, channel_name, cropped_channel, (
            left,
            right,
            top,
            bottom,
        ) in data_model.mask_bbox_iter(left, right, top, bottom):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (left, right, top, bottom),
            )
    else:
        assert isinstance(polygon_or_bbox, np.ndarray), f"Invalid type: {type(polygon_or_bbox)}"
        for path, reader, channel_index, channel_name, cropped_channel, (
            left,
            right,
            top,
            bottom,
        ) in data_model.mask_polygon_iter(polygon_or_bbox):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (left, right, top, bottom),
            )


def export_mask_regions(
    data_model: DataModel,
    output_dir: Path,
    regions: list[tuple[int, int, int, int] | np.ndarray],
    tile_size: int = 512,
    as_uint8: bool = True,
) -> ty.Generator[tuple[Path, int, int], None, None]:
    """Export mask images."""
    from image2image_io.writers import OmeTiffWrapper

    n = len(regions)
    for current, polygon_or_bbox in enumerate(regions, start=1):
        for path, reader in data_model.wrapper.path_reader_iter():
            output_path, dtype, hash_str, shape, rgb = None, None, None, None, []
            wrapper = OmeTiffWrapper()
            for channel, hash_str in reader.mask_region_iter(polygon_or_bbox):
                output_path = output_dir / f"{path.stem}_{hash_str}".replace(".ome", "")
                dtype = reader.dtype
                shape = reader.shape
                break

            if dtype is None or hash_str is None or shape is None or output_path is None:
                logger.warning(f"Skipping {path} as it has no data.")
                continue
            if output_path.with_suffix(".ome.tiff").exists():
                logger.warning(f"Skipping {output_path} as it already exists.")
                yield output_path, current, n
            else:
                with wrapper.write(
                    channel_names=reader.channel_names,
                    resolution=reader.resolution,
                    dtype=dtype,
                    shape=shape,
                    name=output_path.name,
                    output_dir=output_dir,
                    tile_size=tile_size,
                    as_uint8=as_uint8,
                ):
                    channel_index = 0
                    for channel, _ in reader.mask_region_iter(polygon_or_bbox):
                        if channel is None:
                            continue
                        if reader.is_rgb:
                            rgb.append(channel)
                        else:
                            wrapper.add_channel(channel_index, reader.channel_names[channel_index], channel)
                            channel_index += 1
                    if rgb:
                        wrapper.add_channel([0, 1, 2], ["R", "G", "B"], np.dstack(rgb))
                yield wrapper.path, current, n


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
