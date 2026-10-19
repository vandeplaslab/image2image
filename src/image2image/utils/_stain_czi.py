"""Native CZI region access without whole-slide conversion to a temporary Zarr store."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np


class CziStainArray:
    """Read intersecting full-resolution CZI subblocks with a bounded decoded-block cache."""

    def __init__(self, czi: Any, shape: tuple[int, int], mapping: tuple[int, int, int]) -> None:
        self.czi = czi
        self.shape = (*shape, 3)
        self.mapping = mapping
        self.dtype = np.dtype(czi.dtype)
        self._cache: OrderedDict[int, tuple[np.ndarray, tuple[int, ...]]] = OrderedDict()
        self._cache_bytes = 0
        self._entries = []
        boxes = []
        y_axis, x_axis = czi.axes.index("Y"), czi.axes.index("X")
        origin_y, origin_x = czi.start[y_axis], czi.start[x_axis]
        for entry in czi.filtered_subblock_directory:
            # Directory metadata is sufficient; opening/decoding pixels is deferred until a region intersects.
            if entry.shape[y_axis] != entry.stored_shape[y_axis] or entry.shape[x_axis] != entry.stored_shape[x_axis]:
                continue
            if np.prod(entry.stored_shape, dtype=np.int64) * self.dtype.itemsize > 128 * 1024**2:
                raise ValueError(
                    "This CZI contains a subblock larger than 128 MiB. Convert it to tiled OME-TIFF first."
                )
            y, x = int(entry.start[y_axis] - origin_y), int(entry.start[x_axis] - origin_x)
            height, width = int(entry.shape[y_axis]), int(entry.shape[x_axis])
            boxes.append((y, x, y + height, x + width))
            self._entries.append(entry)
        if not boxes:
            raise ValueError("The selected CZI scene has no native-resolution image subblocks.")
        self._boxes = np.asarray(boxes, dtype=np.int64)

    def _tile(self, index: int) -> tuple[np.ndarray, tuple[int, ...]]:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        entry = self._entries[index]
        raw = entry.data_segment().data(resize=False)
        selected: list[int | slice] = []
        axes = ""
        for axis in self.czi.axes:
            if axis in "YX":
                selected.append(slice(None))
                axes += axis.lower()
            elif (axis == "0" and self.czi.is_rgb) or (axis == "C" and not self.czi.is_rgb):
                selected.append(slice(None))
                axes += "c"
            else:
                selected.append(0)
        tile = np.asarray(raw[tuple(selected)]).transpose(*(axes.index(axis) for axis in "yxc"))
        if self.czi.is_rgb:
            channels = (0, 1, 2)
        else:
            axis = self.czi.axes.index("C")
            start = int(entry.start[axis] - self.czi.start[axis])
            channels = tuple(range(start, start + tile.shape[-1]))
        # A copy avoids retaining a large raw array via a small view.
        tile = np.ascontiguousarray(tile)
        while self._cache and self._cache_bytes + tile.nbytes > 32 * 1024**2:
            old, _ = self._cache.popitem(last=False)[1]
            self._cache_bytes -= old.nbytes
        if tile.nbytes <= 32 * 1024**2:
            self._cache[index] = tile, channels
            self._cache_bytes += tile.nbytes
        return tile, channels

    def __getitem__(self, slices: tuple[slice, slice, slice]) -> np.ndarray:
        y_slice, x_slice, _ = slices
        y, x, bottom, right = y_slice.start, x_slice.start, y_slice.stop, x_slice.stop
        # Match the existing reader's zero fill for gaps between mosaic subblocks.
        result = np.zeros((bottom - y, right - x, 3), dtype=self.dtype)
        boxes = self._boxes
        indices = np.flatnonzero((boxes[:, 0] < bottom) & (boxes[:, 1] < right) & (boxes[:, 2] > y) & (boxes[:, 3] > x))
        for index in indices:
            top, left, stop_y, stop_x = boxes[index]
            y0, x0, y1, x1 = max(y, top), max(x, left), min(bottom, stop_y), min(right, stop_x)
            tile, channels = self._tile(int(index))
            for output_channel, input_channel in enumerate(self.mapping):
                if input_channel in channels:
                    result[y0 - y : y1 - y, x0 - x : x1 - x, output_channel] = tile[
                        y0 - top : y1 - top, x0 - left : x1 - left, channels.index(input_channel)
                    ]
        return result
