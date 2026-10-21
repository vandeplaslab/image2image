"""Bounded, display-aware region growing for interactive annotation."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numba import njit
from rasterio.features import shapes
from rasterio.transform import Affine
from scipy.ndimage import binary_fill_holes, gaussian_filter, map_coordinates
from shapely.geometry import Polygon, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from image2image.models.annotation import ImagePatch
from image2image.utils.annotation.images import PatchReader


@dataclass(frozen=True)
class WandLayer:
    """Immutable source-channel and display settings captured on the GUI thread."""

    channels: tuple[int, ...]
    contrast: tuple[float, float]
    gamma: float
    colors: np.ndarray | None = None
    opacity: float = 1.0
    blending: str = "additive"


def _composite(result: np.ndarray, data: np.ndarray, layer: WandLayer, *, first: bool) -> np.ndarray:
    values = data.astype(np.float32)
    low, high = layer.contrast
    valid = np.isfinite(values).all(axis=-1)
    values = np.nan_to_num(values, nan=low, posinf=high, neginf=low)
    values = np.clip((values - low) / max(high - low, np.finfo(np.float32).eps), 0, 1) ** layer.gamma
    if layer.colors is None:
        color = values[..., :3]
        alpha = data[..., 3:4].astype(np.float32) if data.shape[-1] == 4 else np.ones_like(values[..., :1])
        if data.shape[-1] == 4 and np.issubdtype(data.dtype, np.integer):
            alpha /= np.iinfo(data.dtype).max
    else:
        positions = values[..., 0] * (len(layer.colors) - 1)
        left = positions.astype(np.intp)
        right = np.minimum(left + 1, len(layer.colors) - 1)
        fraction = (positions - left)[..., None]
        rgba = layer.colors[left] * (1 - fraction) + layer.colors[right] * fraction
        color, alpha = rgba[..., :3], rgba[..., 3:4]
    alpha = np.clip(alpha * layer.opacity, 0, 1)
    if first and layer.blending == "minimum":
        result = color.copy()
    elif first and layer.blending == "additive":
        result = color * alpha
    elif first:
        result = color * alpha + result * (1 - alpha)
    elif layer.blending == "opaque":
        result = color.copy()
    elif layer.blending == "minimum":
        result = np.minimum(result, color)
    elif layer.blending == "additive":
        result = result + color * alpha
    elif layer.blending == "multiplicative":
        result = result * color
    else:
        result = color * alpha + result * (1 - alpha)
    result[~valid] = np.nan
    return np.clip(result, 0, 1).astype(np.float32)


def display_patch(
    data: np.ndarray,
    layers: tuple[WandLayer, ...],
    channels: tuple[int, ...],
    *,
    background: tuple[float, float, float] = (0, 0, 0),
) -> np.ndarray:
    """Composite HWC source pixels into normalized RGB using image-layer settings."""
    result = np.empty((*data.shape[:2], 3), dtype=np.float32)
    result[:] = background
    for index, layer in enumerate(layers):
        values = data[..., [channels.index(channel) for channel in layer.channels]]
        result = _composite(result, values, layer, first=index == 0)
    return result


@njit(cache=True, nogil=True)
def _connected_mask(
    smooth: np.ndarray,
    valid: np.ndarray,
    row: int,
    column: int,
    x: float,
    y: float,
    sx: float,
    sy: float,
    radius: float,
    sensitivity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse eligibility checks and visit only the seed's four-connected component."""
    height, width = valid.shape
    disk = np.empty((height, width), dtype=np.bool_)
    allowed = np.empty((height, width), dtype=np.bool_)
    for r in range(height):
        dy2 = ((r + 0.5 - y) * sy) ** 2
        for c in range(width):
            inside = ((c + 0.5 - x) * sx) ** 2 + dy2 <= radius**2 or (r == row and c == column)
            disk[r, c] = inside
            accept = inside and valid[r, c]
            if accept:
                for channel in range(smooth.shape[2]):
                    if abs(smooth[r, c, channel] - smooth[row, column, channel]) > sensitivity:
                        accept = False
                        break
            allowed[r, c] = accept
    mask = np.zeros((height, width), dtype=np.bool_)
    queue = np.empty(height * width, dtype=np.intp)
    queue[0] = row * width + column
    allowed[row, column] = False
    mask[row, column] = True
    head, tail = 0, 1
    while head < tail:
        index = queue[head]
        head += 1
        r, c = index // width, index % width
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width and allowed[nr, nc]:
                allowed[nr, nc] = False
                mask[nr, nc] = True
                queue[tail] = nr * width + nc
                tail += 1
    return mask, disk


def wand_region(patch: ImagePatch, seed: tuple[float, float], radius: float, sensitivity: float) -> BaseGeometry:
    """Grow a filled, four-connected region near a native XY seed in displayed RGB.

    Sensitivity is a normalized maximum per-channel difference from the smoothed
    seed color. The patch transform maps output pixel edges back to the slide.
    """
    data = patch.data
    height, width = data.shape[:2]
    sx, sy = patch.downsample
    x, y = (seed[0] - patch.offset[0]) / sx, (seed[1] - patch.offset[1]) / sy
    column, row = math.floor(x), math.floor(y)
    if not (0 <= row < height and 0 <= column < width) or radius <= 0:
        return Polygon()
    valid = np.isfinite(data).all(axis=-1)
    if not valid[row, column]:
        return Polygon()
    if valid.all():
        smooth = gaussian_filter(data, (1, 1, 0), mode="nearest")
    else:
        # Normalize around missing pixels so they cannot contaminate nearby seeds.
        weights = gaussian_filter(valid.astype(np.float32), 1, mode="nearest")
        smooth = gaussian_filter(np.where(valid[..., None], data, 0), (1, 1, 0), mode="nearest")
        smooth /= np.maximum(weights[..., None], np.finfo(np.float32).eps)
    mask, disk = _connected_mask(smooth, valid, row, column, x, y, sx, sy, radius, smooth.dtype.type(sensitivity))
    mask = binary_fill_holes(mask) & disk & valid
    transform = Affine.translation(*patch.offset) * Affine.scale(sx, sy)
    polygons = [shape(geometry) for geometry, _ in shapes(mask.astype(np.uint8), mask=mask, transform=transform)]
    return unary_union(polygons).simplify(min(sx, sy) * 0.25, preserve_topology=True)


class WandSource:
    """Read aligned display tiles with a 32 MiB cache and bounded growing patches.

    The caller owns the reader and uses this object exclusively in its worker.
    """

    TILE = 256
    MAX_BYTES = 32 * 1024 * 1024
    MAX_SIDE = 512

    def __init__(
        self, reader: PatchReader, layers: tuple[WandLayer, ...], *, background: tuple[float, float, float] = (0, 0, 0)
    ) -> None:
        self.reader = reader
        self.layers = layers
        self.background = background
        self._tiles: OrderedDict[tuple[int, int, int], np.ndarray] = OrderedDict()
        self.cache_bytes = 0

    def set_display(self, layers: tuple[WandLayer, ...], background: tuple[float, float, float]) -> None:
        """Reuse image reads across strokes, discarding composites when the display changes."""
        if self.background == background and len(self.layers) == len(layers):
            for old, new in zip(self.layers, layers, strict=True):
                if (old.channels, old.contrast, old.gamma, old.opacity, old.blending) != (
                    new.channels,
                    new.contrast,
                    new.gamma,
                    new.opacity,
                    new.blending,
                ) or not np.array_equal(old.colors, new.colors):
                    break
            else:
                return
        self.layers, self.background = layers, background
        self._tiles.clear()
        self.cache_bytes = 0

    def _tile(self, x: int, y: int, downsample: int) -> np.ndarray:
        key = (x, y, downsample)
        if key in self._tiles:
            self._tiles.move_to_end(key)
            return self._tiles[key]
        step = self.TILE * downsample
        # Read channels separately so many-channel inputs keep bounded allocations.
        composite = np.empty((self.TILE, self.TILE, 3), dtype=np.float32)
        composite[:] = self.background
        for index, layer in enumerate(self.layers):
            patch = self.reader.read((x, y, x + step, y + step), downsample=downsample, channels=layer.channels)
            h, w = patch.data.shape[:2]
            # The final partial tile has its own exact pixel-center spacing.
            yy = ((np.arange(h) + 0.5) * downsample / patch.downsample[1]) - 0.5
            xx = ((np.arange(w) + 0.5) * downsample / patch.downsample[0]) - 0.5
            gy, gx = np.meshgrid(yy, xx, indexing="ij")
            values = np.stack(
                [
                    map_coordinates(patch.data[..., c].astype(np.float32), (gy, gx), order=1, mode="nearest")
                    for c in range(patch.data.shape[-1])
                ],
                axis=-1,
            )
            if len(layer.channels) == 4 and np.issubdtype(patch.data.dtype, np.integer):
                values[..., 3] /= np.iinfo(patch.data.dtype).max
            composite[:h, :w] = _composite(composite[:h, :w], values, layer, first=index == 0)
        composite = np.clip(composite, 0, 1)
        while self._tiles and self.cache_bytes + composite.nbytes > self.MAX_BYTES:
            self.cache_bytes -= self._tiles.popitem(last=False)[1].nbytes
        self._tiles[key] = composite
        self.cache_bytes += composite.nbytes
        return composite

    def region(self, seed: tuple[float, float], radius: float, zoom: float, sensitivity: float) -> BaseGeometry:
        """Return a local polygon using screen-pixel radius and current camera zoom."""
        height, width = self.reader.shape
        if not self.layers or not (0 <= seed[0] < width and 0 <= seed[1] < height):
            return Polygon()
        native_radius = radius / zoom
        ds = max(1, math.ceil(1 / zoom), math.ceil(2 * native_radius / (self.MAX_SIDE - 6)))
        x0, y0 = (
            max(0, math.floor((seed[0] - native_radius) / ds) - 2),
            max(0, math.floor((seed[1] - native_radius) / ds) - 2),
        )
        x1, y1 = (
            min(math.ceil(width / ds), math.ceil((seed[0] + native_radius) / ds) + 2),
            min(math.ceil(height / ds), math.ceil((seed[1] + native_radius) / ds) + 2),
        )
        data = np.empty((y1 - y0, x1 - x0, 3), dtype=np.float32)
        for ty in range(y0 // self.TILE, math.ceil(y1 / self.TILE)):
            for tx in range(x0 // self.TILE, math.ceil(x1 / self.TILE)):
                tile = self._tile(tx * self.TILE * ds, ty * self.TILE * ds, ds)
                left, top, right, bottom = (
                    max(x0, tx * self.TILE),
                    max(y0, ty * self.TILE),
                    min(x1, (tx + 1) * self.TILE),
                    min(y1, (ty + 1) * self.TILE),
                )
                data[top - y0 : bottom - y0, left - x0 : right - x0] = tile[
                    top - ty * self.TILE : bottom - ty * self.TILE, left - tx * self.TILE : right - tx * self.TILE
                ]
        patch = ImagePatch(data, (x0 * ds, y0 * ds), (ds, ds), (0, 1, 2))
        return wand_region(patch, seed, native_radius, sensitivity).intersection(box(0, 0, width, height))
