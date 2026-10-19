"""Bounded viewport rendering and cached low-magnification class overviews."""

from __future__ import annotations

import math
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image2image.utils.annotation.store import AnnotationStore


@dataclass
class OverlayFrame:
    """RGBA display pixels in a native-pixel viewport; never authoritative geometry."""

    rgba: np.ndarray
    offset: tuple[float, float]
    downsample: float
    revision: int


def _color(value: str, alpha: int = 100) -> tuple[int, int, int, int]:
    try:
        return (*bytes.fromhex(value.lstrip("#")[:6]), alpha)
    except ValueError:
        return 230, 115, 115, alpha


class AnnotationRenderer:
    """Render fixed tiles with a bounded LRU; dense overview queries avoid WKB decoding."""

    TILE = 256

    def __init__(self, project: str | Path, *, max_bytes: int = 128 * 1024 * 1024) -> None:
        self.project = Path(project)
        self.max_bytes = max_bytes
        self._tiles: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._revision: int | None = None
        self._invalidations: deque[tuple] = deque()

    @property
    def cache_bytes(self) -> int:
        """Return actual resident annotation tile bytes."""
        return self._bytes

    def invalidate(
        self, slide_id: str | None = None, bounds: tuple | None = None, *, revision: int | None = None
    ) -> None:
        """Invalidate only intersecting tiles when an edit supplies its changed bounds."""
        self._invalidations.append((slide_id, bounds, revision))

    def _invalidate_now(self, slide_id: str | None, bounds: tuple | None, revision: int | None) -> None:
        for key in list(self._tiles):
            slide, _set, _review, ds, x, y = key
            intersects = bounds is None or not (
                x + self.TILE * ds < bounds[0] or x > bounds[2] or y + self.TILE * ds < bounds[1] or y > bounds[3]
            )
            if (slide_id is None or slide == slide_id) and intersects:
                self._bytes -= self._tiles.pop(key).nbytes
        if revision is not None:
            self._revision = revision

    def render(
        self,
        slide_id: str,
        bounds: tuple[float, float, float, float],
        *,
        pixels: int = 1024,
        set_id: str | None = None,
        review: str = "accepted",
    ) -> OverlayFrame:
        """Render an aligned viewport using at most approximately pixels squared output pixels."""
        with AnnotationStore(self.project) as store:
            while self._invalidations:
                self._invalidate_now(*self._invalidations.popleft())
            revision = store.revision
            if self._revision is not None and self._revision != revision:
                self._invalidate_now(None, None, revision)
            self._revision = revision
            height, width = store.slide(slide_id).geometry.shape
            x0, y0, x1, y1 = max(0, bounds[0]), max(0, bounds[1]), min(width, bounds[2]), min(height, bounds[3])
            if x1 <= x0 or y1 <= y0:
                return OverlayFrame(np.zeros((1, 1, 4), dtype=np.uint8), (0, 0), 1, revision)
            ds = 2 ** max(0, math.ceil(math.log2(max(x1 - x0, y1 - y0) / max(1, pixels))))
            step = ds * self.TILE
            left, top = int(x0 // step) * step, int(y0 // step) * step
            right, bottom = math.ceil(x1 / step) * step, math.ceil(y1 / step) * step
            frame = np.zeros(((bottom - top) // ds, (right - left) // ds, 4), dtype=np.uint8)
            colors = {item.id: _color(item.color) for item in store.classes() if item.visible}
            for y in range(top, bottom, step):
                for x in range(left, right, step):
                    key = (slide_id, set_id, review, ds, x, y)
                    if key in self._tiles:
                        tile = self._tiles.pop(key)
                        self._tiles[key] = tile
                    else:
                        tile = self._render_tile(store, slide_id, set_id, review, x, y, ds, colors)
                        while self._tiles and self._bytes + tile.nbytes > self.max_bytes:
                            self._bytes -= self._tiles.popitem(last=False)[1].nbytes
                        if tile.nbytes <= self.max_bytes:
                            self._tiles[key] = tile
                            self._bytes += tile.nbytes
                    yy, xx = (y - top) // ds, (x - left) // ds
                    frame[yy : yy + self.TILE, xx : xx + self.TILE] = tile
        return OverlayFrame(frame, (left, top), ds, revision)

    def _render_tile(
        self,
        store: AnnotationStore,
        slide: str,
        set_id: str | None,
        review: str,
        x: int,
        y: int,
        ds: int,
        colors: dict,
    ) -> np.ndarray:
        from rasterio.features import rasterize
        from rasterio.transform import Affine
        from scipy.ndimage import binary_erosion

        size = self.TILE
        bounds = (x, y, x + size * ds, y + size * ds)
        tile = np.zeros((size, size, 4), dtype=np.uint8)
        args = [slide, review, bounds[2], x, bounds[3], y]
        query = (
            " FROM bounds b CROSS JOIN objects o ON b.id=o.id JOIN annotation_sets s ON s.id=o.set_id "
            "WHERE s.slide_id=? AND s.state='ready' AND o.review=? AND b.minx<=? AND b.maxx>=? AND b.miny<=? AND b.maxy>=? "
            "AND (o.class_id IS NULL OR o.class_id NOT IN (SELECT id FROM classes WHERE json_extract(data,'$.visible')=0))"
        )
        if set_id:
            query += " AND o.set_id=?"
            args.append(set_id)
        # Limit the count operation; only density mode needs a full streaming SQL pass.
        dense = (
            store.db.execute("SELECT count(*) FROM (SELECT o.id" + query + " LIMIT 5001)", args).fetchone()[0] > 5000  # noqa: S608 - identifiers are fixed internal SQL; all data is bound
        )
        if dense:
            sql = (
                "SELECT CAST(((b.minx+b.maxx)/2-?)/? AS INTEGER) px,"
                "CAST(((b.miny+b.maxy)/2-?)/? AS INTEGER) py,o.class_id,count(*) n"
                + query
                + " GROUP BY px,py,o.class_id"
            )
            for px, py, class_id, count in store.db.execute(sql, [x, ds, y, ds, *args]):
                if 0 <= px < size and 0 <= py < size:
                    color = colors.get(class_id, (230, 230, 230, 100))
                    tile[py, px] = (*color[:3], min(220, 70 + count * 12))
        else:
            transform = Affine(ds, 0, x, 0, ds, y)
            for obj in store.iter_objects(
                slide_id=slide, set_id=set_id, review=review, bounds=bounds, visible_only=True
            ):
                color = colors.get(obj.class_id, (230, 230, 230, 100))
                geometry = obj.geometry.simplify(ds * 0.25, preserve_topology=True) if ds > 1 else obj.geometry
                mask = rasterize([(geometry, 1)], out_shape=(size, size), transform=transform, dtype=np.uint8).astype(
                    bool
                )
                tile[mask] = color
                if ds <= 4:
                    edge = mask & ~binary_erosion(mask)
                    tile[edge] = (*color[:3], 230)
        return tile
