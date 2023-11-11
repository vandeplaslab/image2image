"""GeoJSON reader for image2image."""
from __future__ import annotations

import typing as ty

import numpy as np
from koyo.typing import PathLike

from image2image.readers._base_reader import BaseReader
from image2image.readers.geojson_utils import read_geojson, shape_reader


class GeoJSONReader(BaseReader):
    """GeoJSON reader for image2image."""

    reader_type = "shapes"
    _channel_names: list[str]

    def __init__(self, path: PathLike):
        super().__init__(path)
        self._channel_names = [self.path.stem]

        self.geojson_data, self.shape_data = read_geojson(self.path)

    def to_mask(self, output_shape: tuple[int, int]) -> np.ndarray:
        """Convert to mask."""
        from image2image.utils.mask import polygons_to_mask, shapes_to_polygons

        polygons = shapes_to_polygons(self.shape_data)
        mask = polygons_to_mask(polygons, output_shape)
        return mask

    def to_shapes(self) -> tuple[str, dict[str, np.ndarray | str]]:
        """Convert to shapes that can be exported to Shapes layer."""
        _, shape_types, shape_names, shape_arrays, *_ = self.parse_data()
        shape_types = [s.lower() for s in shape_types]  # expected polygon not Polygon
        return shape_names[0], {"shape_types": shape_types, "shape_data": shape_arrays}

    def parse_data(self) -> tuple:
        """Parse data."""
        shape_data = self.shape_data
        shapes_geojson, shapes = shape_reader(shape_data)

        n_shapes = len(shapes_geojson)
        shape_types = [sh["geometry"]["type"] for sh in shapes_geojson]
        shape_names = [sh["properties"]["classification"]["name"] for sh in shapes_geojson]
        shape_arrays = [s["array"][:, [1, 0]] for s in self.shape_data]
        shape_props = {"name": shape_names}
        shape_text = {
            "text": "{name}",
            "color": "white",
            "anchor": "center",
            "size": 12,
            "visible": False,
        }
        return n_shapes, shape_types, shape_names, shape_arrays, shape_props, shape_text

    def to_shapes_kwargs(self, **kwargs: ty.Any) -> dict:
        """Return data so it's compatible with Shapes layer."""
        *_, shape_arrays, shape_props, shape_text = self.parse_data()
        kws = {
            "data": shape_arrays,
            "properties": shape_props,
            "text": shape_text,
            "shape_type": "polygon",
            "scale": self.scale,
            "affine": self.transform,
        }
        kws.update(kwargs)
        return kws

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def to_csv(self, path: PathLike) -> str:
        """Export data as CSV file."""
        raise NotImplementedError("Must implement method")
