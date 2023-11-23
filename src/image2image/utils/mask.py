"""Create mask(s) for image(s) based on shapes."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from image2image_reader.enums import TIME_FORMAT
from koyo.typing import PathLike
from loguru import logger
from rasterio.features import rasterize
from scipy.ndimage import affine_transform
from shapely import Polygon
from skimage.transform import AffineTransform

logger = logger.bind(src="Mask")


def shapes_to_polygons(
    shape_data: list[np.ndarray], with_index: bool = False
) -> list[Polygon] | list[tuple[Polygon, int]]:
    """Convert shapes to polygons."""
    polygons = []
    for index, shape in enumerate(shape_data, start=1):
        if isinstance(shape, dict):
            shape = shape["array"]
        yx = shape
        if with_index:
            polygons.append((Polygon(yx), index))
        else:
            polygons.append(Polygon(yx))
    return polygons


def polygons_to_mask(polygons: list[Polygon], output_shape: tuple[int, int]) -> np.ndarray:
    """Convert polygons to mask."""
    mask: np.ndarray = rasterize(polygons, out_shape=output_shape)
    return mask


def transform_mask(mask: np.ndarray, transform: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Transform mask."""
    tform = AffineTransform(matrix=transform)
    transformed_mask: np.ndarray = affine_transform(
        mask, tform.params, output_shape=output_shape, mode="constant", order=0
    )
    return transformed_mask


def write_masks(
    path: PathLike,
    name: str,
    mask: np.ndarray,
    shapes_data: dict[str, np.ndarray | str],
    display_name: str | None = None,
    creation_date: datetime | None = None,
    color: np.ndarray | None = None,
    metadata: dict[str, np.ndarray] | None = None,
) -> None:
    """Write masks to file."""
    import h5py

    # need the following attributes to be valid
    assert mask.max() == 1, "Expected binary mask."
    assert mask.ndim == 2, "Expected 2D mask."
    # turn into a binary mask
    mask = mask.astype(bool)
    shape = mask.shape

    if display_name is None:
        display_name = name
    creation_date = creation_date or datetime.now()
    if isinstance(creation_date, datetime):
        creation_date = creation_date.strftime(TIME_FORMAT)
    if color is None:
        color = np.asarray([1.0, 0.0, 0.0, 1.0])
    if shapes_data is None:
        shapes_data = {}
    if shapes_data:
        assert "shape_data" in shapes_data, "Expected 'shape_data' key in shapes_data."
        assert "shape_types" in shapes_data, "Expected 'shape_types' key in shapes_data."

    # write to file
    with h5py.File(path, "w") as f:
        # write attributes first
        grp = f.create_group("Mask")
        grp.attrs["shape"] = tuple(shape)
        grp.attrs["name"] = name
        grp.attrs["display_name"] = display_name
        grp.attrs["creation_date"] = creation_date
        # write mask data
        grp.create_dataset("mask", data=mask)
        grp.create_dataset("color", data=color)
        # write shapes data
        if shapes_data:
            for index, (data, shape_type) in enumerate(zip(shapes_data["shape_data"], shapes_data["shape_types"])):
                grp = f.create_group(f"Mask/Shapes/{index}")
                grp.attrs["shape_type"] = shape_type
                grp.create_dataset("data", data=data)
        # add extra data
        if metadata:
            grp = f.create_group("Mask/Metadata")
            for meta_name, meta_data in metadata.items():
                if meta_data.ndim != mask.ndim or meta_data.shape != mask.shape:
                    logger.warning(
                        f"Skipped metadata key={meta_name} as it had wrong shape or dimension. {meta_data.shape}"
                    )
                    continue
                grp.create_dataset(f"{meta_name}", data=meta_data)
