"""Utilities."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from napari.layers.points._points_mouse_bindings import select as _select

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


DRAG_DIST_THRESHOLD = 5

np.seterr(divide="ignore", invalid="ignore")


def sanitize_path(path: PathLike) -> ty.Optional[Path]:
    """Sanitize path."""
    if path is None:
        return None
    path = Path(path)
    if path.is_dir():
        return path
    suffix = path.suffix.lower()
    if suffix == ".h5":
        path = path.parent
        assert path.suffix == ".data", "Expected .data file"
    elif suffix in [".tsf", ".tdf"]:
        path = path.parent
        assert path.suffix == ".d", "Expected .d file"
    return path


def round_to_half(*values):
    """Round values to nearest .5."""
    return np.round(np.asarray(values) * 2) / 2


def select(layer, event):
    """Select and move points."""
    yield from _select(layer, event)
    layer.events.move()


def _get_text_properties():
    return {
        "text": "{name}",
        "color": "red",
        "anchor": "center",
        "size": 12,
    }


def _get_text_data(data: np.ndarray) -> ty.Dict[str, ty.List[str]]:
    """Get data."""
    n_pts = data.shape[0]
    return {"name": [str(i + 1) for i in range(n_pts)]}


def add(layer, event):
    """Add a new point at the clicked position."""
    if event.type == "mouse_press":
        start_pos = event.pos

    while event.type != "mouse_release":
        yield

    dist = np.linalg.norm(start_pos - event.pos)
    if dist < DRAG_DIST_THRESHOLD:
        coordinates = round_to_half(layer.world_to_data(event.position))
        # add point
        layer.add(coordinates)
        # update text with index
        layer.properties = _get_text_data(layer.data)
        layer.text = _get_text_properties()
        layer.events.add_point()


def compute_transform(src: np.ndarray, dst: np.ndarray, transform_type: str = "affine") -> "ProjectiveTransform":
    """Compute transform."""
    from skimage.transform import estimate_transform

    if len(dst) != len(src):
        raise ValueError(f"The number of fixed and moving points is not equal. (moving={len(dst)}; fixed={len(src)})")
    return estimate_transform(transform_type, src, dst)


def transform_image(moving_image: np.ndarray, transform) -> np.ndarray:
    """Transform an image."""
    from skimage.transform import warp

    return warp(moving_image, transform, clip=False)
