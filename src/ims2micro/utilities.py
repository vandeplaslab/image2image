"""Utilities."""
import typing as ty

import numpy as np
from napari.layers.points._points_mouse_bindings import select as _select

if ty.TYPE_CHECKING:
    from skimage.transform._geometric import ProjectiveTransform


DRAG_DIST_THRESHOLD = 5


def round_to_half(*values):
    """Round values to nearest .5."""
    return np.round(np.asarray(values) * 2) / 2


def select(layer, event):
    """Select and move points."""
    yield from _select(layer, event)
    layer.events.move()


def add(layer, event):
    """Add a new point at the clicked position."""
    if event.type == "mouse_press":
        start_pos = event.pos

    while event.type != "mouse_release":
        yield

    dist = np.linalg.norm(start_pos - event.pos)
    if dist < DRAG_DIST_THRESHOLD:
        coordinates = round_to_half(layer.world_to_data(event.position))
        # update text with index
        label = np.asarray([str(v + 1) for v in range(layer.data.shape[0] + 1)])
        layer.text.values = label
        # add point
        layer.add(coordinates)
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
