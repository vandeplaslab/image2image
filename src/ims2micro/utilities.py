"""Utilities."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from napari._vispy.layers.points import VispyPointsLayer
from napari.layers.points._points_mouse_bindings import select as _select
from napari.layers.points.points import Mode, Points
from napari.utils.events import Event

from ims2micro.config import CONFIG

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


DRAG_DIST_THRESHOLD = 5

np.seterr(divide="ignore", invalid="ignore")

PREFERRED_COLORMAPS = [
    "red",
    "green",
    "blue",
    "magenta",
    "yellow",
    "cyan",
]


def format_mz(mz: float) -> str:
    """Format m/z value."""
    return f"m/z {mz:.3f}"


def is_debug() -> bool:
    """Return whether in debug mode."""
    import os

    return os.environ.get("IMS2MICRO_DEV_MODE", "0") == "1"


def log_exception(message_or_error):
    """Log exception message. If in 'DEBUG mode' raise exception."""
    if is_debug():
        logger.exception(message_or_error)
    else:
        logger.warning(message_or_error)


def open_link(url: str):
    """Open link."""
    import webbrowser

    webbrowser.open(url)


def open_docs():
    """Open documentation site."""
    open_link("https://ims2micro.readthedocs.io/en/latest/")


def open_github():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/ims2micro")


def open_request():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/ims2micro/issues/new")


def open_bug_report():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/ims2micro/issues/new")


def style_form_layout(layout):
    """Override certain styles for macOS."""
    from qtextra.utils.utilities import IS_MAC

    if IS_MAC:
        layout.setVerticalSpacing(4)


def get_colormap(index: int, used: ty.List[str]):
    """Get colormap that has not been used yet."""
    if index < len(PREFERRED_COLORMAPS):
        colormap = PREFERRED_COLORMAPS[index]
        if colormap not in used:
            return colormap
    for colormap in PREFERRED_COLORMAPS:
        if colormap not in used:
            return colormap
    return "gray"


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


def init_points_layer(layer: Points, visual: VispyPointsLayer):
    """Initialize points layer."""
    layer._drag_modes[Mode.ADD] = add
    layer._drag_modes[Mode.SELECT] = select
    layer.edge_width = 0
    layer.events.add(move=Event, add_point=Event)

    visual._highlight_color = (0, 0.6, 1, 0.3)


def _get_text_format():
    return {
        "text": "{name}",
        "color": CONFIG.label_color,
        "anchor": "center",
        "size": CONFIG.label_size,
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
        layer.text = _get_text_format()
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


def write_xml_registration(output_path: PathLike, affine: np.ndarray):
    """Export affine matrix as XML file."""
    from xml.dom.minidom import parseString

    from dicttoxml import dicttoxml

    assert affine.ndim == 2, "Affine matrix must be 2D."
    assert affine.shape == (3, 3), "Affine matrix must be 3x3."

    temp = [
        " ".join([str(x) for x in affine[0]]),
        " ".join([str(x) for x in affine[1]]),
        " ".join([str(x) for x in affine[2]]),
    ]
    affine = "\n".join(temp)
    meta = {"affine_transformation_matrix": affine}

    xml = dicttoxml(meta, custom_root="data_source_registration", attr_type=False)

    with open(output_path, "w") as f:
        f.write(parseString(xml).toprettyxml())
