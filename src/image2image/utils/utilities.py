"""Utilities."""
from functools import partial

import random
import typing as ty
from pathlib import Path

import numba
import numpy as np
from koyo.typing import PathLike
from loguru import logger
from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.layers.shapes import VispyShapesLayer
from napari.layers import Image, Points, Shapes
from napari.layers.points._points_mouse_bindings import select as _select
from napari.layers.points.points import Mode as PointsMode
from napari.utils.colormaps.colormap_utils import convert_vispy_colormap
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.events import Event
from vispy.color import Colormap as VispyColormap

from image2image.config import CONFIG

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform

    from image2image.readers._base_reader import BaseReader


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


def log_exception_or_error(exc_or_error: Exception):
    """Log exception or error and send it to Sentry."""
    from sentry_sdk import capture_exception, capture_message

    if isinstance(exc_or_error, str):
        capture_message(exc_or_error)
        logger.error(exc_or_error)
    else:
        capture_exception(exc_or_error)
        logger.exception(exc_or_error)


def update_affine(matrix: np.ndarray, min_resolution: float, resolution: float) -> np.ndarray:
    """Update affine transformation."""
    from napari.utils.transforms import Affine

    # create copy
    matrix = np.asarray(matrix).copy().astype(np.float64)
    if resolution == min_resolution or resolution == 1 or min_resolution == 1:
        return matrix
    affine = Affine(affine_matrix=matrix)
    affine.scale = (1, 1)
    # affine.scale = affine.scale.astype(np.float64) / (resolution / min_resolution)
    # affine.translate = affine.translate * min_resolution
    return affine.affine_matrix


def format_mz(mz: float) -> str:
    """Format m/z value."""
    return f"m/z {mz:.3f}"


def is_debug() -> bool:
    """Return whether in debug mode."""
    import os

    return os.environ.get("IMAGE2IMAGE_DEV_MODE", "0") == "1"


def log_exception(message_or_error: ty.Union[str, Exception]) -> None:
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
    open_link("https://vandeplaslab.github.io/image2image-docs/")


def open_github():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image")


def open_request():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image/issues/new")


def open_bug_report():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image/issues/new")


def get_random_hex_color() -> str:
    """Return random hex color."""
    return "#%06x" % random.randint(0, 0xFFFFFF)


def get_used_colormaps(layer_list) -> ty.List[str]:
    """Return list of used colormaps based on their name."""
    used = []
    for layer in layer_list:
        if isinstance(layer, Image):
            if hasattr(layer.colormap, "name"):
                used.append(layer.colormap.name)
            else:
                used.append(layer.colormap)
    return used


def get_colormap(index: int, layer_list):
    """Get colormap that has not been used yet."""
    used = get_used_colormaps(layer_list)
    if index < len(PREFERRED_COLORMAPS):
        colormap = PREFERRED_COLORMAPS[index]
        if colormap not in used:
            return colormap
    for colormap in PREFERRED_COLORMAPS:
        if colormap not in used:
            return colormap
    return vispy_colormap(get_random_hex_color())


def vispy_colormap(color) -> VispyColormap:
    """Return vispy colormap."""
    return convert_vispy_colormap(
        VispyColormap([np.asarray([0.0, 0.0, 0.0, 1.0]), transform_color(color)[0]]), name=str(color)
    )


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


def round_to_half(*values: tuple[float, ...]) -> np.ndarray:
    """Round values to nearest .5."""
    return np.round(np.asarray(values) * 2) / 2


def select(layer, event):
    """Select and move points."""
    yield from _select(layer, event)
    layer.events.move()


def init_points_layer(layer: Points, visual: VispyPointsLayer, snap: bool = True) -> None:
    """Initialize points layer."""
    layer._drag_modes[PointsMode.ADD] = partial(add, snap=snap)
    layer._drag_modes[PointsMode.SELECT] = select
    layer.edge_width = 0
    layer.events.add(move=Event, add_point=Event)

    # adjust the highlight
    visual._highlight_color = (1.0, 0.0, 0.0, 0.7)


def init_shapes_layer(layer: Shapes, visual: VispyShapesLayer) -> None:
    """Initialize shapes layer."""
    layer._highlight_color = (1.0, 0.0, 0.0, 0.7)


def _get_text_format() -> ty.Dict[str, ty.Any]:
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


def add(layer, event, snap = True) -> None:
    """Add a new point at the clicked position."""
    start_pos = event.pos
    dist = 0
    yield

    while event.type == "mouse_move":
        dist = np.linalg.norm(start_pos - event.pos)
        if dist < DRAG_DIST_THRESHOLD:
            # prevent vispy from moving the canvas if we're below threshold
            event.handled = True
        yield

    dist = np.linalg.norm(start_pos - event.pos)
    if dist < DRAG_DIST_THRESHOLD:
        coordinates = layer.world_to_data(event.position)
        if snap:
            coordinates = round_to_half(coordinates)
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


def write_reader_to_xml(path_or_tiff: ty.Union[PathLike, "BaseReader"], filename: PathLike):
    """Get filename."""
    from xml.dom.minidom import parseString

    from dicttoxml import dicttoxml

    if isinstance(path_or_tiff, (str, Path)):
        from image2image._reader import TiffImageReader

        reader = TiffImageReader(path_or_tiff)
    else:
        reader = path_or_tiff

    _, _, image_shape = get_shape_of_image(reader.pyramid[0])
    shape = get_flat_shape_of_image(reader.pyramid[0])

    meta = {
        "modality": "microscopy",
        "data_label": reader.path.stem,
        "nr_spatial_dims": 2,
        "spatial_grid_size": f"{image_shape[0]} {image_shape[1]}",
        "nr_spatial_grid_elems": image_shape[0] * image_shape[1],
        "spatial_resolution_um": reader.resolution,
        "nr_obs": shape[0],
        "nr_vars": shape[1],
    }
    xml = dicttoxml(meta, custom_root="data_source", attr_type=False)

    # filename = xml_output_path.parent / (reader.path.stem + ".xml")
    filename = Path(filename).with_suffix(".xml")
    with open(filename, "w") as f:
        f.write(parseString(xml).toprettyxml())
    logger.debug(f"Saved XML file to {filename}")


def reshape_fortran(x: np.ndarray, shape: ty.Tuple[int, int]) -> np.ndarray:
    """Reshape data to Fortran (MATLAB) ordering."""
    return x.T.reshape(shape[::-1]).T


@numba.njit(nogil=True, cache=True)
def int_format_to_row(row: np.ndarray) -> str:
    """Format string to row."""
    return ",".join([str(v) for v in row]) + ",\n"


def float_format_to_row(row: np.ndarray) -> str:
    """Format string to row."""
    return ",".join(
        [
            ",".join([str(int(v)) for v in row[0:2]]),
            ",".join([f"{v:.2f}" for v in row[2:]]),
            "\n",
        ]
    )


def get_shape_of_image(array: np.ndarray) -> tuple[int, ty.Optional[int], tuple[int, ...]]:
    """Return shape of an image."""
    if array.ndim == 3:
        shape = list(array.shape)
        channel_axis = int(np.argmin(shape))
        n_channels = int(shape[channel_axis])
        shape.pop(channel_axis)
    else:
        shape = list(array.shape)
        n_channels = 1
        channel_axis = None
    return n_channels, channel_axis, tuple(shape)


def get_flat_shape_of_image(array: np.ndarray) -> tuple[int, int]:
    """Return shape of an image."""
    n_channels, _, shape = get_shape_of_image(array)
    n_px = int(np.prod(shape))
    return n_channels, n_px


def get_dtype_for_array(array: np.ndarray) -> np.dtype:
    """Return smallest possible data type for shape."""
    n = array.shape[1]
    if np.issubdtype(array.dtype, np.integer):
        if n < np.iinfo(np.uint8).max:
            return np.uint8
        elif n < np.iinfo(np.uint16).max:
            return np.uint16
        elif n < np.iinfo(np.uint32).max:
            return np.uint32
        elif n < np.iinfo(np.uint64).max:
            return np.uint64
    else:
        if n < np.finfo(np.float32).max:
            return np.float32
        elif n < np.finfo(np.float64).max:
            return np.float64


def _insert_indices(array: np.ndarray, shape: ty.Tuple[int, int]) -> np.ndarray:
    n = array.shape[1]
    y, x = np.indices(shape)
    y = y.ravel(order="F")
    x = x.ravel(order="F")
    dtype = get_dtype_for_array(array)
    res = np.zeros((x.size, n + 2), dtype=dtype)
    res[:, 0] = y + 1
    res[:, 1] = x + 1
    res[:, 2:] = reshape_fortran(array, (-1, n))
    return res


def write_reader_to_txt(
    reader: "BaseReader", path: PathLike, update_freq: int = 100_000
) -> ty.Generator[tuple[int, int, str], None, None]:
    """Write image data to text."""
    array, shape = reader.flat_array()
    array = _insert_indices(array, shape)

    if reader.n_channels == 3 and reader.dtype == np.uint8:
        yield from write_rgb_to_txt(path, array, update_freq=update_freq)
    elif np.issubdtype(reader.dtype, np.integer):
        yield from write_int_to_txt(path, array, reader.channel_names, update_freq=update_freq)
    else:
        yield from write_float_to_txt(path, array, reader.channel_names, update_freq=update_freq)


def write_rgb_to_txt(
    path: PathLike, array: np.ndarray, update_freq: int = 100_000
) -> ty.Generator[tuple[int, int, str], None, None]:
    """Write IMS data to text."""
    columns = ["row", "col", "Red (R)", "Green (G)", "Blue (B)"]
    yield from _write_txt(path, columns, array, int_format_to_row, update_freq=update_freq)


def write_int_to_txt(
    path: PathLike, array: np.ndarray, channel_names: ty.List[str], update_freq: int = 100_000
) -> ty.Generator[tuple[int, int, str], None, None]:
    """Write IMS data to text."""
    columns = ["row", "col", *channel_names]
    yield from _write_txt(path, columns, array, int_format_to_row, update_freq=update_freq)


def write_float_to_txt(
    path: PathLike, array: np.ndarray, channel_names: ty.List[str], update_freq: int = 100_000
) -> ty.Generator[tuple[int, int, str], None, None]:
    """Write IMS data to text."""
    columns = ["row", "col", *channel_names]
    yield from _write_txt(path, columns, array, float_format_to_row, update_freq=update_freq)


def _write_txt(
    path: PathLike,
    columns: ty.List[str],
    array: np.ndarray,
    str_func: ty.Callable,
    update_freq: int = 100_000,
) -> ty.Generator[tuple[int, int, str], None, None]:
    """Write data to csv file."""
    from tqdm import tqdm

    path = Path(path)
    assert path.suffix == ".txt", "Path must have .txt extension."

    n = array.shape[0]
    logger.debug(
        f"Exporting array with {array.shape[0]:,} observations, {array.shape[1]:,} features and {array.dtype} data"
        f" type to '{path}'"
    )
    with open(path, "w", newline="\n", encoding="cp1252") as f:
        f.write(",".join(columns) + ",\n")
        with tqdm(array, total=n, mininterval=1) as pbar:
            for i, row in enumerate(pbar):
                f.write(str_func(row))
                if i % update_freq == 0:
                    if i != 0:
                        d = pbar.format_dict
                        if d["rate"]:
                            eta = pbar.format_interval((d["total"] - d["n"]) / d["rate"])
                        else:
                            eta = ""
                        yield i, n, eta
        yield n, n, ""


def write_project(path: PathLike, data: dict) -> None:
    """Export project in appropriate format."""
    path = Path(path)
    if path.suffix == ".json":
        from koyo.json import write_json_data

        write_json_data(path, data)
    elif path.suffix == ".toml":
        from koyo.toml import write_toml_data

        write_toml_data(path, data)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
