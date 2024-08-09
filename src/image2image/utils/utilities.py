"""Utilities."""

from __future__ import annotations

import random
import re
import typing as ty
from functools import partial
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.layers.shapes import VispyShapesLayer
from napari.layers import Image, Layer, Points, Shapes
from napari.layers.points._points_mouse_bindings import select as _select
from napari.layers.points.points import Mode as PointsMode
from napari.utils.colormaps.colormap_utils import convert_vispy_colormap
from napari.utils.colormaps.standardize_color import transform_color
from napari.utils.events import Event
from natsort import natsorted
from vispy.color import Colormap as VispyColormap

DRAG_DIST_THRESHOLD = 5

np.seterr(divide="ignore", invalid="ignore")

PREFERRED_COLORMAPS = [
    "red",
    "green",
    "blue",
    "magenta",
    "yellow",
    "cyan",
    "#469990",
    "#bfef45",
    "#f58231",
    "#42d4f4",
    "#800000",
]


def format_reader_metadata(reader_metadata: dict) -> str:
    """Format metadata."""
    metadata = []
    has_scenes = len(reader_metadata) > 1
    for _index, (scene_index, scene_metadata) in enumerate(reader_metadata.items()):
        channel_ids = scene_metadata["channel_ids"]
        channel_names = scene_metadata["channel_names"]
        if has_scenes and channel_ids:
            metadata.append(f"scene {scene_index}")
        for channel_index, channel_name in zip(channel_ids, channel_names):
            metadata.append(f"- {channel_name}: {channel_index}")
    return "\n".join(metadata)


def format_shape(shape: tuple[int, ...]) -> str:
    """Format shape."""
    return " x ".join(str(x) for x in shape)


def get_groups(filenames: list[str], keyword: str, by_slide: bool = False) -> dict[str, list[str]]:
    """Get groups."""
    groups: dict[str, list[str]] = {"no group": []}
    for index, filename in enumerate(filenames):
        group = str(index) if by_slide else extract_number(filename, keyword)

        if group is None:
            groups["no group"].append(filename)
        else:
            if group not in groups:
                groups[group] = []
            groups[group].append(filename)
    # remove empty groups
    for group in list(groups.keys()):
        if len(groups[group]) == 0:
            del groups[group]
    return groups


def groups_to_group_id(groups: dict[str, list[str]]) -> dict[str, int]:
    """Convert groups to group ID."""
    dataset_to_group_map = {}
    for i, group in enumerate(natsorted(groups)):
        try:
            key = int(group)
        except (TypeError, ValueError):
            key = i
        for filename in natsorted(groups[group]):
            dataset_to_group_map[filename] = key
    return dataset_to_group_map


def format_group_info(groups: dict[str, list[str]]) -> str:
    """Format group info."""
    res = ""
    for group in groups:
        res += f"<b>Group '{group}'</b> - {len(groups[group])} in group<br>"
        for filename in groups[group]:
            res += f"  - {filename}<br>"
        res += "<br>"
    return res


def extract_number(filename: str, keyword: str) -> str | None:
    """Extract number from filename."""
    # Define a regular expression pattern to capture the specified keyword and its associated number
    pattern = re.compile(rf"{keyword}(\d+)")

    # Use the pattern to find the match in the filename
    match = pattern.search(filename)

    # Return the extracted number, or None if not found
    return match.group(1) if match else None


def extract_extension(available_formats: str) -> list[str]:
    """Extract suffix/extension from available formats."""
    res = []
    for row in available_formats.split(";;"):
        for row_ in row.split("*"):
            if row_.startswith("."):
                res.append(row_.replace("(", "").replace(")", "").replace(" ", "").replace("*", ""))
    return list(set(res))


def log_exception_or_error(exc_or_error: Exception) -> None:
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


def is_debug() -> bool:
    """Return whether in debug mode."""
    import os

    return os.environ.get("IMAGE2IMAGE_DEV_MODE", "0") == "1"


def log_exception(message_or_error: str | Exception) -> None:
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


def get_used_colormaps(layer_list: list[Layer]) -> list[str]:
    """Return list of used colormaps based on their name."""
    used = []
    for layer in layer_list:
        if isinstance(layer, Image):
            if hasattr(layer.colormap, "name"):
                used.append(layer.colormap.name)
            else:
                used.append(layer.colormap)
    return used


def get_colormap(index: int, layer_list, preferred: str | None = None) -> VispyColormap | str:
    """Get colormap that has not been used yet."""
    used = get_used_colormaps(layer_list)
    if preferred is not None and isinstance(preferred, str) and preferred not in used:
        if preferred.startswith("#"):
            return vispy_colormap(preferred)
        return preferred

    if index < len(PREFERRED_COLORMAPS):
        colormap = PREFERRED_COLORMAPS[index]
        if colormap not in used:
            if colormap.startswith("#"):
                return vispy_colormap(colormap)
            return colormap
    for colormap in PREFERRED_COLORMAPS:
        if colormap not in used:
            if colormap.startswith("#"):
                return vispy_colormap(colormap)
            return colormap
    return vispy_colormap(get_random_hex_color())


def get_contrast_limits(array: list[np.ndarray]) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Estimate contrast limits."""
    from napari.layers.utils.layer_utils import calc_data_range

    if len(array) == 0:
        return None, None

    if len(array) == 1:
        array_ = array[0]
    else:
        mid = len(array) // 2
        array_ = array[mid]

    data_range, max_range = None, None
    if 1e5 > array_.size < 1e7:
        array_ = array_[::50, ::50]
    elif array_.size > 1e7:
        array_ = array_[::100, ::100]
    # if array_.dtype == np.uint8:
    #     data_range = max_range = (0, 255)
    elif array_.dtype in [np.int16, np.int32, np.uint16]:
        max_range = np.iinfo(array_.dtype).min, np.iinfo(array_.dtype).max

    if data_range is None:
        data_range = calc_data_range(array_.astype(np.float32))
    return data_range, max_range


def vispy_colormap(color: str | np.ndarray) -> VispyColormap:
    """Return vispy colormap."""
    return convert_vispy_colormap(
        VispyColormap([np.asarray([0.0, 0.0, 0.0, 1.0]), transform_color(color)[0]]), name=str(color)
    )


def sanitize_path(path: PathLike) -> Path | None:
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


def _get_text_format() -> dict[str, ty.Any]:
    from image2image.config import REGISTER_CONFIG

    return {
        "text": "{name}",
        "color": REGISTER_CONFIG.label_color,
        "anchor": "center",
        "size": REGISTER_CONFIG.label_size,
    }


def _get_text_data(data: np.ndarray) -> dict[str, list[str]]:
    """Get data."""
    n_pts = data.shape[0]
    return {"name": [str(i + 1) for i in range(n_pts)]}


def add(layer, event, snap=True) -> None:
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


def write_xml_registration(filename: PathLike, affine: np.ndarray):
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
    meta = {"affine_transformation_matrix": "\n".join(temp)}

    xml = dicttoxml(meta, custom_root="data_source_registration", attr_type=False)

    with open(filename, "w") as f:
        f.write(parseString(xml).toprettyxml())


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


def ensure_extension(path: PathLike, extension: str) -> Path:
    """Ensure that path has a specific extension."""
    path = Path(path)
    if extension not in path.name:
        suffix = path.suffix
        path = path.with_suffix(f".{extension}{suffix}")
    return path


def get_cli_path(name: str) -> str:
    """Get path to imimspy executable.

    The path is determined in the following order:
    1. First, we check whether environment variable `AUTOIMS_{name.upper()}_PATH` is set.
    2. If not, we check whether we are running as a PyInstaller app.
    3. If not, we check whether we are running as a Python app.
    4. If not, we raise an error.
    """
    import os
    import sys

    from koyo.system import IS_LINUX, IS_MAC, IS_WIN
    from koyo.utilities import running_as_pyinstaller_app

    env_var = f"IMAGE2IMAGE_{name.upper()}_PATH"
    if os.environ.get(env_var, None):
        script_path = Path(os.environ[env_var])
        if script_path.exists():
            return str(script_path)

    base_path = Path(sys.executable).parent
    if running_as_pyinstaller_app():
        if IS_WIN:
            script_path = base_path / f"{name}.exe"
            if script_path.exists():
                return str(script_path)
            script_path = base_path / "image2image.exe"
        elif IS_MAC or IS_LINUX:
            script_path = base_path / name
            if script_path.exists():
                return str(script_path)
            return str(script_path.parent / "image2image_")
        else:
            raise NotImplementedError(f"Unsupported OS: {sys.platform}")
        if script_path.exists():
            return str(script_path)
    else:
        # on Windows, {name} lives under the `Scripts` directory
        if IS_WIN:
            script_path = base_path / "Scripts"
            if script_path.exists() and (script_path / f"{name}.exe").exists():
                return str(script_path / f"{name}.exe")
        elif IS_MAC or IS_LINUX:
            script_path = base_path / name
            if script_path.exists():
                return str(script_path)
        else:
            script_path = base_path / f"{name}.exe"
            if script_path.exists():
                return str(script_path)
    raise RuntimeError(f"Could not find '{name}' executable.")


def get_i2reg_path() -> str:
    """Get image2image-reg path.

    You can force a specific path by setting the environment variable `IMAGE2IMAGE_I2REG`.
    """
    return get_cli_path("i2reg")
