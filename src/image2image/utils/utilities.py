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
from napari.utils.events import Event

if ty.TYPE_CHECKING:
    from image2image_io.wrapper import ImageWrapper
    from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls
    from napari._vispy.layers.points import VispyPointsLayer
    from napari._vispy.layers.shapes import VispyShapesLayer
    from napari.components import LayerList
    from napari.layers import Points, Shapes
    from qtextraplot._napari.image.wrapper import NapariImageView
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
PREFERRED_COLORS = [
    "#ff00ff",  # magenta
    "#ffff00",  # yellow
    "#00ffff",  # cyan
    "#ff0000",  # red
    "#00ff00",  # green
    "#0000ff",  # blue
    "#469990",
    "#bfef45",
    "#f58231",
    "#42d4f4",
    "#800000",
]


def check_image_size(
    image: np.ndarray,
    scale: tuple[float, float],
    pyramid_level: int,
    channel_axis: int,
    max_size: int = 1024,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Check image size.

    If pyramid level == -1 then check and ensure that image is not too large. If it is, let's scale it down,
     maintaining aspect ratio.
    """
    if pyramid_level == -1:
        while max(image.shape) > max_size:
            scale = (scale[0] * 0.5, scale[1] * 0.5)
            if channel_axis is None:
                image = np.asarray(
                    [
                        np.asarray([image[int(i / 2), int(j / 2)] for j in range(0, image.shape[1], 2)])
                        for i in range(0, image.shape[0], 2)
                    ]
                )
            elif channel_axis == 1 or channel_axis == 0:
                image = np.asarray(
                    [
                        np.asarray([image[k, int(i / 2), int(j / 2)] for j in range(0, image.shape[2], 2)])
                        for i in range(0, image.shape[1], 2)
                        for k in range(image.shape[0])
                    ]
                )
            elif channel_axis == 2:
                image = np.asarray(
                    [
                        np.asarray([image[k, int(i / 2), int(j / 2)] for j in range(0, image.shape[1], 2)])
                        for i in range(0, image.shape[0], 2)
                        for k in range(image.shape[2])
                    ]
                )
    return image, scale


def pad_str(value: ty.Any) -> str:
    """Pad string with quotes around out."""
    from koyo.system import IS_MAC

    if IS_MAC:
        return str(value)
    return f'"{value!s}"'


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


def format_reader_metadata_alt(scene_index: int, scene_metadata: dict) -> str:
    """Format metadata."""
    metadata = []
    channel_ids = scene_metadata["channel_ids"]
    channel_names = scene_metadata["channel_names"]
    if scene_index is not None and channel_ids:
        metadata.append(f"scene {scene_index}")
    for channel_index, channel_name in zip(channel_ids, channel_names):
        metadata.append(f"- {channel_name}: {channel_index}")
    return "\n".join(metadata)


def format_shape(shape: tuple[int, ...]) -> str:
    """Format shape."""
    return " x ".join(f"{x:,}" for x in shape)


def format_shape_with_pyramid(shape: tuple[int, ...], n_pyramid: int | None = None) -> str:
    """Format shape and optionally add pyramid information."""
    return format_shape(shape) + (f" ({n_pyramid})" if n_pyramid is not None else "")


def format_size(shape: tuple[int, ...], dtype: np.dtype) -> str:
    """Format size in GB."""
    from koyo.utilities import human_readable_byte_size

    dtype = np.dtype(dtype)
    count = np.prod(np.asarray(shape, dtype=np.float64)) if len(shape) > 1 else np.float64(shape[0])
    n_bytes = np.multiply(count, dtype.itemsize)
    return f"{human_readable_byte_size(n_bytes)} ({dtype.name})"


def ensure_list(value: ty.Any) -> list[ty.Any]:
    """Ensure that value is a list."""
    if not isinstance(value, list):
        return [value]
    return value


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
    from natsort import natsorted

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


def log_exception_or_error(exc_or_error: Exception, message: str = "") -> None:
    """Log exception or error and send it to Sentry."""
    from sentry_sdk import capture_exception, capture_message

    message = f"{exc_or_error}" if not message else f"{message}: {exc_or_error}"

    if isinstance(exc_or_error, str):
        capture_message(exc_or_error)
        logger.error(message)
    else:
        capture_exception(exc_or_error)
        logger.exception(message)


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


def open_link(url: str) -> None:
    """Open link."""
    import webbrowser

    webbrowser.open(url)


def open_docs(app: str = "", dialog: str = "", page: str = "") -> None:
    """Open documentation site."""
    if app:
        open_link(f"https://vandeplaslab.github.io/image2image-docs/apps/{app}")
    elif dialog:
        open_link(f"https://vandeplaslab.github.io/image2image-docs/dialogs/{dialog}")
    elif page:
        open_link(f"https://vandeplaslab.github.io/image2image-docs/{page}")
    else:
        open_link("https://vandeplaslab.github.io/image2image-docs/")


def open_github() -> None:
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image")


def open_request() -> None:
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image/issues/new")


def open_bug_report() -> None:
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image/issues/new")


def get_random_hex_color() -> str:
    """Return random hex color."""
    return f"#{random.randint(0, 0xFFFFFF):06x}"


def get_used_colormaps(layer_list: LayerList) -> list[str]:
    """Return list of used colormaps based on their name."""
    from napari.layers import Image

    used = []
    for layer in layer_list:
        if isinstance(layer, Image):
            if hasattr(layer.colormap, "name"):
                used.append(layer.colormap.name)
            else:
                if hasattr(layer, "rgb") and layer.rgb:
                    continue
                used.append(layer.colormap)
    return used


def get_colormap(index: int, layer_list: LayerList, preferred: str | None = None) -> VispyColormap | str:
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


def get_used_colors(layer_list: LayerList, kind: ty.Literal["shapes", "points"]) -> list[str]:
    """Return list of colors."""
    from koyo.color import rgb_1_to_hex
    from napari.layers import Points, Shapes

    cls = Shapes if kind == "shapes" else Points
    attr = "edge_color" if kind == "shapes" else "face_color"

    used = []
    for layer in layer_list:
        if isinstance(layer, cls):
            color = getattr(layer, attr)
            if len(color) > 0:
                color = color[0]
                used.append(rgb_1_to_hex(color))
    return list(set(used))


def get_next_color(index: int, layer_list: LayerList, kind: ty.Literal["shapes", "points"]) -> str:
    """Get shapes color."""
    used = get_used_colors(layer_list, kind)
    if index < len(PREFERRED_COLORS):
        color = PREFERRED_COLORS[index]
        if color not in used:
            return color
    for color in PREFERRED_COLORS:
        if color not in used:
            return color
    return get_random_hex_color()


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


def get_simple_contrast_limits(
    array: list[np.ndarray],
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """Estimate contrast limits."""
    import dask.array as da

    if len(array) == 0:
        return None, None

    array_ = array[0] if len(array) == 1 else array[-1]

    data_range, max_range = None, None
    if 1e5 > array_.size < 1e7:
        array_ = array_[::50, ::50]
    elif array_.size > 1e7:
        array_ = array_[::100, ::100]
    elif array_.dtype in [np.int16, np.int32, np.uint16]:
        max_range = np.iinfo(array_.dtype).min, np.iinfo(array_.dtype).max

    if data_range is None:
        data_range = (0, np.nanquantile(np.array(array_), 0.99))
    return data_range, max_range


def vispy_colormap(color: str | np.ndarray) -> VispyColormap:
    """Return vispy colormap."""
    from napari.utils.colormaps.colormap_utils import convert_vispy_colormap
    from napari.utils.colormaps.standardize_color import transform_color
    from vispy.color import Colormap as VispyColormap

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
    from napari.layers.points._points_mouse_bindings import select as _select

    yield from _select(layer, event)
    layer.events.move()


def init_points_layer(
    layer: Points, visual: VispyPointsLayer, snap: bool = True, add_func: ty.Callable | None = None
) -> None:
    """Initialize points layer."""
    from napari.layers.points.points import Mode as PointsMode

    if add_func is None:
        add_func = partial(add, snap=snap)
    layer._drag_modes[PointsMode.ADD] = add_func
    layer._drag_modes[PointsMode.SELECT] = select
    layer.border_width = 0
    layer.events.add(move=Event, add_point=Event)

    # adjust the highlight
    visual._highlight_color = (1.0, 0.0, 0.0, 0.7)


def init_shapes_layer(layer: Shapes, visual: VispyShapesLayer | None = None) -> None:
    """Initialize shapes layer."""
    layer._highlight_color = (1.0, 0.0, 0.0, 0.7)


def replace_shapes_layer(widget: QtShapesControls, layer: Shapes) -> None:
    """Set new layer for this container."""
    import weakref

    from napari.utils.events import disconnect_events
    from qtextra.helpers import disable_widgets

    def _replace_layer_in_button(btn):
        if hasattr(btn, "layer_ref"):
            btn.layer_ref = weakref.ref(layer)
        else:
            btn.layer = layer

    if layer == widget.layer:
        return
    disconnect_events(widget.layer.events, widget)

    widget.layer = layer
    # update values
    widget._on_opacity_change()
    # widget._on_mode_change()
    widget._on_current_edge_color_change()
    widget._on_current_face_color_change()
    widget._on_edge_width_change()
    widget._on_text_visibility_change()
    widget._on_editable_or_visible_change()
    # for button in widget.button_group.buttons():
    #     _replace_layer_in_button(button)
    for button in widget._MODE_BUTTONS.values():
        _replace_layer_in_button(button)
    for button in widget._EDIT_BUTTONS:
        _replace_layer_in_button(button)

    # connect new events
    widget.layer.events.mode.connect(widget._on_mode_change)
    widget.layer.events.editable.connect(widget._on_editable_or_visible_change)
    widget.layer.events.visible.connect(widget._on_editable_or_visible_change)
    widget.layer.events.blending.connect(widget._on_blending_change)
    widget.layer.events.opacity.connect(widget._on_opacity_change)
    widget.layer.events.edge_width.connect(widget._on_edge_width_change)
    widget.layer.events.current_edge_color.connect(widget._on_current_edge_color_change)
    widget.layer.events.current_face_color.connect(widget._on_current_face_color_change)
    widget.layer.text.events.visible.connect(widget._on_text_visibility_change)

    disable_widgets(
        widget.line_button, widget.path_button, widget.ellipse_button, widget.polyline_button, disabled=True
    )


def _get_text_format() -> dict[str, ty.Any]:
    from image2image.config import get_register_config

    return {
        "text": "{name}",
        "color": get_register_config().label_color,
        "anchor": "center",
        "size": get_register_config().label_size,
    }


def _get_text_data(data: np.ndarray) -> dict[str, list[str]]:
    """Get data."""
    n_pts = data.shape[0]
    return {"name": [str(i + 1) for i in range(n_pts)]}


def get_extents_from_layers(viewer: NapariImageView) -> tuple[float, float, float, float]:
    """Calculate extents from all layers."""
    from napari.layers import Image, Points, Shapes

    extents = []
    for layers in [
        viewer.get_layers_of_type(Image),
        viewer.get_layers_of_type(Points),
        viewer.get_layers_of_type(Shapes),
    ]:
        for layer in layers:
            mins = np.min(layer._extent_data, axis=0)
            maxs = np.max(layer._extent_data, axis=0)
            extents.append((mins[0], maxs[0], mins[1], maxs[1]))
    if not extents:
        extents = [(0, 512, 0, 512)]
    extents = np.asarray(extents)
    return np.nanmin(extents[:, 0]), np.nanmax(extents[:, 1]), np.nanmin(extents[:, 2]), np.nanmax(extents[:, 3])


def get_multiplier(xmax: float, ymax: float) -> float:
    """Based on the maximum value, get a multiplier."""
    from koyo.utilities import find_nearest_value

    max_size = max(xmax, ymax)
    range_to_multiplier = {
        1_000: 0.75,
        2_500: 0.5,
        5_000: 0.35,
        10_000: 0.25,
        15_000: 0.05,
        23_000: 0.03,
        100_000: 0.01,
        250_000: 0.005,
        float("inf"): 0.005,
    }
    nearest = find_nearest_value(list(range_to_multiplier.keys()), max_size)
    return range_to_multiplier[nearest]


def calculate_zoom(
    shape: np.ndarray, viewer: NapariImageView, multiplier: float | None = None
) -> tuple[float, float, float]:
    """Calculate zoom for specified region."""
    # calculate min/max for y, x coordinates
    mins = np.min(shape, axis=0)
    maxs = np.max(shape, axis=0)
    y_fixed = (maxs[0] + mins[0]) / 2
    x_fixed = (maxs[1] + mins[1]) / 2
    # calculate extents for the view
    xmin, xmax, ymin, ymax = get_extents_from_layers(viewer)
    if multiplier is None:
        multiplier = get_multiplier(xmax, ymax)

    # calculate zoom as fraction of the extent
    if ymax > xmax:
        zoom = ((ymax - ymin) / (maxs[0] - mins[0])) * multiplier
    else:
        zoom = ((xmax - xmin) / (maxs[1] - mins[1])) * multiplier
    return zoom, y_fixed, x_fixed


def add(layer: Points, event, snap: bool = True) -> None:
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
        if script_path.exists() and script_path.is_file():
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
            script_path = base_path
            if script_path.name != "Scripts":
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
    try:
        return get_cli_path("i2reg")
    except RuntimeError as exc:
        log_exception_or_error(exc)
        return "i2reg"


def get_resolution_options(wrapper: ImageWrapper) -> dict[str, float]:
    """Get resolution options."""
    resolutions: dict[float, list[str]] = {}
    for reader in wrapper.reader_iter():
        if reader.reader_type != "image":
            continue
        if reader.resolution not in resolutions:
            resolutions[reader.resolution] = []
        resolutions[reader.resolution].append(reader.name)

    options = {"Apply no scaling.": 1.0}
    for resolution, names in resolutions.items():
        if resolution == 1.0:
            continue
        datasets = "<br>".join(names)
        # if len(datasets) > 120:
        #     datasets = f"{datasets[:120]}..."
        options[f"{resolution:.5f}µm<br>Like<br>{datasets}"] = resolution
    return options
