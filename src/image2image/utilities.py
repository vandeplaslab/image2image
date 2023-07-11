"""Utilities."""
import typing as ty
from pathlib import Path

import numba
import numpy as np
from koyo.typing import PathLike
from loguru import logger
from napari._vispy.layers.points import VispyPointsLayer
from napari._vispy.layers.shapes import VispyShapesLayer
from napari.layers.points._points_mouse_bindings import select as _select
from napari.layers.points.points import Mode as PointsMode
from napari.layers.points.points import Points
from napari.layers.shapes.shapes import Shapes
from napari.utils.events import Event

from image2image.config import CONFIG

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


def update_affine(matrix: np.ndarray, min_resolution: float, resolution: float) -> np.ndarray:
    """Update affine transformation."""
    from napari.utils.transforms import Affine

    # create copy
    matrix = np.asarray(matrix).copy().astype(np.float64)
    if resolution == min_resolution or resolution == 1 or min_resolution == 1:
        return matrix
    affine = Affine(affine_matrix=matrix)
    affine.scale = affine.scale.astype(np.float64) / (resolution / min_resolution)
    affine.translate = affine.translate * min_resolution
    return affine.affine_matrix


def format_mz(mz: float) -> str:
    """Format m/z value."""
    return f"m/z {mz:.3f}"


def is_debug() -> bool:
    """Return whether in debug mode."""
    import os

    return os.environ.get("IMAGE2IMAGE_DEV_MODE", "0") == "1"


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
    open_link("https://image2image.readthedocs.io/en/latest/")


def open_github():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image")


def open_request():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image/issues/new")


def open_bug_report():
    """Open GitHub website."""
    open_link("https://github.com/vandeplaslab/image2image/issues/new")


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
    layer._drag_modes[PointsMode.ADD] = add
    layer._drag_modes[PointsMode.SELECT] = select
    layer.edge_width = 0
    layer.events.add(move=Event, add_point=Event)

    visual._highlight_color = (0, 0.6, 1, 0.3)


def init_shapes_layer(layer: Shapes, visual: VispyShapesLayer):
    """Initialize shapes layer."""
    layer._highlight_color = (0, 0.6, 1, 0.3)


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


def write_xml_micro_metadata(path_or_tiff: PathLike, xml_output_path: Path):
    """Get filename."""
    from xml.dom.minidom import parseString

    from dicttoxml import dicttoxml

    if isinstance(path_or_tiff, (str, Path)):
        from image2image._reader import TiffImageReader

        reader = TiffImageReader(path_or_tiff)
    else:
        reader = path_or_tiff

    image_shape = reader.im_dims[0:2]
    shape = reader.pyramid[0].reshape(-1, 3).shape

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

    filename = xml_output_path.parent / (reader.path.stem + ".xml")
    with open(filename, "w") as f:
        f.write(parseString(xml).toprettyxml())


def reshape_fortran(x, shape):
    """Reshape data to Fortran (MATLAB) ordering."""
    return x.T.reshape(shape[::-1]).T


@numba.njit()
def micro_format_to_row(row: np.ndarray) -> str:
    """Format string to row."""
    return ",".join([str(v) for v in row]) + ",\n"


def prepare_micro_to_fusion(path_or_tiff: PathLike, output_dir: PathLike, get_minimal: bool = False):
    """Export microscopy data to text."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(path_or_tiff, (str, Path)):
        path = path_or_tiff
    else:
        path = path_or_tiff.path
    micro_path = output_dir / (Path(path).stem + ".txt")
    if get_minimal:
        return None, micro_path

    if isinstance(path_or_tiff, (str, Path)):
        from image2image._reader import TiffImageReader

        reader = TiffImageReader(path_or_tiff)
    else:
        reader = path_or_tiff
    image = reader.pyramid[0]
    shape = image.shape[0:2]
    n = image.shape[2]

    y, x = np.indices(shape)
    y = y.ravel(order="F")
    x = x.ravel(order="F")

    res = np.zeros((x.size, n + 2), dtype=np.uint16)
    res[:, 0] = y + 1
    res[:, 1] = x + 1
    res[:, 2:] = reshape_fortran(image, (-1, n))
    return res, micro_path


def write_micro_to_txt(path: PathLike, array: np.ndarray, chunk_size: int = 5000):
    """Write IMS data to text."""
    columns = ["row", "col", "Red (R)", "Green (G)", "Blue (B)"]
    _write_txt(path, columns, array, micro_format_to_row, chunk_size=chunk_size)


def _write_txt(
    path: PathLike,
    columns: ty.List[str],
    array: np.ndarray,
    str_func: ty.Callable,
    chunk_size: int = 5000,
    in_chunks: bool = False,
):
    """Write data to csv file."""
    from koyo.utilities import chunks
    from tqdm.auto import tqdm

    columns = ",".join(columns) + ",\n"

    path = Path(path)
    assert path.suffix == ".txt", "Path must have .txt extension."

    logger.debug(f"Exporting array with {array.shape[0]:,} observations and {array.shape[1]:,} features to '{path}'")
    with open(path, "w", newline="\n", encoding="cp1252") as f:
        f.write(columns)
        if not in_chunks:
            for row in tqdm(array, total=array.shape[0], mininterval=1):
                f.write(str_func(row))
        else:
            write_chunks = chunks(array, n_items=chunk_size)
            with tqdm(total=array.shape[0], mininterval=1) as pbar:
                for chunk in write_chunks:
                    temp = []
                    for row in chunk:
                        temp.append(str_func(row))
                    f.write("".join(temp))
                    pbar.update(chunk.shape[0])
