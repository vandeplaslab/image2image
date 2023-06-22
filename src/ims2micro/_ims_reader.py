"""IMS reader."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

from ims2micro.models import DataWrapper


def set_dimensions(reader: "IMSWrapper"):
    """Set dimension information."""
    x, y = reader.x, reader.y
    reader.xmin, reader.xmax = np.min(x), np.max(x)
    reader.ymin, reader.ymax = np.min(y), np.max(y)
    reader.image_shape = (reader.ymax - reader.ymin + 1, reader.xmax - reader.xmin + 1)


class IMSWrapper(DataWrapper):
    """Wrapper around IMS data."""

    xmin: int
    xmax: int
    ymin: int
    ymax: int
    image_shape: ty.Tuple[int, int]
    data = ty.Dict[str, ty.Optional[np.ndarray]]

    def __init__(
        self,
        regions: ty.Optional[np.ndarray],
        x: np.ndarray,
        y: np.ndarray,
        resolution: float = 1.0,
        data: ty.Optional[ty.Dict[str, ty.Optional[np.ndarray]]] = None,
    ):
        super().__init__(data)
        self.regions = regions
        self.x = x
        self.y = y
        self.resolution = resolution
        set_dimensions(self)

    def filter_to_roi(self, index: int):
        """Filter to a single region of interest."""
        if self.regions is None:
            return
        mask = self.regions == index
        self.x = self.x[mask]
        self.y = self.y[mask]
        if self.data:
            data = {}
            for key in self.data:
                data[key] = self.data[key][mask]
            self.data = data
        set_dimensions(self)

    def image_iter(self, view_type: ty.Optional[str] = None):
        """Iterator of image channels."""
        if view_type is None or str(view_type).lower() == "random":
            yield self.get_image(None)
        else:
            for channel_name in self.channel_names():
                yield self.get_image(channel_name)

    def image(self, channel: ty.Optional[str] = None) -> ty.Dict[str, np.ndarray]:
        """Return IMS image."""
        array = self.get_image(channel)
        return {"name": "IMS", "data": array, "blending": "additive", "channel_axis": None}

    def get_image(self, channel: ty.Optional[str] = None):
        """Return image."""
        if channel is None or channel not in self.data:
            array = np.full(self.image_shape, np.nan)
            array[self.y - self.ymin, self.x - self.xmin] = np.random.randint(5, 255, size=len(self.x))
        else:
            array = self.data[channel]
        return array

    def channel_names(self, view_type: ty.Optional[str] = None) -> ty.List[str]:
        """Return list of channel names."""
        if view_type is None or str(view_type).lower() == "random":
            return ["IMS"]
        return list(self.data.keys())


def read_imaging(path: PathLike, wrapper: ty.Optional[IMSWrapper] = None) -> IMSWrapper:
    """Read imaging data."""
    path = Path(path)
    if path.suffix.lower() in [".tsf", ".tdf"]:
        region, x, y, resolution, data = _read_tsf_tdf_coordinates(path)
    elif path.suffix.lower() == ".imzml":
        region, x, y, resolution, data = _read_imzml_coordinates(path)
    elif path.suffix.lower() == ".h5":
        region, x, y, resolution, data = _read_metadata_h5_coordinates(path)
    elif path.suffix.lower() in [".npy"]:
        region, x, y, resolution, data = _read_npy_coordinates(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    if wrapper is None:
        wrapper = IMSWrapper(region, x, y, resolution, data)
    else:
        for key, value in data.items():
            wrapper.add(key, value)
    wrapper.add_path(Path(path))
    return wrapper


def _read_npy_coordinates(
    path: PathLike,
) -> ty.Tuple[ty.Optional[np.ndarray], np.ndarray, np.ndarray, float, ty.Dict[str, np.ndarray]]:
    """Read data from npz or npy file."""
    with open(path, "rb") as f:
        image = np.load(f)
    assert image.ndim == 2, "Only 2D images are supported"
    y, x = get_yx_coordinates_from_shape(image.shape)
    return None, x, y, 1.0, {path.name: image}


def _read_metadata_h5_coordinates(
    path: PathLike,
) -> ty.Tuple[ty.Optional[np.ndarray], np.ndarray, np.ndarray, float, ty.Dict[str, np.ndarray]]:
    """Read coordinates from HDF5 file."""
    import h5py
    from koyo.json import read_json_data

    path = Path(path)
    assert path.suffix in [".h5"], "Only .h5 files are supported"

    # read coordinates
    with h5py.File(path, "r") as f:
        yx = f["Dataset/Spectral/Coordinates/yx"][:]
        tic = f["Dataset/Spectral/Sum/y"][:]
    y = yx[:, 0]
    x = yx[:, 1]
    # read pixel size (resolution)
    resolution = 1.0
    if (path.parent / "metadata.json").exists():
        metadata = read_json_data(path.parent / "metadata.json")
        resolution = metadata["metadata.experimental"]["pixel_size"]
    return None, x, y, resolution, {path.name: reshape(x, y, tic)}


def _read_tsf_tdf_coordinates(
    path: PathLike,
) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, float, ty.Dict[str, np.ndarray]]:
    """Read coordinates from TSF file."""
    import sqlite3

    path = Path(path)
    assert path.suffix in [".tsf", ".tdf"], "Only .tsf and .tdf files are supported"

    # get reader
    conn = sqlite3.connect(path)

    try:
        cursor = conn.execute("SELECT SpotSize FROM MaldiFrameLaserInfo")
        resolution = float(cursor.fetchone()[0])
    except sqlite3.OperationalError:
        resolution = 1.0

    # get coordinates
    cursor = conn.cursor()
    cursor.execute("SELECT Frame, RegionNumber, XIndexPos, YIndexPos FROM MaldiFrameInfo")
    frame_index_position = np.array(cursor.fetchall())
    regions = frame_index_position[:, 1]
    x = frame_index_position[:, 2]
    x = x - np.min(x)  # minimized
    y = frame_index_position[:, 3]
    y = y - np.min(y)  # minimized

    # get tic
    cursor = conn.execute("SELECT SummedIntensities FROM Frames")
    tic = np.array(cursor.fetchall())
    tic = tic[:, 0]
    return regions, x, y, resolution, {path.name: reshape(x, y, tic)}


def _read_imzml_coordinates(
    path: PathLike,
) -> ty.Tuple[ty.Optional[np.ndarray], np.ndarray, np.ndarray, float, ty.Dict[str, np.ndarray]]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    path = Path(path)
    assert path.suffix == ".imzML", "Only .imzML files are supported"

    # get reader
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    tic = reader.get_tic()
    return None, x, y, 1.0, {path.name: reader.reshape(tic)}


def reshape(x, y, array):
    """Reshape array."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    shape = (ymax - ymin + 1, xmax - xmin + 1)
    new_array = np.full(shape, np.nan)
    new_array[y - ymin, x - xmin] = array
    return new_array


def get_yx_coordinates_from_shape(shape: ty.Tuple[int, int]) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Get coordinates from image shape."""
    _y, _x = np.indices(shape)
    yx_coordinates = np.c_[np.ravel(_y), np.ravel(_x)]
    return yx_coordinates[:, 0], yx_coordinates[:, 1]
