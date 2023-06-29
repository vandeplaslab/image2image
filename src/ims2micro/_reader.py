"""Generic reader."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

if ty.TYPE_CHECKING:
    from ims2micro.readers.base import BaseImageReader

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
TIFF_EXTENSIONS = [".scn", ".ome.tiff", ".tif", ".tiff", ".svs", ".ndpi"]
CZI_EXTENSIONS = [".czi"]
BRUKER_EXTENSIONS = [".tsf", ".tdf"]
IMZML_EXTENSIONS = [".imzml"]
H5_EXTENSIONS = [".h5", ".hdf5"]
NPY_EXTENSIONS = [".npy"]


class ImageWrapper:
    """Wrapper around microscopy data."""

    data: ty.Dict[str, ty.Optional[ty.Union["BaseImageReader", np.ndarray]]]
    paths: ty.List[Path]
    resolution: float = 1.0

    def __init__(self, reader_or_array: ty.Dict[str, ty.Union[np.ndarray, "BaseImageReader"]]):
        self.data = reader_or_array or {}
        self.paths = []

        resolution = [1.0]
        for _, _reader_or_array in reader_or_array.items():
            if hasattr(_reader_or_array, "base_layer_pixel_res"):
                resolution.append(_reader_or_array.base_layer_pixel_res)
        self.resolution = np.min(resolution)

    def add(self, key: str, array: ty.Union[np.ndarray, ty.Any]):
        """Add data to wrapper."""
        self.data[key] = array

    def add_path(self, path: PathLike):
        """Add the path to wrapper."""
        self.paths.append(Path(path))

    def remove_path(self, path: PathLike):
        """Remove the path from wrapper."""
        path = Path(path)
        if path in self.paths:
            self.paths.remove(path)
        path = str(path.name)
        if path in self.data:
            reader = self.data[path]
            if hasattr(reader, "close"):
                reader.close()
            del self.data[path]

    def is_loaded(self, path: PathLike):
        """Check if the path is loaded."""
        return Path(path) in self.paths

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        return len(self.channel_names())

    def channel_names_for_names(self, names: ty.List[PathLike]) -> ty.List[str]:
        """Return list of channel names for a given reader/dataset."""
        clean_names = []
        for name in names:
            if isinstance(name, Path):
                name = str(name.name)
            if "| " not in name:
                name = f"| {name}"
            clean_names.append(name)
        channel_names = []
        for channel_name in self.channel_names():
            for name in clean_names:
                if channel_name.endswith(name):
                    channel_names.append(channel_name)
        return channel_names

    def map_channel_to_index(self, dataset: str, channel_name: str) -> int:
        """Map channel name to index."""
        dataset_to_channel_map = {}
        for name in self.channel_names():
            dataset, channel = name.split(" | ")
            dataset_to_channel_map.setdefault(dataset, []).append(channel)
        return dataset_to_channel_map[dataset].index(channel_name)

    def channel_image_iter(self, view_type: ty.Optional[str] = None) -> ty.Iterator[ty.Tuple[str, np.ndarray]]:
        """Iterator of channel name + image."""
        yield from zip(self.channel_names(view_type), self.image_iter(view_type))

    def reader_image_iter(
        self, view_type: ty.Optional[str] = None
    ) -> ty.Iterator[ty.Tuple[str, ty.Any, np.ndarray, int]]:
        """Iterator to add channels."""
        for key, reader_or_array in self.data.items():
            # image is a numpy array
            if isinstance(reader_or_array, np.ndarray):
                if reader_or_array.ndim == 2:
                    reader_or_array = np.atleast_3d(reader_or_array, axis=-1)
                    self.data[key] = reader_or_array  # replace to ensure it's 3d array
                array = [reader_or_array]
            elif hasattr(reader_or_array, "pyramid"):
                temp = reader_or_array.pyramid
                array = temp if isinstance(temp, list) else [temp]
            else:
                raise ValueError("Cannot read image")

            shape = array[0].shape
            channel_axis = int(np.argmin(shape))
            n_channels = shape[channel_axis]
            for ch in range(n_channels):
                if channel_axis == 0:
                    yield key, self.data[key], [a[ch] for a in array], ch
                elif channel_axis == 1:
                    yield key, self.data[key], [a[:, ch] for a in array], ch
                else:
                    yield key, self.data[key], [a[..., ch] for a in array], ch

    def image_iter(self, view_type: ty.Optional[str] = None) -> ty.Iterator[np.ndarray]:
        """Iterator to add channels."""
        for _, _, image, _ in self.reader_image_iter():
            yield image

    def channel_names(self, view_type: ty.Optional[str] = None) -> ty.List[str]:
        """Return list of channel names."""
        names = []
        for key, reader_or_array, _, index in self.reader_image_iter():
            if isinstance(reader_or_array, np.ndarray):
                channel_names = [f"C{index}"]
            else:
                try:
                    channel_names = [reader_or_array.channel_names[index]]
                except IndexError:
                    channel_names = [f"C{index}"]
            names.extend([f"{name} | {key}" for name in channel_names])
        return names


def read_image(path: PathLike, wrapper: ty.Optional["ImageWrapper"] = None) -> "ImageWrapper":
    """Read microscopy data."""
    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    assert (
        path.suffix.lower()
        in TIFF_EXTENSIONS
        + IMAGE_EXTENSIONS
        + CZI_EXTENSIONS
        + NPY_EXTENSIONS
        + BRUKER_EXTENSIONS
        + IMZML_EXTENSIONS
        + H5_EXTENSIONS
    ), f"Unsupported file format: {path.suffix}"

    suffix = path.suffix.lower()
    if suffix in TIFF_EXTENSIONS:
        data = _read_tiff(path)
    elif suffix in CZI_EXTENSIONS:
        data = _read_czi(path)
    elif suffix in IMAGE_EXTENSIONS:
        data = _read_image(path)
    elif suffix in NPY_EXTENSIONS:
        data = _read_npy_coordinates(path)
    elif suffix in BRUKER_EXTENSIONS:
        data = _read_tsf_tdf_coordinates(path)
    elif suffix in IMZML_EXTENSIONS:
        data = _read_imzml_coordinates(path)
    elif suffix in H5_EXTENSIONS:
        if path.name.startswith("peaks_"):
            data = _read_centroids_h5_coordinates(path)
        else:
            data = _read_metadata_h5_coordinates(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    if wrapper is None:
        wrapper = ImageWrapper(data)
    else:
        for key, value in data.items():
            wrapper.add(key, value)
    wrapper.add_path(path)
    return wrapper


def _read_czi(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read CZI file."""
    from ims2micro.readers.czi_reader import CziImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return {path.name: CziImageReader(path)}


def _read_tiff(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read TIFF file."""
    from ims2micro.readers.tiff_reader import TiffImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return {path.name: TiffImageReader(path)}


def _read_image(path: PathLike) -> ty.Dict[str, np.ndarray]:
    """Read image."""
    from skimage.io import imread

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return {path.name: imread(path)}


def _read_npy_coordinates(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read data from npz or npy file."""
    from ims2micro.readers.coordinate_reader import CoordinateReader

    with open(path, "rb") as f:
        image = np.load(f)  # noqa
    assert image.ndim == 2, "Only 2D images are supported"
    y, x = get_yx_coordinates_from_shape(image.shape)
    return {path.name: CoordinateReader(path, x, y, array_or_reader=image)}


def _read_metadata_h5_coordinates(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read coordinates from HDF5 file."""
    import h5py
    from koyo.json import read_json_data

    from ims2micro.readers.coordinate_reader import CoordinateReader

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
    return {path.name: CoordinateReader(path, x, y, resolution, array_or_reader=tic)}


def _read_centroids_h5_coordinates(path: PathLike):
    """Read centroids data from HDF5 file."""


def _read_tsf_tdf_coordinates(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read coordinates from TSF file."""
    import sqlite3

    from ims2micro.readers.coordinate_reader import CoordinateReader

    path = Path(path)
    assert path.suffix in BRUKER_EXTENSIONS, "Only .tsf and .tdf files are supported"

    # get reader
    conn = sqlite3.connect(path)

    try:
        cursor = conn.execute("SELECT SpotSize FROM MaldiFrameLaserInfo")  # noqa
        resolution = float(cursor.fetchone()[0])
    except sqlite3.OperationalError:
        resolution = 1.0

    # get coordinates
    cursor = conn.cursor()
    cursor.execute("SELECT Frame, RegionNumber, XIndexPos, YIndexPos FROM MaldiFrameInfo")  # noqa
    frame_index_position = np.array(cursor.fetchall())
    # regions = frame_index_position[:, 1]
    x = frame_index_position[:, 2]
    x = x - np.min(x)  # minimized
    y = frame_index_position[:, 3]
    y = y - np.min(y)  # minimized

    # get tic
    cursor = conn.execute("SELECT SummedIntensities FROM Frames")  # noqa
    tic = np.array(cursor.fetchall())
    tic = tic[:, 0]
    return {path.name: CoordinateReader(path, x, y, resolution, array_or_reader=reshape(x, y, tic))}


def _read_tsf_tdf_reader(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read coordinates from Bruker file."""
    from imzy import get_reader

    from ims2micro.readers.coordinate_reader import CoordinateReader

    path = Path(path)
    assert path.suffix in BRUKER_EXTENSIONS, "Only .tsf and .tdf files are supported"

    # get reader
    reader = get_reader(path)
    x = reader.x_coordinates
    y = reader.y_coordinates
    return {path.name: CoordinateReader(path, x, y, array_or_reader=reader)}


def _read_imzml_coordinates(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    from ims2micro.readers.coordinate_reader import CoordinateReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"

    # get reader
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    tic = reader.get_tic()
    return {path.name: CoordinateReader(path, x, y, array_or_reader=reshape(x, y, tic))}


def _read_imzml_reader(path: PathLike) -> ty.Dict[str, "BaseImageReader"]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    from ims2micro.readers.coordinate_reader import CoordinateReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"

    # get reader
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    return {path.name: CoordinateReader(path, x, y, array_or_reader=reader)}


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
