"""Generic wrapper."""
from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger

from image2image.models.transform import TransformData

if ty.TYPE_CHECKING:
    from image2image.readers._base_reader import BaseReader
    from image2image.readers.array_reader import ArrayImageReader
    from image2image.readers.coordinate_reader import CoordinateImageReader
    from image2image.readers.czi_reader import CziImageReader
    from image2image.readers.geojson_reader import GeoJSONReader
    from image2image.readers.tiff_reader import TiffImageReader

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
TIFF_EXTENSIONS = [".scn", ".ome.tiff", ".tif", ".tiff", ".svs", ".ndpi"]
CZI_EXTENSIONS = [".czi"]
BRUKER_EXTENSIONS = [".tsf", ".tdf", ".d"]
IMZML_EXTENSIONS = [".imzml"]
H5_EXTENSIONS = [".h5", ".hdf5"]
NPY_EXTENSIONS = [".npy"]
GEOJSON_EXTENSIONS = [".geojson", ".json"]


class ImageWrapper:
    """Wrapper around image data."""

    data: dict[str, BaseReader]
    paths: list[Path]
    resolution: float = 1.0

    def __init__(self, reader_or_array: ty.Optional[ty.Mapping[str, BaseReader]] = None):
        self.data = reader_or_array or {}
        self.paths = []

        resolution = [1.0]
        for _, _reader_or_array in self.data.items():
            if hasattr(_reader_or_array, "base_layer_pixel_res"):
                resolution.append(_reader_or_array.base_layer_pixel_res)
        self.resolution = np.min(resolution)

    def add(self, key: str, array: ty.Union[np.ndarray, ty.Any]) -> None:
        """Add data to wrapper."""
        self.data[key] = array
        logger.trace(f"Added '{key}' to wrapper data.")

    def add_path(self, path: PathLike) -> None:
        """Add the path to wrapper."""
        self.paths.append(Path(path))
        logger.trace(f"Added '{path}' to wrapper paths.")

    def remove_path(self, path: PathLike) -> None:
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

    def is_loaded(self, path: PathLike) -> bool:
        """Check if the path is loaded."""
        return Path(path) in self.paths

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        return len(self.channel_names())

    def channel_names_for_names(self, names: ty.Sequence[PathLike]) -> list[str]:
        """Return list of channel names for a given wrapper/dataset."""
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
        dataset_to_channel_map: dict[str, list[str]] = {}
        for name in self.channel_names():
            dataset, channel = name.split(" | ")
            dataset_to_channel_map.setdefault(dataset, []).append(channel)
        channels: list[str] = dataset_to_channel_map[dataset]
        return channels.index(channel_name)

    def channel_image_iter(self) -> ty.Generator[tuple[str, list[np.ndarray]], None, None]:
        """Iterator of channel name + image."""
        yield from zip(self.channel_names(), self.image_iter())

    def channel_image_reader_iter(self) -> ty.Generator[tuple[str, list[np.ndarray], BaseReader], None, None]:
        """Iterator of channel name + image."""
        for channel_name, (_, reader_or_array, image, _) in zip(self.channel_names(), self.reader_data_iter()):
            yield channel_name, image, reader_or_array

    def path_reader_iter(self) -> ty.Generator[tuple[Path, BaseReader], None, None]:
        """Iterator of a path + reader."""
        for path in self.paths:
            yield path, self.data[path.name]

    def reader_channel_iter(
        self,
    ) -> ty.Generator[tuple[str, BaseReader, int], None, None]:
        """Iterator to add channels."""
        for reader_name, reader_or_array in self.data.items():
            if reader_or_array.reader_type == "shapes":
                yield reader_name, reader_or_array, 0
            else:
                for reader_name_, reader_or_array_, _, index in self._reader_image_iter(reader_name, reader_or_array):
                    yield reader_name_, reader_or_array_, index

    def reader_data_iter(
        self,
    ) -> ty.Generator[tuple[str, BaseReader, list[np.ndarray] | None, int], None, None]:
        """Iterator to add channels."""
        for reader_name, reader_or_array in self.data.items():
            if reader_or_array.reader_type == "shapes":
                yield reader_name, reader_or_array, None, 0
            else:
                yield from self._reader_image_iter(reader_name, reader_or_array)

    def reader_image_iter(
        self,
    ) -> ty.Generator[tuple[str, BaseReader, list[np.ndarray], int], None, None]:
        """Iterator to add channels."""
        for reader_name, reader_or_array in self.data.items():
            yield from self._reader_image_iter(reader_name, reader_or_array)

    def _reader_image_iter(
        self, reader_name: str, reader_or_array: BaseReader
    ) -> ty.Generator[tuple[str, BaseReader, list[np.ndarray], int], None, None]:
        # image is a numpy array
        if isinstance(reader_or_array, np.ndarray):
            logger.error("THIS SHOULD NOT HAPPEN!")
            if reader_or_array.ndim == 2:
                reader_or_array = np.atleast_3d(reader_or_array, axis=-1)
                self.data[reader_name] = reader_or_array  # replace to ensure it's 3d array
            array = [reader_or_array]
        # microscopy or ims data wrapper
        elif hasattr(reader_or_array, "pyramid"):
            temp = reader_or_array.pyramid
            array = temp if isinstance(temp, list) else [temp]
        else:
            raise ValueError("Cannot read image")

        # get the shape of the 'largest image in pyramid'
        shape = array[0].shape
        # get the number of dimensions, which determines how images are split into channels
        ndim = len(shape)
        # 2D images will be returned as they are
        if ndim == 2:
            channel_axis = None
            n_channels = 1
        # 3D images will be split into channels
        else:
            channel_axis = int(np.argmin(shape))
            n_channels = shape[channel_axis]
        for channel_index in range(n_channels):
            # 2D image
            if channel_axis is None:
                yield reader_name, reader_or_array, array, channel_index
            # 3D image where the first axis corresponds to different channels
            elif channel_axis == 0:
                yield reader_name, reader_or_array, [a[channel_index] for a in array], channel_index
            # 3D image where the second axis corresponds to different channels
            elif channel_axis == 1:
                yield reader_name, reader_or_array, [a[:, channel_index] for a in array], channel_index
            # 3D image where the last axis corresponds to different channels
            elif channel_axis == 2:
                yield reader_name, reader_or_array, [a[..., channel_index] for a in array], channel_index
            else:
                raise ValueError(f"Cannot read image with {ndim} dimensions")

    def image_iter(self) -> ty.Iterator[list[np.ndarray]]:
        """Iterator to add channels."""
        for _, _, image, _ in self.reader_image_iter():
            yield image

    def reader_iter(self) -> ty.Iterator[BaseReader]:
        """Iterator of readers."""
        for _, reader_or_array in self.data.items():
            yield reader_or_array

    def channel_names(self) -> list[str]:
        """Return list of channel names."""
        names = []
        for key, reader_or_array, index in self.reader_channel_iter():
            if isinstance(reader_or_array, np.ndarray):
                channel_names = [f"C{index}"]
            else:
                try:
                    channel_names = [reader_or_array.channel_names[index]]
                except IndexError:
                    channel_names = [f"C{index}"]
            names.extend([f"{name} | {key}" for name in channel_names])
        return names

    @property
    def min_resolution(self) -> float:
        """Return minimum resolution."""
        return min([reader.resolution for reader in self.reader_iter()])

    def update_affine(self, affine: np.ndarray, resolution: float) -> np.ndarray:
        """Update affine matrix."""
        from image2image.utils.utilities import update_affine

        return update_affine(affine, self.min_resolution, resolution)

    @staticmethod
    def get_affine(reader: BaseReader, moving_resolution: float) -> np.ndarray:
        """Get affine transformation for specified reader."""
        transform_data = reader.transform_data
        if not transform_data:
            raise ValueError("No transformation data found")
        affine = transform_data.compute(moving_resolution=moving_resolution, px=False)
        return np.asarray(affine.params)


def sanitize_path(path: PathLike) -> Path:
    """Sanitize a path, so it has a unified format across models."""
    path = Path(path).resolve()
    if path.is_file():
        if path.suffix in [".tsf", ".tdf"]:
            path = path.parent
    return path


def read_data(
    path: PathLike,
    wrapper: ty.Optional[ImageWrapper] = None,
    is_fixed: bool = False,
    transform_data: ty.Optional[TransformData] = None,
    resolution: ty.Optional[float] = None,
) -> ImageWrapper:
    """Read image data."""
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
        + GEOJSON_EXTENSIONS
    ), f"Unsupported file format: {path.suffix} ({path})"

    suffix = path.suffix.lower()
    reader: BaseReader
    if suffix in TIFF_EXTENSIONS:
        path, reader = _read_tiff(path)
    elif suffix in CZI_EXTENSIONS:
        path, reader = _read_czi(path)
    elif suffix in IMAGE_EXTENSIONS:
        path, reader = _read_image(path)
    elif suffix in NPY_EXTENSIONS:
        path, reader = _read_npy_coordinates(path)
    elif suffix in BRUKER_EXTENSIONS:
        path, reader = _read_tsf_tdf_reader(path)
    elif suffix in IMZML_EXTENSIONS:
        path, reader = _read_imzml_reader(path)
    elif suffix in H5_EXTENSIONS:
        if path.name.startswith("peaks_"):
            path, reader = _read_centroids_h5_coordinates(path)
        else:
            path, reader = _read_metadata_h5_coordinates(path)
    elif suffix in GEOJSON_EXTENSIONS:
        path, reader = _read_geojson(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    if transform_data is not None:
        reader.transform_data = transform_data
        reader.transform_name = path.name
    if resolution is not None:
        reader.resolution = resolution

    if wrapper is None:
        wrapper = ImageWrapper(None)

    # specify whether the model is fixed
    if hasattr(reader, "is_fixed"):
        reader.is_fixed = is_fixed

    wrapper.add(path.name, reader)
    wrapper.add_path(path)
    return wrapper


def _read_geojson(path: PathLike) -> tuple[Path, GeoJSONReader]:
    """Read GeoJSON file."""
    from image2image.readers.geojson_reader import GeoJSONReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return path, GeoJSONReader(path)


def _read_czi(path: PathLike) -> tuple[Path, CziImageReader]:
    """Read CZI file."""
    from image2image.readers.czi_reader import CziImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return path, CziImageReader(path)


def _read_tiff(path: PathLike) -> tuple[Path, TiffImageReader]:
    """Read TIFF file."""
    from image2image.readers.tiff_reader import TiffImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return path, TiffImageReader(path)


def _read_image(path: PathLike) -> tuple[Path, ArrayImageReader]:
    """Read image."""
    from skimage.io import imread

    from image2image.readers.array_reader import ArrayImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return path, ArrayImageReader(path, imread(path))


def _read_npy_coordinates(path: PathLike) -> tuple[Path, CoordinateImageReader]:
    """Read data from npz or npy file."""
    from image2image.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    with open(path, "rb") as f:
        image = np.load(f)  # noqa
    assert image.ndim == 2, "Only 2D images are supported"
    y, x = get_yx_coordinates_from_shape(image.shape)
    return path, CoordinateImageReader(path, x, y, array_or_reader=image)


def _read_metadata_h5_coordinates(path: PathLike) -> tuple[Path, CoordinateImageReader]:
    """Read coordinates from HDF5 file."""
    import h5py
    from koyo.json import read_json_data

    from image2image.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    # read coordinates
    with h5py.File(path, "r") as f:
        try:
            yx = f["Dataset/Spectral/Coordinates/yx"][:]
            tic = f["Dataset/Spectral/Sum/y"][:]
        except KeyError:
            yx = f["Dataset/Spatial/Coordinates/yx"][:]
            tic = f["Dataset/Spatial/Sum/y"][:]
    y = yx[:, 0]
    x = yx[:, 1]
    # read pixel size (resolution)
    resolution = 1.0
    if (path.parent / "metadata.json").exists():
        metadata = read_json_data(path.parent / "metadata.json")
        resolution = metadata["metadata.experimental"]["pixel_size"]
    return path, CoordinateImageReader(path, x, y, resolution, array_or_reader=reshape(x, y, tic))


def _read_centroids_h5_coordinates(path: PathLike) -> tuple[Path, CoordinateImageReader]:
    """Read centroids data from HDF5 file."""
    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    metadata_file = path.parent / "dataset.metadata.h5"
    if not metadata_file.exists():
        data_dir = path.parent.parent.with_suffix(".data")
        metadata_file = data_dir / "dataset.metadata.h5"
    if metadata_file.exists():
        return _read_centroids_h5_coordinates_with_metadata(path, metadata_file)
    raise NotImplementedError("Not implemented yet.")
    # return _read_centroids_h5_coordinates_without_metadata(path)


def _read_centroids_h5_coordinates_with_metadata(path: Path, metadata_file: Path) -> tuple[Path, CoordinateImageReader]:
    import h5py

    from image2image.utils.utilities import format_mz

    assert metadata_file.exists(), f"File does not exist: {metadata_file}"
    _, reader = _read_metadata_h5_coordinates(metadata_file)
    x = reader.x  # noqa
    y = reader.y  # noqa

    with h5py.File(path, "r") as f:
        ys = f["Array"]["ys"][:]
        indices = np.argsort(ys)[::-1]
        indices = indices[0:10]  # take the top 10 images
        indices = np.sort(indices)  # sort so they are ordered otherwise h5py will throw an error
        mzs = f["Array"]["xs"][indices]  # retrieve m/zs
        centroids = f["Array"]["array"][:, indices]  # retrieve ion images
    mzs = [format_mz(mz) for mz in mzs]  # generate labels
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(mzs, centroids)))
    return path, reader


def _read_centroids_h5_coordinates_without_metadata(path: Path) -> tuple[Path, CoordinateImageReader]:
    # TODO: implement this
    pass
    # import h5py
    #
    # from image2image.utils.utilities import format_mz
    #
    # _, reader = _read_metadata_h5_coordinates(metadata_file)
    # x = reader.x  # noqa
    # y = reader.y  # noqa
    #
    # with h5py.File(path, "r") as f:
    #     ys = f["Array"]["ys"][:]
    #     indices = np.argsort(ys)[::-1]
    #     indices = indices[0:10]  # take the top 10 images
    #     indices = np.sort(indices)  # sort so they are ordered otherwise h5py will throw an error
    #     mzs = f["Array"]["xs"][indices]  # retrieve m/zs
    #     centroids = f["Array"]["array"][:, indices]  # retrieve ion images
    # mzs = [format_mz(mz) for mz in mzs]  # generate labels
    # centroids = reshape_batch(x, y, centroids)  # reshape images
    # reader.data.update(dict(zip(mzs, centroids)))
    # return path, reader


def _read_tsf_tdf_coordinates(path: PathLike) -> tuple[Path, CoordinateImageReader]:
    """Read coordinates from TSF file."""
    import sqlite3

    from image2image.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix in BRUKER_EXTENSIONS, "Only .tsf and .tdf files are supported"

    if path.suffix == ".d":
        if (path / "analysis.tsf").exists():
            path = path / "analysis.tsf"
        else:
            path = path / "analysis.tdf"

    # get wrapper
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
    return path.parent, CoordinateImageReader(path, x, y, resolution, array_or_reader=reshape(x, y, tic))


def _read_tsf_tdf_reader(path: PathLike) -> tuple[Path, CoordinateImageReader]:
    """Read coordinates from Bruker file."""
    import sqlite3

    from imzy import get_reader

    from image2image.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix in BRUKER_EXTENSIONS, "Only .tsf and .tdf files are supported"

    if path.suffix == ".d":
        if (path / "analysis.tsf").exists():
            path = path / "analysis.tsf"
        else:
            path = path / "analysis.tdf"

    conn = sqlite3.connect(path)
    try:
        cursor = conn.execute("SELECT SpotSize FROM MaldiFrameLaserInfo")  # noqa
        resolution = float(cursor.fetchone()[0])
    except sqlite3.OperationalError:
        resolution = 1.0
    conn.close()

    # get wrapper
    path = path.parent
    reader = get_reader(path)
    x = reader.x_coordinates
    y = reader.y_coordinates
    return path, CoordinateImageReader(path.parent, x, y, resolution, array_or_reader=reader)


def _read_imzml_coordinates(path: PathLike) -> tuple[Path, CoordinateImageReader]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    from image2image.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"

    # get wrapper
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    tic = reader.get_tic()
    return path, CoordinateImageReader(path, x, y, array_or_reader=reshape(x, y, tic))


def _read_imzml_reader(path: PathLike) -> tuple[Path, CoordinateImageReader]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    from image2image.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    assert path.suffix.lower() in IMZML_EXTENSIONS, "Only .imzML files are supported"

    # get wrapper
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    return path, CoordinateImageReader(path, x, y, array_or_reader=reader)


def reshape(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Reshape array."""
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    shape = (ymax - ymin + 1, xmax - xmin + 1)
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    new_array = np.full(shape, fill_value=fill_value, dtype=dtype)
    new_array[y - ymin, x - xmin] = array
    return new_array


def reshape_batch(x: np.ndarray, y: np.ndarray, array: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """Batch reshaping of images."""
    if array.ndim != 2:
        raise ValueError("Expected 2-D array.")
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    y = y - ymin
    x = x - xmin
    n = array.shape[1]
    shape = (n, ymax - ymin + 1, xmax - xmin + 1)
    dtype = np.float32 if np.isnan(fill_value) else array.dtype
    im = np.full(shape, fill_value=fill_value, dtype=dtype)
    for i in range(n):
        im[i, y, x] = array[:, i]
    return im


def get_yx_coordinates_from_shape(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Get coordinates from image shape."""
    _y, _x = np.indices(shape)
    yx_coordinates = np.c_[np.ravel(_y), np.ravel(_x)]
    return yx_coordinates[:, 0], yx_coordinates[:, 1]
