"""Generic wrapper."""
from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger

from image2image.exceptions import UnsupportedFileFormatError
from image2image.models.transform import TransformData

if ty.TYPE_CHECKING:
    from image2image.readers._base_reader import BaseReader
    from image2image.readers.array_reader import ArrayImageReader
    from image2image.readers.coordinate_reader import CoordinateImageReader
    from image2image.readers.czi_reader import CziImageReader, CziSceneImageReader
    from image2image.readers.geojson_reader import GeoJSONReader
    from image2image.readers.tiff_reader import TiffImageReader

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]
TIFF_EXTENSIONS = [".scn", ".ome.tiff", ".tif", ".tiff", ".svs", ".ndpi"]
CZI_EXTENSIONS = [".czi"]
BRUKER_EXTENSIONS = [".tsf", ".tdf", ".d"]
IMZML_EXTENSIONS = [".imzml"]
H5_EXTENSIONS = [".h5", ".hdf5"]
IMSPY_EXTENSIONS = [".data"]
NPY_EXTENSIONS = [".npy"]
GEOJSON_EXTENSIONS = [".geojson", ".json"]


class ImageWrapper:
    """Wrapper around image data."""

    data: dict[str, BaseReader]
    resolution: float = 1.0

    def __init__(self, reader: ty.Mapping[str, BaseReader] | None = None):
        self.data = reader or {}

        resolution = [1.0]
        for _, _reader_or_array in self.data.items():
            if hasattr(_reader_or_array, "base_layer_pixel_res"):
                resolution.append(_reader_or_array.resolution)
        self.resolution = np.min(resolution)

    def add(self, reader: BaseReader) -> None:
        """Add data to wrapper."""
        self.data[reader.key] = reader
        logger.trace(f"Added '{reader.key}' to ImageWrapper.")

    def remove(self, key_or_reader: str | BaseReader) -> None:
        """Remove data from wrapper."""
        if hasattr(key_or_reader, "key"):
            key_or_reader = key_or_reader.key
        if key_or_reader in self.data:
            reader = self.data.pop(key_or_reader, None)
            if reader and hasattr(reader, "close"):
                reader.close()
            del reader

    def remove_path(self, path: PathLike) -> list[str]:
        """Remove readers for specified path."""
        keys = self.get_key_for_path(path)
        for key in keys:
            self.remove(key)
        return keys

    def get_key_for_path(self, path: PathLike) -> list[str]:
        """Return key(s) for the specified path."""
        path = Path(path)
        keys = []
        for reader in self.reader_iter():
            if reader.path == path:
                keys.append(reader.key)
        return keys

    def is_loaded(self, path: PathLike) -> bool:
        """Check if the path is loaded."""
        path = Path(path)
        for reader in self.reader_iter():
            if reader.path == path or reader.path == sanitize_read_path(path, raise_error=False):
                return True
        return False

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        return len(self.channel_names())

    def channel_names_for_names(self, names: ty.Sequence[str]) -> list[str]:
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
        for reader in self.reader_iter():
            yield reader.path, reader

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

    @staticmethod
    def _reader_image_iter(
        reader_name: str, reader: BaseReader
    ) -> ty.Generator[tuple[str, BaseReader, list[np.ndarray], int], None, None]:
        # image is a numpy array
        # microscopy or ims data wrapper
        if hasattr(reader, "pyramid"):
            temp = reader.pyramid
            array = temp if isinstance(temp, list) else [temp]
        else:
            raise ValueError("Cannot read image")

        # get the shape of the 'largest image in pyramid'
        shape = array[0].shape
        # get the number of dimensions, which determines how images are split into channels
        ndim = len(shape)
        channel_axis, n_channels = reader.get_channel_axis_and_n_channels()

        for channel_index in range(n_channels):
            # 2D image
            if channel_axis is None:
                yield reader_name, reader, array, channel_index
            # 3D image where the first axis corresponds to different channels
            elif channel_axis == 0:
                yield reader_name, reader, [a[channel_index] for a in array], channel_index
            # 3D image where the second axis corresponds to different channels
            elif channel_axis == 1:
                yield reader_name, reader, [a[:, channel_index] for a in array], channel_index
            # 3D image where the last axis corresponds to different channels
            elif channel_axis == 2:
                yield reader_name, reader, [a[..., channel_index] for a in array], channel_index
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
        elif path.name == "dataset.metadata.h5":
            path = path.parent
    return path


def get_alternative_path(path: PathLike) -> Path:
    """Retrieve alternative name."""
    path = Path(path)
    if path.is_dir() and path.suffix == ".data":
        return path / "dataset.metadata.h5"
    return path


def sanitize_read_path(path: PathLike, raise_error: bool = True) -> Path | None:
    """Sanitize path that can be read by the reader."""
    path = Path(path)
    if not path.exists():
        if raise_error:
            raise FileNotFoundError(f"File does not exist: {path}")
        return None
    suffix = path.suffix.lower()
    if suffix not in (
        TIFF_EXTENSIONS
        + IMAGE_EXTENSIONS
        + CZI_EXTENSIONS
        + NPY_EXTENSIONS
        + BRUKER_EXTENSIONS
        + IMZML_EXTENSIONS
        + H5_EXTENSIONS
        + IMSPY_EXTENSIONS
        + GEOJSON_EXTENSIONS
    ):
        if raise_error:
            raise ValueError(f"Unsupported file format: {path.suffix} ({path})")
        return None
    if suffix in BRUKER_EXTENSIONS:
        if path.suffix == ".d":
            if (path / "analysis.tsf").exists():
                path = path / "analysis.tsf"
            else:
                path = path / "analysis.tdf"
    elif suffix in IMSPY_EXTENSIONS:
        path = path / "dataset.metadata.h5"
    return path


def read_data(
    path: PathLike,
    wrapper: ImageWrapper | None = None,
    is_fixed: bool = False,
    transform_data: TransformData | None = None,
    resolution: float | None = None,
) -> tuple[ImageWrapper, list[str], dict[Path, Path]]:
    """Read image data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in (
        TIFF_EXTENSIONS
        + IMAGE_EXTENSIONS
        + CZI_EXTENSIONS
        + NPY_EXTENSIONS
        + BRUKER_EXTENSIONS
        + IMZML_EXTENSIONS
        + H5_EXTENSIONS
        + IMSPY_EXTENSIONS
        + GEOJSON_EXTENSIONS
    ):
        raise ValueError(f"Unsupported file format: {path.suffix} ({path})")

    reader: dict[str, BaseReader]
    name = path.name
    original_path = path
    if suffix in TIFF_EXTENSIONS:
        logger.trace(f"Reading TIFF file: {path}")
        path, reader = _read_tiff(path)  # type: ignore
    elif suffix in CZI_EXTENSIONS:
        if _check_multi_scene_czi(path):
            logger.trace(f"Reading multi-scene CZI file: {path}")
            path, reader = _read_multi_scene_czi(path)  # type: ignore
        else:
            logger.trace(f"Reading single-scene CZI file: {path}")
            path, reader = _read_single_scene_czi(path)  # type: ignore
    elif suffix in IMAGE_EXTENSIONS:
        logger.trace(f"Reading image file: {path}")
        path, reader = _read_image(path)  # type: ignore
    elif suffix in NPY_EXTENSIONS:
        logger.trace(f"Reading NPY file: {path}")
        path, reader = _read_npy_coordinates(path)  # type: ignore
    elif suffix in BRUKER_EXTENSIONS:
        logger.trace(f"Reading Bruker file: {path}")
        path, reader = _read_tsf_tdf_reader(path)  # type: ignore
    elif suffix in IMZML_EXTENSIONS:
        logger.trace(f"Reading imzML file: {path}")
        path, reader = _read_imzml_reader(path)  # type: ignore
    elif suffix in H5_EXTENSIONS + IMSPY_EXTENSIONS:
        logger.trace(f"Reading HDF5 file: {path}")
        if path.suffix == ".data":
            path = path / "dataset.metadata.h5"
        if path.name.startswith("dataset.metadata"):
            path, reader = _read_metadata_h5_coordinates(path)  # type: ignore
        else:
            path, reader = _read_centroids_h5_coordinates(path)  # type: ignore
    elif suffix in GEOJSON_EXTENSIONS:
        logger.trace(f"Reading GeoJSON file: {path}")
        path, reader = _read_geojson(path)  # type: ignore
    else:
        raise UnsupportedFileFormatError(f"Unsupported file format: '{suffix}'")
    # add transformation information if provided
    if transform_data is not None:
        for reader_ in reader.values():
            reader_.transform_data = transform_data
            reader_.transform_name = name
    # add resolution information if provided
    if resolution is not None:
        for reader_ in reader.values():
            reader_.resolution = resolution
    # specify whether the model is fixed
    if hasattr(reader, "is_fixed"):
        reader.is_fixed = is_fixed

    # initialize image wrapper, unless it's been provided
    if wrapper is None:
        wrapper = ImageWrapper(None)
    # add readers to the wrapper
    just_added = []
    for reader_ in reader.values():
        wrapper.add(reader_)
        just_added.append(reader_.key)
    return wrapper, just_added, {original_path: path}


def get_key(path: Path, scene_index: int | None = None) -> str:
    """Return representative key."""
    name = path.name
    if name == "dataset.metadata.h5":
        name = path.parent.name
    if scene_index is not None:
        name = f"Scene={scene_index}; {name}"
    return name


def _check_multi_scene_czi(path: PathLike) -> bool:
    """Check whether this is a multi-scene CZI file."""
    from image2image.readers._czi import CziSceneFile

    path = Path(path)
    return bool(CziSceneFile.get_num_scenes(path) > 1)


def _read_geojson(path: PathLike) -> tuple[Path, dict[str, GeoJSONReader]]:
    """Read GeoJSON file."""
    from image2image.readers.geojson_reader import GeoJSONReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: GeoJSONReader(path, key=key)}


def _read_single_scene_czi(path: PathLike) -> tuple[Path, dict[str, CziImageReader]]:
    """Read CZI file."""
    from image2image.readers.czi_reader import CziImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: CziImageReader(path, key=key)}


def _read_multi_scene_czi(path: PathLike) -> tuple[Path, dict[str, CziSceneImageReader]]:
    """Read CZI file."""
    from image2image.readers._czi import CziSceneFile
    from image2image.readers.czi_reader import CziSceneImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    n = CziSceneFile.get_num_scenes(path)
    logger.trace(f"Found {n} scenes in CZI file: {path}")
    return path, {
        f"S{i}_{path.name}": CziSceneImageReader(path, scene_index=i, key=get_key(path, scene_index=i))
        for i in range(n)
    }


def _read_tiff(path: PathLike) -> tuple[Path, dict[str, TiffImageReader]]:
    """Read TIFF file."""
    from image2image.readers.tiff_reader import TiffImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: TiffImageReader(path, key=key)}


def _read_image(path: PathLike) -> tuple[Path, dict[str, ArrayImageReader]]:
    """Read image."""
    from skimage.io import imread

    from image2image.readers.array_reader import ArrayImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    key = get_key(path)
    return path, {path.name: ArrayImageReader(path, imread(path), key=key)}


def _read_npy_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read data from npz or npy file."""
    from image2image.readers.coordinate_reader import CoordinateImageReader

    path = Path(path)
    with open(path, "rb") as f:
        image = np.load(f)  # noqa
    assert image.ndim == 2, "Only 2D images are supported"
    y, x = get_yx_coordinates_from_shape(image.shape)
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=image, key=key)}


def _read_metadata_h5_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
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
    key = get_key(path)
    return path, {
        path.name: CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=reshape(x, y, tic), key=key)
    }


def _read_centroids_h5_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
    """Read centroids data from HDF5 file."""
    path = Path(path)
    assert path.suffix in H5_EXTENSIONS, "Only .h5 files are supported"

    metadata_file = path.parent / "dataset.metadata.h5"
    if not metadata_file.exists():
        data_dir = path.parent.parent.with_suffix(".data")
        metadata_file = data_dir / "dataset.metadata.h5"
    if metadata_file.exists():
        return _read_centroids_h5_coordinates_with_metadata(path, metadata_file)
    return _read_centroids_h5_coordinates_without_metadata(path)


def _read_centroids_h5_coordinates_with_metadata(
    path: Path, metadata_file: Path
) -> tuple[Path, dict[str, CoordinateImageReader]]:
    import h5py

    from image2image.utils.utilities import format_mz

    assert metadata_file.exists(), f"File does not exist: {metadata_file}"
    _, reader_ = _read_metadata_h5_coordinates(metadata_file)
    reader = reader_[path.name]

    x = reader.x  # noqa
    y = reader.y  # noqa

    with h5py.File(path, "r") as f:
        mzs = f["Array"]["xs"][:]  # retrieve m/zs
        centroids = f["Array"]["array"][:]  # retrieve ion images
    mzs = [format_mz(mz) for mz in mzs]  # generate labels
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(mzs, centroids)))
    return path, {path.name: reader}


def _read_centroids_h5_coordinates_without_metadata(path: Path) -> tuple[Path, dict[str, CoordinateImageReader]]:
    import h5py

    from image2image.readers.coordinate_reader import CoordinateImageReader
    from image2image.utils.utilities import format_mz

    with h5py.File(path, "r") as f:
        # get coordinate metadata
        x = f["Misc/Spatial/x_coordinates"][:]
        y = f["Misc/Spatial/y_coordinates"][:]
        resolution = float(f["Misc/Spatial"].attrs["pixel_size"])
        mzs = f["Array"]["xs"][:]  # retrieve m/zs
        centroids = f["Array"]["array"][:]  # retrieve ion images
    tic = np.random.randint(128, 255, len(x), dtype=np.uint8)
    tic = reshape(x, y, tic)
    key = get_key(path)
    reader = CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=tic, key=key)
    mzs = [format_mz(mz) for mz in mzs]  # generate labels
    centroids = reshape_batch(x, y, centroids)  # reshape images
    reader.data.update(dict(zip(mzs, centroids)))
    return path, {path.name: reader}


def _read_tsf_tdf_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
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
    key = get_key(path)
    return path.parent, {
        path.name: CoordinateImageReader(path, x, y, resolution=resolution, array_or_reader=reshape(x, y, tic), key=key)
    }


def _read_tsf_tdf_reader(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
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
    key = get_key(path)
    return path, {
        path.name: CoordinateImageReader(path.parent, x, y, resolution=resolution, array_or_reader=reader, key=key)
    }


def _read_imzml_coordinates(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
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
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=reshape(x, y, tic), key=key)}


def _read_imzml_reader(path: PathLike) -> tuple[Path, dict[str, CoordinateImageReader]]:
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
    key = get_key(path)
    return path, {path.name: CoordinateImageReader(path, x, y, array_or_reader=reader, key=key)}


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


def write_ome_tiff(path: PathLike, array: np.ndarray, reader: BaseReader) -> Path:
    """Write OME-TIFF."""
    from wsireg.reg_images import NumpyRegImage
    from wsireg.writers.ome_tiff_writer import OmeTiffWriter

    if array.ndim == 2:
        array = np.atleast_3d(array)

    reg = NumpyRegImage(
        array,
        reader.resolution,
        channel_names=reader.channel_names,
    )

    path = Path(path)
    writer = OmeTiffWriter(reg)
    filename = writer.write_image_by_plane(path.stem, path.parent, write_pyramid=True)
    return filename
