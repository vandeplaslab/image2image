"""Microscopy reader."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

from ims2micro.models import DataWrapper

if ty.TYPE_CHECKING:
    from ims2micro.readers.tiff_reader import TiffImageReader

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


class MicroWrapper(DataWrapper):
    """Wrapper around microscopy data."""

    def __init__(self, reader_or_array: ty.Dict[str, ty.Any]):
        super().__init__(reader_or_array)
        self.resolution = (
            reader_or_array.base_layer_pixel_res if hasattr(reader_or_array, "base_layer_pixel_res") else 1.0
        )

    def reader_image_iter(self) -> ty.Iterator[ty.Tuple[str, ty.Any, np.ndarray, int]]:
        """Iterator to add channels."""
        for key, reader_or_array in self.data.items():
            if isinstance(reader_or_array, np.ndarray):
                if reader_or_array.ndim == 2:
                    reader_or_array = np.atleast_3d(reader_or_array, axis=-1)
                    self.data[key] = reader_or_array  # replace to ensure it's 3d array
                array = reader_or_array
            else:
                temp = reader_or_array.get_dask_pyr()
                array = temp[0] if isinstance(temp, list) else temp
            channel_axis = np.argmin(array.shape)
            for ch in range(array.shape[channel_axis]):
                if channel_axis == 0:
                    yield key, self.data[key], array[ch], ch
                elif channel_axis == 1:
                    yield key, self.data[key], array[:, ch], ch
                else:
                    yield key, self.data[key], array[..., ch], ch

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
                channel_names = [reader_or_array.channel_names[index]]
            names.extend([f"{name} | {key}" for name in channel_names])
        return names


def read_microscopy(path: PathLike, wrapper: ty.Optional["MicroWrapper"] = None) -> "MicroWrapper":
    """Read microscopy data."""
    from ims2micro.readers.tiff_reader import TIFF_EXTENSIONS

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    assert path.suffix.lower() in TIFF_EXTENSIONS + IMAGE_EXTENSIONS, f"Unsupported file format: {path.suffix}"

    if path.suffix in TIFF_EXTENSIONS:
        data = _read_tiff(path)
    elif path.suffix in IMAGE_EXTENSIONS:
        data = _read_image(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    if wrapper is None:
        wrapper = MicroWrapper(data)
    else:
        for key, value in data.items():
            wrapper.add(key, value)
    wrapper.add_path(path)
    return wrapper


def _read_tiff(path: PathLike) -> ty.Dict[str, "TiffImageReader"]:
    """Read TIFF file."""
    from ims2micro.readers.tiff_reader import TiffImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return {path.name: TiffImageReader(path)}


def _read_image(path: PathLike) -> ty.Dict[str, "TiffImageReader"]:
    """Read image."""
    from skimage.io import imread

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return {path.name: imread(path)}
