"""Microscopy reader."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


class MicroWrapper:
    """Wrapper around microscopy data."""

    def __init__(self, reader_or_image):
        self.reader_or_image = reader_or_image
        self.resolution = (
            reader_or_image.base_layer_pixel_res if hasattr(reader_or_image, "base_layer_pixel_res") else 1.0
        )

    def image(self) -> ty.Dict[str, ty.Any]:
        """Return image."""
        if isinstance(self.reader_or_image, np.ndarray):
            array = self.reader_or_image
            channel_names = None
        else:
            temp = self.reader_or_image.get_dask_pyr()
            array = temp[0] if isinstance(temp, list) else temp
            channel_names = self.reader_or_image.channel_names

        if array.ndim == 2:
            array = np.atleast_3d(array, axis=-1)
        
        # get channel axis with the lowest number of elements
        channel_axis = np.argmin(array.shape)
        n_ch = array.shape[channel_axis]
        if channel_names is None or len(channel_names) != n_ch:
            channel_names = [f"C{i}" for i in range(n_ch)]

        return {
            "name": channel_names,
            "data": array,
            "blending": "additive",
            "channel_axis": channel_axis,
        }

    def channel_names(self) -> ty.List[str]:
        """Return list of channel names."""
        metadata = self.image()
        return metadata["name"]


def read_microscopy(path: PathLike):
    """Read microscopy data."""
    from ims2micro.readers.tiff_reader import TIFFFILE_EXTS, TiffImageReader

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    assert path.suffix.lower() in TIFFFILE_EXTS + IMAGE_EXTS, f"Unsupported file format: {path.suffix}"

    if path.suffix in TIFFFILE_EXTS:
        return MicroWrapper(TiffImageReader(path))
    elif path.suffix in IMAGE_EXTS:
        return MicroWrapper(_read_image(path))
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _read_image(path: PathLike):
    """Read image."""
    from skimage.io import imread

    path = Path(path)
    assert path.exists(), f"File does not exist: {path}"
    return imread(path)
