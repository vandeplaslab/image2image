"""Microscopy reader."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

IMAGE_EXTS = [".jpg", ".png"]


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
            return {
                "name": "Microscopy",
                "data": self.reader_or_image,
                "blending": "additive",
                "channel_axis": None,
            }
        else:
            array = self.reader_or_image.get_dask_pyr()
            temp = array[0] if isinstance(array, list) else array
            channel_axis = None if (self.reader_or_image.is_rgb or temp.shape[0] <= 2) else 0
            if self.reader_or_image.is_rgb or channel_axis is None:
                channel_names = self.reader_or_image.channel_names[0]
            else:
                channel_names = self.reader_or_image.channel_names
            return {
                "name": channel_names,
                "data": array,
                "blending": "additive",
                "channel_axis": channel_axis,
            }

    def channel_names(self) -> ty.List[str]:
        """Return list of channel names."""
        if isinstance(self.reader_or_image, np.ndarray):
            return ["Microscopy"]
        return self.reader_or_image.channel_names


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
