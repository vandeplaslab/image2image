"""Coordinate reader."""
import typing as ty

import numpy as np
from koyo.typing import PathLike
from loguru import logger

from ims2micro.config import CONFIG
from ims2micro.readers.base import BaseImageReader

if ty.TYPE_CHECKING:
    from imzy._readers._base import BaseReader


def set_dimensions(reader: "CoordinateReader"):
    """Set dimension information."""
    x, y = reader.x, reader.y
    reader.xmin, reader.xmax = np.min(x), np.max(x)
    reader.ymin, reader.ymax = np.min(y), np.max(y)
    reader.image_shape = (reader.ymax - reader.ymin + 1, reader.xmax - reader.xmin + 1)


def get_image(array_or_reader):
    """Return image for the array/image."""
    if isinstance(array_or_reader, np.ndarray):
        return array_or_reader
    else:
        return array_or_reader.reshape(array_or_reader.get_tic())


class CoordinateReader(BaseImageReader):
    """Reader for data that has defined coordinates."""

    xmin: int
    xmax: int
    ymin: int
    ymax: int
    image_shape: ty.Tuple[int, int]

    def __init__(
        self,
        path: PathLike,
        x: np.ndarray,
        y: np.ndarray,
        resolution: float = 1.0,
        array_or_reader: ty.Optional[ty.Union[np.ndarray, "BaseReader"]] = None,
        data: ty.Optional[ty.Dict[str, np.ndarray]] = None,
    ):
        super().__init__(path)
        self.x = x
        self.y = y
        self.resolution = resolution
        self.reader = None if isinstance(array_or_reader, np.ndarray) else array_or_reader
        self.allow_extraction = self.reader is not None
        self.data = data or {}
        if self.name not in self.data:
            name = "tic" if self.reader is not None else self.name
            self.data[name] = get_image(array_or_reader)
        print(self.data.keys())
        set_dimensions(self)

    @property
    def channel_names(self) -> ty.List[str]:
        """List of channel names."""
        return list(self.data.keys())

    @property
    def pyramid(self) -> ty.List:
        """Pyramid."""
        return self.get_dask_pyr()

    def extract(self, mzs: np.ndarray, ppm: float = 10.0):
        """Extract ion images."""
        if self.reader is None:
            raise ValueError("Cannot extract ion images from a numpy array.")
        mzs = np.atleast_1d(mzs)
        images = self.reader.get_ion_images(mzs, ppm=ppm)
        for i, mz in enumerate(mzs):
            self.data[f"{mz:.4f}"] = images[i]

    def get_dask_pyr(self) -> ty.List[np.ndarray]:
        """Get dask representation of the pyramid."""
        if CONFIG.view_type == "random":
            return [self.get_random_image()]
        return [self.get_image()]

    def get_random_image(self):
        """Return random ion image."""
        array = np.full(self.image_shape, np.nan)
        array[self.y - self.ymin, self.x - self.xmin] = np.random.randint(5, 255, size=len(self.x))
        return array

    def get_image(self):
        """Return image as a stack."""
        array = np.dstack([self.data[key] for key in self.data])
        print("dstack", array.shape)
        return array
