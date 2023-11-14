"""Coordinate wrapper."""
from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger

from image2image.config import CONFIG
from image2image.readers._base_reader import BaseReader
from image2image.utils.utilities import format_mz

if ty.TYPE_CHECKING:
    from imzy._readers._base import BaseReader as BaseImzyReader


def set_dimensions(reader: CoordinateImageReader) -> None:
    """Set dimension information."""
    x, y = reader.x, reader.y
    reader.xmin, reader.xmax = np.min(x), np.max(x)
    reader.ymin, reader.ymax = np.min(y), np.max(y)
    reader.image_shape = (reader.ymax - reader.ymin + 1, reader.xmax - reader.xmin + 1)


def get_image(array_or_reader: ty.Union[np.ndarray, BaseImzyReader]) -> np.ndarray:
    """Return image for the array/image."""
    if isinstance(array_or_reader, np.ndarray):
        return array_or_reader
    return array_or_reader.reshape(array_or_reader.get_tic())


class CoordinateImageReader(BaseReader):
    """Reader for data that has defined coordinates."""

    xmin: int
    xmax: int
    ymin: int
    ymax: int
    image_shape: tuple[int, int]
    is_fixed: bool = False
    lazy = True

    def __init__(
        self,
        path: PathLike,
        x: np.ndarray,
        y: np.ndarray,
        resolution: float = 1.0,
        array_or_reader: ty.Optional[ty.Union[np.ndarray, BaseImzyReader]] = None,
        data: ty.Optional[dict[str, np.ndarray]] = None,
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
        set_dimensions(self)

    @property
    def channel_names(self) -> list[str]:
        """List of channel names."""
        return list(self.data.keys())

    @property
    def pyramid(self) -> list:
        """Pyramid."""
        return self.get_dask_pyr()

    def extract(self, mzs: np.ndarray, ppm: float = 10.0) -> tuple[Path, list[str]]:
        """Extract ion images."""
        if self.reader is None:
            raise ValueError("Cannot extract ion images from a numpy array.")
        mzs = np.atleast_1d(mzs)
        logger.trace(f"Extracting {len(mzs)} ion images from {self.path.name} ({self.name})")

        with MeasureTimer() as timer:
            images = self.reader.get_ion_images(mzs, ppm=ppm)
        logger.trace(f"Extracted {len(mzs)} ion images in {timer}")

        labels = []
        for i, mz in enumerate(mzs):
            label = format_mz(mz)
            self.data[label] = images[i]
            labels.append(f"{label} | {self.name}")
        return self.path, labels

    def get_dask_pyr(self) -> list[np.ndarray]:
        """Get dask representation of the pyramid."""
        if not self.is_fixed and CONFIG.view_type == "random":
            return [self.get_random_image()]
        return [self.get_image()]

    def get_channel_axis_and_n_channels(self) -> tuple[ty.Optional[int], int]:
        """Return channel axis and number of channels."""
        return 2, len(self.data)

    def get_random_image(self):
        """Return random ion image."""
        array = np.full(self.image_shape, np.nan)
        array[self.y - self.ymin, self.x - self.xmin] = np.random.randint(5, 255, size=len(self.x))
        return array

    def get_image(self):
        """Return image as a stack."""
        array = np.dstack([self.data[key] for key in self.data])
        return array
