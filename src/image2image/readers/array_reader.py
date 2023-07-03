"""Numpy array wrapper."""
import typing as ty

import numpy as np
from koyo.typing import PathLike

from image2image.readers.base import BaseImageReader


class ArrayReader(BaseImageReader):
    """Reader for data that has defined coordinates."""

    is_fixed: bool = False

    def __init__(self, path: PathLike, array: np.ndarray):
        super().__init__(path)
        self.array = array

    @property
    def channel_names(self) -> ty.List[str]:
        """List of channel names."""
        if self.array.ndim == 2:
            return ["C0"]
        return [f"C{i}" for i in range(self.array.shape[2])]

    @property
    def pyramid(self) -> ty.List:
        """Pyramid."""
        return [self.array]
