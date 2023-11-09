"""Numpy array wrapper."""
import typing as ty

import numpy as np
from koyo.typing import PathLike

from image2image.readers._base_reader import BaseReader


class ArrayImageReader(BaseReader):
    """Reader for data that has defined coordinates."""

    is_fixed: bool = False

    def __init__(self, path: PathLike, array: np.ndarray, resolution: float = 1.0):
        super().__init__(path)
        self.array = array
        self.resolution = resolution

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
