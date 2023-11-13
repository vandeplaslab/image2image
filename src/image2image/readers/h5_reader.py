"""HDF5-based reader for AutoIMS / imspy data."""
from __future__ import annotations

import typing as ty

import h5py
import hdf5plugin
import numpy as np
from koyo.typing import PathLike

from image2image.readers._base_reader import BaseReader


class H5ImageReader(BaseReader):
    """GeoJSON reader for image2image."""

    _channel_names: list[str]

    def __init__(self, path: PathLike):
        super().__init__(path)

        # read channel names
