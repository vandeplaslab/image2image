"""Base image wrapper."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike


class BaseImageReader:
    """Base class for some of the other image readers."""

    _pyramid = None
    fh = None
    allow_extraction: bool = False
    base_layer_pixel_res: float = 1.0
    channel_names: ty.List[str]
    channel_colors: ty.Optional[ty.List[str]]

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.base_layer_idx = 0
        self.transform: np.ndarray = np.eye(3, dtype=np.float64)
        self.transform_name = "Identity matrix"

    @property
    def n_channels(self) -> int:
        """Return number of channels"""
        return len(self.channel_names)

    @property
    def dtype(self) -> np.dtype:
        """Return dtype."""
        return self.pyramid[0].dtype

    @property
    def scale(self) -> ty.Tuple[float, float]:
        """Return scale."""
        return self.resolution, self.resolution

    @property
    def resolution(self) -> float:
        """Return resolution."""
        return self.base_layer_pixel_res

    @resolution.setter
    def resolution(self, value: float):
        self.base_layer_pixel_res = value

    @property
    def name(self) -> str:
        """Return name of the input path."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Return name of the input path."""
        return self.path.stem

    def flat_array(self, index: int = 0):
        """Return a flat array."""
        array = self.pyramid[index]
        if array.ndim == 3:
            n_channels = np.min(array.shape)
            array = array.reshape(-1, n_channels)
        else:
            array = array.reshape(-1, 1)
        return array

    def close(self):
        """Close the file handle."""
        if self.fh and hasattr(self.fh, "close"):
            self.fh.close()
        self.fh = None
        self._pyramid = None

    @property
    def pyramid(self) -> ty.List:
        """Pyramid."""
        if self._pyramid is None:
            self._pyramid = self.get_dask_pyr()
        return self._pyramid

    def get_dask_pyr(self) -> ty.List[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def to_csv(self, path: PathLike) -> str:
        """Export data as CSV file."""
        from image2image.utilities import write_rgb_to_txt, write_xml_micro_metadata

        write_xml_micro_metadata(self, path.with_suffix(".xml"))
        yield from write_rgb_to_txt(path, self.pyramid[0])
        return self.path.name
