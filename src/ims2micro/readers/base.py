"""Base image reader."""
import typing as ty
from pathlib import Path

from koyo.typing import PathLike


class BaseImageReader:
    """Base class for some of the other image readers."""

    _pyramid = None
    fh = None
    base_layer_pixel_res: float
    channel_names: ty.List[str]
    channel_colors: ty.Optional[ty.List[str]]

    def __init__(self, path: PathLike):
        self.path = Path(path)
        self.base_layer_idx = 0

    def close(self):
        """Close the file handle."""
        if self.fh and hasattr(self.fh, "close"):
            self.fh.close()
        del self.fh, self._pyramid

    @property
    def pyramid(self) -> ty.List:
        """Pyramid."""
        if self._pyramid is None:
            self._pyramid = self.get_dask_pyr()
        return self._pyramid

    def get_dask_pyr(self):
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")
