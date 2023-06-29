"""Coordinate reader."""
import typing as ty

import numpy as np
from koyo.typing import PathLike

from ims2micro.readers.base import BaseImageReader

if ty.TYPE_CHECKING:
    from imzy._readers._base import BaseReader


def set_dimensions(reader: "CoordinateReader"):
    """Set dimension information."""
    x, y = reader.x, reader.y
    reader.xmin, reader.xmax = np.min(x), np.max(x)
    reader.ymin, reader.ymax = np.min(y), np.max(y)
    reader.image_shape = (reader.ymax - reader.ymin + 1, reader.xmax - reader.xmin + 1)


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
    ):
        super().__init__(path)
        self.x = x
        self.y = y
        self.resolution = resolution
        self.array_or_reader = array_or_reader
        self.allow_extraction = not isinstance(array_or_reader, np.ndarray)
        set_dimensions(self)

    @property
    def pyramid(self) -> ty.List:
        """Pyramid."""
        return self.get_dask_pyr()

    def get_dask_pyr(self):
        """Get dask representation of the pyramid."""
        # if CONFIG.view_type == "random":
        #     return [self.get_image(None)]
        # return [self.get_image(i) for i in range(self.num_layers)]

    def get_image(self, channel: ty.Optional[str] = None):
        """Return image."""

    #     if isinstance(channel, str) and " | " in channel:
    #         channel, dataset = channel.split(" | ")
    #         if self.data[dataset].ndim > 2:
    #             index = self.map_channel_to_index(channel, dataset)
    #             return self.data[dataset][..., index]
    #         return self.data[dataset]
    #
    #     if channel is None or channel not in self.data:
    #         array = np.full(self.image_shape, np.nan)
    #         array[self.y - self.ymin, self.x - self.xmin] = np.random.randint(5, 255, size=len(self.x))
    #     else:
    #         array = self.data[channel]
    #     return array
