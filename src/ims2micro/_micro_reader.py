"""Microscopy reader."""
import typing as ty

import numpy as np

if ty.TYPE_CHECKING:
    from ims2micro.readers.base import BaseImageReader

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


class MicroWrapper:
    """Wrapper around microscopy data."""

    def __init__(self, reader_or_array: ty.Dict[str, "BaseImageReader"]):
        super().__init__(reader_or_array)

        resolution = [1.0]
        for _, _reader_or_array in reader_or_array.items():
            if hasattr(_reader_or_array, "base_layer_pixel_res"):
                resolution.append(_reader_or_array.base_layer_pixel_res)
        self.resolution = np.min(resolution)

    def reader_image_iter(self) -> ty.Iterator[ty.Tuple[str, ty.Any, np.ndarray, int]]:
        """Iterator to add channels."""
        for key, reader_or_array in self.data.items():
            if isinstance(reader_or_array, np.ndarray):
                if reader_or_array.ndim == 2:
                    reader_or_array = np.atleast_3d(reader_or_array, axis=-1)
                    self.data[key] = reader_or_array  # replace to ensure it's 3d array
                array = [reader_or_array]
            else:
                temp = reader_or_array.pyramid
                array = temp if isinstance(temp, list) else [temp]
                shape = temp[0].shape

            channel_axis = np.argmin(shape)
            n_channels = shape[channel_axis]
            for ch in range(n_channels):
                if channel_axis == 0:
                    yield key, self.data[key], [a[ch] for a in array], ch
                elif channel_axis == 1:
                    yield key, self.data[key], [a[:, ch] for a in array], ch
                else:
                    yield key, self.data[key], [a[..., ch] for a in array], ch

    def image_iter(self, view_type: ty.Optional[str] = None) -> ty.Iterator[np.ndarray]:
        """Iterator to add channels."""
        for _, _, image, _ in self.reader_image_iter():
            yield image

    def channel_names(self, view_type: ty.Optional[str] = None) -> ty.List[str]:
        """Return list of channel names."""
        names = []
        for key, reader_or_array, _, index in self.reader_image_iter():
            if isinstance(reader_or_array, np.ndarray):
                channel_names = [f"C{index}"]
            else:
                try:
                    channel_names = [reader_or_array.channel_names[index]]
                except IndexError:
                    channel_names = [f"C{index}"]
            names.extend([f"{name} | {key}" for name in channel_names])
        return names
