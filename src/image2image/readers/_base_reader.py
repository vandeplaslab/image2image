"""Base image wrapper."""
from __future__ import annotations

import math
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike

from image2image.enums import DEFAULT_TRANSFORM_NAME
from image2image.models.transform import TransformData


class BaseReader:
    """Base class for some of the other image readers."""

    _pyramid = None
    _image_shape: tuple[int, int] | None = None
    reader_type: str = "image"
    lazy: bool = False
    fh: ty.Any | None = None
    allow_extraction: bool = False
    _resolution: float = 1.0
    _channel_names: list[str]

    def __init__(self, path: PathLike, key: str | None = None, reader_kws: dict | None = None):
        # This is the direct path to the image
        self.path = Path(path)
        # This is the attribute we will use to identify the image
        self.key = key or self.path.name
        self.reader_kws = reader_kws or {}
        self.transform_data: TransformData = TransformData()
        self.transform_name = DEFAULT_TRANSFORM_NAME

    @property
    def transform(self) -> np.ndarray:
        """Return transform."""
        transform: np.ndarray = self.transform_data.transform.params
        return transform

    @transform.setter
    def transform(self, value: np.ndarray) -> None:
        assert value.shape == (3, 3)
        self.transform_data._transform = value

    def is_identity_transform(self) -> bool:
        """Return whether transform is identity."""
        if self.transform_data.transform:
            return np.allclose(self.transform_data.transform.params, np.eye(3))
        return np.allclose(self.transform, np.eye(3))

    @property
    def inv_resolution(self) -> float:
        """Return inverse resolution."""
        return 1 / self.resolution

    @property
    def image_shape(self) -> tuple[int, int]:
        """Image shape."""
        if self._image_shape is None:
            from image2image.utils.utilities import get_shape_of_image

            return get_shape_of_image(self.pyramid[0])[-1]
        return self._image_shape

    @image_shape.setter
    def image_shape(self, value: tuple[int, int]) -> None:
        self._image_shape = value

    @property
    def channel_names(self) -> list[str]:
        """Return channel names."""
        return self._channel_names

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        return len(self.channel_names)

    @property
    def dtype(self) -> np.dtype:
        """Return dtype."""
        return self.pyramid[0].dtype

    @property
    def scale(self) -> tuple[float, float]:
        """Return scale."""
        return self.resolution, self.resolution

    @property
    def resolution(self) -> float:
        """Return resolution."""
        return self._resolution

    @resolution.setter
    def resolution(self, value: float) -> None:
        self._resolution = value
        if self.transform_data:
            self.transform_data.moving_resolution = value

    @property
    def name(self) -> str:
        """Return name of the input path."""
        return self.path.name

    @property
    def stem(self) -> str:
        """Return name of the input path."""
        return self.path.stem

    def flat_array(self, index: int = 0) -> tuple[np.ndarray, tuple[int, int]]:
        """Return a flat array."""
        from image2image.utils.utilities import get_shape_of_image

        array = self.pyramid[index]
        if hasattr(array, "compute"):
            array = array.compute()
        n_channels, _, shape = get_shape_of_image(array)
        if array.ndim == 3:
            array = array.reshape(-1, n_channels)
        else:
            array = array.reshape(-1, 1)
        return array, shape

    def close(self) -> None:
        """Close the file handle."""
        if self.fh and hasattr(self.fh, "close"):
            self.fh.close()
        self.fh = None
        self._pyramid = None

    @property
    def pyramid(self) -> list:
        """Pyramid."""
        if self._pyramid is None:
            self._pyramid = self.get_dask_pyr()
        return self._pyramid

    def get_dask_pyr(self) -> list[ty.Any]:
        """Get dask representation of the pyramid."""
        raise NotImplementedError("Must implement method")

    def crop(self, left: int, right: int, top: int, bottom: int) -> np.ndarray:
        """Crop image."""
        top, bottom = sorted([top, bottom])
        left, right = sorted([left, right])
        left = math.floor(left * self.inv_resolution)
        right = math.ceil(right * self.inv_resolution)
        top = math.floor(top * self.inv_resolution)
        bottom = math.ceil(bottom * self.inv_resolution)
        array = self.pyramid[0]
        if array.ndim == 2:
            array_ = array[top:bottom, left:right]
        elif array.ndim == 3:
            shape = array.shape
            channel_axis = int(np.argmin(shape))
            if channel_axis == 0:
                array_ = array[:, top:bottom, left:right]
            elif channel_axis == 1:
                array_ = array[top:bottom, :, left:right]
            elif channel_axis == 2:
                array_ = array[top:bottom, left:right, :]
            else:
                raise ValueError(f"Array has unsupported shape: {array.shape}")
        else:
            raise ValueError(f"Array has unsupported shape: {array.shape}")
        # check whether an array is dask array - if so, we need to compute it
        if hasattr(array_, "compute"):
            array_ = array_.compute()
        return array_  # type: ignore[no-any-return]

    def warp(self, array: np.ndarray) -> np.ndarray:
        """Warp array."""
        from image2image.utils.mask import transform_mask

        transform = self.transform_data.compute(yx=True, px=True).params
        transformed_mask = transform_mask(array, transform, self.image_shape)
        return transformed_mask

    def get_channel_axis_and_n_channels(self) -> tuple[int | None, int]:
        """Return channel axis and number of channels."""
        shape = self.pyramid[0].shape
        ndim = len(shape)
        # 2D images will be returned as they are
        if ndim == 2:
            channel_axis = None
            n_channels = 1
        # 3D images will be split into channels
        else:
            channel_axis = int(np.argmin(shape))
            n_channels = shape[channel_axis]
        return channel_axis, n_channels
