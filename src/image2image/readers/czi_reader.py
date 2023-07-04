import typing as ty

import dask.array as da
import numpy as np
import zarr
from koyo.typing import PathLike
from tifffile import xml2dict

from image2image.readers.base_reader import BaseImageReader
from image2image.readers.czi import CziFile
from image2image.readers.utilities import guess_rgb


class CziImageReader(BaseImageReader):
    """CZI file wrapper."""

    def __init__(self, path: PathLike, init_pyramid: bool = True):
        super().__init__(path)
        self.fh = CziFile(self.path)

        (
            self.ch_dim_idx,
            self.y_dim_idx,
            self.x_dim_idx,
            self.im_dims,
            self.im_dtype,
        ) = self._get_image_info()

        self.im_dims = tuple(self.im_dims)
        self.is_rgb = guess_rgb(self.im_dims)
        self.n_ch = self.im_dims[2] if self.is_rgb else self.im_dims[0]

        czi_meta = xml2dict(self.fh.metadata())
        pixel_scaling_str = czi_meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"][0]["Value"]
        pixel_scaling = float(pixel_scaling_str) * 1000000
        self.resolution = pixel_scaling
        channels_meta = czi_meta["ImageDocument"]["Metadata"]["DisplaySetting"]["Channels"]["Channel"]

        channel_names = []
        for ch in channels_meta:
            channel_names.append(ch.get("ShortName"))
        self.channel_names = channel_names

        channel_colors = []
        for ch in channels_meta:
            channel_colors.append(ch.get("Color"))
        self.channel_colors = channel_colors

        self.base_layer_idx = 0
        if init_pyramid:
            self._pyramid = self.pyramid

    def _prepare_dask_image(self):
        ch_dim = self.im_dims[1:] if not self.is_rgb else self.im_dims[:2]
        chunks = ((1,) * self.n_ch, (ch_dim[0],), (ch_dim[1],))
        d_image = da.map_blocks(
            self.read_single_channel,
            chunks=chunks,
            dtype=self.im_dtype,
            meta=np.array((), dtype=self.im_dtype),
        )
        return d_image

    def get_dask_pyr(self):
        """Get instance of Dask pyramid."""
        return self.fh.zarr_pyramidalize_czi(zarr.storage.TempStore())

    def _get_image_info(self):
        # if RGB need to get 0
        if self.fh.shape[-1] > 1:
            ch_dim_idx = self.fh.axes.index("0")
        else:
            ch_dim_idx = self.fh.axes.index("C")
        y_dim_idx = self.fh.axes.index("Y")
        x_dim_idx = self.fh.axes.index("X")
        if self.fh.shape[-1] > 1:
            im_dims = np.array(self.fh.shape)[[y_dim_idx, x_dim_idx, ch_dim_idx]]
        else:
            im_dims = np.array(self.fh.shape)[[ch_dim_idx, y_dim_idx, x_dim_idx]]
        return ch_dim_idx, y_dim_idx, x_dim_idx, im_dims, self.fh.dtype

    def read_single_channel(self, block_id: ty.Tuple[int, ...]):
        """Read a single channel from CZI file."""
        channel_idx = block_id[0]
        if self.is_rgb is False:
            image = self.fh.sub_asarray(
                channel_idx=[channel_idx],
            )
        else:
            image = self.fh.sub_asarray_rgb(channel_idx=[channel_idx], greyscale=False)

        return np.expand_dims(np.squeeze(image), axis=0)
