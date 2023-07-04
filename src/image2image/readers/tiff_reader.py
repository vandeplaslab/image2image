"""Tiff file wrapper.

Copied from:
https://github.com/NHPatterson/napari-imsmicrolink/blob/master/src/napari_imsmicrolink/data/tifffile_reader.py
"""
import warnings
from pathlib import Path

from ome_types import from_xml
from tifffile import TiffFile

from image2image.readers.base_reader import BaseImageReader
from image2image.readers.tiff_meta import (
    ometiff_ch_names,
    ometiff_xy_pixel_sizes,
    svs_xy_pixel_sizes,
    tifftag_xy_pixel_sizes,
)
from image2image.readers.utilities import (
    get_tifffile_info,
    guess_rgb,
    tf_zarr_read_single_ch,
    tifffile_to_dask,
)


class TiffImageReader(BaseImageReader):
    """TIFF image wrapper."""

    def __init__(self, path, init_pyramid: bool = True):
        super().__init__(path)
        self.fh = TiffFile(self.path)
        self.reader = "tifffile"

        self.im_dims, self.im_dtype, self.largest_series = self._get_image_info()

        self.im_dims = tuple(self.im_dims)
        self.is_rgb = guess_rgb(self.im_dims)
        self.n_channels = self.im_dims[2] if self.is_rgb else self.im_dims[0]

        self.resolution = self._get_im_res()
        self.channel_names = self._get_channel_names()
        self.channel_colors = None

        if init_pyramid:
            self._pyramid = self.pyramid

    def get_dask_pyr(self):
        """Get instance of Dask pyramid."""
        d_pyr = tifffile_to_dask(self.path, self.largest_series)
        if self.is_rgb and guess_rgb(d_pyr[0].shape) is True:
            d_pyr[0] = d_pyr[0].rechunk((2048, 2048, 1))
        elif len(d_pyr[0].shape) > 2:
            d_pyr[0] = d_pyr[0].rechunk((1, 2048, 2048))
        else:
            d_pyr[0] = d_pyr[0].rechunk((2048, 2048))
        return d_pyr

    def _get_im_res(self):
        if Path(self.path).suffix.lower() in [".scn", ".ndpi"]:
            return tifftag_xy_pixel_sizes(
                self.fh,
                self.largest_series,
                0,
            )[0]
        elif Path(self.path).suffix.lower() in [".svs"]:
            return svs_xy_pixel_sizes(
                self.fh,
                self.largest_series,
                0,
            )[0]
        elif self.fh.ome_metadata:
            return ometiff_xy_pixel_sizes(
                from_xml(self.fh.ome_metadata, parser="lxml"),
                self.largest_series,
            )[0]
        else:
            try:
                return tifftag_xy_pixel_sizes(
                    self.fh,
                    self.largest_series,
                    0,
                )[0]
            except KeyError:
                warnings.warn(
                    "Unable to parse pixel resolution information from file" " defaulting to 1",
                    stacklevel=2,
                )
                return 1.0

    def _get_channel_names(self):
        if self.fh.ome_metadata:
            channel_names = ometiff_ch_names(from_xml(self.fh.ome_metadata), self.largest_series)
        else:
            channel_names = []
            if self.is_rgb:
                channel_names.append("C01 - RGB")
            else:
                for idx, _ch in enumerate(range(self.n_channels)):
                    channel_names.append(f"C{str(idx + 1).zfill(2)}")

        return channel_names

    def _get_image_info(self):
        if len(self.fh.series) > 1:
            warnings.warn(
                "The tiff contains multiple series, " "the largest series will be read by default",
                stacklevel=2,
            )

        im_dims, im_dtype, largest_series = get_tifffile_info(self.path)

        return im_dims, im_dtype, largest_series

    def read_single_channel(self, channel_idx: int):
        """Read data from a single channel."""
        if channel_idx > (self.n_channels - 1):
            warnings.warn(
                "channel_idx exceeds number of channels, reading channel at channel_idx == 0",
                stacklevel=2,
            )
            channel_idx = 0
        image = tf_zarr_read_single_ch(self.path, channel_idx, self.is_rgb)
        return image
