import multiprocessing
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import dask.array as da
import numpy as np
import zarr
from czifile import CziFile as _CziFile
from tifffile import create_output

from image2image.readers.utilities import compute_sub_res


class CziFile(_CziFile):
    """Sub-class of CziFile with added functionality to only read certain channels."""

    def sub_asarray(
        self,
        resize=True,
        order=0,
        out=None,
        max_workers=None,
        channel_idx=None,
        as_uint8=False,
        zarr_fp=None,
        ds_factor=1,
    ):
        """Return image data from file(s) as numpy array.

        Parameters
        ----------
        resize: bool
            If True (default), resize sub/supersampled subblock data.
        order: int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out: numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers: int
            Maximum number of threads to read and decode subblock data. By default up to half the CPU cores are used.
        channel_idx: int or list of int
            The indices of the channels to extract
        as_uint8: bool
            byte-scale image data to np.uint8 data type
        zarr_fp: str
            path to zarr file to save data to
        ds_factor: int
            Downsampling factor to apply to data

        Parameters
        ----------
        out:np.ndarray
            image read with selected parameters as np.ndarray
        """
        out_shape = list(self.shape)
        start = list(self.start)

        ch_dim_idx = self.axes.index("C")

        if channel_idx is not None:
            if isinstance(channel_idx, int):
                channel_idx = [channel_idx]

            if out_shape[ch_dim_idx] == 1:
                channel_idx = None

            else:
                out_shape[ch_dim_idx] = len(channel_idx)
                min_ch_seq = {}
                for idx, i in enumerate(channel_idx):
                    min_ch_seq.update({i: idx})

        if as_uint8 is True:
            out_dtype = np.uint8
        else:
            out_dtype = self.dtype

        if zarr_fp is not None:
            if ds_factor > 1:
                out_shape[3] = out_shape[3] // ds_factor
                out_shape[4] = out_shape[4] // ds_factor
                out_shape = tuple(out_shape)
                start[3] = start[3] // ds_factor
                start[4] = start[4] // ds_factor
            rgb_chunk = self.shape[-1] if self.shape[-1] > 2 else 1
            root = zarr.open_group(zarr_fp, mode="a")
            pyramid_seq = str(int(np.log2(ds_factor)))
            chunking = (1, 1, 1, 1024, 1024, rgb_chunk)
            out = root.create_dataset(
                pyramid_seq,
                shape=tuple(out_shape),
                chunks=chunking,
                dtype=out_dtype,
                overwrite=True,
            )

        elif out is None:
            out = create_output(None, tuple(out_shape), out_dtype)

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        def func(directory_entry, resize=resize, order=order, start=start, out=out):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            dvstart = list(directory_entry.start)
            czi_c_idx = [de.dimension for de in subblock.dimension_entries].index("C")
            subblock_ch_idx = subblock.dimension_entries[czi_c_idx].start
            if channel_idx is not None:
                if subblock_ch_idx in channel_idx:
                    subblock.dimension_entries[czi_c_idx].start
                    tile = subblock.data(resize=resize, order=order)
                    dvstart[ch_dim_idx] = min_ch_seq.get(subblock_ch_idx)
                else:
                    return
            else:
                tile = subblock.data(resize=resize, order=order)

            if ds_factor > 1:
                tile = np.squeeze(tile)

                w = tile.shape[0] // ds_factor
                h = tile.shape[0] // ds_factor
                tile_ds = cv2.resize(np.squeeze(tile), dsize=(w, h), interpolation=cv2.INTER_LINEAR)

                tile_ds = np.reshape(tile_ds, (1, 1, 1, tile_ds.shape[0], tile_ds.shape[1], rgb_chunk))
                tile = tile_ds
                dvstart[3] = dvstart[3] // ds_factor
                dvstart[4] = dvstart[4] // ds_factor

            if as_uint8 is True:
                tile = (tile / 256).astype("uint8")

            index = tuple(slice(i - j, i - j + k) for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape))

            try:
                out[index] = tile
            except ValueError as e:
                error = e
                corr_shape = str(error).split("shape ")[1].split(", got")[0].strip("(").strip(")")
                corr_shape.split(",")
                cor_shape = tuple([slice(int(t)) for t in corr_shape.split(",")])
                tile = tile[cor_shape]
                index = tuple(slice(i - j, i - j + k) for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape))
                out[index] = tile

        if max_workers > 1:
            self._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:
            for directory_entry in self.filtered_subblock_directory:
                func(directory_entry)

        if hasattr(out, "flush"):
            out.flush()
        return out

    def sub_asarray_rgb(
        self,
        resize=True,
        order=0,
        out=None,
        max_workers=None,
        channel_idx=None,
        as_uint8=False,
        greyscale=False,
    ):
        """Image data from file(s) as numpy array.

        Parameters
        ----------
        resize : bool
            If True (default), resize sub/supersampled subblock data.
        order : int
            The order of spline interpolation used to resize sub/supersampled
            subblock data. Default is 0 (nearest neighbor).
        out : numpy.ndarray, str, or file-like object; optional
            Buffer where image data will be saved.
            If numpy.ndarray, a writable array of compatible dtype and shape.
            If str or open file, the file name or file object used to
            create a memory-map to an array stored in a binary file on disk.
        max_workers : int
            Maximum number of threads to read and decode subblock data.
            By default up to half the CPU cores are used.
        channel_idx : int or list of int
            The indices of the channels to extract
        as_uint8 : bool
            byte-scale image data to np.uint8 data type
        greyscale : bool
            return greyscale image data

        Parameters
        ----------
        out:np.ndarray
            image read with selected parameters as np.ndarray
        """
        out_shape = list(self.shape)
        start = list(self.start)
        ch_dim_idx = self.axes.index("0")

        if channel_idx is not None:
            if isinstance(channel_idx, int):
                channel_idx = [channel_idx]
            out_shape[ch_dim_idx] = len(channel_idx)

        if greyscale is True:
            out_shape[ch_dim_idx] = 1

        if as_uint8 is True:
            out_dtype = np.uint8
        else:
            out_dtype = self.dtype

        if out is None:
            out = create_output(None, tuple(out_shape), out_dtype)

        if max_workers is None:
            max_workers = multiprocessing.cpu_count() - 1

        def func(directory_entry, resize=resize, order=order, start=start, out=out):
            """Read, decode, and copy subblock data."""
            subblock = directory_entry.data_segment()
            dvstart = list(directory_entry.start)
            tile = subblock.data(resize=resize, order=order)

            if greyscale is True:
                tile = czi_tile_grayscale(tile)

            if channel_idx is not None:
                tile = tile[:, :, :, :, :, channel_idx]

            index = tuple(slice(i - j, i - j + k) for i, j, k in zip(tuple(dvstart), tuple(start), tile.shape))

            try:
                out[index] = tile
            except ValueError as e:
                warnings.warn(str(e), stacklevel=1)

        if max_workers > 1:
            self._fh.lock = True
            with ThreadPoolExecutor(max_workers) as executor:
                executor.map(func, self.filtered_subblock_directory)
            self._fh.lock = None
        else:
            for directory_entry in self.filtered_subblock_directory:
                func(directory_entry)

        if hasattr(out, "flush"):
            out.flush()
        return out

    def zarr_pyramidalize_czi(self, zarr_fp):
        """Create a pyramidal zarr store from a CZI file."""
        dask_pyr = []
        root = zarr.open_group(zarr_fp, mode="a")

        root.attrs["axes_names"] = list(self.axes)
        root.attrs["orig_shape"] = list(self.shape)
        all_axes = list(self.axes)
        yx_dims = np.where(np.isin(all_axes, ["Y", "X"]) == 1)[0].tolist()
        yx_shape = np.array(self.shape[slice(yx_dims[0], yx_dims[1] + 1)])

        ds = 1
        while np.min(yx_shape) // 2**ds >= 512:
            ds += 1

        self.sub_asarray(zarr_fp=zarr_fp, resize=True, order=0, ds_factor=1, max_workers=4)
        zarray = da.squeeze(da.from_zarr(zarr.open(zarr_fp)[0]))
        dask_pyr.append(da.squeeze(zarray))
        for ds_factor in range(1, ds):
            zres = zarr.storage.TempStore()
            rgb_chunk = self.shape[-1] if self.shape[-1] > 2 else 1
            is_rgb = True if rgb_chunk > 1 else False

            sub_res_image = compute_sub_res(zarray, ds_factor, 512, is_rgb, self.dtype)

            da.to_zarr(sub_res_image, zres, component="0")

            dask_pyr.append(da.squeeze(da.from_zarr(zres, component="0")))

        return dask_pyr


def czi_tile_grayscale(rgb_image):
    """
    convert RGB image data to greyscale.

    Parameters
    ----------
    rgb_image: np.ndarray
        image data

    Returns
    -------
    image:np.ndarray
        returns 8-bit greyscale image for 24-bit RGB image
    """
    result = (
        (rgb_image[..., 0] * 0.2125).astype(np.uint8)
        + (rgb_image[..., 1] * 0.7154).astype(np.uint8)
        + (rgb_image[..., 2] * 0.0721).astype(np.uint8)
    )

    return np.expand_dims(result, axis=-1)
