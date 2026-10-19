import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
from scipy.ndimage import (
    gaussian_filter,
    zoom,
)


def generate_cool_volume(
    shape=(128, 256, 256),
    seed=42,
    n_blobs=35,
    n_filaments=20,
):
    """
    Generate a synthetic 3D microscopy-like volume.

    Parameters
    ----------
    shape : tuple
        Volume shape as (z, y, x).
    seed : int
        Random seed.
    n_blobs : int
        Number of Gaussian-ish blobs.
    n_filaments : int
        Number of elongated structures.

    Returns
    -------
    volume : np.ndarray
        Float32 array with values in [0, 1].
    """
    rng = np.random.default_rng(seed)

    z_size, y_size, x_size = shape

    # Coordinate grids normalized to [-1, 1]
    z = np.linspace(-1, 1, z_size, dtype=np.float32)
    y = np.linspace(-1, 1, y_size, dtype=np.float32)
    x = np.linspace(-1, 1, x_size, dtype=np.float32)

    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

    volume = np.zeros(shape, dtype=np.float32)

    # ------------------------------------------------------------
    # 1. Random 3D Gaussian blobs
    # ------------------------------------------------------------

    for _ in range(n_blobs):
        cz, cy, cx = rng.uniform(-0.8, 0.8, size=3)

        sz = rng.uniform(0.03, 0.15)
        sy = rng.uniform(0.03, 0.15)
        sx = rng.uniform(0.03, 0.15)

        amplitude = rng.uniform(0.3, 1.2)

        blob = amplitude * np.exp(-(((Z - cz) / sz) ** 2 + ((Y - cy) / sy) ** 2 + ((X - cx) / sx) ** 2) / 2)

        volume += blob.astype(np.float32)

    # ------------------------------------------------------------
    # 2. Add some shell / membrane-like structures
    # ------------------------------------------------------------

    for _ in range(6):
        cz, cy, cx = rng.uniform(-0.5, 0.5, size=3)

        rz = rng.uniform(0.15, 0.45)
        ry = rng.uniform(0.15, 0.45)
        rx = rng.uniform(0.15, 0.45)

        radius = np.sqrt(((Z - cz) / rz) ** 2 + ((Y - cy) / ry) ** 2 + ((X - cx) / rx) ** 2)

        thickness = rng.uniform(0.02, 0.06)

        shell = np.exp(-((radius - 1.0) ** 2) / (2 * thickness**2))

        volume += shell.astype(np.float32) * rng.uniform(0.2, 0.7)

    # ------------------------------------------------------------
    # 3. Random filament-like structures
    # ------------------------------------------------------------

    filament_mask = np.zeros(shape, dtype=np.float32)

    for _ in range(n_filaments):
        # random start
        p = np.array(
            [
                rng.uniform(10, z_size - 10),
                rng.uniform(10, y_size - 10),
                rng.uniform(10, x_size - 10),
            ]
        )

        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        length = rng.integers(30, 120)

        for _i in range(length):
            # slowly curve the filament
            direction += rng.normal(scale=0.05, size=3)
            direction /= np.linalg.norm(direction)

            p += direction

            zi, yi, xi = np.round(p).astype(int)

            if 0 <= zi < z_size and 0 <= yi < y_size and 0 <= xi < x_size:
                filament_mask[zi, yi, xi] = 1

    filament_mask = gaussian_filter(
        filament_mask,
        sigma=(1.5, 2.5, 2.5),
    )

    if filament_mask.max() > 0:
        filament_mask /= filament_mask.max()

    volume += filament_mask * 0.8

    # ------------------------------------------------------------
    # 4. Add correlated / cloudy noise
    # ------------------------------------------------------------

    noise = rng.normal(size=shape).astype(np.float32)

    cloudy_noise = gaussian_filter(
        noise,
        sigma=(4, 8, 8),
    )

    cloudy_noise -= cloudy_noise.min()

    if cloudy_noise.max() > 0:
        cloudy_noise /= cloudy_noise.max()

    volume += cloudy_noise * 0.25

    # Fine-grained detector noise
    volume += rng.normal(
        scale=0.025,
        size=shape,
    ).astype(np.float32)

    # ------------------------------------------------------------
    # 5. Non-linear intensity transformation
    # ------------------------------------------------------------

    volume = np.clip(volume, 0, None)

    # percentile normalization avoids a few bright blobs
    # dominating everything
    p1, p99 = np.percentile(volume, [1, 99.8])

    volume = (volume - p1) / (p99 - p1)
    volume = np.clip(volume, 0, 1)

    # Gamma gives it a nice microscopy appearance
    volume = volume**0.7

    return volume.astype(np.float32)


def write_ome_zarr(
    volume,
    output_path="cool_volume.ome.zarr",
):
    """Write a ZYX volume as a multiscale OME-Zarr image."""
    # Add channel dimension:
    #
    # ZYX -> CZYX
    #
    data = volume[np.newaxis, ...]

    store = parse_url(
        output_path,
        mode="w",
    ).store

    root = zarr.group(store=store)

    axes = [
        {
            "name": "c",
            "type": "channel",
        },
        {
            "name": "z",
            "type": "space",
            "unit": "micrometer",
        },
        {
            "name": "y",
            "type": "space",
            "unit": "micrometer",
        },
        {
            "name": "x",
            "type": "space",
            "unit": "micrometer",
        },
    ]

    # Example voxel size:
    #
    # z = 2 µm
    # y = 0.5 µm
    # x = 0.5 µm
    #
    coordinate_transformations = [
        [
            {
                "type": "scale",
                "scale": [
                    1.0,
                    2.0,
                    0.5,
                    0.5,
                ],
            }
        ]
    ]

    write_image(
        image=data,
        group=root,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options={
            "chunks": (1, 16, 128, 128),
        },
    )

    print(f"Wrote OME-Zarr to: {output_path}")


if __name__ == "__main__":
    volume = generate_cool_volume(
        shape=(128, 256, 256),
        seed=42,
        n_blobs=40,
        n_filaments=25,
    )

    write_ome_zarr(
        volume,
        "cool_volume.ome.zarr",
    )
