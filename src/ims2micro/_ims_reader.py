"""IMS reader."""
from pathlib import Path
import sqlite3
from koyo.typing import PathLike
import numpy as np
import typing as ty


class IMSWrapper:
    """Wrapper around IMS data."""

    def __init__(self, regions: ty.Optional[np.ndarray], x: np.ndarray, y: np.ndarray, resolution: float = 1.0):
        self.regions = regions
        self.x = x
        self.y = y
        self.xmin, self.xmax = np.min(x), np.max(x)
        self.ymin, self.ymax = np.min(y), np.max(y)
        self.image_shape = (self.ymax - self.ymin + 1, self.xmax - self.xmin + 1)
        self.resolution = resolution

    def image(self) -> ty.Dict[str, np.ndarray]:
        """Return IMS image."""
        array = np.full(self.image_shape, np.nan)
        array[self.y - self.ymin, self.x - self.xmin] = np.random.randint(5, 255, size=len(self.x))
        return {"name": "IMS", "data": array, "blending": "additive", "channel_axis": None}


def read_imaging(path: PathLike):
    """Read imaging data."""
    path = Path(path)
    if path.suffix.lower() in [".tsf", ".tdf"]:
        return IMSWrapper(*_read_tsf_tdf_coordinates(path))
    elif path.suffix.lower() == ".imzml":
        return IMSWrapper(*_read_imzml_coordinates(path))
    elif path.suffix.lower() == ".h5":
        raise ValueError("Reading of .h5 files is not supported yet")
    raise ValueError(f"Unsupported file format: {path.suffix}")


def _read_tsf_tdf_coordinates(path: PathLike) -> ty.Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Read coordinates from TSF file."""
    path = Path(path)
    assert path.suffix in [".tsf", ".tdf"], "Only .tsf and .tdf files are supported"

    # get reader
    conn = sqlite3.connect(path)

    try:
        cursor = conn.execute("SELECT SpotSize FROM MaldiFrameLaserInfo")
        resolution = float(cursor.fetchone()[0])
    except sqlite3.OperationalError:
        resolution = 1.0

    cursor = conn.cursor()
    cursor.execute("SELECT Frame, RegionNumber, XIndexPos, YIndexPos FROM MaldiFrameInfo")

    frame_index_position = np.array(cursor.fetchall())

    regions = frame_index_position[:, 1]
    x = frame_index_position[:, 2]
    x = x - np.min(x)  # minimized
    y = frame_index_position[:, 3]
    y = y - np.min(y)  # minimized
    return regions, x, y, resolution


def _read_imzml_coordinates(path: PathLike) -> ty.Tuple[ty.Optional[np.ndarray], np.ndarray, np.ndarray]:
    """Read coordinates from imzML file."""
    from imzy import get_reader

    path = Path(path)
    assert path.suffix == ".imzML", "Only .imzML files are supported"

    # get reader
    reader = get_reader(path)
    x = reader.x_coordinates
    x = x - np.min(x)  # minimized
    y = reader.y_coordinates
    y = y - np.min(y)  # minimized
    return None, x, y
