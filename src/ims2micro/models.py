"""Registration model."""
import typing as ty
from datetime import datetime
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from pydantic import BaseModel, validator
from skimage.transform import ProjectiveTransform

if ty.TYPE_CHECKING:
    from ims2micro._ims_reader import IMSWrapper
    from ims2micro._micro_reader import MicroWrapper


class DataWrapper:
    """Base class for IMS and microscopy wrappers."""

    data: ty.Dict[str, ty.Optional[np.ndarray]]
    paths: ty.List[Path]

    def __init__(self, data: ty.Optional[ty.Dict[str, ty.Optional[ty.Union[np.ndarray, ty.Any]]]] = None):
        self.data = data or {}
        self.paths = []

    def add(self, key: str, array: ty.Union[np.ndarray, ty.Any]):
        """Add data to wrapper."""
        self.data[key] = array

    def add_path(self, path: PathLike):
        """Add path to wrapper."""
        self.paths.append(Path(path))

    def remove_path(self, path: PathLike):
        """Remove path from wrapper."""
        path = Path(path)
        if path in self.paths:
            self.paths.remove(path)
        path = str(path.name)
        if path in self.data:
            del self.data[path]

    def is_loaded(self, path: PathLike):
        """Check if path is loaded."""
        return Path(path) in self.paths

    @property
    def n_channels(self) -> int:
        """Return number of channels."""
        return len(self.channel_names())

    def channel_names(self, view_type: ty.Optional[str] = None) -> ty.List[str]:
        """Return list of channel names."""
        raise NotImplementedError("Must implement method")

    def map_channel_to_index(self, dataset: str, channel_name: str) -> int:
        """Map channel name to index."""
        dataset_to_channel_map = {}
        for name in self.channel_names():
            dataset, channel = name.split(" | ")
            dataset_to_channel_map.setdefault(dataset, []).append(channel)
        return dataset_to_channel_map[dataset].index(channel_name)

    def image_iter(self, view_type: ty.Optional[str] = None):
        """Iterator to add channels."""
        raise NotImplementedError("Must implement method")

    def channel_image_iter(self, view_type: ty.Optional[str] = None) -> ty.Iterator[ty.Tuple[str, np.ndarray]]:
        """Iterator of channel name + image."""
        yield from zip(self.channel_names(view_type), self.image_iter(view_type))


class DataModel(BaseModel):
    """Base model."""

    paths: ty.Optional[ty.List[Path]] = None
    resolution: float = 1.0
    reader: ty.Optional[DataWrapper] = None

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    @validator("paths", pre=True, allow_reuse=True)
    def _validate_path(value: ty.Union[PathLike, ty.List[PathLike]]) -> ty.List[Path]:
        """Validate path."""
        if isinstance(value, (str, Path)):
            value = [Path(value)]
        value = [Path(path) for path in value]
        assert all(path.exists() for path in value), "Path does not exist."
        return value

    def get_filename(self) -> str:
        """Get representative filename."""
        if self.n_paths == 1:
            return self.paths[0].parent.stem
        return self.paths[-1].parent.stem + "_multiple_files"

    def add_paths(self, path_or_paths: ty.Union[PathLike, ty.List[PathLike]]):
        """Add paths to model."""
        if isinstance(path_or_paths, (str, Path)):
            path_or_paths = [path_or_paths]
        if self.paths is None:
            self.paths = []
        for path in path_or_paths:
            path = Path(path)
            if path not in self.paths:
                self.paths.append(path)

    def load(self):
        """Load data into memory."""
        with MeasureTimer() as timer:
            self.get_reader()
            logger.info(f"Loaded data in {timer()}")
        return self

    def get_reader(self) -> DataWrapper:
        """Read data from file."""
        raise NotImplementedError("Must implement method")

    def channel_names(self) -> ty.List[str]:
        """Return list of channel names."""
        return self.get_reader().channel_names()

    @property
    def n_paths(self) -> int:
        """Return number of paths."""
        return len(self.paths) if self.paths is not None else 0

    def close(self, paths: ty.List[PathLike]):
        """Close certain (or all) paths."""
        if paths is None:
            return
        for path in paths:
            path = Path(path)
            if path in self.paths:
                self.paths.remove(path)
            if self.reader:
                self.reader.remove_path(path)


class ImagingModel(DataModel):
    """IMS model."""

    def get_reader(self) -> "IMSWrapper":
        """Read data from file."""
        from ims2micro._ims_reader import read_imaging

        for path in self.paths:
            if self.reader is None or not self.reader.is_loaded(path):
                self.reader = read_imaging(path, self.reader)
        self.resolution = self.reader.resolution
        return self.reader


class MicroscopyModel(DataModel):
    """IMS model."""

    def get_reader(self) -> "MicroWrapper":
        """Read data from file."""
        from ims2micro._micro_reader import read_microscopy

        for path in self.paths:
            if self.reader is None or not self.reader.is_loaded(path):
                self.reader = read_microscopy(path, self.reader)
        self.resolution = self.reader.resolution
        return self.reader


class Transformation(BaseModel):
    """Temporary object that holds transformation information."""

    # Transformation object
    transform: ProjectiveTransform = None
    # Type of transformation
    transformation_type: str = ""
    # Path to the image
    micro_model: ty.Optional[MicroscopyModel] = None
    ims_model: ty.Optional[ImagingModel] = None
    # Date when the registration was created
    time_created: ty.Optional[datetime] = None
    # Arrays of fixed and moving points
    fixed_points: ty.Optional[np.ndarray] = None
    moving_points: ty.Optional[np.ndarray] = None

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    def __call__(self, coords: np.ndarray):
        """Transform coordinates."""
        return self.transform(coords)

    def inverse(self, coords: np.ndarray):
        """Inverse transformation of coordinates."""
        return self.transform.inverse(coords)

    @property
    def matrix(self):
        """Retrieve transformation array."""
        return self.transform.params

    def compute(self, yx: bool = True, px: bool = True):
        """Compute transformation matrix."""
        from ims2micro.utilities import compute_transform

        moving_points = self.moving_points
        fixed_points = self.fixed_points
        if not yx:
            moving_points = moving_points[:, ::-1]
            fixed_points = fixed_points[:, ::-1]
        if not px:
            moving_points = moving_points * self.micro_model.resolution
            fixed_points = fixed_points * self.ims_model.resolution

        transform = compute_transform(
            moving_points,  # source
            fixed_points,  # destination
            self.transformation_type,
        )
        return transform

    def about(self) -> str:
        """Retrieve information about the model in textual format."""
        info = ""
        if self.transformation_type:
            info += f"Transformation type: {self.transformation_type}"
        transform = self.transform
        if transform:
            if hasattr(transform, "params"):
                info += "\nTransformation matrix:"
                info += f"\n{transform.params}"
            if hasattr(transform, "scale"):
                scale = transform.scale
                scale = (scale, scale) if isinstance(scale, float) else scale
                info += f"\nScale: {scale[0]:.3f}, {scale[1]:.3f}"
            if hasattr(transform, "translation"):
                translation = transform.translation
                translation = (translation, translation) if isinstance(translation, float) else translation
                f"\nTranslation: {translation[0]:.3f}, {translation[1]:.3f}"
            if hasattr(transform, "rotation"):
                rotation = transform.rotation
                info += f"\nRotation: {rotation:.3f}"
        if self.fixed_points is not None:
            info += f"\nNumber of fixed points: {len(self.fixed_points)}"
        if self.moving_points is not None:
            info += f"\nNumber of moving points: {len(self.moving_points)}"
        return info

    def to_dict(self):
        """Convert to dict."""
        return {
            "schema_version": "1.0",
            "time_created": self.time_created.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "fixed_points_yx_px": self.fixed_points.tolist(),  # default
            "fixed_points_yx_um": (self.micro_model.resolution * self.fixed_points).tolist(),
            "moving_points_yx_px": self.moving_points.tolist(),  # default
            "moving_points_yx_um": (self.ims_model.resolution * self.moving_points).tolist(),
            "transformation_type": self.transformation_type,
            "micro_paths": [str(path) for path in self.micro_model.paths],
            "micro_resolution_um": self.micro_model.resolution,
            "ims_paths": [str(path) for path in self.ims_model.paths],
            "ims_resolution_um": self.ims_model.resolution,
            "matrix_yx_px": self.compute(yx=True, px=True).params.tolist(),
            "matrix_yx_um": self.compute(yx=True, px=False).params.tolist(),
            "matrix_xy_px": self.compute(yx=False, px=True).params.tolist(),
            "matrix_xy_um": self.compute(yx=False, px=False).params.tolist(),
            "matrix_yx_px_inv": self.compute(yx=True, px=True)._inv_matrix.tolist(),
            "matrix_yx_um_inv": self.compute(yx=True, px=False)._inv_matrix.tolist(),
            "matrix_xy_px_inv": self.compute(yx=False, px=True)._inv_matrix.tolist(),
            "matrix_xy_um_inv": self.compute(yx=False, px=False)._inv_matrix.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dict."""
        raise NotImplementedError("Must implement method")

    def to_json(self, path: PathLike):
        """Export data as JSON."""
        from koyo.json import write_json_data

        path = Path(path)
        write_json_data(path, self.to_dict())

    @classmethod
    def from_json(cls, path: PathLike):
        """Create from JSON."""
        from koyo.json import read_json_data

        path = Path(path)
        return cls.from_dict(read_json_data(path))

    def to_toml(self, path: PathLike):
        """Export data as TOML."""
        from koyo.toml import write_toml_data

        path = Path(path)
        write_toml_data(path, self.to_dict())

    @classmethod
    def from_toml(cls, path: PathLike):
        """Create from TOML."""
        from koyo.toml import read_toml_data

        path = Path(path)
        return cls.from_dict(read_toml_data(path))

    def to_file(self, path: PathLike):
        """Export data as any supported format."""
        path = Path(path)
        if path.suffix == ".json":
            self.to_json(path)
        elif path.suffix == ".toml":
            self.to_toml(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")

    @classmethod
    def from_file(cls, path: PathLike):
        """Create from file."""
        path = Path(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix == ".toml":
            return cls.from_toml(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")


def load_from_file(
    path: PathLike, micro: bool = True, ims: bool = True, fixed: bool = True, moving: bool = True
) -> ty.Tuple[
    str,
    ty.Optional[ty.List[Path]],
    ty.Optional[ty.List[Path]],
    ty.Optional[np.ndarray],
    ty.Optional[ty.List[Path]],
    ty.Optional[ty.List[Path]],
    ty.Optional[np.ndarray],
]:
    """Load registration from file."""
    path = Path(path)
    if path.suffix not in [".json", ".toml"]:
        raise ValueError(f"Unknown file format: {path.suffix}")

    if path.suffix == ".json":
        from koyo.json import read_json_data

        data = read_json_data(path)
    else:
        from koyo.toml import read_toml_data

        data = read_toml_data(path)

    # ims2micro config
    if "schema_version" in data:
        (
            transformation_type,
            micro_paths,
            micro_paths_missing,
            fixed_points,
            ims_paths,
            ims_paths_missing,
            moving_points,
        ) = _read_ims2micro_config(data)
    # imsmicrolink config
    elif "Project name" in data:
        (
            transformation_type,
            micro_paths,
            micro_paths_missing,
            fixed_points,
            ims_paths,
            ims_paths_missing,
            moving_points,
        ) = _read_imsmicrolink_config(data)
    else:
        raise ValueError(f"Unknown file format: {path.suffix}")

    # apply config
    micro_paths, micro_paths_missing = (micro_paths, micro_paths_missing) if micro else (None, None)
    ims_paths, ims_paths_missing = (ims_paths, ims_paths_missing) if ims else (None, None)
    fixed_points = fixed_points if fixed else None
    moving_points = moving_points if moving else None
    return (
        transformation_type,
        micro_paths,
        micro_paths_missing,
        fixed_points,
        ims_paths,
        ims_paths_missing,
        moving_points,
    )


def _get_paths(paths: ty.List[PathLike]):
    _paths_exist, _paths_missing = [], []
    for path in paths:
        path = Path(path)
        if path.exists():
            _paths_exist.append(path)
        else:
            _paths_missing.append(path)
    if not _paths_exist:
        _paths_exist = None
    return _paths_exist, _paths_missing


def _read_ims2micro_config(config: ty.Dict):
    """Read ims2micro configuration file."""
    # read important fields
    micro_paths, micro_paths_missing = _get_paths(config["micro_paths"])
    ims_paths, ims_paths_missing = _get_paths(config["ims_paths"])
    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        micro_paths,
        micro_paths_missing,
        fixed_points,
        ims_paths,
        ims_paths_missing,
        moving_points,
    )


def _read_imsmicrolink_config(config: ty.Dict):
    """Read imsmicrolink configuration file."""
    micro_paths, micro_paths_missing = _get_paths([config["PostIMS microscopy image"]])  # need to be a list
    ims_paths, ims_paths_missing = _get_paths(config["Pixel Map Datasets Files"])
    fixed_points = np.array(config["PAQ microscopy points (xy, px)"])
    moving_points = np.array(config["IMS pixel map points (xy, px)"])
    transformation_type = "Affine"
    return (
        transformation_type,
        micro_paths,
        micro_paths_missing,
        fixed_points,
        ims_paths,
        ims_paths_missing,
        moving_points,
    )
