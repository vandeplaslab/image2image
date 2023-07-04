"""Registration model."""
import typing as ty
from datetime import datetime
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import ArrayLike, PathLike
from loguru import logger
from pydantic import BaseModel as _BaseModel
from pydantic import validator
from skimage.transform import ProjectiveTransform

from image2image._reader import ImageWrapper, sanitize_path

if ty.TYPE_CHECKING:
    from image2image.readers.base_reader import BaseImageReader


class BaseModel(_BaseModel):
    """Base model."""

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    def update(self, **kwargs):
        """Update transformation."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        """Convert to dict."""
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

    @classmethod
    def from_dict(cls, data: dict):
        """Create from dict."""
        raise NotImplementedError("Must implement method")

    @classmethod
    def from_file(cls, path: PathLike):
        """Create from file."""
        path = Path(path)
        if path.suffix == ".json":
            return cls.from_json(path)
        elif path.suffix == ".toml":
            return cls.from_toml(path)
        raise ValueError(f"Unknown file format: '{path.suffix}'")


class DataModel(BaseModel):
    """Base model."""

    paths: ty.Optional[ty.List[Path]] = None
    just_added: ty.Optional[ty.List[Path]] = None
    resolution: float = 1.0
    wrapper: ty.Optional[ImageWrapper] = None
    is_fixed: bool = False

    # noinspection PyMethodFirstArgAssignment,PyMethodParameters
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
        if self.n_paths == 0:
            return "no_files"
        elif self.n_paths == 1:
            return self.paths[0].parent.stem
        return self.paths[-1].parent.stem + "_multiple_files"

    def add_paths(self, path_or_paths: ty.Union[PathLike, ty.List[PathLike]]):
        """Add paths to model."""
        if isinstance(path_or_paths, (str, Path)):
            path_or_paths = [path_or_paths]
        if self.paths is None:
            self.paths = []
        just_added = []
        for path in path_or_paths:
            path = sanitize_path(path)
            if path not in self.paths:
                self.paths.append(path)
                just_added.append(path)
                logger.trace(f"Added '{path}' to model paths.")
        self.just_added = just_added

    def remove_paths(self, path_or_paths: ty.Union[PathLike, ty.List[PathLike]]):
        """Remove paths."""
        if path_or_paths is None:
            return
        if isinstance(path_or_paths, (str, Path)):
            path_or_paths = [path_or_paths]
        for path in path_or_paths:
            path = sanitize_path(path)
            if path in self.paths:
                self.paths.remove(path)
                logger.trace(f"Removed '{path}' from model paths.")
            if self.wrapper:
                self.wrapper.remove_path(path)
                logger.trace(f"Removed '{path}' from reader.")
            if self.just_added and path in self.just_added:
                self.just_added.remove(path)
                logger.trace(f"Removed '{path}' from just_added.")
        # remove wrapper
        if not self.paths:
            self.wrapper = None

    def load(self):
        """Load data into memory."""
        with MeasureTimer() as timer:
            self.get_wrapper()
            logger.info(f"Loaded data in {timer()}")
        return self

    def get_wrapper(self) -> ty.Optional["ImageWrapper"]:
        """Read data from file."""
        from image2image._reader import read_image

        if self.paths is None:
            return None

        just_added = []
        for path in self.paths:
            if self.wrapper is None or not self.wrapper.is_loaded(path):
                self.wrapper = read_image(path, self.wrapper, self.is_fixed)
                just_added.append(path)
        if self.wrapper:
            self.resolution = self.wrapper.resolution
        if just_added:
            self.just_added = just_added
        return self.wrapper

    def get_reader(self, path: PathLike) -> ty.Optional["BaseImageReader"]:
        """Get reader for the path."""
        path = Path(path)
        wrapper = self.get_wrapper()
        if wrapper:
            return wrapper.data[path.name]
        return None

    def get_extractable_paths(self) -> ty.List[Path]:
        """Get a list of paths which are extractable."""
        paths = []
        wrapper = self.get_wrapper()
        if wrapper:
            for path, reader in wrapper.path_reader_iter():
                if hasattr(reader, "allow_extraction") and reader.allow_extraction:
                    paths.append(path)
        return paths

    def channel_names(self) -> ty.List[str]:
        """Return list of channel names."""
        wrapper = self.get_wrapper()
        if wrapper:
            return wrapper.channel_names()
        return []

    @property
    def n_paths(self) -> int:
        """Return number of paths."""
        return len(self.paths) if self.paths is not None else 0

    def to_file(self, path: PathLike):
        """Save model to file."""
        path = Path(path)
        if path.suffix == ".json":
            self.to_json(path)
        elif path.suffix == ".toml":
            self.to_toml(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        logger.info(f"Exported to '{path}'")

    def to_dict(self) -> ty.Dict:
        """Return dictionary of values to export."""
        return {
            "schema_version": "1.0",
            "images": [
                {"path": str(path), "matrix_yx_px": reader.transform.tolist()}
                for path, reader in self.get_wrapper().path_reader_iter()
            ],
        }


class Transformation(BaseModel):
    """Temporary object that holds transformation information."""

    # Transformation object
    transform: ty.Optional[ProjectiveTransform] = None
    # Type of transformation
    transformation_type: str = ""
    # Path to the image
    fixed_model: ty.Optional[DataModel] = None
    moving_model: ty.Optional[DataModel] = None
    # Date when the registration was created
    time_created: ty.Optional[datetime] = None
    # Arrays of fixed and moving points
    fixed_points: ty.Optional[np.ndarray] = None
    moving_points: ty.Optional[np.ndarray] = None

    def is_valid(self):
        """Returns True if the transformation is valid."""
        return self.transform is not None

    def clear(self):
        """Clear transformation."""
        self.transform = None
        self.transformation_type = ""
        self.fixed_model = None
        self.moving_model = None
        self.time_created = None
        self.fixed_points = None
        self.moving_points = None

    def __call__(self, coords: np.ndarray):
        """Transform coordinates."""
        return self.transform(coords)

    def inverse(self, coords: np.ndarray):
        """Inverse transformation of coordinates."""
        return self.transform.inverse(coords)

    @property
    def matrix(self):
        """Retrieve the transformation array."""
        return self.transform.params

    def compute(self, yx: bool = True, px: bool = True):
        """Compute transformation matrix."""
        from image2image.utilities import compute_transform

        moving_points = self.moving_points
        fixed_points = self.fixed_points
        if not yx:
            moving_points = moving_points[:, ::-1]
            fixed_points = fixed_points[:, ::-1]
        if not px:
            moving_points = moving_points * self.fixed_model.resolution
            fixed_points = fixed_points * self.moving_model.resolution

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
            "schema_version": "1.2",
            "time_created": self.time_created.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "fixed_points_yx_px": self.fixed_points.tolist(),  # default
            "fixed_points_yx_um": (self.fixed_model.resolution * self.fixed_points).tolist(),
            "moving_points_yx_px": self.moving_points.tolist(),  # default
            "moving_points_yx_um": (self.moving_model.resolution * self.moving_points).tolist(),
            "transformation_type": self.transformation_type,
            "fixed_paths": [
                {
                    "path": str(path),
                    "resolution_um": resolution,
                }
                for (path, resolution) in self.fixed_model.path_resolution_iter()
            ],
            "moving_paths": [
                {
                    "path": str(path),
                    "resolution_um": resolution,
                }
                for (path, resolution) in self.moving_model.path_resolution_iter()
            ],
            "matrix_yx_px": self.compute(yx=True, px=True).params.tolist(),
            "matrix_yx_um": self.compute(yx=True, px=False).params.tolist(),
            "matrix_xy_px": self.compute(yx=False, px=True).params.tolist(),
            "matrix_xy_um": self.compute(yx=False, px=False).params.tolist(),
            "matrix_yx_px_inv": self.compute(yx=True, px=True)._inv_matrix.tolist(),
            "matrix_yx_um_inv": self.compute(yx=True, px=False)._inv_matrix.tolist(),
            "matrix_xy_px_inv": self.compute(yx=False, px=True)._inv_matrix.tolist(),
            "matrix_xy_um_inv": self.compute(yx=False, px=False)._inv_matrix.tolist(),
        }

    def to_file(self, path: PathLike):
        """Export data as any supported format."""
        path = Path(path)
        if path.suffix == ".json":
            self.to_json(path)
        elif path.suffix == ".toml":
            self.to_toml(path)
        elif path.suffix == ".xml":
            self.to_xml(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        logger.info(f"Exported to '{path}'")

    def to_xml(self, path: PathLike):
        """Export dat aas fusion file."""
        from image2image.utilities import write_xml_registration

        affine = self.compute(yx=False, px=True).params
        affine = affine.flatten("F").reshape(3, 3)
        write_xml_registration(path, affine)


class TransformModel(BaseModel):
    """Model containing transformation data."""

    transforms: ty.Optional[ty.Dict[Path, np.ndarray]] = None

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    @property
    def name_to_path_map(self):
        """Returns dictionary that maps transform name to path."""
        if self.transforms is None:
            return {}

        mapping = {}
        for name in self.transforms:
            mapping[name.name] = name
            mapping[Path(name.name)] = name
            mapping[name] = name
        return mapping

    def add_transform(self, name_or_path: PathLike, matrix: ArrayLike):
        """Add transformation matrix."""
        if self.transforms is None:
            self.transforms = {}

        path = Path(name_or_path)
        matrix = np.asarray(matrix)
        assert matrix.shape == (3, 3), "Expected (3, 3) matrix"
        self.transforms[path] = matrix
        logger.info(f"Added '{path.name}' to list of transformations")

    def remove_transform(self, name_or_path: PathLike):
        """Remove transformation matrix."""
        if self.transforms is None:
            return

        name_or_path = Path(name_or_path)
        if name_or_path in self.transforms:
            del self.transforms[name_or_path]

    def get_matrix(self, name_or_path: PathLike):
        """Get transformation matrix."""
        if self.transforms is None:
            return None

        name_or_path = Path(name_or_path)
        name_or_path = self.name_to_path_map.get(name_or_path, None)
        if name_or_path in self.transforms:
            return self.transforms[name_or_path]
        return None


def load_transform_from_file(
    path: PathLike,
    fixed_image: bool = True,
    moving_image: bool = True,
    fixed_points: bool = True,
    moving_points: bool = True,
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

    # image2image config
    if "schema_version" in data:
        (
            transformation_type,
            fixed_paths,
            fixed_missing_paths,
            _fixed_points,
            moving_paths,
            moving_missing_paths,
            _moving_points,
        ) = _read_image2image_config(data)
    # imsmicrolink config
    elif "Project name" in data:
        (
            transformation_type,
            fixed_paths,
            fixed_missing_paths,
            _fixed_points,
            moving_paths,
            moving_missing_paths,
            _moving_points,
        ) = _read_imsmicrolink_config(data)
    else:
        raise ValueError(f"Unknown file format: {path.suffix}.")

    # apply config
    fixed_paths, fixed_missing_paths = (fixed_paths, fixed_missing_paths) if fixed_image else (None, None)
    moving_paths, moving_missing_paths = (moving_paths, moving_missing_paths) if moving_image else (None, None)
    _fixed_points = _fixed_points if fixed_points else None
    _moving_points = _moving_points if moving_points else None
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        _fixed_points,
        moving_paths,
        moving_missing_paths,
        _moving_points,
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


def _read_image2image_config(config: ty.Dict):
    """Read image2image configuration file."""
    schema_version = config["schema_version"]
    if schema_version == "1.0":
        return _read_image2image_v1_0_config(config)
    elif schema_version == "1.1":
        return _read_image2image_v1_1_config(config)
    return _read_image2image_latest_config(config)


def _read_image2image_latest_config(config: ty.Dict):
    # read important fields
    paths = [temp["path"] for temp in config["fixed_paths"]]
    fixed_paths, fixed_missing_paths = _get_paths(paths)
    paths = [temp["path"] for temp in config["moving_paths"]]
    moving_paths, moving_missing_paths = _get_paths(paths)
    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
    )


def _read_image2image_v1_1_config(config: ty.Dict):
    # read important fields
    fixed_paths, fixed_missing_paths = _get_paths(config["fixed_paths"])
    moving_paths, moving_missing_paths = _get_paths(config["moving_paths"])
    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
    )


def _read_image2image_v1_0_config(config: ty.Dict):
    # read important fields
    fixed_paths, fixed_missing_paths = _get_paths(config["micro_paths"])
    moving_paths, moving_missing_paths = _get_paths(config["ims_paths"])
    fixed_points = np.array(config["fixed_points_yx_px"])
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
    )


def _read_imsmicrolink_config(config: ty.Dict):
    """Read imsmicrolink configuration file."""
    fixed_paths, fixed_missing_paths = _get_paths([config["PostIMS microscopy image"]])  # need to be a list
    moving_paths, moving_missing_paths = _get_paths(config["Pixel Map Datasets Files"])
    fixed_points = np.array(config["PAQ microscopy points (xy, px)"])[:, ::-1]
    moving_points = np.array(config["IMS pixel map points (xy, px)"])[:, ::-1]
    padding = config["padding"]
    if padding["x_left_padding (px)"]:
        moving_points[:, 1] -= padding["x_left_padding (px)"]
    if padding["y_top_padding (px)"]:
        moving_points[:, 0] -= padding["y_top_padding (px)"]
    transformation_type = "Affine"
    return (
        transformation_type,
        fixed_paths,
        fixed_missing_paths,
        fixed_points,
        moving_paths,
        moving_missing_paths,
        moving_points,
    )
