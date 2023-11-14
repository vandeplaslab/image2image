"""Image data models."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from pydantic import Field, validator

from image2image._reader import ImageWrapper, get_alternative_path, sanitize_path, sanitize_read_path
from image2image.models.base import BaseModel
from image2image.models.transform import TransformData
from image2image.models.utilities import _get_paths, _read_config_from_file
from image2image.readers._base_reader import BaseReader
from image2image.utils.utilities import log_exception_or_error

I2V_METADATA = ty.Tuple[ty.List[Path], ty.List[Path], ty.Dict[str, TransformData], ty.Dict[str, float]]
I2C_METADATA = ty.Tuple[
    ty.List[Path], ty.List[Path], ty.Dict[str, TransformData], ty.Dict[str, float], list[dict[str, int]]
]


class DataModel(BaseModel):
    """Base model."""

    keys: ty.List[str] = Field(default_factory=list)
    just_added_keys: ty.List[str] = Field(default_factory=list)
    paths: ty.List[Path] = Field(default_factory=list)
    resolution: float = 1.0
    wrapper: ty.Optional[ImageWrapper] = None
    is_fixed: bool = False

    # noinspection PyMethodFirstArgAssignment,PyMethodParameters
    @validator("paths", pre=True, allow_reuse=True)
    def _validate_path(value: ty.Union[PathLike, ty.List[PathLike]]) -> ty.List[Path]:  # type: ignore[misc]
        """Validate path."""
        if isinstance(value, (str, Path)):
            value = [Path(value)]
        value_ = [Path(path) for path in value]
        assert all(p.exists() for p in value_), "Path does not exist."
        return value_

    def get_filename(self) -> str:
        """Get representative filename."""
        if self.n_paths == 0:
            return "no_files"
        elif self.n_paths == 1:
            return self.paths[0].parent.stem
        return self.paths[-1].parent.stem + "_multiple_files"

    def add_paths(self, path_or_paths: ty.Union[PathLike, ty.Sequence[PathLike]]) -> None:
        """Add paths to model."""
        if isinstance(path_or_paths, (str, Path)):
            path_or_paths = [path_or_paths]
        for path in path_or_paths:
            path_ = path
            path = sanitize_read_path(path, raise_error=False)
            if path is None:
                logger.warning(f"Failed to add '{path_}' to model paths.")
            elif path not in self.paths:
                self.paths.append(path)
                logger.trace(f"Added '{path}' to model paths.")
            else:
                logger.warning(f"Path '{path}' already in model paths.")

    def remove_keys(self, key_or_keys: ty.Union[str, ty.Sequence[str]]) -> None:
        """Remove keys."""
        if not key_or_keys:
            return
        if isinstance(key_or_keys, str):
            key_or_keys = [key_or_keys]
        logger.trace(f"Removing keys... - {key_or_keys}")
        # make sure that paths are in sync
        wrapper = self.wrapper
        for key in key_or_keys:
            if key in self.keys:
                self.keys.remove(key)
                logger.trace(f"Removed '{key}' from model keys.")
            if wrapper:
                wrapper.remove(key)
                logger.trace(f"Removed '{key}' from reader.")
                if self.just_added_keys:
                    if key in self.just_added_keys:
                        self.just_added_keys.remove(key)
                        logger.trace(f"Removed '{key}' from just_added_keys.")
        # synchronize paths
        if wrapper:
            all_paths = [reader.path for reader in wrapper.reader_iter()]
            paths = []
            for path in self.paths:
                if path in all_paths:
                    paths.append(path)
            self.paths = paths

    def remove_paths(self, path_or_paths: ty.Union[PathLike, ty.Sequence[PathLike]]) -> None:
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
                keys = self.wrapper.remove_path(path)
                logger.trace(f"Removed '{path}' from reader.")
                if self.just_added_keys:
                    for key in keys:
                        if key in self.just_added_keys:
                            self.just_added_keys.remove(key)
                            logger.trace(f"Removed '{key}' from just_added_keys.")
        # remove wrapper
        if not self.paths:
            self.wrapper = None

    def has_key(self, key: str) -> bool:
        """Check if key is in the model."""
        wrapper = self.wrapper
        if wrapper:
            return key in wrapper.data
        return False

    def load(
        self,
        transform_data: ty.Optional[ty.Dict[str, TransformData]] = None,
        resolution: ty.Optional[ty.Dict[str, float]] = None,
    ) -> "DataModel":
        """Load data into memory."""
        logger.trace(f"Loading data for '{self.paths}'")
        with MeasureTimer() as timer:
            self.get_wrapper(transform_data, resolution)
        logger.info(f"Loaded data in {timer()}")
        return self

    def get_wrapper(
        self,
        transform_data: ty.Optional[ty.Dict[str, TransformData]] = None,
        resolution: ty.Optional[ty.Dict[str, float]] = None,
    ) -> ty.Optional["ImageWrapper"]:
        """Read data from file."""
        from image2image._reader import read_data

        if not self.paths:
            return None

        transform_data = transform_data or {}
        resolution = resolution or {}

        just_added_keys = []
        for path in self.paths:
            if self.wrapper is None or not self.wrapper.is_loaded(path):
                logger.trace(f"Loading '{path}'...")
                transform_data_ = transform_data.get(path.name, None)
                if not transform_data_:
                    transform_data_ = transform_data.get(get_alternative_path(path).name, None)
                if transform_data and not transform_data_:
                    logger.trace(f"Failed to retrieve transform data for '{path}'")
                pixel_size = resolution.get(path.name, None)
                if not pixel_size:
                    pixel_size = resolution.get(get_alternative_path(path).name, None)
                if resolution and not pixel_size:
                    logger.trace(f"Failed to retrieve resolution for '{path}'")
                try:
                    self.wrapper, just_added_keys_, path_map = read_data(
                        path,
                        self.wrapper,
                        self.is_fixed,
                        transform_data=transform_data_,
                        resolution=pixel_size,
                    )
                    just_added_keys.extend(just_added_keys_)
                    if path_map:
                        for original_path, new_path in path_map.items():
                            if original_path == new_path:
                                continue
                            if original_path in self.paths:
                                index = self.paths.index(original_path)
                                self.paths[index] = new_path
                                logger.trace(f"Updated path '{original_path}' to '{new_path}'")
                except Exception as e:  # noqa
                    log_exception_or_error(e)
                    logger.error(f"Failed to load '{path}'")
                    self.remove_paths(path)

        if self.wrapper:
            self.resolution = self.wrapper.resolution
        if just_added_keys:
            self.just_added_keys = just_added_keys
            logger.trace(f"Added keys: {just_added_keys}")
        return self.wrapper

    def get_reader(self, path: PathLike) -> ty.Optional["BaseReader"]:
        """Get reader for the path."""
        path = Path(path)
        wrapper = self.wrapper
        if wrapper:
            for reader in wrapper.data.values():
                if reader.path == path:
                    return reader
        return None

    def get_reader_for_key(self, key: str) -> ty.Optional["BaseReader"]:
        """Get reader for the specified key."""
        wrapper = self.wrapper
        if wrapper:
            return wrapper.data[key]
        return None

    def get_extractable_paths(self) -> ty.List[Path]:
        """Get a list of paths which are extractable."""
        paths = []
        wrapper = self.wrapper
        if wrapper:
            for path, reader in wrapper.path_reader_iter():
                if hasattr(reader, "allow_extraction") and reader.allow_extraction:
                    paths.append(path)
        return paths

    def path_resolution_iter(self) -> ty.Generator[ty.Tuple[Path, float], None, None]:
        """Iterator of path and pixel size."""
        wrapper = self.wrapper
        if wrapper:
            for reader in wrapper.reader_iter():
                yield reader.path, reader.resolution

    def export_iter(self) -> ty.Generator[dict[str, ty.Union[Path, float, str, tuple[int, int], dict]], None, None]:
        """Export iterator."""
        wrapper = self.wrapper
        if wrapper:
            for reader in wrapper.reader_iter():
                yield {
                    "path": reader.path,
                    "pixel_size_um": reader.resolution,
                    "image_shape": reader.image_shape,
                    "type": reader.reader_type,
                    "reader_kws": reader.reader_kws,
                }

    def channel_names(self) -> ty.List[str]:
        """Return list of channel names."""
        wrapper = self.wrapper
        if wrapper:
            return wrapper.channel_names()
        return []

    @property
    def n_paths(self) -> int:
        """Return number of paths."""
        return len(self.paths) if self.paths is not None else 0

    def to_file(self, path: PathLike) -> None:
        """Save model to file."""
        path = Path(path)
        if path.suffix == ".json":
            self.to_json(path)
        elif path.suffix == ".toml":
            self.to_toml(path)
        else:
            raise ValueError(f"Unknown file format: {path.suffix}")
        logger.info(f"Exported to '{path}'")

    @property
    def min_resolution(self) -> float:
        """Return minimum resolution."""
        wrapper = self.wrapper
        if wrapper:
            return wrapper.min_resolution
        return 1.0

    def to_dict(self) -> ty.Dict:
        """Return dictionary of values to export."""
        wrapper = self.wrapper
        if not wrapper:
            raise ValueError("No wrapper found.")
        return {
            "schema_version": "1.1",
            "images": [
                {
                    "path": str(path),
                    "pixel_size_um": reader.resolution,
                    **reader.transform_data.to_dict(),
                }
                for path, reader in wrapper.path_reader_iter()
            ],
        }

    def is_valid(self) -> bool:
        """Returns True if there is some data on the model."""
        return self.n_paths > 0

    def crop(
        self, left: int, right: int, top: int, bottom: int
    ) -> ty.Generator[ty.Tuple[Path, "BaseReader", np.ndarray], None, None]:
        """Crop image(s) to the specified region."""
        wrapper = self.wrapper
        if wrapper:
            for path, reader in wrapper.path_reader_iter():
                cropped = reader.crop(left, right, top, bottom)
                yield path, reader, cropped


def load_viewer_setup_from_file(path: PathLike) -> I2V_METADATA:
    """Load configuration from config file."""
    data = _read_config_from_file(path)

    if "schema_version" not in data:
        raise ValueError("Cannot read config file.")
    if data["schema_version"] == "1.0":
        return _read_image2viewer_v1_0_config(data)
    return _read_image2viewer_latest_config(data)


def _read_image2viewer_latest_config(config: ty.Dict) -> I2V_METADATA:
    """Read config file."""
    paths = [temp["path"] for temp in config["images"]]
    transform_data = {
        Path(temp["path"]).name: TransformData(
            fixed_points=np.asarray(temp["fixed_points"]),
            moving_points=np.asarray(temp["moving_points"]),
            fixed_resolution=temp["fixed_pixel_size_um"],
            moving_resolution=temp["moving_pixel_size_um"],
            affine=np.asarray(temp["matrix_yx_um"]),
        )
        for temp in config["images"]
    }
    resolution = {Path(temp["path"]).name: temp["pixel_size_um"] for temp in config["images"]}
    paths, paths_missing = _get_paths(paths)
    if not paths:
        paths = []
    if not paths_missing:
        paths_missing = []
    return paths, paths_missing, transform_data, resolution


def _read_image2viewer_v1_0_config(config: ty.Dict) -> I2V_METADATA:
    # read important fields
    paths = [temp["path"] for temp in config["images"]]
    transform_data = {
        Path(temp["path"]).name: TransformData(affine=np.asarray(temp["matrix_yx_px"])) for temp in config["images"]
    }
    resolution = {Path(temp["path"]).name: temp["pixel_size_um"] for temp in config["images"]}
    paths, paths_missing = _get_paths(paths)
    if not paths:
        paths = []
    if not paths_missing:
        paths_missing = []
    return paths, paths_missing, transform_data, resolution


def load_crop_setup_from_file(path: PathLike) -> I2C_METADATA:
    """Load configuration from config file."""
    data = _read_config_from_file(path)

    if "schema_version" not in data:
        raise ValueError("Cannot read config file.")
    if data["schema_version"] == "1.0":
        return _read_image2crop_latest_config(data)


def _read_image2crop_latest_config(config: ty.Dict) -> I2C_METADATA:
    """Read config file."""
    paths = [temp["path"] for temp in config["images"]]
    transform_data = {
        Path(temp["path"]).name: TransformData(
            fixed_points=np.asarray(temp["fixed_points"]),
            moving_points=np.asarray(temp["moving_points"]),
            fixed_resolution=temp["fixed_pixel_size_um"],
            moving_resolution=temp["moving_pixel_size_um"],
            affine=np.asarray(temp["matrix_yx_um"]),
        )
        for temp in config["images"]
    }
    resolution = {Path(temp["path"]).name: temp["pixel_size_um"] for temp in config["images"]}
    paths, paths_missing = _get_paths(paths)
    crop = config["crop"][0]
    if not paths:
        paths = []
    if not paths_missing:
        paths_missing = []
    return paths, paths_missing, transform_data, resolution, crop
