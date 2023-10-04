import typing as ty
from pathlib import Path

import numpy as np
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from pydantic import Field, validator

from image2image._reader import ImageWrapper, sanitize_path
from image2image.models.base import BaseModel
from image2image.models.utilities import _get_paths, _read_config_from_file
from image2image.readers.base_reader import BaseImageReader


class DataModel(BaseModel):
    """Base model."""

    paths: ty.List[Path] = Field(default_factory=list)
    just_added: ty.List[Path] = Field(default_factory=list)
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
        just_added = []
        for path in path_or_paths:
            path = sanitize_path(path)
            if path not in self.paths:
                self.paths.append(path)
                just_added.append(path)
                logger.trace(f"Added '{path}' to model paths.")
        self.just_added = just_added

    def remove_paths(self, path_or_paths: ty.Union[PathLike, ty.List[PathLike]]) -> None:
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

    def get_path(self, path: PathLike) -> ty.Optional[Path]:
        """Get path."""
        path = sanitize_path(path)
        if path not in self.paths:
            for path_ in self.paths:
                if path_.name == path.name:
                    return Path(path_)
        if path.exists():
            return path
        return None

    def has_path(self, path: PathLike) -> bool:
        """Check if path is in model."""
        path = self.get_path(path)
        return path is not None

    def load(
        self, affine: ty.Optional[ty.Dict[str, np.ndarray]] = None, resolution: ty.Optional[ty.Dict[str, float]] = None
    ) -> "DataModel":
        """Load data into memory."""
        with MeasureTimer() as timer:
            self.get_wrapper(affine, resolution)
            logger.info(f"Loaded data in {timer()}")
        return self

    def get_wrapper(
        self,
        affine: ty.Optional[ty.Dict[str, np.ndarray]] = None,
        resolution: ty.Optional[ty.Dict[str, float]] = None,
    ) -> ty.Optional["ImageWrapper"]:
        """Read data from file."""
        from image2image._reader import read_image

        if not self.paths:
            return None

        affine = affine or {}
        resolution = resolution or {}

        just_added = []
        for path in self.paths:
            matrix = affine.get(path.name, None)
            pixel_size = resolution.get(path.name, None)
            if self.wrapper is None or not self.wrapper.is_loaded(path):
                self.wrapper = read_image(path, self.wrapper, self.is_fixed, affine=matrix, resolution=pixel_size)
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

    def path_resolution_iter(self) -> ty.Iterator[ty.Tuple[Path, float]]:
        """Iterator of path and pixel size."""
        for path in self.paths:
            reader = self.get_reader(path)
            if reader is None:
                raise ValueError(f"Cannot find reader for path '{path}'")
            yield path, reader.resolution

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
        wrapper = self.get_wrapper()
        if wrapper:
            return wrapper.min_resolution
        return 1.0

    def to_dict(self) -> ty.Dict:
        """Return dictionary of values to export."""
        wrapper = self.get_wrapper()
        if not wrapper:
            raise ValueError("No wrapper found.")
        return {
            "schema_version": "1.0",
            "images": [
                {"path": str(path), "matrix_yx_px": reader.transform.tolist(), "pixel_size_um": reader.resolution}
                for path, reader in wrapper.path_reader_iter()
            ],
        }

    def is_valid(self) -> bool:
        """Returns True if there is some data on the model."""
        return self.n_paths > 0


def load_viewer_setup_from_file(
    path: PathLike,
) -> ty.Tuple[ty.List[Path], ty.List[Path], ty.Dict[str, np.ndarray], ty.Dict[str, float]]:
    """Load configuration from config file."""
    data = _read_config_from_file(path)

    if "schema_version" in data:
        return _read_image2viewer_latest_config(data)
    raise ValueError("Cannot read config file.")


def _read_image2viewer_latest_config(
    config: ty.Dict,
) -> ty.Tuple[ty.List[Path], ty.List[Path], ty.Dict[str, np.ndarray], ty.Dict[str, float]]:
    # read important fields
    paths = [temp["path"] for temp in config["images"]]
    affine = {Path(temp["path"]).name: np.asarray(temp["matrix_yx_px"]) for temp in config["images"]}
    resolution = {Path(temp["path"]).name: temp["pixel_size_um"] for temp in config["images"]}
    paths, paths_missing = _get_paths(paths)
    if not paths:
        paths = []
    if not paths_missing:
        paths_missing = []
    return paths, paths_missing, affine, resolution
