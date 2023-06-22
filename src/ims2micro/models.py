"""Registration model."""
import typing as ty
from datetime import datetime
from pathlib import Path

from loguru import logger
import numpy as np
from koyo.typing import PathLike
from koyo.timer import MeasureTimer
from pydantic import BaseModel, validator
from skimage.transform import ProjectiveTransform

if ty.TYPE_CHECKING:
    from ims2micro._ims_reader import IMSWrapper
    from ims2micro._micro_reader import MicroWrapper


class DataModel(BaseModel):
    """Base model."""

    path: Path
    resolution: float = 1.0
    reader: ty.Optional[ty.Any] = None

    @validator("path", pre=True, allow_reuse=True)
    def _validate_path(value: PathLike) -> Path:
        """Validate path."""
        path = Path(value)
        assert path.exists(), f"Path {path} does not exist."
        return path

    def load(self):
        """Load data into memory."""
        with MeasureTimer() as timer:
            self.get_reader()
            logger.info(f"Loaded data in {timer()}")
        return self

    def get_reader(self):
        """Read data from file."""
        raise NotImplementedError("Must implement method")

    def channel_names(self) -> ty.List[str]:
        """Return list of channel names."""
        return self.get_reader().channel_names()


class ImagingModel(DataModel):
    """IMS model."""

    def get_reader(self) -> "IMSWrapper":
        """Read data from file."""
        from ims2micro._ims_reader import read_imaging

        if self.reader is None:
            self.reader = read_imaging(self.path)
            self.resolution = self.reader.resolution
        return self.reader


class MicroscopyModel(DataModel):
    """IMS model."""

    def get_reader(self) -> "MicroWrapper":
        """Read data from file."""
        from ims2micro._micro_reader import read_microscopy

        if self.reader is None:
            self.reader = read_microscopy(self.path)
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
            "micro_path": str(self.micro_model.path),
            "micro_resolution_um": self.micro_model.resolution,
            "ims_path": str(self.ims_model.path),
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


def load_from_file(path: PathLike):
    """Load registration from file."""
    path = Path(path)
    if path.suffix not in [".json", ".toml"]:
        raise ValueError(f"Unknown file format: {path.suffix}")

    if path.suffix == ".json":
        from koyo.json import read_json_data

        config = read_json_data(path)
    else:
        from koyo.toml import read_toml_data

        config = read_toml_data(path)

    # read important fields
    micro_path = Path(config["micro_path"])
    if not micro_path.exists():
        micro_path = None
    fixed_points = np.array(config["fixed_points_yx_px"])
    ims_path = Path(config["ims_path"])
    if not ims_path.exists():
        ims_path = None
    moving_points = np.array(config["moving_points_yx_px"])
    transformation_type = config["transformation_type"]
    return transformation_type, micro_path, fixed_points, ims_path, moving_points
