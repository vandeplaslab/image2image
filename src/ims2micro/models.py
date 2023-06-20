"""Registration model."""
import typing as ty
from datetime import datetime
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
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

    @validator("path", pre=True)
    def _validate_path(value: PathLike) -> Path:
        """Validate path."""
        path = Path(value)
        assert path.exists(), f"Path {path} does not exist."
        return path

    def load(self):
        """Load data into memory."""
        print("Started loading data...")
        self.get_reader()
        return self

    def get_reader(self):
        """Read data from file."""
        raise NotImplementedError("Must implement method")


class ImagingModel(DataModel):
    """IMS model."""

    def get_reader(self) -> "IMSWrapper":
        """Read data from file."""
        from ims2micro._ims_reader import read_imaging

        if self.reader is None:
            self.reader = read_imaging(self.path)
        return self.reader


class MicroscopyModel(DataModel):
    """IMS model."""

    def get_reader(self) -> "MicroWrapper":
        """Read data from file."""
        from ims2micro._micro_reader import read_microscopy

        if self.reader is None:
            self.reader = read_microscopy(self.path)
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

    def about(self) -> str:
        """Retrieve information about the model in textual format."""
        info = ""
        if self.transformation_type:
            info += f"Transformation type: {self.transformation_type}"
        transform = self.transform
        if transform:
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
        # TODO: add xy, yx, inverse
        return {
            "matrix": self.matrix,
            "transformation_type": self.transformation_type,
            "fixed_points": self.fixed_points,
            "moving_points": self.moving_points,
            "time_created": self.time_created,
            "micro_path": self.micro_model.path,
            "ims_path": self.ims_model.path,
            "micro_resolution": self.micro_model.resolution,
            "ims_resolution": self.ims_model.resolution,
        }

    def to_json(self):
        """Export data as JSON."""
