"""Registration model."""
import os
import typing as ty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from pydantic import BaseModel, PrivateAttr, validator

if ty.TYPE_CHECKING:
    from skimage.transform._geometric import GeometricTransform
    from ims2micro._ims_reader import IMSWrapper
    from ims2micro._micro_reader import MicroWrapper


@dataclass
class Transformation:
    """Temporary object that holds transformation information."""

    # Transformation object
    transform: "GeometricTransform" = None
    # Type of transformation
    transformation_type: str = ""
    # Path to the image
    path: str = ""
    # Date when the registration was created
    time_created: ty.Optional[datetime] = None
    # Arrays of fixed and moving points
    fixed_points: ty.Optional[np.ndarray] = None
    moving_points: ty.Optional[np.ndarray] = None

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


class RegistrationModel(BaseModel):
    """Metadata associate with registration item."""

    # Initial name based on the filename
    name: str = ""
    # Name to be displayed to the user
    display_name: str = ""
    # Path to the RegistrationStore instance
    path: str = ""
    # Path to the image
    image_path: str = ""
    # Date when the registration was created
    time_created: ty.Optional[datetime] = None
    # Flag to indicate whether this registration is saved to file
    is_exported: bool = False
    locked: bool = False

    # Temporary data
    temporary_transform: ty.Optional[Transformation] = None
    _transform: ty.Optional["GeometricTransform"] = PrivateAttr(None)

    # Arrays of fixed and moving points
    _fixed_points: ty.Optional[np.ndarray] = PrivateAttr(None)
    _moving_points: ty.Optional[np.ndarray] = PrivateAttr(None)
    _transformation_type: ty.Optional[str] = PrivateAttr("")

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_temporary(self) -> Transformation:
        """Create temporary registration item."""
        if self.temporary_transform:
            return self.temporary_transform
        return Transformation(
            transform=self.transform,
            transformation_type=self.transformation_type,
            fixed_points=self.fixed_points,
            moving_points=self.moving_points,
            path=self.image_path,
        )

    @property
    def time_created_str(self) -> str:
        """Return the time the model finished fitting."""
        time_created = self.time_created
        if time_created is None:
            time_created = datetime.now()
        return time_created.strftime("%d-%m-%Y %H:%M:%S")

    @property
    def transform_store(self):
        """Retrieve transform store."""
        from imimspy.storage.registration import RegistrationStore

        if self.path != "":
            if not os.path.exists(self.path):
                return None
            store = RegistrationStore(self.path, mode="r")
            return store

    @property
    def transform_data(self) -> ty.Dict[str, ty.Any]:
        """Retrieve transformation data."""
        store = self.transform_store
        if store:
            data = store.get_registration()
            return data
        return {}

    @property
    def transform(self) -> ty.Optional["GeometricTransform"]:
        """Return transformation."""
        if self.temporary_transform:
            return self.temporary_transform.transform
        else:
            if self._transform:
                return self._transform
            store = self.transform_store
            if store:
                self._transform = store.get_transform()
            return self._transform

    @property
    def fixed_points(self) -> np.ndarray:
        """Fixed points."""
        if self._fixed_points is None:
            data = self.transform_data
            self._fixed_points, self._moving_points = data["fixed_points"], data["moving_points"]
            self._transformation_type = data["transformation_type"]
        return self._fixed_points

    @property
    def moving_points(self) -> np.ndarray:
        """Moving points."""
        if self._moving_points is None:
            _ = self.fixed_points
        return self._moving_points

    @property
    def transformation_type(self) -> str:
        """Moving points."""
        if self._transformation_type == "":
            _ = self.fixed_points
        return self._transformation_type

    def __call__(self, coords: np.ndarray):
        """Transform coordinates."""
        return self.transform(coords)

    def inverse(self, coords: np.ndarray):
        """Inverse transformation of coordinates."""
        return self.transform.inverse(coords)

    @property
    def matrix(self) -> np.ndarray:
        """Return matrix."""
        transform = self.transform
        if transform:
            return transform.params
        return np.ones((3, 3))

    @property
    def image_path_exists(self) -> bool:
        """Check whether image path exists."""
        if self.image_path is None:
            return False
        return os.path.exists(self.image_path)

    @property
    def path_exists(self) -> bool:
        """Check whether store path exists."""
        return os.path.exists(self.path)

    def from_hdf5(self):
        """Load transformation data to memory."""
        from imimspy.storage.registration import RegistrationStore

        if self.path:
            store = RegistrationStore(self.path, mode="r")
            data = store.get_registration()
            self._fixed_points = data["fixed_points"]
            self._moving_points = data["moving_points"]
            temp_transform = Transformation(
                transform=store.get_transform(),
                transformation_type=data["transformation_type"],
                fixed_points=data["fixed_points"],
                moving_points=data["moving_points"],
            )
            self.temporary_transform = temp_transform

    def to_hdf5(self, path: str) -> str:
        """Export transformation to file."""
        from imimspy.storage.registration import RegistrationStore

        if self.temporary_transform is None:
            raise ValueError("Cannot export transformation if there is no transform to save.")
        temporary_transform = self.temporary_transform
        store = RegistrationStore(path, mode="a")
        store.add_registration(
            name=self.name,
            image_path=self.image_path,
            transformation_type=temporary_transform.transformation_type,
            matrix=temporary_transform.matrix,
            fixed_points=temporary_transform.fixed_points,
            moving_points=temporary_transform.moving_points,
            display_name=self.display_name,
        )
        self._transform = temporary_transform.transform
        self.path = path
        self.is_exported = True
        self.temporary_transform = None
        return path

    def from_registration(self, other: "RegistrationModel"):
        """Modify registration based on another registration model."""
        temporary = other.to_temporary()
        temporary.path = self.image_path
        self.temporary_transform = temporary

    def update_hdf5(self):
        """Update contents of store transformation."""
        if self.temporary_transform:
            self.to_hdf5(self.path)
        else:
            from imimspy.storage.registration import RegistrationStore

            store = RegistrationStore(self.path, mode="a")
            store.update_registration(image_path=self.image_path, display_name=self.display_name)

    def save_filename(self, output_dir: PathLike, extension: str) -> str:
        """Generate appropriate filename."""
        import os.path

        path = os.path.join(output_dir, os.path.splitext(self.name)[0] + extension)
        return path

    def about(self) -> str:
        """Retrieve information about the model in textual format."""
        info = ""
        if self.path_exists:
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

    def delete(self):
        """Delete item."""
        if self.path and os.path.exists(self.path):
            os.remove(self.path)


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
