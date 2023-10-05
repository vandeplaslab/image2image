"""Transform."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from skimage.transform import AffineTransform, ProjectiveTransform

from image2image.models.base import BaseModel


class TransformData(BaseModel):
    """Transformation data."""

    # Transformation object
    _transform: ty.Optional[ProjectiveTransform] = None
    # this value should never change
    fixed_resolution: float = 1.0
    # this value can change
    moving_resolution: float = 1.0
    # Arrays of fixed and moving points
    fixed_points: ty.Optional[np.ndarray] = None
    moving_points: ty.Optional[np.ndarray] = None
    # affine transformation matrix
    affine: ty.Optional[np.ndarray] = None

    # Type of transformation
    transformation_type: str = ""

    @property
    def transform(self) -> ProjectiveTransform:
        """Retrieve the transformation object."""
        if self._transform is None:
            if self.fixed_points is None or self.moving_points is None:
                self._transform = AffineTransform(matrix=np.eye(3))
            else:
                self._transform = self.compute()
        return self._transform

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        """Transform coordinates."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        return self.transform(coords)  # type: ignore[no-any-return]

    def inverse(self, coords: np.ndarray) -> np.ndarray:
        """Inverse transformation of coordinates."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        return self.transform.inverse(coords)  # type: ignore[no-any-return]

    @property
    def matrix(self) -> np.ndarray:
        """Retrieve the transformation array."""
        if self.transform is None:
            raise ValueError("No transformation found.")
        return self.transform.params  # type: ignore[no-any-return]

    def compute(self, moving_resolution: float = 1.0, yx: bool = True, px: bool = True) -> ProjectiveTransform:
        """Compute transformation matrix."""
        from image2image.utilities import compute_transform

        moving_points = self.moving_points
        fixed_points = self.fixed_points
        if moving_points is None or fixed_points is None:
            raise ValueError("No points found.")

        if not yx:
            moving_points = moving_points[:, ::-1]
            fixed_points = fixed_points[:, ::-1]
        if not px:
            moving_points = moving_points * moving_resolution
            fixed_points = fixed_points * self.fixed_resolution

        transform = compute_transform(
            moving_points,  # source
            fixed_points,  # destination
            self.transformation_type,
        )
        return transform

    @classmethod
    def from_i2r(cls, path: PathLike) -> "TransformData":
        """Load directly from i2r."""
        from image2image.models.transformation import load_transform_from_file

        (
            transformation_type,
            _fixed_paths,
            _fixed_paths_missing,
            fixed_points,
            _moving_paths,
            _moving_paths_missing,
            moving_points,
            fixed_resolution,
            _moving_resolution,
        ) = load_transform_from_file(path)
        return TransformData(
            fixed_points=fixed_points,
            moving_points=moving_points,
            transformation_type=transformation_type,
            fixed_resolution=fixed_resolution,
        )

    @classmethod
    def from_array(cls, matrix: np.ndarray) -> "TransformData":
        """Load from array."""
        matrix = np.asarray(matrix)
        assert matrix.shape == (3, 3), "Expected (3, 3) matrix"
        return TransformData(affine=matrix)


class TransformModel(BaseModel):
    """Model containing transformation data."""

    transforms: ty.Optional[ty.Dict[PathLike, TransformData]] = None

    class Config:
        """Config."""

        arbitrary_types_allowed = True

    @property
    def name_to_path_map(self) -> ty.Dict[ty.Union[str, Path], Path]:
        """Returns dictionary that maps transform name to path."""
        if self.transforms is None:
            return {}

        mapping: ty.Dict[PathLike, Path] = {}
        for name in self.transforms:
            if isinstance(name, str):
                name = Path(name)
            mapping[name.name] = name
            mapping[Path(name.name)] = name
            mapping[name] = name
        return mapping

    @property
    def transform_names(self) -> ty.List[str]:
        """Return list of transform names."""
        return [t.name for t in self.transforms] if self.transforms else []

    def add_transform(self, name_or_path: PathLike, transform_data: TransformData) -> None:
        """Add transformation matrix."""
        if self.transforms is None:
            self.transforms = {}

        path = Path(name_or_path)
        # matrix = np.asarray(matrix)
        # assert matrix.shape == (3, 3), "Expected (3, 3) matrix"
        self.transforms[path] = transform_data
        logger.info(f"Added '{path.name}' to list of transformations")

    def remove_transform(self, name_or_path: PathLike) -> None:
        """Remove transformation matrix."""
        if self.transforms is None:
            return

        name_or_path = Path(name_or_path)
        if name_or_path in self.transforms:
            del self.transforms[name_or_path]

    def get_matrix(self, name_or_path: PathLike) -> ty.Optional[TransformData]:
        """Get transformation matrix."""
        if self.transforms is None:
            return None

        name_or_path = Path(name_or_path)
        name_or_path = self.name_to_path_map.get(name_or_path, None)  # type: ignore
        if name_or_path is None:
            return None
        if name_or_path in self.transforms:
            return self.transforms[name_or_path]
        return None
