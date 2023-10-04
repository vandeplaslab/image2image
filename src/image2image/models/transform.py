"""Transform."""
import typing as ty
from pathlib import Path

import numpy as np
from koyo.typing import ArrayLike, PathLike
from loguru import logger

from image2image.models.base import BaseModel


class TransformModel(BaseModel):
    """Model containing transformation data."""

    transforms: ty.Optional[ty.Dict[PathLike, np.ndarray]] = None

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

    def add_transform(self, name_or_path: PathLike, matrix: ArrayLike) -> None:
        """Add transformation matrix."""
        if self.transforms is None:
            self.transforms = {}

        path = Path(name_or_path)
        matrix = np.asarray(matrix)
        assert matrix.shape == (3, 3), "Expected (3, 3) matrix"
        self.transforms[path] = matrix
        logger.info(f"Added '{path.name}' to list of transformations")

    def remove_transform(self, name_or_path: PathLike) -> None:
        """Remove transformation matrix."""
        if self.transforms is None:
            return

        name_or_path = Path(name_or_path)
        if name_or_path in self.transforms:
            del self.transforms[name_or_path]

    def get_matrix(self, name_or_path: PathLike) -> ty.Optional[np.ndarray]:
        """Get transformation matrix."""
        if self.transforms is None:
            return None

        name_or_path = Path(name_or_path)
        name_or_path = self.name_to_path_map.get(name_or_path, None)
        if name_or_path is None:
            return None
        if name_or_path in self.transforms:
            return self.transforms[name_or_path]
        return None
