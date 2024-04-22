"""Focus model."""

import typing as ty

from image2image.models.base import BaseModel


class Focus(BaseModel):
    """Focus model."""

    position: ty.Tuple[float, float, float] = (0, 0, 0)
    zoom: float = 1.0


class Bounds(BaseModel):
    """Bounds model."""

    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
