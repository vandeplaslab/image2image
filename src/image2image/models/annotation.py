"""Public models for calibrated, persistent two-dimensional annotations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from shapely.affinity import affine_transform
from shapely.geometry.base import BaseGeometry


class AnnotationModel(BaseModel):
    """Validate project records without accepting non-finite coordinates."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class AnnotationProject(AnnotationModel):
    """Versioned manifest for a directory containing the object database and model assets."""

    tool: Literal["annotate"] = "annotate"
    schema_version: Literal[1] = 1


class SlideGeometry(AnnotationModel):
    """Pixel-edge coordinates; shape is YX, spacing and origin are XY in um."""

    shape: tuple[int, int]
    spacing: tuple[float, float] | None = None
    origin: tuple[float, float] = (0.0, 0.0)

    @field_validator("shape", "spacing")
    @classmethod
    def positive(cls, value: tuple | None) -> tuple | None:
        """Require positive dimensions and known pixel sizes."""
        if value is not None and min(value) <= 0:
            raise ValueError("Dimensions and pixel sizes must be positive.")
        return value

    def to_physical(self, xy: np.ndarray) -> np.ndarray:
        """Convert full-resolution pixel XY coordinates to physical XY."""
        if self.spacing is None:
            raise ValueError("Physical coordinates require calibrated X and Y pixel sizes.")
        return np.asarray(xy, dtype=float) * self.spacing + self.origin

    def to_pixels(self, xy: np.ndarray) -> np.ndarray:
        """Convert physical XY coordinates to full-resolution pixel XY."""
        if self.spacing is None:
            raise ValueError("Physical coordinates require calibrated X and Y pixel sizes.")
        return (np.asarray(xy, dtype=float) - self.origin) / self.spacing


class CoordinateSpec(AnnotationModel):
    """Explicit import/export frame; pixel offset is full-resolution slide XY."""

    units: Literal["px", "um"] = "px"
    offset: tuple[float, float] = (0.0, 0.0)
    downsample: tuple[float, float] = (1.0, 1.0)

    @field_validator("downsample")
    @classmethod
    def positive(cls, value: tuple[float, float]) -> tuple[float, float]:
        """Reject reflected or empty pixel grids."""
        if min(value) <= 0:
            raise ValueError("Downsample must be positive on both axes.")
        return value

    def transform(self, geometry: BaseGeometry, slide: SlideGeometry, *, inverse: bool = False) -> BaseGeometry:
        """Transform an external geometry to slide pixels, or invert for export."""
        if self.units == "um":
            if slide.spacing is None:
                raise ValueError("Set X/Y calibration before importing or exporting physical coordinates.")
            if self.offset != (0.0, 0.0) or self.downsample != (1.0, 1.0):
                raise ValueError("Patch offsets/downsampling apply to pixel coordinates only.")
            sx, sy = 1 / slide.spacing[0], 1 / slide.spacing[1]
            tx, ty = -slide.origin[0] * sx, -slide.origin[1] * sy
        else:
            sx, sy = self.downsample
            tx, ty = self.offset
        if inverse:
            sx, sy, tx, ty = 1 / sx, 1 / sy, -tx / sx, -ty / sy
        return affine_transform(geometry, (sx, 0, 0, sy, tx, ty))


class AnnotationSlide(AnnotationModel):
    """Referenced image and its explicit native calibration."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    path: str
    geometry: SlideGeometry
    channel_names: list[str] = Field(default_factory=list)
    fingerprint: str = ""
    series: int = 0


class AnnotationClass(AnnotationModel):
    """Shared class with visibility, edit protection, and optional Ignore masking."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(min_length=1)
    color: str = Field(default="#e57373", pattern=r"^#[0-9a-fA-F]{6}$")
    visible: bool = True
    locked: bool = False
    is_ignore: bool = False


@dataclass
class AnnotationObject:
    """One complete polygonal object; public IDs are independent of database keys."""

    set_id: str
    object_id: str | int
    geometry: BaseGeometry
    class_id: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    review: Literal["accepted", "proposed", "rejected"] = "accepted"
    training: bool = False
    key: int = 0


@dataclass
class EditCommand:
    """Named object changes; None removes the corresponding database key."""

    label: str
    changes: Sequence[tuple[int | None, AnnotationObject | None]]


@dataclass
class ImagePatch:
    """HWC pixels and their pixel-edge transform into the source slide."""

    data: np.ndarray
    offset: tuple[float, float]
    downsample: tuple[float, float]
    channels: tuple[int, ...]

    @property
    def coordinates(self) -> CoordinateSpec:
        """Return the patch-to-slide coordinate conversion."""
        return CoordinateSpec(offset=self.offset, downsample=self.downsample)


class FeatureSettings(AnnotationModel):
    """Reproducible random-forest feature extraction and sampling settings."""

    channels: tuple[int, ...] = (0,)
    downsample: int = Field(default=4, ge=1)
    sigma_min: float = Field(default=1.0, gt=0)
    sigma_max: float = Field(default=8.0, gt=0)
    samples_per_class: int = Field(default=10000, ge=10, le=100000)
    trees: int = Field(default=100, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    max_depth: int = Field(default=24, ge=1)
    class_ids: tuple[str, ...] | None = None


class FineTuneSettings(AnnotationModel):
    """Explicit inputs for fine-tuning an instance model from accepted annotations."""

    backend: Literal["cellpose", "micro-sam"]
    model: str
    checkpoint: str = ""
    class_ids: tuple[str, ...]
    channels: tuple[int, ...] = (0,)
    downsample: int = Field(default=1, ge=1)
    epochs: int = Field(default=100, ge=1, le=10000)
    learning_rate: float = Field(default=1e-5, gt=0, le=1)
    batch_size: int = Field(default=1, ge=1, le=32)
    patch_size: int = Field(default=256, ge=256, le=1024)
    max_patches: int = Field(default=128, ge=2, le=1024)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    freeze_encoder: bool = True


class PredictionRequest(AnnotationModel):
    """Immutable inputs for a cancellable worker operation."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    project: str
    slide_id: str
    revision: int
    backend: Literal["forest", "micro-sam", "cellpose"] = "forest"
    sam_mode: Literal["prompts", "automatic"] = "prompts"
    model: str = ""
    checkpoint: str = ""
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    channels: tuple[int, ...] = (0,)
    downsample: int = Field(default=1, ge=1)
    tile_size: int = Field(default=512, ge=64, le=2048)
    halo: int = Field(default=64, ge=8, le=256)
    bounds: tuple[float, float, float, float] | None = None
    points: list[tuple[float, float, int]] = Field(default_factory=list)
    box: tuple[float, float, float, float] | None = None
    boxes: list[tuple[float, float, float, float]] = Field(default_factory=list)
    class_id: str | None = None
    allow_download: bool = False


class PredictionBatch(AnnotationModel):
    """A staged review set and the request that produced it."""

    id: str
    set_id: str
    request: PredictionRequest
    count: int
    ambiguous: int = 0
