"""Serializable inputs and numerical results for independent registration validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ValidationModel(BaseModel):
    """Reject non-finite coordinates and unknown session fields."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, validate_assignment=True)


class ImageGeometry(ValidationModel):
    """Axis-aligned native image geometry; shape is YX, spacing and origin are XY in um."""

    shape: tuple[int, int]
    spacing: tuple[float, float] = (1.0, 1.0)
    origin: tuple[float, float] = (0.0, 0.0)

    @field_validator("shape", "spacing")
    @classmethod
    def positive(cls, value: tuple) -> tuple:
        """Require positive image dimensions and pixel spacing."""
        if min(value) <= 0:
            raise ValueError("Image shape and spacing must be positive.")
        return value

    def to_physical(self, xy: np.ndarray) -> np.ndarray:
        """Convert native pixel centers to physical XY coordinates."""
        return np.asarray(xy, dtype=float) * self.spacing + self.origin

    def to_pixels(self, xy: np.ndarray) -> np.ndarray:
        """Convert physical XY coordinates to native pixel centers."""
        return (np.asarray(xy, dtype=float) - self.origin) / self.spacing

    def contains(self, xy: np.ndarray) -> np.ndarray:
        """Return coverage of pixel cells, including their half-pixel margins."""
        pixels = self.to_pixels(xy)
        return (
            np.isfinite(pixels).all(axis=-1)
            & (pixels >= -0.5).all(axis=-1)
            & (pixels < np.asarray(self.shape[::-1]) - 0.5).all(axis=-1)
        )


class ValidationImage(ValidationModel):
    """Image or label mask reference with explicit native calibration."""

    path: str
    geometry: ImageGeometry
    channel: int = Field(0, ge=0)
    display_channel: int | Literal["rgb"] | None = None
    is_rgb: bool = False
    channel_names: list[str] = Field(default_factory=list)
    reader_kws: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str = ""
    calibration_source: Literal["unconfirmed", "registration", "user"] = "unconfirmed"
    calibration_confirmed: bool = False

    @field_validator("display_channel")
    @classmethod
    def valid_display_channel(cls, value: int | Literal["rgb"] | None) -> int | Literal["rgb"] | None:
        """Keep display selection separate from the scalar channel used for analysis."""
        if isinstance(value, int) and value < 0:
            raise ValueError("Display channel must be nonnegative.")
        return value


class ValidationLandmark(ValidationModel):
    """A corresponding point pair, always stored in physical XY coordinates."""

    id: str = Field(default_factory=lambda: uuid4().hex[:8], min_length=1)
    fixed: tuple[float, float]
    moving: tuple[float, float]
    enabled: bool = True
    note: str = ""
    source: Literal["manual", "assisted", "automatic"] = "manual"
    score: float | None = None


class ValidationCandidate(ValidationModel):
    """A frozen registration reference; source selects an Elastix registration path."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    kind: Literal["register", "elastix", "aligned"]
    path: str
    source: str = ""
    fingerprint: str = ""


class MaskCorrespondence(ValidationModel):
    """Two non-background labels representing the same anatomical region."""

    name: str = ""
    fixed_label: int = Field(gt=0)
    moving_label: int = Field(gt=0)


class ValidationSession(ValidationModel):
    """Versioned comparison of one image pair using shared validation evidence."""

    schema_version: Literal["1.0"] = "1.0"
    tool: Literal["validate"] = "validate"
    fixed: ValidationImage | None = None
    moving: ValidationImage | None = None
    candidates: list[ValidationCandidate] = Field(default_factory=list)
    landmarks: list[ValidationLandmark] = Field(default_factory=list)
    landmark_units: Literal["px", "um"] = "um"
    fixed_mask: ValidationImage | None = None
    moving_mask: ValidationImage | None = None
    mask_pairs: list[MaskCorrespondence] = Field(default_factory=list)
    max_pixels: int = Field(4_000_000, ge=9, le=4_000_000)
    downsample: int = Field(1, ge=1)
    auto_residual_map: bool = False
    auto_tile_um: float | None = Field(None, gt=0)
    independence_note: str = "Validation independence is user-declared; known fitting landmarks are excluded."

    @field_validator("landmarks", "candidates")
    @classmethod
    def unique_ids(cls, values: list) -> list:
        """Keep stable, unique row identifiers across UI sorting and exports."""
        if len({value.id for value in values}) != len(values):
            raise ValueError("Identifiers must be unique.")
        return values


class ValidationPair(ValidationModel):
    """An identified image pair with its own annotations and candidates."""

    id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    session: ValidationSession = Field(default_factory=ValidationSession)

    @property
    def name(self) -> str:
        """Return the moving image filename for display and reports."""
        return Path(self.session.moving.path).name if self.session.moving else "New moving image"


class ValidationStudy(ValidationModel):
    """A collection of independent pair sessions sharing one fixed reference."""

    schema_version: Literal["2.0", "2.1"] = "2.1"
    tool: Literal["validate"] = "validate"
    pairs: list[ValidationPair] = Field(default_factory=lambda: [ValidationPair()], min_length=1)

    @model_validator(mode="after")
    def shared_reference(self) -> ValidationStudy:
        """Reject duplicate pair identifiers or incompatible fixed references."""
        if len({pair.id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("Pair identifiers must be unique.")
        references = [pair.session.fixed for pair in self.pairs if pair.session.fixed is not None]
        if references:
            fixed = references[0]
            for image in references[1:]:
                if (
                    Path(image.path).resolve() != Path(fixed.path).resolve()
                    or image.geometry != fixed.geometry
                    or image.reader_kws != fixed.reader_kws
                ):
                    raise ValueError("Every pair must use the same fixed image and calibration.")
        return self


@dataclass
class CandidateAnalysis:
    """Small result record; dense maps are stored separately on disk."""

    id: str
    name: str
    fingerprint: str = ""
    status: str = "complete"
    error: str = ""
    landmarks: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    fitting: dict[str, Any] = field(default_factory=dict)
    masks: list[dict[str, Any]] = field(default_factory=list)
    mask_reason: str = ""
    deformation: dict[str, Any] = field(default_factory=dict)
    residual_map: dict[str, Any] = field(default_factory=dict)
    automatic: dict[str, Any] = field(default_factory=dict)
    map_path: Path | None = None


@dataclass
class ValidationAnalysis:
    """An immutable input snapshot and its comparison results."""

    session: ValidationSession
    grid: ImageGeometry
    candidates: list[CandidateAnalysis]
    versions: dict[str, str]
    created: str


@dataclass
class PairAnalysis:
    """Completed measurements or a failure for one identified study pair."""

    pair_id: str
    name: str
    analysis: ValidationAnalysis | None = None
    error: str = ""


@dataclass
class StudyAnalysis:
    """A study snapshot with separate outcomes for each analyzed pair."""

    study: ValidationStudy
    pairs: list[PairAnalysis]
