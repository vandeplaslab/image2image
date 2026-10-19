"""Versioned profiles and batch recipes for brightfield stain separation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from image2image.models.illumination import IlluminationSource


class StainSource(IlluminationSource):
    """A 2D source with an explicit red, green, blue channel mapping."""

    rgb_channels: tuple[int, int, int] | None = None

    @property
    def key(self) -> str:
        """Return the stable scene key used by manual overrides."""
        return f"{self.path.resolve()}::{self.scene}"


class StainProfile(BaseModel):
    """Named unit optical-density vectors and white values as fractions of the dtype maximum."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = "H&E"
    names: tuple[str, ...] = ("Hematoxylin", "Eosin")
    vectors: tuple[tuple[float, float, float], ...] = ((0.65, 0.70, 0.29), (0.07, 0.99, 0.11))
    white: tuple[float, float, float] = (1.0, 1.0, 1.0)

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        """Normalize vectors and reject dependent or nonphysical stain profiles."""
        values = np.asarray(self.vectors, dtype=float)
        if len(self.names) not in (2, 3) or values.shape != (len(self.names), 3):
            raise ValueError("Supply two or three named RGB stain vectors.")
        if any(not name.strip() for name in self.names) or len(set(self.names)) != len(self.names):
            raise ValueError("Stain names must be nonempty and unique.")
        if "Residual" in self.names:
            raise ValueError("Residual is reserved for the automatically generated component.")
        lengths = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(lengths < 1e-8):
            raise ValueError("Stain vectors must be finite, nonnegative, and nonzero.")
        values /= np.where(np.isclose(lengths, 1, rtol=0, atol=1e-12), 1, lengths)[:, None]
        if np.linalg.cond(values) > 100:
            raise ValueError("Stain vectors are too similar; choose independent stain colors.")
        if any(not 0 < value <= 1 for value in self.white):
            raise ValueError("Background RGB values must be greater than zero and at most 1.")
        object.__setattr__(self, "vectors", tuple(tuple(float(v) for v in row) for row in values))
        return self

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Return the three output component names, including the two-stain residual."""
        return (*self.names, "Residual") if len(self.names) == 2 else self.names


def stain_preset(name: str = "H&E") -> StainProfile:
    """Return a fresh H&E, H-DAB, or custom starting profile."""
    if name == "H&E":
        return StainProfile()
    if name == "H-DAB":
        return StainProfile(name=name, names=("Hematoxylin", "DAB"), vectors=((0.65, 0.70, 0.29), (0.27, 0.57, 0.78)))
    if name == "Custom (2 stains)":
        return StainProfile(name=name, names=("Stain 1", "Stain 2"))
    if name == "Custom (3 stains)":
        return StainProfile(
            name=name,
            names=("Stain 1", "Stain 2", "Stain 3"),
            vectors=((0.65, 0.70, 0.29), (0.07, 0.99, 0.11), (0.27, 0.57, 0.78)),
        )
    raise ValueError(f"Unknown stain preset: {name}")


class StainEstimation(BaseModel):
    """Bounded tissue sampling and robust fitting settings, using natural-log optical density."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    min_od: float = Field(default=0.15, gt=0, lt=3)
    max_od: float = Field(default=3.0, gt=0, le=12)
    angular_percentile: float = Field(default=1, gt=0, lt=25)
    concentration_percentile: float = Field(default=99, ge=90, lt=100)
    max_samples: int = Field(default=100_000, ge=64, le=100_000)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        """Require a nonempty optical-density interval."""
        if self.max_od <= self.min_od:
            raise ValueError("Maximum optical density must exceed the minimum.")
        return self


class StainOverride(BaseModel):
    """An explicit source profile tied to the inspected file identity and scene."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: StainSource
    profile: StainProfile


class StainRecipe(BaseModel):
    """A frozen batch recipe; source coefficients are exported independently of normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    profile: StainProfile = Field(default_factory=StainProfile)
    vector_mode: Literal["auto", "shared"] = "auto"
    estimation: StainEstimation = Field(default_factory=StainEstimation)
    overrides: tuple[StainOverride, ...] = ()
    reference: StainSource | None = None
    reference_profile: StainProfile | None = None
    normalization: Literal["colors", "colors_intensity"] = "colors"
    export_channels: tuple[int, ...] = (0, 1)
    channel_dtype: Literal["float32", "uint16"] = "float32"
    export_normalized: bool = False

    @model_validator(mode="after")
    def validate_recipe(self) -> Self:
        """Reject inconsistent channel selections and profile identities."""
        if len(self.profile.names) == 3 and self.vector_mode == "auto":
            raise ValueError("Automatic estimation supports two stains. Use shared/manual vectors for three stains.")
        if len(set(self.export_channels)) != len(self.export_channels) or any(
            i not in (0, 1, 2) for i in self.export_channels
        ):
            raise ValueError("Select each output channel at most once.")
        if not self.export_channels and not self.export_normalized:
            raise ValueError("Select stain channels or normalized RGB to export.")
        if self.export_normalized and self.reference is None:
            raise ValueError("Select a reference before exporting normalized RGB.")
        if len({item.source.key for item in self.overrides}) != len(self.overrides):
            raise ValueError("Only one override is allowed per image scene.")
        for profile in (self.reference_profile, *(item.profile for item in self.overrides)):
            if profile is not None and profile.names != self.profile.names:
                raise ValueError("Source and reference stain names/order must match the selected profile.")
        return self


class StainOutcome(BaseModel):
    """Completed output paths or an actionable per-image batch failure."""

    source: StainSource
    outputs: tuple[Path, ...] = ()
    error: str = ""
