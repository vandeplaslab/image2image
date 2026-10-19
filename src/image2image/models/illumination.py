"""Serializable inputs and recipes for illumination correction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class IlluminationSource(BaseModel):
    """One explicitly selected 2D scene with native channel and spatial metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: Path
    scene: int = 0
    shape: tuple[int, int]
    channels: tuple[str, ...]
    dtype: str
    spacing: tuple[float, float] | None = None
    rgb: bool = False
    size: int
    modified_ns: int


class CorrectionRecipe(BaseModel):
    """A shared channel-ordered recipe; radii use native pixels and heights use intensity units."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    channels: tuple[str, ...] = ()
    dark: IlluminationSource | None = None
    flat: IlluminationSource | None = None
    dark_channels: tuple[int, ...] = ()
    flat_channels: tuple[int, ...] = ()
    flat_dark_corrected: bool = False
    minimum_response: float = Field(default=1e-6, gt=0, lt=1)
    background: Literal["none", "gaussian", "morphology", "rolling_ball"] = "none"
    polarity: Literal["bright", "dark"] = "bright"
    sigma: float = Field(default=50, gt=0, le=4096)
    radius: int = Field(default=50, ge=1, le=4096)
    height: float = Field(default=50, gt=0)
    output_dtype: Literal["float32", "source", "uint8", "uint16"] = "float32"


class CorrectionOutcome(BaseModel):
    """The completed path or a recoverable per-image batch failure."""

    source: IlluminationSource
    output: Path | None = None
    error: str = ""
