"""Editable volume inputs and immutable export snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

MissingChannelPolicy = Literal["channels", "slides"]


@dataclass(frozen=True)
class VolumeSource:
    """Metadata for one 2D scene, without owning an open reader."""

    path: Path
    scene_index: int
    shape: tuple[int, int]
    pixel_size: float
    channel_names: tuple[str, ...]
    dtype: str
    rgb: bool = False


@dataclass
class VolumeSlide:
    """One file with an explicitly selected scene and physical section number."""

    path: Path
    sources: tuple[VolumeSource, ...]
    scene_index: int | None
    detected_number: int | None
    position: int | None
    detection_error: str = ""
    included: bool = True
    manual_position: bool = False

    @property
    def key(self) -> str:
        """Return a stable identifier independent of row order and scene selection."""
        return str(self.path)

    @property
    def source(self) -> VolumeSource | None:
        """Return the selected scene, or None until a scene has been chosen."""
        return next((source for source in self.sources if source.scene_index == self.scene_index), None)


@dataclass
class VolumeChannel:
    """An output channel whose identity is independent of its final label."""

    key: str
    name: str
    selected: bool = True
    overrides: dict[str, int | None] = field(default_factory=dict)


@dataclass(frozen=True)
class VolumeSelection:
    """Effective inclusion and explicit reasons for missing-channel exclusions."""

    slide_keys: tuple[str, ...]
    channel_keys: tuple[str, ...]
    excluded_slides: tuple[str, ...]
    excluded_channels: tuple[str, ...]
    missing: tuple[tuple[str, tuple[str, ...]], ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class VolumeExportSlide:
    """Frozen source and channel indices for a numbered output position."""

    path: Path
    position: int
    source: VolumeSource | None
    channel_indices: tuple[int, ...]
    included: bool


@dataclass(frozen=True)
class VolumeExportPlan:
    """Validated assembly settings detached from all editable GUI state."""

    slides: tuple[VolumeExportSlide, ...]
    channel_names: tuple[str, ...]
    shape: tuple[int, int, int, int]
    dtype: str
    spacing: tuple[float, float, float]
    first_position: int
    present: tuple[bool, ...]
    tile_size: int
    policy: MissingChannelPolicy
    excluded_channels: tuple[str, ...]
    missing: tuple[tuple[str, tuple[str, ...]], ...]
    sharding: bool = False
    dot_separator: bool = False
