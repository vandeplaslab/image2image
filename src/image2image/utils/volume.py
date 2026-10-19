"""Inspect, reconcile, and stream already aligned slides into OME-Zarr."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from image2image.models.volume import (
    MissingChannelPolicy,
    VolumeChannel,
    VolumeExportPlan,
    VolumeExportSlide,
    VolumeSelection,
    VolumeSlide,
    VolumeSource,
)

if TYPE_CHECKING:
    from image2image_io.readers import BaseReader

DEFAULT_SECTION_PATTERN = r"(?i)(?:^|[^a-z0-9])(?:slide|section|sec)[_ .-]*(?P<section>\d+)(?=$|[^0-9])"


def detect_slide_number(filename: str, pattern: str = "") -> tuple[int | None, str]:
    """Extract exactly one section number; never guess among multiple matches."""
    try:
        regex = re.compile(pattern or DEFAULT_SECTION_PATTERN)
    except re.error as exc:
        raise ValueError(f"Invalid filename expression: {exc}") from exc
    if not regex.groups:
        raise ValueError("The filename expression needs a 'section' group or one capture group.")
    matches = list(regex.finditer(filename))
    if len(matches) != 1:
        return None, "No slide number found" if not matches else "Ambiguous slide numbers"
    value = matches[0].group("section" if "section" in regex.groupindex else 1)
    if value is None or not re.fullmatch(r"[0-9]+", value):
        return None, "Slide number must be a nonnegative integer"
    return int(value), ""


def open_volume_source(path: Path, scene_index: int = 0) -> BaseReader:
    """Open an independent, lazy 2D reader for the exact selected scene."""
    from image2image_io.config import CONFIG
    from image2image_io.readers import (
        TIFF_EXTENSIONS,
        CziImageReader,
        CziSceneImageReader,
        OmeZarrImageReader,
        TiffImageReader,
        get_czi_metadata,
        get_simple_reader,
    )

    if path.suffix.lower() in TIFF_EXTENSIONS:
        return TiffImageReader(path, series_index=scene_index, init_pyramid=False, auto_pyramid=False)
    if path.suffix.lower() == ".czi":
        if len(get_czi_metadata(path)) > 1:
            return CziSceneImageReader(path, scene_index=scene_index, init_pyramid=False, auto_pyramid=False)
        return CziImageReader(path, init_pyramid=False, auto_pyramid=False)
    if path.suffix.lower() == ".zarr":
        return OmeZarrImageReader(path, init_pyramid=False, auto_pyramid=False)
    settings = CONFIG.init_pyramid, CONFIG.auto_pyramid, CONFIG.quiet
    try:
        return get_simple_reader(path, scene_index=scene_index, init_pyramid=False, auto_pyramid=False)
    finally:
        CONFIG.init_pyramid, CONFIG.auto_pyramid, CONFIG.quiet = settings


def inspect_volume_file(path: str | Path, pattern: str = "") -> VolumeSlide:
    """Read scene metadata and close every reader; multi-scene selection stays unset."""
    from image2image_io.readers import TIFF_EXTENSIONS, get_czi_metadata

    path = Path(path).resolve()
    scenes = [0]
    if path.suffix.lower() in TIFF_EXTENSIONS:
        from tifffile import TiffFile

        with TiffFile(path) as handle:
            scenes = list(range(len(handle.series)))
            for series in handle.series:
                if any(axis in "ZT" and size > 1 for axis, size in zip(series.axes, series.shape, strict=True)):
                    raise ValueError(f"{path.name}: Z/T stacks are not supported; supply 2D slides.")
    elif path.suffix.lower() == ".czi":
        scenes = [item["scene_index"] or 0 for item in get_czi_metadata(path).values()]
    sources = []
    for scene in scenes:
        reader = open_volume_source(path, scene)
        try:
            czi_handle = reader.fh
            if (
                path.suffix.lower() == ".czi"
                and czi_handle is not None
                and any(axis in "ZT" and size > 1 for axis, size in zip(czi_handle.axes, czi_handle.shape, strict=True))
            ):
                raise ValueError(f"{path.name}: Z/T stacks are not supported; supply 2D slides.")
            if reader.reader_type != "image":
                raise ValueError(f"{path.name}: supply a microscopy image.")
            shape = tuple(reader.image_shape)
            if len(shape) != 2 or min(shape) <= 0:
                raise ValueError(f"{path.name}, scene {scene}: select a 2D image, not a Z/T stack.")
            channels = tuple(reader.channel_names)
            if reader.is_rgb:
                if reader.n_channels != 3:
                    raise ValueError(f"{path.name}: only three-channel RGB images are supported.")
                channels = ("R", "G", "B")
            if not channels:
                raise ValueError(f"{path.name}: no channels found.")
            sources.append(
                VolumeSource(
                    path,
                    scene,
                    (int(shape[0]), int(shape[1])),
                    float(reader.resolution),
                    channels,
                    np.dtype(reader.dtype).name,
                    reader.is_rgb,
                )
            )
        finally:
            reader.close()
    number, error = detect_slide_number(path.name, pattern)
    return VolumeSlide(path, tuple(sources), scenes[0] if len(scenes) == 1 else None, number, number, error)


def channel_key(name: str) -> str:
    """Normalize source labels without changing their scientific meaning."""
    return name.strip().casefold()


def source_channel(slide: VolumeSlide, channel: VolumeChannel) -> tuple[int | None, str]:
    """Resolve an explicit override or a unique normalized source name."""
    source = slide.source
    if source is None:
        return None, "Select a scene"
    if slide.key in channel.overrides:
        index = channel.overrides[slide.key]
        if index is None:
            return None, ""
        if type(index) is not int or not 0 <= index < len(source.channel_names):
            return None, "Invalid source channel"
        return index, ""
    matches = [i for i, name in enumerate(source.channel_names) if channel_key(name) == channel.key]
    if len(matches) > 1:
        return None, "Ambiguous source channels; choose one explicitly"
    return (matches[0] if matches else None), ""


def reconcile_channels(slides: Sequence[VolumeSlide], channels: Sequence[VolumeChannel]) -> list[VolumeChannel]:
    """Append newly discovered channels while preserving existing edits and choices."""
    if not slides:
        return []
    result = list(channels)
    known = {channel.key for channel in result}
    included = [slide for slide in slides if slide.included]
    if any(slide.source is None for slide in included):
        return result
    for slide in slides:
        if slide.source is None:
            continue
        for name in slide.source.channel_names:
            key = channel_key(name)
            if key not in known:
                channel = VolumeChannel(key, name.strip())
                channel.selected = bool(included) and all(
                    source_channel(item, channel)[0] is not None for item in included
                )
                result.append(channel)
                known.add(key)
    return result


def resolve_volume_selection(
    slides: Sequence[VolumeSlide], channels: Sequence[VolumeChannel], policy: MissingChannelPolicy
) -> VolumeSelection:
    """Calculate effective inclusion without mutating any user selection."""
    if policy not in ("channels", "slides"):
        raise ValueError("Unknown missing-channel policy.")
    included = [slide for slide in slides if slide.included]
    selected = [channel for channel in channels if channel.selected]
    errors, missing = [], []
    for slide in included:
        absent = []
        for channel in selected:
            index, error = source_channel(slide, channel)
            if error:
                errors.append(f"{slide.path.name} / {channel.name}: {error}")
            if index is None:
                absent.append(channel.key)
        missing.append((slide.key, tuple(absent)))
    incomplete = {key for _, names in missing for key in names}
    channel_keys = tuple(channel.key for channel in selected if policy == "slides" or channel.key not in incomplete)
    slide_keys = tuple(key for key, names in missing if policy == "channels" or not names)
    return VolumeSelection(
        slide_keys,
        channel_keys,
        tuple(slide.key for slide in slides if slide.key not in slide_keys),
        tuple(channel.key for channel in channels if channel.key not in channel_keys),
        tuple(missing),
        tuple(errors),
    )


def _positions(slides: Sequence[VolumeSlide]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for slide in slides:
        position = slide.position
        if position is None or type(position) is not int or position < 0:
            raise ValueError("Every loaded slide needs a nonnegative integer Z position.")
        positions[slide.key] = position
    if len(set(positions.values())) != len(positions):
        raise ValueError("Duplicate Z positions: assign a unique position to each slide.")
    return positions


def reorder_slides(slides: Sequence[VolumeSlide], keys: Sequence[str]) -> list[VolumeSlide]:
    """Move images between existing numbered slots, retaining every gap."""
    if len(keys) != len(slides) or set(keys) != {slide.key for slide in slides}:
        raise ValueError("Reordering must contain every slide exactly once.")
    positions = _positions(slides)
    lookup = {slide.key: slide for slide in slides}
    result = [lookup[key] for key in keys]
    for slide, position in zip(result, sorted(positions.values()), strict=True):
        slide.position = position
        slide.manual_position = True
    return result


def build_volume_plan(
    slides: Sequence[VolumeSlide],
    channels: Sequence[VolumeChannel],
    *,
    policy: MissingChannelPolicy = "channels",
    z_spacing: float,
    xy_spacing: float | None = None,
    tile_size: int = 512,
) -> VolumeExportPlan:
    """Validate inputs and freeze source mappings, geometry, and exclusions."""
    if not slides or len({slide.key for slide in slides}) != len(slides):
        raise ValueError("Load at least one slide; each file can appear only once.")
    positions = _positions(slides)
    if len({channel.key for channel in channels}) != len(channels):
        raise ValueError("Output channel identities must be unique.")
    selection = resolve_volume_selection(slides, channels, policy)
    if selection.errors:
        raise ValueError("\n".join(selection.errors))
    if not selection.channel_keys or not selection.slide_keys:
        raise ValueError("Select at least one complete channel and slide for export.")
    active_channels = [channel for channel in channels if channel.key in selection.channel_keys]
    names = tuple(channel.name.strip() for channel in active_channels)
    if any(not name for name in names) or len({channel_key(name) for name in names}) != len(names):
        raise ValueError("Final channel names must be nonempty and unique.")
    active_slides = [slide for slide in slides if slide.key in selection.slide_keys]
    sources = []
    for slide in active_slides:
        source = slide.source
        if source is None:
            raise ValueError("Select one 2D scene for every included file.")
        sources.append(source)
    if any(not math.isfinite(source.pixel_size) or source.pixel_size <= 0 for source in sources):
        raise ValueError("Source pixel sizes must be finite and positive.")
    if any(len(source.shape) != 2 or min(source.shape) <= 0 for source in sources):
        raise ValueError("Every source must have positive 2D dimensions.")
    dtypes = {np.dtype(source.dtype) for source in sources}
    if len(dtypes) != 1 or next(iter(dtypes)).kind not in "uif":
        raise ValueError("Included slides must have the same real integer or floating point dtype.")
    xy = max(source.pixel_size for source in sources) if xy_spacing is None else xy_spacing
    if not all(math.isfinite(value) and value > 0 for value in (z_spacing, xy)):
        raise ValueError("Set positive, finite Z and XY spacing in um.")
    if type(tile_size) is not int or tile_size <= 0:
        raise ValueError("Tile size must be a positive integer.")
    first, last = min(positions.values()), max(positions.values())
    if last - first >= 1_000_000:
        raise ValueError("The Z range exceeds one million slices. Check the detected section numbers.")
    shape = (
        len(names),
        last - first + 1,
        math.ceil(max(source.shape[0] * source.pixel_size for source in sources) / xy),
        math.ceil(max(source.shape[1] * source.pixel_size for source in sources) / xy),
    )
    snapshots = []
    for slide in sorted(slides, key=lambda slide: positions[slide.key]):
        included = slide.key in selection.slide_keys
        indices = []
        if included:
            for channel in active_channels:
                index, _ = source_channel(slide, channel)
                if index is None:
                    raise ValueError(f"Missing channel {channel.name} in {slide.path.name}.")
                indices.append(index)
        snapshots.append(VolumeExportSlide(slide.path, positions[slide.key], slide.source, tuple(indices), included))
    present_positions = {slide.position for slide in snapshots if slide.included}
    return VolumeExportPlan(
        tuple(snapshots),
        names,
        shape,
        next(iter(dtypes)).name,
        (z_spacing, xy, xy),
        first,
        tuple(position in present_positions for position in range(first, last + 1)),
        tile_size,
        policy,
        tuple(channel.name for channel in channels if channel.key in selection.excluded_channels),
        tuple(
            (path, tuple(next(channel.name for channel in channels if channel.key == key) for key in missing))
            for path, missing in selection.missing
        ),
    )


def export_volume(
    plan: VolumeExportPlan,
    output: str | Path,
    *,
    progress: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Stream a frozen plan using independent readers and transactional storage."""
    import SimpleITK as sitk
    from image2image_io.volume import write_volume
    from image2image_reg.elastix.tiles import resample_region

    output = Path(output)
    if not output.name.endswith(".ome.zarr"):
        raise ValueError("The output name must end with .ome.zarr.")
    by_z = {slide.position - plan.first_position: slide for slide in plan.slides if slide.included}
    reader: BaseReader | None = None
    current_z: int | None = None
    identity = sitk.Transform(2, sitk.sitkIdentity)

    def read_tile(c: int, z: int, y: int, x: int, height: int, width: int) -> np.ndarray:
        nonlocal reader, current_z
        slide = by_z[z]
        source = slide.source
        if source is None:
            raise ValueError(f"Missing source scene for {slide.path}.")
        if current_z != z:
            if reader is not None:
                reader.close()
                reader = None
            reader = open_volume_source(slide.path, source.scene_index)
            current_z = z
            if (tuple(reader.image_shape), np.dtype(reader.dtype).name, reader.resolution) != (
                source.shape,
                source.dtype,
                source.pixel_size,
            ):
                raise ValueError(f"Source metadata changed after inspection: {slide.path}")
            if not source.rgb and tuple(reader.channel_names) != source.channel_names:
                raise ValueError(f"Source channels changed after inspection: {slide.path}")
        if reader is None:
            raise ValueError(f"Source reader is unavailable: {slide.path}")
        return resample_region(
            reader,
            identity,
            slide.channel_indices[c],
            origin=(0.0, 0.0),
            spacing=plan.spacing[1:],
            region=(y, x, height, width),
        )

    try:
        return write_volume(
            output,
            shape=plan.shape,
            dtype=np.dtype(plan.dtype),
            spacing=plan.spacing,
            origin=(0.0, 0.0, 0.0),
            channel_names=plan.channel_names,
            present=plan.present,
            read_tile=read_tile,
            tile_size=plan.tile_size,
            progress=progress,
            cancelled=cancelled,
            provenance={
                "app": "volume",
                "first_section": plan.first_position,
                "missing_channel_policy": plan.policy,
                "excluded_channels": list(plan.excluded_channels),
                "missing_channels": dict(plan.missing),
                "spacing_um": list(plan.spacing),
                "channel_names": list(plan.channel_names),
                "slides": [
                    {
                        "path": str(slide.path),
                        "scene_index": slide.source.scene_index if slide.source else None,
                        "position": slide.position,
                        "included": slide.included,
                        "channel_indices": list(slide.channel_indices),
                        "source_channel_names": list(slide.source.channel_names) if slide.source else [],
                    }
                    for slide in plan.slides
                ],
            },
        )
    finally:
        if reader is not None:
            reader.close()
