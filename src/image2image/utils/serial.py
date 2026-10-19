"""Headless registration and reconstruction of serial sections."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import numpy as np

from image2image.models.wsiprep import Registration

if TYPE_CHECKING:
    from image2image_io.readers import BaseReader
    from image2image_reg.workflows import ElastixReg


@dataclass(frozen=True)
class SectionInfo:
    """Native image metadata used to validate and calibrate a stack."""

    shape: tuple[int, int]
    spacing: float
    channels: tuple[str, ...]
    dtype: np.dtype
    rgb: bool


def load_serial_project(path: str | Path) -> Registration:
    """Read a wsiprep project, resolving relative source and result paths."""
    from image2image.models.utilities import _read_config_from_file

    path = Path(path).resolve()
    data = _read_config_from_file(path)
    if data.get("tool") != "wsiprep" or not data.get("serial"):
        raise ValueError("Expected a wsiprep project with Serial Sections settings.")
    registration = Registration.model_validate(data)
    for model in registration.images.values():
        if not model.path.is_absolute():
            model.path = (path.parent / model.path).resolve()
    for attr in ("result_dir", "volume_path"):
        value = getattr(registration.serial, attr)
        if value and not Path(value).is_absolute():
            setattr(registration.serial, attr, str((path.parent / value).resolve()))
    return registration


def save_serial_project(path: str | Path, registration: Registration) -> None:
    """Atomically save serial settings alongside the existing wsiprep schema."""
    from image2image.utils.utilities import write_project

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid4().hex}{path.suffix}")
    try:
        write_project(temporary, registration.to_dict())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_sections(registration: Registration) -> dict[str, SectionInfo]:
    """Check channel compatibility and read physical metadata without loading pixels."""
    from image2image_io.readers import get_simple_reader

    settings = registration.serial
    if settings is None:
        raise ValueError("Serial settings are missing.")
    settings.validate_sections(registration)
    infos = {}
    for key in settings.ordered_keys(registration):
        if not registration.images[key].path.exists():
            raise FileNotFoundError(f"Section source is missing: {registration.images[key].path}")
        reader = get_simple_reader(registration.images[key].path, init_pyramid=False, auto_pyramid=False)
        try:
            if reader.is_rgb and reader.shape[-1] != 3:
                raise ValueError(f"Expected three-channel RGB for section {key}.")
            channels = tuple(reader.channel_names)
            if reader.is_rgb and len(channels) != 3:
                channels = ("R", "G", "B")
            info = SectionInfo(
                tuple(int(size) for size in reader.image_shape),
                float(reader.resolution),
                channels,
                np.dtype(reader.dtype),
                bool(reader.is_rgb),
            )
            if len(info.shape) != 2 or not np.isfinite(info.spacing) or info.spacing <= 0:
                raise ValueError(f"Invalid 2D image calibration: {key}")
            if info.dtype.kind not in "uif":
                raise ValueError(f"Unsupported intensity type: {info.dtype}")
            if infos:
                first = next(iter(infos.values()))
                if (info.channels, info.dtype, info.rgb) != (first.channels, first.dtype, first.rgb):
                    raise ValueError(f"Channel names/order, RGB layout and dtype must match across sections: {key}")
            if settings.registration_channel >= len(info.channels):
                raise ValueError(f"Registration channel is unavailable in {key}.")
            infos[key] = info
        finally:
            reader.close()
    return infos


def registration_fingerprint(registration: Registration, infos: dict[str, SectionInfo] | None = None) -> str:
    """Identify registration inputs, excluding volume geometry and display settings."""
    settings = registration.serial
    if settings is None:
        raise ValueError("Serial settings are missing.")
    infos = infos or inspect_sections(registration)
    images = []
    for key in settings.ordered_keys(registration):
        model = registration.images[key]
        stat = model.path.stat()
        info = infos[key]
        images.append(
            {
                "path": str(model.path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "number": settings.section_numbers[key],
                "rotate": float(model.rotate),
                "flip": bool(model.flip_lr),
                "translate": [float(model.translate_x), float(model.translate_y)],
                "spacing": info.spacing,
                "shape": info.shape,
                "channels": info.channels,
                "dtype": str(info.dtype),
                "rgb": info.rgb,
            }
        )
        if model.path.is_dir():
            # Zarr chunks can change without updating the root directory's mtime.
            contents = hashlib.sha256()
            for parent, directories, files in os.walk(model.path):
                directories.sort()
                for name in sorted(files):
                    item = Path(parent) / name
                    item_stat = item.stat()
                    contents.update(
                        json.dumps(
                            [str(item.relative_to(model.path)), item_stat.st_size, item_stat.st_mtime_ns]
                        ).encode()
                    )
            images[-1]["store_contents"] = contents.hexdigest()
    data = {
        "images": images,
        "reference": settings.section_numbers[settings.reference],
        "preset": settings.preset,
        "channel": settings.registration_channel,
        "downsample": settings.registration_downsample,
        "pixel_cap": settings.max_registration_pixels,
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def reference_canvas(registration: Registration, info: SectionInfo) -> tuple[float, float, float, float]:
    """Return the prepared reference extent in physical XY coordinates."""
    from image2image_reg.elastix.transform_utils import compute_rotation_bounds

    model = registration.images[registration.serial.reference]
    width, height = compute_rotation_bounds(info.shape, model.rotate % 360)
    width = np.ceil(np.ceil(width) + model.translate_x / info.spacing) * info.spacing
    height = np.ceil(np.ceil(height) + model.translate_y / info.spacing) * info.spacing
    if min(width, height) <= 0:
        raise ValueError("Reference translation removes the entire output canvas.")
    return 0.0, 0.0, float(width), float(height)


def build_serial_registration(
    registration: Registration, output_dir: Path, infos: dict[str, SectionInfo]
) -> ElastixReg:
    """Create an Elastix project with neighboring sections converging on the reference."""
    from image2image_reg.enums import CoordinateFlip
    from image2image_reg.models import Preprocessing
    from image2image_reg.workflows import ElastixReg

    settings = registration.serial
    settings.validate_sections(registration)
    project = ElastixReg(name="sections", output_dir=output_dir, cache=True, merge=False)
    presets = {"rigid": ["rigid"], "affine": ["rigid", "affine"], "deformable": ["rigid", "affine", "nl"]}
    names = {key: f"section_{settings.section_numbers[key]}" for key in infos}
    for key, info in infos.items():
        model = registration.images[key]
        pre = Preprocessing.brightfield() if info.rgb else Preprocessing.fluorescence()
        if not info.rgb:
            pre.channel_indices = [settings.registration_channel]
            pre.channel_names = [info.channels[settings.registration_channel]]
        pre.downsample = settings.registration_downsample
        pre.rotate_counter_clockwise = model.rotate % 360
        if model.flip_lr:
            pre.flip = CoordinateFlip.HORIZONTAL
        # Preparation controls are forward transforms; resampling uses their inverse.
        pre.translate_x = -model.translate_x
        pre.translate_y = -model.translate_y
        project.add_modality(names[key], model.path, pixel_size=info.spacing, preprocessing=pre)
    for source, target, through in settings.registration_paths(registration):
        project.add_registration_path(
            names[source], names[target], transform=presets[settings.preset], through=names.get(through)
        )
    project.save()
    return project


def register_serial(path: str | Path, progress: Callable[[str], None] = print) -> Path:
    """Register a saved stack and persist a manifest only after all transforms succeed."""
    import SimpleITK as sitk

    path = Path(path).resolve()
    registration = load_serial_project(path)
    infos = inspect_sections(registration)
    settings = registration.serial
    fingerprint = registration_fingerprint(registration, infos)
    run_dir = path.parent / "serial-runs" / uuid4().hex
    run_dir.mkdir(parents=True)
    save_serial_project(run_dir / "input.i2wsiprep.json", registration)
    project = build_serial_registration(registration, run_dir, infos)
    progress("Registering serial sections")
    project.register(max_registration_pixels=settings.max_registration_pixels, minimum_pyramid_size=16)
    if not project.is_registered:
        raise RuntimeError("Registration did not complete for every section.")
    transforms = {}
    for key in infos:
        name = f"section_{settings.section_numbers[key]}"
        transform = project.get_resampling_transform(name, max_registration_pixels=settings.max_registration_pixels)
        output = run_dir / f"{name}.h5"
        sitk.WriteTransform(transform, str(output))
        transforms[str(settings.section_numbers[key])] = output.name
    manifest = {"fingerprint": fingerprint, "transforms": transforms}
    (run_dir / "transforms.json").write_text(json.dumps(manifest, indent=2))
    settings.result_dir = str(run_dir)
    settings.registered_fingerprint = fingerprint
    settings.volume_path = None
    # Do not replace edits made while the subprocess was running.
    latest = load_serial_project(path)
    if registration_fingerprint(latest) != fingerprint:
        raise RuntimeError(f"Project changed during registration. Results are retained in {run_dir}.")
    latest.serial.result_dir = settings.result_dir
    latest.serial.registered_fingerprint = fingerprint
    latest.serial.volume_path = None
    save_serial_project(path, latest)
    progress(f"Registration complete: {run_dir}")
    return run_dir


def reconstruct_serial(
    path: str | Path,
    output: str | Path,
    progress: Callable[[str], None] = print,
    cancelled: Callable[[], bool] | None = None,
) -> Path:
    """Stream registered native channels into a calibrated OME-Zarr volume."""
    import SimpleITK as sitk
    from image2image_io.readers import get_simple_reader
    from image2image_io.volume import write_volume
    from image2image_reg.elastix.tiles import resample_region

    registration = load_serial_project(path)
    infos = inspect_sections(registration)
    settings = registration.serial
    fingerprint = registration_fingerprint(registration, infos)
    if fingerprint != settings.registered_fingerprint or not settings.result_dir:
        raise ValueError("Registration is missing or stale. Register the current sections first.")
    run_dir = Path(settings.result_dir)
    manifest = json.loads((run_dir / "transforms.json").read_text())
    section_numbers = {key: str(settings.section_numbers[key]) for key in infos}
    if manifest["fingerprint"] != fingerprint or set(manifest["transforms"]) != set(section_numbers.values()):
        raise ValueError("Registration manifest does not match the current sections.")
    info = infos[settings.reference]
    spacing = settings.output_spacing_um or max(item.spacing for item in infos.values())
    left, top, width, height = settings.canvas_xywh or reference_canvas(registration, info)
    first, last = min(settings.section_numbers.values()), max(settings.section_numbers.values())
    by_z = {settings.section_numbers[key] - first: key for key in infos}
    shape = (len(info.channels), last - first + 1, int(np.ceil(height / spacing)), int(np.ceil(width / spacing)))
    snapshot = Path(path).resolve().parent / "serial-runs" / uuid4().hex / "reconstruction.i2wsiprep.json"
    save_serial_project(snapshot, registration)
    reader: BaseReader | None = None
    transform: sitk.Transform | None = None
    current_key: str | None = None

    def read_tile(c: int, z: int, y: int, x: int, h: int, w: int) -> np.ndarray:
        nonlocal reader, transform, current_key
        key = by_z[z]
        if current_key != key:
            if reader is not None:
                reader.close()
                reader = None
            reader = get_simple_reader(registration.images[key].path, init_pyramid=False, auto_pyramid=False)
            transform = sitk.ReadTransform(str(run_dir / manifest["transforms"][section_numbers[key]]))
            current_key = key
        return resample_region(
            reader, transform, c, origin=(left, top), spacing=(spacing, spacing), region=(y, x, h, w)
        )

    try:
        result = write_volume(
            output,
            shape=shape,
            dtype=info.dtype,
            spacing=(settings.spacing_um, spacing, spacing),
            origin=(0.0, top, left),
            channel_names=info.channels,
            present=[z in by_z for z in range(shape[1])],
            read_tile=read_tile,
            provenance={
                "project": registration.to_dict(),
                "project_snapshot": str(snapshot),
                "fingerprint": fingerprint,
                "first_section": first,
                "rgb": info.rgb,
            },
            progress=progress,
            cancelled=cancelled,
        )
    finally:
        if reader is not None:
            reader.close()
    latest = load_serial_project(path)
    if latest.to_dict() == registration.to_dict():
        latest.serial.volume_path = str(result.resolve())
        save_serial_project(path, latest)
    return result
