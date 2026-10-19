"""Physical-coordinate validation, bounded image sampling, and session persistence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

from image2image.models.validation import (
    CandidateAnalysis,
    ImageGeometry,
    ValidationAnalysis,
    ValidationCandidate,
    ValidationImage,
    ValidationLandmark,
    ValidationSession,
)

CSV_COLUMNS = ("id", "fixed_x", "fixed_y", "moving_x", "moving_y")


class AnalysisCancelled(Exception):
    """Raised at bounded work boundaries when the user cancels analysis."""


def check_cancel(cancel: Event | None) -> None:
    """Stop work without publishing partial results."""
    if cancel is not None and cancel.is_set():
        raise AnalysisCancelled("Analysis cancelled.")


def image_fingerprint(path: str | Path) -> str:
    """Fingerprint image file metadata without reading entire whole-slide images."""
    path = Path(path).resolve()
    paths = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    if not path.exists():
        raise FileNotFoundError(path)
    payload = [
        (str(p.relative_to(path) if path.is_dir() else p.name), p.stat().st_size, p.stat().st_mtime_ns) for p in paths
    ]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def transform_fingerprint(candidate: ValidationCandidate) -> str:
    """Hash registration definitions, transform chains, and cached geometry metadata."""
    path = Path(candidate.path)
    if candidate.kind == "register":
        paths = [path]
        root = path.parent
    else:
        root = path if path.is_dir() else path.parent
        paths = sorted(
            {
                root / "project.config.json",
                root / "registered-project.config.json",
                *(root / "Transformations").glob("*.json"),
                *(root / "Cache").glob("*.json"),
            }
        )
    if not paths:
        raise ValueError(f"No registration definitions found in {path}.")
    digest = hashlib.sha256(candidate.source.encode())
    for item in paths:
        digest.update(str(item.relative_to(root)).encode())
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def open_validation_reader(image: ValidationImage) -> Any:
    """Open a lazy 2D reader without generating image pyramids."""
    from image2image_io.readers import get_simple_reader

    if not Path(image.path).exists():
        raise FileNotFoundError(f"Image missing: {image.path}")
    reader = get_simple_reader(
        image.path, init_pyramid=False, auto_pyramid=False, quick=True, scene_index=image.reader_kws.get("scene_index")
    )
    if tuple(reader.image_shape) != image.geometry.shape:
        reader.close()
        raise ValueError(f"Image shape changed for {image.path}.")
    return reader


def inspect_validation_image(
    path: str | Path, spacing: tuple[float, float] | None = None, reader_kws: dict | None = None
) -> ValidationImage:
    """Read native shape and calibration; explicit registration spacing takes precedence."""
    from image2image_io.readers import get_simple_reader

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image missing: {path}")
    reader_kws = reader_kws or {}
    reader = get_simple_reader(
        path, init_pyramid=False, auto_pyramid=False, quick=True, scene_index=reader_kws.get("scene_index")
    )
    try:
        if reader.get_channel(0, split_rgb=True).ndim != 2:
            raise ValueError("Only 2D image channels are supported by Validate.")
        return ValidationImage(
            path=str(path),
            geometry=ImageGeometry(
                shape=reader.image_shape, spacing=spacing or tuple(float(v) for v in reader.scale[::-1])
            ),
            channel_names=reader.get_channel_names(split_rgb=True),
            reader_kws=reader_kws,
            fingerprint=image_fingerprint(path),
        )
    finally:
        reader.close()


def _resolve_image_path(path: str, parent: Path) -> Path:
    result = Path(path).expanduser()
    return result.resolve() if result.is_absolute() else (parent / result).resolve()


def discover_validation_candidates(
    path: str | Path,
) -> list[tuple[ValidationCandidate, ValidationImage, ValidationImage]]:
    """Discover native image pairs in a Register file or an Elastix project."""
    from image2image.models.utilities import _read_config_from_file

    path = Path(path).resolve()
    if ".i2r." in path.name:
        config = _read_config_from_file(path)
        fixed_paths = config.get("fixed_paths", config.get("micro_paths"))
        moving_paths = config.get("moving_paths", config.get("ims_paths"))
        if not fixed_paths or not moving_paths:
            raise ValueError("Registration has no image references. Export it with the current Register app.")
        # A matrix has one native source/target grid. Multi-image projects require explicit selection in Register.
        if len(fixed_paths) != 1 or len(moving_paths) != 1:
            raise ValueError("Export one fixed/moving image pair from Register before validation.")
        images = []
        for paths in (fixed_paths, moving_paths):
            entry = paths[0]
            if isinstance(entry, str):
                entry = {"path": entry}
            spacing = float(entry["pixel_size_um"]) if "pixel_size_um" in entry else None
            images.append(
                inspect_validation_image(
                    _resolve_image_path(entry["path"], path.parent),
                    (spacing, spacing) if spacing is not None else None,
                    entry.get("reader_kws"),
                )
            )
            if entry.get("image_shape") and tuple(entry["image_shape"]) != images[-1].geometry.shape:
                raise ValueError("Image geometry no longer matches the saved Register image shape.")
        candidate = ValidationCandidate(
            name=path.name.removesuffix(".i2r.json").removesuffix(".i2r.toml"), kind="register", path=str(path)
        )
        candidate.fingerprint = transform_fingerprint(candidate)
        return [(candidate, *images)]
    root = path if path.is_dir() else path.parent
    config = json.loads((root / "project.config.json").read_text())
    if not (root / "registered-project.config.json").exists():
        raise ValueError("Elastix registration is incomplete: registered-project.config.json is missing.")
    result = []
    for registration in config["registration_paths"].values():
        source, target = registration["source"], registration["target"]
        images = []
        for name in (target, source):
            entry = config["modalities"][name]
            spacing = float(entry["pixel_size"])
            images.append(
                inspect_validation_image(
                    _resolve_image_path(entry["path"], root), (spacing, spacing), entry.get("reader_kws")
                )
            )
        candidate = ValidationCandidate(
            name=f"{root.stem}: {source} to {target}", kind="elastix", path=str(root), source=source
        )
        candidate.fingerprint = transform_fingerprint(candidate)
        result.append((candidate, *images))
    if not result:
        raise ValueError("No registration paths were found.")
    return result


def require_same_pair(session: ValidationSession, fixed: ValidationImage, moving: ValidationImage) -> None:
    """Reject a candidate belonging to another pair or native coordinate system."""
    for selected, actual in ((session.fixed, fixed), (session.moving, moving)):
        if selected is None:
            raise ValueError("Select both images before analysis.")
        if (
            Path(selected.path).resolve() != Path(actual.path).resolve()
            or selected.geometry != actual.geometry
            or selected.reader_kws != actual.reader_kws
        ):
            raise ValueError("Candidate image pair or native calibration does not match this session.")


@dataclass
class ValidationTransform:
    """A fixed-to-moving physical transform and its original fitting evidence."""

    candidate: ValidationCandidate
    matrix: np.ndarray | None = None
    itk_transform: Any = None
    fitting: list[ValidationLandmark] = field(default_factory=list)

    def map_points(self, xy: np.ndarray) -> np.ndarray:
        """Map physical XY points without mutating their input array."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.matrix is not None:
            homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ self.matrix.T
            with np.errstate(divide="ignore", invalid="ignore"):
                return homogeneous[:, :2] / homogeneous[:, 2:3]
        if self.itk_transform is None:
            raise ValueError("No transformation was loaded.")
        return np.asarray([self.itk_transform.TransformPoint(tuple(p)) for p in xy]).reshape(-1, 2)

    def map_grid(self, grid: ImageGeometry) -> np.ndarray:
        """Sample a rectangular physical grid, using ITK's vectorized field filter when available."""
        yy, xx = np.indices(grid.shape, dtype=np.float64)
        xy = grid.to_physical(np.stack((xx, yy), axis=-1))
        if self.itk_transform is None:
            return self.map_points(xy.reshape(-1, 2)).reshape(*grid.shape, 2)
        import SimpleITK as sitk

        field_image = sitk.TransformToDisplacementField(
            self.itk_transform, sitk.sitkVectorFloat64, list(grid.shape[::-1]), grid.origin, grid.spacing
        )
        return xy + sitk.GetArrayFromImage(field_image)


def load_validation_transform(candidate: ValidationCandidate, session: ValidationSession) -> ValidationTransform:
    """Load a complete transform, checking its pair and saved fingerprint before use."""
    from image2image.models.utilities import _read_config_from_file

    discovered = discover_validation_candidates(candidate.path)
    match = next((item for item in discovered if item[0].source == candidate.source), None)
    if match is None:
        raise ValueError(f"Registration path {candidate.source!r} no longer exists.")
    current, fixed, moving = match
    require_same_pair(session, fixed, moving)
    if candidate.fingerprint and candidate.fingerprint != current.fingerprint:
        raise ValueError("Registration changed since it was added. Remove and re-add this candidate to accept it.")
    loaded = ValidationTransform(candidate.model_copy(update={"fingerprint": current.fingerprint}))
    if candidate.kind == "elastix":
        import SimpleITK as sitk
        from image2image_reg.elastix.transform_sequence import TransformSequence
        from image2image_reg.workflows.elastix import ElastixReg

        project = ElastixReg.from_path(candidate.path)
        if candidate.source not in project.transformations or not project.is_registered:
            raise ValueError("Elastix registration or its transformation chain is incomplete.")
        # The workflow loader initializes these entries from an unprocessed wrapper.
        # Restore the saved original-reference mapping before composing the sampling transform.
        registered = json.loads((project.project_dir / project.REGISTERED_CONFIG_NAME).read_text())
        original = registered.get("original_size_transforms") or {}
        project.original_size_transforms.update({key: value for key, value in original.items() if value})
        target = project.registration_paths[candidate.source][-1]
        if not project.original_size_transforms.get(target):
            from image2image_reg.wrapper import ImageWrapper

            restored = ImageWrapper.load_original_size_transform(
                project.modalities[target], project.project_dir / "Cache"
            )
            if restored:
                project.original_size_transforms[target] = restored
            else:
                preprocessing = project.modalities[target].preprocessing
                if preprocessing and (
                    preprocessing.flip
                    or preprocessing.rotate_counter_clockwise
                    or preprocessing.translate_x
                    or preprocessing.translate_y
                    or preprocessing.affine is not None
                    or preprocessing.is_cropped()
                ):
                    raise ValueError("The preprocessed fixed image is missing its original-reference transform.")
        transform_name = f"{candidate.source}_to_{target}_transformations.json"
        transform_path = project.transformations_dir / transform_name
        if not transform_path.exists():
            transform_path = project.transformations_dir / f"{project.name}-{transform_name}"
        saved_chain = json.loads(transform_path.read_text())
        for index, edge in enumerate(project.transform_path_map[candidate.source]):
            if not saved_chain.get(f"{index:03d}-to-{edge['target']}"):
                raise ValueError("Elastix registration is missing a stage of its transformation chain.")
        preprocessing = project.modalities[candidate.source].preprocessing
        if (
            preprocessing
            and (
                preprocessing.flip
                or preprocessing.rotate_counter_clockwise
                or preprocessing.translate_x
                or preprocessing.translate_y
                or preprocessing.affine is not None
                or preprocessing.is_cropped()
            )
            and not any(value for key, value in saved_chain.items() if key.startswith("initial"))
        ):
            raise ValueError("The preprocessed moving image is missing its initial transform.")
        sequence = project.transformations[candidate.source]["full-transform-seq"].copy()
        restoration = project.original_size_transforms.get(target)
        if restoration:
            sequence.append(
                TransformSequence.from_list(restoration if isinstance(restoration, list) else [restoration])
            )
        loaded.itk_transform = sitk.Transform(sequence.final_transform)
        if loaded.itk_transform.GetDimension() != 2:
            raise ValueError("Only 2D transformations are supported.")
        return loaded
    config = _read_config_from_file(candidate.path)
    points_fixed = np.asarray(config.get("fixed_points_yx_px", []), dtype=float).reshape(-1, 2)[:, ::-1]
    points_moving = np.asarray(config.get("moving_points_yx_px", []), dtype=float).reshape(-1, 2)[:, ::-1]
    if len(points_fixed) != len(points_moving) or not np.isfinite([points_fixed, points_moving]).all():
        raise ValueError("Fitting point arrays are inconsistent or non-finite.")
    if "matrix_yx_px" in config:
        permute = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        matrix = permute @ np.asarray(config["matrix_yx_px"], dtype=float) @ permute
    elif "matrix_xy_px" in config:
        matrix = np.asarray(config["matrix_xy_px"], dtype=float)
    else:
        from image2image.utils.transform import compute_transform

        if len(points_fixed) < 3:
            raise ValueError("Missing matrix and insufficient original fitting landmarks.")
        matrix = compute_transform(points_moving, points_fixed, config["transformation_type"]).params
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("A finite 3x3 registration matrix is required.")
    # Saved fitting coordinates already have the display-only initial affine removed.
    source = np.diag([*moving.geometry.spacing, 1.0])
    target = np.diag([*fixed.geometry.spacing, 1.0])
    source[:2, 2], target[:2, 2] = moving.geometry.origin, fixed.geometry.origin
    try:
        loaded.matrix = source @ np.linalg.inv(matrix) @ np.linalg.inv(target)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Registration matrix is singular.") from exc
    loaded.fitting = [
        ValidationLandmark(id=f"fit-{i + 1}", fixed=tuple(f), moving=tuple(m))
        for i, (f, m) in enumerate(
            zip(fixed.geometry.to_physical(points_fixed), moving.geometry.to_physical(points_moving), strict=True)
        )
    ]
    return loaded


def save_validation_session(path: str | Path, session: ValidationSession) -> Path:
    """Save a session atomically, without changing any registration input."""
    path = Path(path)
    if not path.name.endswith(".i2validate.json"):
        path = path.with_name(path.name + ".i2validate.json")
    validated = ValidationSession.model_validate(session.model_dump())
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(validated.model_dump_json(indent=2) + "\n")
    temporary.replace(path)
    return path


def load_validation_session(path: str | Path) -> tuple[ValidationSession, list[str]]:
    """Reopen annotations and report missing or changed inputs without accepting them silently."""
    session = ValidationSession.model_validate_json(Path(path).read_text())
    return session, validation_input_changes(session)


def validation_input_changes(session: ValidationSession) -> list[str]:
    """Check provenance before analysis or reuse of a saved session."""
    changes = []
    for item in (session.fixed, session.moving, session.fixed_mask, session.moving_mask):
        if item is not None:
            try:
                if item.fingerprint and image_fingerprint(item.path) != item.fingerprint:
                    changes.append(f"Image changed: {item.path}. Reload it to accept the new input.")
            except OSError:
                changes.append(f"Image missing: {item.path}.")
    for candidate in session.candidates:
        try:
            if candidate.fingerprint and transform_fingerprint(candidate) != candidate.fingerprint:
                changes.append(f"Registration changed: {candidate.name}. Remove and re-add it.")
        except (OSError, ValueError):
            changes.append(f"Registration missing: {candidate.name}.")
    return changes


def import_validation_landmarks(path: str | Path, session: ValidationSession, units: str) -> list[ValidationLandmark]:
    """Read paired CSV coordinates; pixels refer to each image's native pixel centers."""
    if session.fixed is None or session.moving is None or units not in ("px", "um"):
        raise ValueError("Select both images and coordinate units (px or um).")
    result = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headings = reader.fieldnames or []
        if not set(CSV_COLUMNS).issubset(headings) or len(headings) != len(set(headings)):
            raise ValueError(f"CSV must contain {', '.join(CSV_COLUMNS)}.")
        for line, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(f"Too many fields on CSV line {line}.")
            try:
                f = np.array([float(row["fixed_x"]), float(row["fixed_y"])])
                m = np.array([float(row["moving_x"]), float(row["moving_y"])])
                if units == "px":
                    f, m = session.fixed.geometry.to_physical(f), session.moving.geometry.to_physical(m)
                result.append(ValidationLandmark(id=row["id"].strip(), fixed=tuple(f), moving=tuple(m)))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid landmark on CSV line {line}: {exc}") from exc
    if len({p.id for p in result}) != len(result):
        raise ValueError("CSV landmark IDs must be unique.")
    return result


def export_validation_landmarks(path: str | Path, session: ValidationSession) -> None:
    """Export all pairs using the session's selected native-pixel or physical units."""
    if session.fixed is None or session.moving is None:
        raise ValueError("Both images are required.")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(CSV_COLUMNS)
        for pair in session.landmarks:
            f, m = np.array(pair.fixed), np.array(pair.moving)
            if session.landmark_units == "px":
                f, m = session.fixed.geometry.to_pixels(f), session.moving.geometry.to_pixels(m)
            writer.writerow([pair.id, *f, *m])


def analysis_grid(session: ValidationSession) -> ImageGeometry:
    """Choose one capped fixed grid, shared by every candidate and mask comparison."""
    if session.fixed is None:
        raise ValueError("A fixed image is required.")
    native = session.fixed.geometry
    factor = max(session.downsample, math.ceil(math.sqrt(math.prod(native.shape) / session.max_pixels)))
    while math.prod(math.ceil(size / factor) for size in native.shape) > session.max_pixels:
        factor += 1
    return ImageGeometry(
        shape=tuple(math.ceil(v / factor) for v in native.shape),
        spacing=tuple(v * factor for v in native.spacing),
        origin=native.origin,
    )


def distance_statistics(distances: np.ndarray) -> dict[str, float | int | None]:
    """Summarize finite distances without representing missing evidence as zero error."""
    values = np.asarray(distances, dtype=float)
    values = values[np.isfinite(values)]
    result: dict[str, float | int | None] = dict.fromkeys(("mean", "median", "rms", "p95", "maximum"))
    result["count"] = len(values)
    if len(values):
        result.update(
            mean=float(np.mean(values)),
            median=float(np.median(values)),
            rms=float(np.sqrt(np.mean(values**2))),
            p95=float(np.percentile(values, 95)),
            maximum=float(np.max(values)),
        )
    return result


def landmark_exclusions(session: ValidationSession, transforms: list[ValidationTransform]) -> dict[str, str]:
    """Use the same validation subset across candidates, excluding any known fitting landmark."""
    excluded = {}
    fitting = [point for transform in transforms for point in transform.fitting]
    for point in session.landmarks:
        if not point.enabled:
            excluded[point.id] = point.note or "Excluded by user"
        elif session.fixed is None or session.moving is None:
            excluded[point.id] = "Image geometry is missing"
        elif not session.fixed.geometry.contains(np.array(point.fixed)) or not session.moving.geometry.contains(
            np.array(point.moving)
        ):
            excluded[point.id] = "Landmark lies outside a native image"
        elif any(
            np.allclose(point.fixed, fit.fixed, rtol=0, atol=1e-6)
            or np.allclose(point.moving, fit.moving, rtol=0, atol=1e-6)
            for fit in fitting
        ):
            excluded[point.id] = "Reuses a known fitting landmark"
    return excluded


def landmark_errors(
    transform: ValidationTransform, landmarks: list[ValidationLandmark], exclusions: dict[str, str] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure physical residuals G(fixed)-moving in the moving image's frame."""
    exclusions = exclusions or {}
    rows = []
    for point in landmarks:
        reason = exclusions.get(point.id, "")
        mapped = transform.map_points(np.array([point.fixed]))[0] if not reason else np.full(2, np.nan)
        residual = mapped - point.moving
        if not reason and not np.isfinite(residual).all():
            reason = "Transform produced non-finite coordinates"
        rows.append(
            {
                "id": point.id,
                "fixed_x": point.fixed[0],
                "fixed_y": point.fixed[1],
                "moving_x": point.moving[0],
                "moving_y": point.moving[1],
                "dx_um": float(residual[0]) if not reason else None,
                "dy_um": float(residual[1]) if not reason else None,
                "error_um": float(np.linalg.norm(residual)) if not reason else None,
                "excluded": reason,
            }
        )
    summary = distance_statistics(np.array([row["error_um"] for row in rows if row["error_um"] is not None]))
    summary.update(total=len(landmarks), eligible=len(landmarks) - len(exclusions), units="um", frame="moving")
    return rows, summary


def sample_pixels(
    array: Any,
    geometry: ImageGeometry,
    xy: np.ndarray,
    *,
    labels: bool = False,
    max_source_bytes: int = 64 * 1024 * 1024,
    cancel: Event | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a lazy 2D array with bounded source crops; labels always use nearest-neighbor."""
    check_cancel(cancel)
    xy = np.asarray(xy)
    shape = xy.shape[:-1]
    flat = geometry.to_pixels(xy.reshape(-1, 2))
    covered = geometry.contains(xy).reshape(-1)
    output = np.zeros(len(flat), dtype=array.dtype if labels else np.float32)
    indices = np.flatnonzero(covered)
    if len(indices):
        coords = flat[indices]
        low = np.maximum(0, np.floor(coords.min(axis=0)).astype(int) - 1)
        high = np.minimum(geometry.shape[::-1], np.ceil(coords.max(axis=0)).astype(int) + 2)
        if math.prod(high - low) * np.dtype(array.dtype).itemsize > max_source_bytes:
            if len(flat) < 2:
                raise MemoryError("A source interpolation footprint exceeds the memory limit.")
            split = len(flat) // 2
            for start, end in ((0, split), (split, len(flat))):
                out, valid = sample_pixels(
                    array,
                    geometry,
                    xy.reshape(-1, 2)[start:end],
                    labels=labels,
                    max_source_bytes=max_source_bytes,
                    cancel=cancel,
                )
                output[start:end], covered[start:end] = out, valid
        else:
            crop = array[low[1] : high[1], low[0] : high[0]]
            if hasattr(crop, "compute"):
                crop = crop.compute()
            crop = np.asarray(crop)
            if crop.ndim != 2:
                raise ValueError("Select a single 2D channel.")
            local = coords - low
            if labels:
                if crop.dtype.kind not in "bui" or (crop.dtype.kind == "i" and np.any(crop < 0)):
                    raise ValueError("Masks must contain nonnegative integer labels.")
                ij = np.floor(local + 0.5).astype(int)
                ij = np.clip(ij, 0, np.asarray(crop.shape[::-1]) - 1)
                output[indices] = crop[ij[:, 1], ij[:, 0]]
            else:
                from scipy.ndimage import map_coordinates

                output[indices] = map_coordinates(
                    crop.astype(np.float32), local[:, ::-1].T, order=1, mode="nearest", prefilter=False
                )
    return output.reshape(shape), covered.reshape(shape)


def sample_image(
    image: ValidationImage,
    grid: ImageGeometry,
    transform: ValidationTransform | None = None,
    *,
    labels: bool = False,
    cancel: Event | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read an image onto the common grid in small output tiles."""
    reader = open_validation_reader(image)
    try:
        array = reader.get_channel(image.channel, split_rgb=True)
        output = np.empty(grid.shape, dtype=array.dtype if labels else np.float32)
        covered = np.empty(grid.shape, dtype=bool)
        for y in range(0, grid.shape[0], 256):
            for x in range(0, grid.shape[1], 256):
                check_cancel(cancel)
                shape = (min(256, grid.shape[0] - y), min(256, grid.shape[1] - x))
                tile = ImageGeometry(
                    shape=shape, spacing=grid.spacing, origin=tuple(grid.to_physical(np.array([x, y])))
                )
                if transform is not None:
                    xy = transform.map_grid(tile)
                else:
                    yy, xx = np.indices(shape)
                    xy = tile.to_physical(np.stack((xx, yy), axis=-1))
                pixels, valid = sample_pixels(array, image.geometry, xy, labels=labels, cancel=cancel)
                output[y : y + shape[0], x : x + shape[1]] = pixels
                covered[y : y + shape[0], x : x + shape[1]] = valid
        return output, covered
    finally:
        reader.close()


def deformation_maps(
    transform: ValidationTransform,
    grid: ImageGeometry,
    moving_geometry: ImageGeometry,
    cancel: Event | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Differentiate G in physical XY coordinates; mask boundaries and unsupported samples."""
    mapped = np.empty((*grid.shape, 2), dtype=np.float64)
    for y in range(0, grid.shape[0], 128):
        check_cancel(cancel)
        tile = ImageGeometry(
            shape=(min(128, grid.shape[0] - y), grid.shape[1]),
            spacing=grid.spacing,
            origin=(grid.origin[0], grid.origin[1] + y * grid.spacing[1]),
        )
        mapped[y : y + tile.shape[0]] = transform.map_grid(tile)
    finite = np.isfinite(mapped).all(axis=-1)
    coverage = moving_geometry.contains(mapped)
    jacobian = np.full(grid.shape, np.nan, dtype=np.float32)
    valid = np.zeros(grid.shape, dtype=bool)
    if min(grid.shape) >= 3:
        dx = (mapped[1:-1, 2:] - mapped[1:-1, :-2]) / (2 * grid.spacing[0])
        dy = (mapped[2:, 1:-1] - mapped[:-2, 1:-1]) / (2 * grid.spacing[1])
        jacobian[1:-1, 1:-1] = dx[..., 0] * dy[..., 1] - dx[..., 1] * dy[..., 0]
        valid[1:-1, 1:-1] = (
            coverage[1:-1, 1:-1]
            & finite[1:-1, 1:-1]
            & finite[1:-1, 2:]
            & finite[1:-1, :-2]
            & finite[2:, 1:-1]
            & finite[:-2, 1:-1]
        )
    valid &= np.isfinite(jacobian)
    jacobian[~valid] = np.nan
    positive = valid & (jacobian > 0)
    area = np.full(grid.shape, np.nan, dtype=np.float32)
    log_area = np.full(grid.shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        area[positive] = 1 / jacobian[positive]
        log_area[positive] = -np.log2(jacobian[positive])
    nonpositive = valid & (jacobian <= 0)
    values = jacobian[valid]
    summary = {
        "valid_count": int(valid.sum()),
        "total_count": int(valid.size),
        "coverage": float(coverage.mean()),
        "derivative_coverage": float(valid.mean()),
        "nonpositive_fraction": float(nonpositive.sum() / valid.sum()) if valid.any() else None,
        "minimum": float(np.min(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
        "direction": "fixed-to-moving",
        "area_direction": "moving-to-fixed",
        "includes_global_affine": True,
        "spacing_xy_um": grid.spacing,
    }
    return {
        "jacobian": jacobian,
        "area_ratio": area,
        "log2_area_ratio": log_area,
        "valid": valid,
        "nonpositive": nonpositive,
        "coverage": coverage,
    }, summary


def mask_overlap(fixed: np.ndarray, moving: np.ndarray, pixel_area: float) -> dict[str, Any]:
    """Compute Dice and IoU for corresponding binary regions on the fixed analysis grid."""
    n_fixed, n_moving = int(np.count_nonzero(fixed)), int(np.count_nonzero(moving))
    intersection = int(np.count_nonzero(fixed & moving))
    union = n_fixed + n_moving - intersection
    return {
        "dice": 2 * intersection / (n_fixed + n_moving) if n_fixed + n_moving else None,
        "iou": intersection / union if union else None,
        "fixed_area_um2": n_fixed * pixel_area,
        "moving_area_um2": n_moving * pixel_area,
        "intersection_area_um2": intersection * pixel_area,
        "reason": "Both labels are empty on the analysis grid" if not union else "",
    }


def analyze_validation_session(
    session: ValidationSession,
    output_dir: str | Path,
    cancel: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> ValidationAnalysis:
    """Analyze a frozen session and stream candidate maps to disk, keeping memory bounded."""
    session = ValidationSession.model_validate(session.model_dump())
    if session.fixed is None or session.moving is None or not session.candidates:
        raise ValueError("Select both images and at least one registration.")
    if bool(session.fixed_mask) != bool(session.moving_mask):
        raise ValueError("Provide both corresponding masks or neither mask.")
    changes = validation_input_changes(session)
    if changes:
        raise ValueError("\n".join(changes))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = analysis_grid(session)
    transforms = []
    # Load all definitions first: otherwise an unreadable candidate could hide fitting-point leakage.
    for candidate in session.candidates:
        check_cancel(cancel)
        if progress:
            progress(f"Loading {candidate.name}")
        transforms.append(load_validation_transform(candidate, session))
    session.candidates = [transform.candidate for transform in transforms]
    exclusions = landmark_exclusions(session, transforms)
    fixed_image, _ = sample_image(session.fixed, grid, cancel=cancel)
    fixed_mask = fixed_mask_coverage = None
    if session.fixed_mask is not None:
        fixed_mask, fixed_mask_coverage = sample_image(session.fixed_mask, grid, labels=True, cancel=cancel)
    analyses = []
    for index, transform in enumerate(transforms):
        check_cancel(cancel)
        candidate = transform.candidate
        if progress:
            progress(f"Analyzing {index + 1}/{len(transforms)}: {candidate.name}")
        result = CandidateAnalysis(id=candidate.id, name=candidate.name, fingerprint=candidate.fingerprint)
        result.landmarks, result.validation = landmark_errors(transform, session.landmarks, exclusions)
        _, result.fitting = landmark_errors(transform, transform.fitting)
        result.fitting["reason"] = "" if transform.fitting else "Original fitting landmarks unavailable"
        maps, result.deformation = deformation_maps(transform, grid, session.moving.geometry, cancel)
        maps["fixed_image"] = fixed_image
        maps["registered_image"], maps["image_coverage"] = sample_image(session.moving, grid, transform, cancel=cancel)
        if fixed_mask is not None and session.moving_mask is not None:
            moving_mask, mask_coverage = sample_image(session.moving_mask, grid, transform, labels=True, cancel=cancel)
            maps.update(
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                mask_coverage=mask_coverage,
                fixed_mask_coverage=fixed_mask_coverage,
            )
            for pair in session.mask_pairs:
                values = mask_overlap(
                    fixed_mask == pair.fixed_label, moving_mask == pair.moving_label, math.prod(grid.spacing)
                )
                result.masks.append(
                    {
                        **pair.model_dump(),
                        **values,
                        "fixed_coverage": float(fixed_mask_coverage.mean()),
                        "moving_coverage": float(mask_coverage.mean()),
                        "joint_coverage": float((fixed_mask_coverage & mask_coverage).mean()),
                    }
                )
        result.map_path = output_dir / f"candidate-{index + 1}.npz"
        check_cancel(cancel)
        np.savez_compressed(result.map_path, **maps, origin_xy_um=grid.origin, spacing_xy_um=grid.spacing)
        analyses.append(result)
        del maps
    check_cancel(cancel)
    changes = validation_input_changes(session)
    if changes:
        raise ValueError("Inputs changed during analysis.\n" + "\n".join(changes))
    versions = {}
    for package in ("image2image", "image2image-io", "image2image-reg", "numpy", "SimpleITK"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return ValidationAnalysis(session, grid, analyses, versions, datetime.now(UTC).isoformat())
