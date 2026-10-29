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
    PairAnalysis,
    StudyAnalysis,
    ValidationAnalysis,
    ValidationCandidate,
    ValidationImage,
    ValidationLandmark,
    ValidationPair,
    ValidationSession,
    ValidationStudy,
)

CSV_COLUMNS = ("id", "fixed_x", "fixed_y", "moving_x", "moving_y")
CSV_OPTIONAL_COLUMNS = ("enabled", "note", "source")


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
    if candidate.kind == "aligned":
        return "identity-physical-v1"
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


def validation_image_pyramid(
    reader: Any, channel: int, max_size: int = 1024, *, rgb: bool = False, labels: bool = False
) -> list[Any]:
    """Return lazy scalar or RGB levels, reducing only spatial axes for large sources.

    Label levels are reduced by striding so that no new label values are created.
    """
    import dask.array as da

    levels = [
        reader.get_channel(0 if rgb else channel, level, split_rgb=not rgb) for level in range(reader.n_in_pyramid)
    ]
    if not levels or any(
        level.ndim != (3 if rgb else 2) or (rgb and level.shape[-1] not in (3, 4)) for level in levels
    ):
        raise ValueError("Validate requires 2D scalar channels or interleaved RGB images.")
    while max(levels[-1].shape[:2]) > max_size and min(levels[-1].shape[:2]) >= 2:
        source = da.asarray(levels[-1])
        if labels:
            levels.append(source[::2, ::2])
        else:
            levels.append(da.coarsen(np.mean, source, {0: 2, 1: 2}, trim_excess=True).astype(source.dtype))
    return levels


def validation_contrast_limits(levels: list[Any]) -> tuple[float, float]:
    """Estimate display contrast from a bounded sample of the coarsest image level."""
    source = levels[-1]
    sample = np.asarray(
        source[:: max(1, math.ceil(source.shape[0] / 256)), :: max(1, math.ceil(source.shape[1] / 256))]
    )
    finite = sample[np.isfinite(sample)]
    if not finite.size:
        return 0.0, 1.0
    low, high = (float(value) for value in np.percentile(finite, (1, 99)))
    if low == high:
        high = low + (1.0 if np.issubdtype(sample.dtype, np.integer) else max(abs(low) * 0.01, 0.001))
    return low, high


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
            geometry=ImageGeometry(shape=reader.image_shape, spacing=spacing or _suggest_image_spacing(reader)),
            channel_names=["R", "G", "B"] if reader.is_rgb else reader.get_channel_names(split_rgb=True),
            is_rgb=reader.is_rgb,
            display_channel="rgb" if reader.is_rgb else None,
            reader_kws=reader_kws,
            fingerprint=image_fingerprint(path),
            calibration_source="user" if spacing is not None else "unconfirmed",
            calibration_confirmed=spacing is not None,
        )
    finally:
        reader.close()


def _suggest_image_spacing(reader: Any) -> tuple[float, float]:
    """Preserve separate TIFF X/Y calibration when the reader exposes a scalar resolution."""
    from lxml import etree

    handle = getattr(reader, "fh", None)
    metadata = getattr(handle, "ome_metadata", None)
    if metadata:
        images = etree.fromstring(
            metadata.encode(), parser=etree.XMLParser(resolve_entities=False, no_network=True)
        ).findall("{*}Image")
        pixels = images[getattr(reader, "series_index", 0)].find("{*}Pixels")
        if pixels is not None and all(f"PhysicalSize{axis}" in pixels.attrib for axis in "XY"):
            factors = {"um": 1.0, "nm": 0.001, "mm": 1000.0, "cm": 10_000.0, "m": 1_000_000.0}
            values = []
            for axis in "XY":
                unit = pixels.get(f"PhysicalSize{axis}Unit", "um").replace("\u00b5", "u").replace("\u03bc", "u")
                if unit not in factors:
                    break
                values.append(float(pixels.attrib[f"PhysicalSize{axis}"]) * factors[unit])
            if len(values) == 2 and all(np.isfinite(value) and value > 0 for value in values):
                return values[0], values[1]
    elif handle is not None and hasattr(handle, "series"):
        tags = handle.series[getattr(reader, "series_index", 0)].levels[0].pages[0].tags
        unit = tags.get("ResolutionUnit")
        factor = {2: 25_400.0, 3: 10_000.0}.get(unit.value if unit else None)
        if factor and "XResolution" in tags and "YResolution" in tags:
            x, y = tags["XResolution"].value, tags["YResolution"].value
            if x[0] > 0 and y[0] > 0:
                return factor * x[1] / x[0], factor * y[1] / y[0]
    return tuple(float(value) for value in reader.scale[::-1])


def _resolve_image_path(path: str, parent: Path) -> Path:
    result = Path(path).expanduser()
    return result.resolve() if result.is_absolute() else (parent / result).resolve()


def discover_validation_candidates(
    path: str | Path,
    *,
    errors: list[str] | None = None,
) -> list[tuple[ValidationCandidate, ValidationImage, ValidationImage]]:
    """Discover native pairs; optionally collect failed Elastix paths instead of aborting import."""
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
        images: list[ValidationImage] = []
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
            if spacing is not None:
                images[-1].calibration_source = "registration"
            if entry.get("image_shape") and tuple(entry["image_shape"]) != images[-1].geometry.shape:
                raise ValueError("Image geometry no longer matches the saved Register image shape.")
        candidate = ValidationCandidate(
            name=path.name.removesuffix(".i2r.json").removesuffix(".i2r.toml"), kind="register", path=str(path)
        )
        candidate.fingerprint = transform_fingerprint(candidate)
        return [(candidate, images[0], images[1])]
    root = path if path.is_dir() else path.parent
    if root.suffix.lower() in (".valis", ".i2valis"):
        raise ValueError("Valis validation is not supported yet. Choose a Register file or Elastix project.")
    config = json.loads((root / "project.config.json").read_text())
    if not (root / "registered-project.config.json").exists():
        raise ValueError("Elastix registration is incomplete: registered-project.config.json is missing.")
    result: list[tuple[ValidationCandidate, ValidationImage, ValidationImage]] = []
    for registration in config["registration_paths"].values():
        try:
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
                images[-1].calibration_source = "registration"
            candidate = ValidationCandidate(
                name=f"{root.stem}: {source} to {target}", kind="elastix", path=str(root), source=source
            )
            candidate.fingerprint = transform_fingerprint(candidate)
            result.append((candidate, images[0], images[1]))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
            if errors is None:
                raise
            errors.append(f"{registration.get('source', '?')} to {registration.get('target', '?')}: {exc}")
    if not result and errors is None:
        raise ValueError("No registration paths were found.")
    return result


def require_same_pair(
    session: ValidationSession, fixed: ValidationImage, moving: ValidationImage, *, allow_user_calibration: bool = False
) -> None:
    """Reject a candidate belonging to another pair or native coordinate system."""
    for selected, actual in ((session.fixed, fixed), (session.moving, moving)):
        if selected is None:
            raise ValueError("Select both images before analysis.")
        if (
            Path(selected.path).resolve() != Path(actual.path).resolve()
            or (
                selected.geometry != actual.geometry
                and not (
                    allow_user_calibration
                    and selected.calibration_source == "user"
                    and selected.calibration_confirmed
                    and selected.geometry.shape == actual.geometry.shape
                )
            )
            or selected.reader_kws != actual.reader_kws
        ):
            raise ValueError("Candidate image pair or native calibration does not match this session.")


def adopt_validation_calibration(image: ValidationImage, reference: ValidationImage | None) -> None:
    """Reuse confirmed geometry only when a registration omits calibration for the same native image."""
    if (
        image.calibration_source == "unconfirmed"
        and reference is not None
        and reference.calibration_confirmed
        and Path(image.path).resolve() == Path(reference.path).resolve()
        and image.geometry.shape == reference.geometry.shape
        and image.reader_kws == reference.reader_kws
    ):
        image.geometry = reference.geometry.model_copy()
        image.calibration_source = reference.calibration_source
        image.calibration_confirmed = True


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

    def inverse_points(
        self, xy: np.ndarray, initial: np.ndarray | None = None, *, tolerance: float = 1e-3, iterations: int = 20
    ) -> np.ndarray:
        """Find fixed points mapping onto moving XY points; non-converged points are NaN.

        Matrix transforms are inverted exactly. Other transforms use Newton iterations with a
        finite-difference Jacobian, starting from ``initial`` (default: the moving coordinates).
        """
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        if self.matrix is not None:
            try:
                inverse = np.linalg.inv(self.matrix)
            except np.linalg.LinAlgError:
                return np.full_like(xy, np.nan)
            homogeneous = np.column_stack((xy, np.ones(len(xy)))) @ inverse.T
            with np.errstate(divide="ignore", invalid="ignore"):
                return homogeneous[:, :2] / homogeneous[:, 2:3]
        estimate = xy.copy() if initial is None else np.asarray(initial, dtype=np.float64).reshape(-1, 2).copy()
        step = max(1e-3, float(np.nanmax(np.abs(xy))) * 1e-6) if xy.size else 1e-3
        converged = np.zeros(len(xy), dtype=bool)
        for _ in range(iterations):
            mapped = self.map_points(estimate)
            residual = mapped - xy
            converged = np.isfinite(residual).all(axis=1) & (np.linalg.norm(residual, axis=1) < tolerance)
            if converged.all():
                break
            columns = [(self.map_points(estimate + offset) - mapped) / step for offset in ((step, 0.0), (0.0, step))]
            jacobian = np.stack(columns, axis=-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                try:
                    delta = np.linalg.solve(jacobian, residual[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    break
            update = ~converged & np.isfinite(delta).all(axis=1)
            if not update.any():
                break
            estimate[update] -= delta[update]
        estimate[~converged] = np.nan
        return estimate

    def jacobian_at(self, xy: np.ndarray, step: float = 1.0) -> np.ndarray:
        """Return the 2x2 finite-difference Jacobian of the fixed-to-moving map at each XY point."""
        xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
        columns = [
            (self.map_points(xy + offset) - self.map_points(xy - offset)) / (2 * step)
            for offset in ((step, 0.0), (0.0, step))
        ]
        return np.stack(columns, axis=-1)

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

    if candidate.kind == "aligned":
        return ValidationTransform(candidate, matrix=np.eye(3))
    discovery_errors: list[str] = []
    discovered = discover_validation_candidates(candidate.path, errors=discovery_errors)
    match = next((item for item in discovered if item[0].source == candidate.source), None)
    if match is None:
        detail = "\n".join(discovery_errors)
        raise ValueError(f"Registration path {candidate.source!r} is unavailable." + (f"\n{detail}" if detail else ""))
    current, fixed, moving = match
    adopt_validation_calibration(fixed, session.fixed)
    adopt_validation_calibration(moving, session.moving)
    require_same_pair(session, fixed, moving, allow_user_calibration=True)
    if candidate.fingerprint and candidate.fingerprint != current.fingerprint:
        raise ValueError("Registration changed since it was added. Remove and re-add this candidate to accept it.")
    loaded = ValidationTransform(candidate.model_copy(update={"fingerprint": current.fingerprint}))
    if candidate.kind == "elastix":
        import SimpleITK as sitk
        from image2image_reg.elastix.transform_sequence import TransformSequence
        from image2image_reg.workflows.elastix import ElastixReg

        project = ElastixReg.from_path(candidate.path, on_error="ignore")
        if candidate.source not in project.transformations:
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
        if session.fixed.geometry != fixed.geometry or session.moving.geometry != moving.geometry:
            transforms = []
            for source_geometry, target_geometry in (
                (moving.geometry, session.moving.geometry),
                (session.fixed.geometry, fixed.geometry),
            ):
                ratio = np.asarray(target_geometry.spacing) / source_geometry.spacing
                affine = sitk.AffineTransform(2)
                affine.SetMatrix(np.diag(ratio).ravel().tolist())
                affine.SetTranslation((np.asarray(target_geometry.origin) - ratio * source_geometry.origin).tolist())
                transforms.append(affine)
            loaded.itk_transform = sitk.CompositeTransform([transforms[0], loaded.itk_transform, transforms[1]])
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
    fixed_geometry, moving_geometry = session.fixed.geometry, session.moving.geometry
    source = np.diag([*moving_geometry.spacing, 1.0])
    target = np.diag([*fixed_geometry.spacing, 1.0])
    source[:2, 2], target[:2, 2] = moving_geometry.origin, fixed_geometry.origin
    try:
        loaded.matrix = source @ np.linalg.inv(matrix) @ np.linalg.inv(target)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Registration matrix is singular.") from exc
    loaded.fitting = [
        ValidationLandmark(id=f"fit-{i + 1}", fixed=tuple(f), moving=tuple(m))
        for i, (f, m) in enumerate(
            zip(fixed_geometry.to_physical(points_fixed), moving_geometry.to_physical(points_moving), strict=True)
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


def save_validation_study(path: str | Path, study: ValidationStudy) -> Path:
    """Atomically save the versioned collection without embedding image data."""
    path = Path(path)
    if not path.name.endswith(".i2validate.json"):
        path = path.with_name(path.name + ".i2validate.json")
    validated = ValidationStudy.model_validate(study.model_dump())
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(validated.model_dump_json(indent=2) + "\n")
    temporary.replace(path)
    return path


def load_validation_study(path: str | Path) -> tuple[ValidationStudy, list[str]]:
    """Open a study or migrate a legacy pair, retaining input-change warnings."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") == "1.0":
        study = ValidationStudy(pairs=[ValidationPair(session=ValidationSession.model_validate(payload))])
    else:
        study = ValidationStudy.model_validate(payload)
    changes = [f"{pair.name}: {change}" for pair in study.pairs for change in validation_input_changes(pair.session)]
    return study, changes


def validation_input_changes(session: ValidationSession, include_masks: bool = True) -> list[str]:
    """Check provenance before analysis or reuse of a saved session."""
    changes = []
    images = (
        (session.fixed, session.moving, session.fixed_mask, session.moving_mask)
        if include_masks
        else (session.fixed, session.moving)
    )
    for item in images:
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


def _parse_enabled(value: str) -> bool:
    value = value.strip().lower()
    if value not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"enabled must be true or false, not {value!r}")
    return value in ("true", "1", "yes")


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
                extra: dict[str, Any] = {}
                if (row.get("enabled") or "").strip():
                    extra["enabled"] = _parse_enabled(row["enabled"])
                if row.get("note"):
                    extra["note"] = row["note"]
                if (row.get("source") or "").strip():
                    extra["source"] = row["source"].strip()
                result.append(ValidationLandmark(id=row["id"].strip(), fixed=tuple(f), moving=tuple(m), **extra))
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid landmark on CSV line {line}: {exc}") from exc
    if len({p.id for p in result}) != len(result):
        raise ValueError("CSV landmark IDs must be unique.")
    return result


def merge_landmarks(
    existing: list[ValidationLandmark], added: list[ValidationLandmark]
) -> tuple[list[ValidationLandmark], dict[str, str]]:
    """Append landmarks, suffixing IDs that collide with existing ones; return the renames."""
    used = {point.id for point in existing} | {point.id for point in added}
    taken = {point.id for point in existing}
    renames: dict[str, str] = {}
    merged = list(existing)
    for point in added:
        if point.id in taken:
            suffix = 2
            while f"{point.id}-{suffix}" in used:
                suffix += 1
            renames[point.id] = f"{point.id}-{suffix}"
            point = point.model_copy(update={"id": renames[point.id]})
            used.add(point.id)
        taken.add(point.id)
        merged.append(point)
    return merged, renames


def export_validation_landmarks(path: str | Path, session: ValidationSession) -> None:
    """Export all pairs using the session's selected native-pixel or physical units."""
    if session.fixed is None or session.moving is None:
        raise ValueError("Both images are required.")
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((*CSV_COLUMNS, *CSV_OPTIONAL_COLUMNS))
        for pair in session.landmarks:
            f, m = np.array(pair.fixed), np.array(pair.moving)
            if session.landmark_units == "px":
                f, m = session.fixed.geometry.to_pixels(f), session.moving.geometry.to_pixels(m)
            writer.writerow([pair.id, *f, *m, str(pair.enabled).lower(), pair.note, pair.source])


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


def landmark_warnings(
    session: ValidationSession, transforms: list[ValidationTransform], pixels: float = 2.0
) -> dict[str, str]:
    """Flag included landmarks lying within a few native pixels of a known fitting landmark.

    These are not excluded: nearby structures can be legitimately independent, but re-clicked
    fitting points would make validation optimistic, so they are reported for review.
    """
    warnings = {}
    fitting = [point for transform in transforms for point in transform.fitting]
    if not fitting or session.fixed is None or session.moving is None:
        return warnings
    fixed_fit = np.array([point.fixed for point in fitting])
    moving_fit = np.array([point.moving for point in fitting])
    fixed_radius = pixels * max(session.fixed.geometry.spacing)
    moving_radius = pixels * max(session.moving.geometry.spacing)
    for point in session.landmarks:
        if (np.linalg.norm(fixed_fit - point.fixed, axis=1) <= fixed_radius).any() or (
            np.linalg.norm(moving_fit - point.moving, axis=1) <= moving_radius
        ).any():
            warnings[point.id] = f"Within {pixels:g} pixels of a fitting landmark; confirm independence"
    return warnings


def landmark_errors(
    transform: ValidationTransform,
    landmarks: list[ValidationLandmark],
    exclusions: dict[str, str] | None = None,
    warnings: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure physical residuals G(fixed)-moving in the moving image's frame."""
    exclusions = exclusions or {}
    warnings = warnings or {}
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
                "warning": "" if reason else warnings.get(point.id, ""),
                "source": point.source,
            }
        )
    summary = distance_statistics(np.array([row["error_um"] for row in rows if row["error_um"] is not None]))
    summary.update(total=len(landmarks), eligible=len(landmarks) - len(exclusions), units="um", frame="moving")
    return rows, summary


def source_statistics(rows: list[dict[str, Any]], exclusions: dict[str, str], *, automatic: bool) -> dict[str, Any]:
    """Summarize independent (manual and assisted) or automatic landmark rows separately.

    Automatic statistics are empty when no automatic landmarks were supplied.
    """
    subset = [row for row in rows if (row.get("source") == "automatic") == automatic]
    if automatic and not subset:
        return {}
    summary: dict[str, Any] = distance_statistics(
        np.array([row["error_um"] for row in subset if row["error_um"] is not None])
    )
    summary.update(
        total=len(subset),
        eligible=sum(row["id"] not in exclusions for row in subset),
        units="um",
        frame="moving",
    )
    if automatic:
        summary["outliers"] = sum(row["excluded"] == "Automatic outlier" for row in subset)
        summary["reason"] = "" if summary["count"] >= 3 else "Fewer than 3 automatic landmarks were measured."
    return summary


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
    if session.fixed is None or session.moving is None:
        raise ValueError("Select both images before analysis.")
    if session.fixed_mask is None and session.moving_mask is None:
        mask_reason = "No segmentation masks supplied. Region overlap is optional."
    elif session.fixed_mask is None or session.moving_mask is None:
        mask_reason = "Provide both fixed and moving segmentation masks to measure region overlap."
    elif not Path(session.fixed_mask.path).exists() or not Path(session.moving_mask.path).exists():
        mask_reason = "A segmentation mask file is missing. Reload both masks to measure region overlap."
    elif not session.mask_pairs:
        mask_reason = "No corresponding region labels selected. Add label pairs in Region overlap (optional)."
    else:
        mask_reason = ""
    changes = validation_input_changes(session, include_masks=not mask_reason)
    if changes:
        raise ValueError("\n".join(changes))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = analysis_grid(session)
    transforms = []
    # Load all definitions first: otherwise an unreadable candidate could hide fitting-point leakage.
    for candidate in session.candidates or [
        ValidationCandidate(id="aligned-images", name="Already aligned images", kind="aligned", path="")
    ]:
        check_cancel(cancel)
        if progress:
            progress(f"Loading {candidate.name}")
        transforms.append(load_validation_transform(candidate, session))
    session.candidates = [transform.candidate for transform in transforms]
    exclusions = landmark_exclusions(session, transforms)
    warnings = landmark_warnings(session, transforms)
    fixed_image, _ = sample_image(session.fixed, grid, cancel=cancel)
    tile_um = session.auto_tile_um or 64 * max(grid.spacing)
    analyses = _analyze_candidates(
        session, grid, transforms, exclusions, warnings, fixed_image, mask_reason, output_dir, tile_um, cancel, progress
    )
    check_cancel(cancel)
    changes = validation_input_changes(session, include_masks=not mask_reason)
    if changes:
        raise ValueError("Inputs changed during analysis.\n" + "\n".join(changes))
    versions = {}
    for package in ("image2image", "image2image-io", "image2image-reg", "numpy", "SimpleITK"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unavailable"
    return ValidationAnalysis(session, grid, analyses, versions, datetime.now(UTC).isoformat())


def _analyze_candidates(
    session: ValidationSession,
    grid: ImageGeometry,
    transforms: list[ValidationTransform],
    exclusions: dict[str, str],
    warnings: dict[str, str],
    fixed_image: np.ndarray,
    mask_reason: str,
    output_dir: Path,
    tile_um: float,
    cancel: Event | None,
    progress: Callable[[str], None] | None,
) -> list[CandidateAnalysis]:
    """Measure each candidate and stream its maps to disk."""
    from image2image.utils.validation_auto import residual_shift_map

    fixed_mask = fixed_mask_coverage = None
    if not mask_reason:
        fixed_mask, fixed_mask_coverage = sample_image(session.fixed_mask, grid, labels=True, cancel=cancel)
    analyses = []
    for index, transform in enumerate(transforms):
        check_cancel(cancel)
        candidate = transform.candidate
        if progress:
            progress(f"Analyzing {index + 1}/{len(transforms)}: {candidate.name}")
        result = CandidateAnalysis(id=candidate.id, name=candidate.name, fingerprint=candidate.fingerprint)
        result.mask_reason = mask_reason
        result.landmarks, _ = landmark_errors(transform, session.landmarks, exclusions, warnings)
        result.validation = source_statistics(result.landmarks, exclusions, automatic=False)
        result.automatic = source_statistics(result.landmarks, exclusions, automatic=True)
        _, result.fitting = landmark_errors(transform, transform.fitting)
        result.fitting["reason"] = "" if transform.fitting else "Original fitting landmarks unavailable"
        if candidate.kind == "aligned":
            result.fitting["reason"] = "Already aligned images: validation landmarks are measured directly, not fitted."
        maps, result.deformation = deformation_maps(transform, grid, session.moving.geometry, cancel)
        result.deformation["reference"] = (
            "Identity mapping in calibrated physical space; original registration deformation is unknown."
            if candidate.kind == "aligned"
            else "Imported registration transform."
        )
        maps["fixed_image"] = fixed_image
        maps["registered_image"], maps["image_coverage"] = sample_image(session.moving, grid, transform, cancel=cancel)
        if session.auto_residual_map:
            if progress:
                progress(f"Residual shift map {index + 1}/{len(transforms)}: {candidate.name}")
            shift_maps, result.residual_map = residual_shift_map(
                fixed_image, maps["registered_image"], maps["image_coverage"], grid, tile_um, cancel=cancel
            )
            maps.update(shift_maps)
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
    return analyses


def analyze_validation_study(
    study: ValidationStudy,
    output_dir: str | Path,
    cancel: Event | None = None,
    progress: Callable[[str], None] | None = None,
) -> StudyAnalysis:
    """Analyze pairs sequentially, isolating pair failures while honoring cancellation."""
    snapshot = ValidationStudy.model_validate(study.model_dump())
    results = []
    for index, pair in enumerate(snapshot.pairs):
        check_cancel(cancel)
        if progress:
            progress(f"Pair {index + 1}/{len(snapshot.pairs)}: {pair.name}")
        try:
            analysis = analyze_validation_session(
                pair.session, Path(output_dir) / f"pair-{index + 1}", cancel, progress
            )
        except AnalysisCancelled:
            raise
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            results.append(PairAnalysis(pair.id, pair.name, error=str(exc)))
        else:
            results.append(PairAnalysis(pair.id, pair.name, analysis=analysis))
    check_cancel(cancel)
    return StudyAnalysis(snapshot, results)
