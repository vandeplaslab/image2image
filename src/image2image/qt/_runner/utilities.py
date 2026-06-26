"""Various utilities."""

from __future__ import annotations

import json
import typing as ty
from pathlib import Path

from loguru import logger

from image2image.qt._runner._constants import (
    REVIEW_FILENAME,
    REVIEW_STATE_FILTER,
    REVIEW_STATES,
    RUN_STATE_FILTER,
    RUN_STATE_FILTERS,
    ProjectKind,
    ReviewState,
    RunnerProject,
)


def has_registration_images(project_dir: Path) -> bool:
    """Return whether a project has completed images on disk."""
    image_dir = Path(project_dir) / "Images"
    if not image_dir.exists():
        return False
    return any(path.is_file() for path in image_dir.iterdir())


def discover_overlap_images(project_dir: Path) -> list[Path]:
    """Return sorted overlap preview PNG files for a project."""
    overlap_dir = Path(project_dir) / "Overlap"
    if not overlap_dir.exists():
        return []
    return sorted((path for path in overlap_dir.glob("*.png") if path.is_file()), key=lambda path: path.name.lower())


def read_review_state(project_dir: Path) -> ReviewState:
    """Read the persisted runner review state for a project."""
    path = Path(project_dir) / REVIEW_FILENAME
    if not path.exists():
        return "unknown"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Could not read runner review state from {path}: {exc}")
        return "unknown"
    state = data.get("review_state")
    if state in REVIEW_STATES:
        return ty.cast(ReviewState, state)
    logger.warning(f"Invalid runner review state in {path}: {state!r}")
    return "unknown"


def write_review_state(project_dir: Path, state: ReviewState) -> None:
    """Persist the runner review state for a project."""
    path = Path(project_dir) / REVIEW_FILENAME
    payload = {"review_state": state}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def project_matches_filters(
    name: str,
    status: str,
    review_state: ReviewState,
    name_filter: str,
    run_state_filter: RUN_STATE_FILTER,
    review_state_filter: REVIEW_STATE_FILTER,
) -> bool:
    """Return whether a project card should be visible."""
    name_filter = name_filter.strip().lower()
    if name_filter and name_filter not in name.lower():
        return False
    status_filter = RUN_STATE_FILTERS[run_state_filter]
    if status_filter and status not in status_filter:
        return False
    return not (review_state_filter != "All" and review_state != review_state_filter.lower())


def _path_to_project_dir(path: Path) -> Path:
    """Return a project directory for a dropped project path."""
    return path.parent if path.is_file() else path


def _preferred_project_kinds(path: Path) -> tuple[ProjectKind, ...]:
    """Return the preferred load order for a registration project path."""
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".valis" or name == "valis.config.json" or ".valis." in name:
        return "valis", "elastix"
    if suffix in {".wsireg", ".i2reg"} or ".i2wsireg." in name or ".i2reg." in name:
        return "elastix", "valis"
    if name.endswith(".config.json"):
        return "elastix", "valis"
    return "elastix", "valis"


def load_registration_project(path: Path) -> RunnerProject:
    """Load an Elastix or Valis registration project from a path."""
    errors = []
    for kind in _preferred_project_kinds(path):
        try:
            if kind == "elastix":
                return _load_elastix_project(path)
            return _load_valis_project(path)
        except (FileNotFoundError, ImportError, ValueError) as exc:
            errors.append(f"{kind}: {exc}")
    message = "; ".join(errors) if errors else "unsupported project"
    raise ValueError(f"Could not load registration project from {path}: {message}")


def _load_elastix_project(path: Path) -> RunnerProject:
    """Load an Elastix registration project."""
    from image2image_reg.workflows.elastix import ElastixReg

    project_dir = _path_to_project_dir(path)
    try:
        project = ElastixReg.from_path(project_dir, quick=True)
    except ValueError:
        ElastixReg.update_paths(project_dir, project_dir.parent)
        project = ElastixReg.from_path(project_dir, quick=True)
    return RunnerProject("elastix", Path(project.project_dir), project)


def _load_valis_project(path: Path) -> RunnerProject:
    """Load a Valis registration project."""
    from image2image_reg.workflows.valis import ValisReg

    project_dir = _path_to_project_dir(path)
    project = ValisReg.from_path(project_dir)
    return RunnerProject("valis", Path(project.project_dir), project)
