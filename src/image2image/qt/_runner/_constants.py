"""Constants."""

from __future__ import annotations

import typing as ty
from dataclasses import dataclass
from pathlib import Path

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.elastix import ElastixReg
    from image2image_reg.workflows.valis import ValisReg

ProjectKind = ty.Literal["elastix", "valis"]
ReviewState = ty.Literal["unknown", "good", "bad"]
RUN_STATE_FILTER = ty.Literal["All", "Finished", "Running", "Queued", "Failed"]
REVIEW_STATE_FILTER = ty.Literal["All", "Unknown", "Good", "Bad"]
REVIEW_FILENAME = ".image2image-runner-review.json"
PROJECT_FILE_FILTER = (
    "Registration projects (*.json *.toml *.i2wsireg.json *.i2wsireg.toml *.wsireg *.i2reg *.config.json "
    "*.valis.json *.valis.toml *.valis valis.config.json);; "
    "Elastix projects (*.i2wsireg.json *.i2wsireg.toml *.wsireg *.i2reg *.config.json);; "
    "Valis projects (*.valis.json *.valis.toml *.valis valis.config.json *.config.json);;"
)
RUN_STATE_FILTERS: dict[RUN_STATE_FILTER, set[str]] = {
    "All": set(),
    "Finished": {"Finished"},
    "Running": {"Running", "In progress"},
    "Queued": {"Queued", "Already queued"},
    "Failed": {"Failed", "Invalid"},
}
REVIEW_STATES: set[str] = {"unknown", "good", "bad"}


@dataclass(frozen=True)
class RunnerProject:
    """Loaded registration project."""

    kind: ProjectKind
    project_dir: Path
    project: ElastixReg | ValisReg
