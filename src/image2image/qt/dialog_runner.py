"""Queue runner for Elastix and Valis registration projects."""

from __future__ import annotations

import typing as ty
from dataclasses import dataclass
from pathlib import Path

import qtextra.helpers as hp
import qtextra.queue.cli_queue as _q
from loguru import logger
from qtextra.config import THEMES
from qtextra.queue.popup import QUEUE, QueuePopup
from qtextra.queue.task import Task
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QListWidget,
    QMenuBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import image2image.constants as C
from image2image import __version__
from image2image.config import STATE, RunnerConfig, get_runner_config
from image2image.qt._dialog_base import Window

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.elastix import ElastixReg
    from image2image_reg.workflows.valis import ValisReg


ProjectKind = ty.Literal["elastix", "valis"]
PROJECT_FILE_FILTER = (
    "Registration projects (*.json *.toml *.i2wsireg.json *.i2wsireg.toml *.wsireg *.i2reg *.config.json "
    "*.valis.json *.valis.toml *.valis valis.config.json);; "
    "Elastix projects (*.i2wsireg.json *.i2wsireg.toml *.wsireg *.i2reg *.config.json);; "
    "Valis projects (*.valis.json *.valis.toml *.valis valis.config.json *.config.json);;"
)


@dataclass(frozen=True)
class RunnerProject:
    """Loaded registration project."""

    kind: ProjectKind
    project_dir: Path
    project: ElastixReg | ValisReg


def is_empty(path: Path) -> bool:
    """Check whether the directory is empty."""
    if not path.exists():
        return True
    return not any(path.iterdir())


class QtRunnerProjectCard(QFrame):
    """Card describing a loaded registration project."""

    evt_queue = Signal(object)
    evt_images = Signal(object)
    evt_network = Signal(object)
    evt_viewer = Signal(object)

    def __init__(self, project: RunnerProject, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.status = "Ready"
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        self.name_label = hp.make_label(
            self,
            f"<b>{project.project.name}</b>",
            enable_url=True,
            object_name="large_text",
        )
        self.summary_label = hp.make_label(self, self._summarize_project(), enable_url=True, wrap=True)
        self.status_label = hp.make_label(self, "Ready", object_name="tip_label")
        self.progress_label = hp.make_label(self, "Waiting to be queued.", wrap=True)

        self.queue_btn = hp.make_btn(
            self,
            "Queue",
            tooltip="Validate and add this project to the queue.",
            func=lambda: self.evt_queue.emit(self.project.project_dir),
        )
        self.images_btn = hp.make_btn(
            self,
            "Images...",
            tooltip="Show the project image list.",
            func=lambda: self.evt_images.emit(self.project.project_dir),
        )
        self.network_btn = hp.make_btn(
            self,
            "Network...",
            tooltip="Show the Elastix registration network."
            if project.kind == "elastix"
            else "Registration network preview is currently available for Elastix projects.",
            func=lambda: self.evt_network.emit(self.project.project_dir),
            disabled=project.kind != "elastix",
        )
        self.viewer_btn = hp.make_btn(
            self,
            "Open in viewer",
            tooltip="Open completed registration images in the viewer.",
            func=lambda: self.evt_viewer.emit(self.project.project_dir),
            disabled=is_empty(self.project.project_dir / "Images"),
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addWidget(self.name_label)
        layout.addWidget(self.summary_label)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Status"),
                self.status_label,
                spacing=2,
                stretch_id=(1,),
            )
        )
        layout.addWidget(self.progress_label)
        layout.addLayout(
            hp.make_h_layout(
                self.queue_btn,
                self.images_btn,
                self.network_btn,
                self.viewer_btn,
                spacing=2,
                stretch_after=True,
            )
        )

    @property
    def registration_model(self) -> ElastixReg | ValisReg:
        """Return the registration model for auxiliary viewers."""
        return self.project.project

    @property
    def modalities(self) -> list[ty.Any]:
        """Return project modalities."""
        return list(self.project.project.modalities.values())

    def set_status(self, status: str, progress: str = "") -> None:
        """Update card status and progress text."""
        self.status = status
        self.status_label.setText(status)
        # hp.disable_widgets(self.viewer_btn, disabled=status != "Finished")
        if progress:
            self.progress_label.setText(progress)

    def image_lines(self) -> list[str]:
        """Return a simple image list for the project."""
        lines = []
        for index, modality in enumerate(self.modalities, start=1):
            lines.append(f"{index}. {modality.name}: {modality.path}")
        return lines

    def _summarize_project(self) -> str:
        """Return a short project summary."""
        project = self.project.project
        n_modalities = len(project.modalities)
        output_dir = hp.hyper(project.output_dir, value=str(project.output_dir))
        project_dir = hp.hyper(project.project_dir, value=str(project.project_dir))
        return (
            f"<b>Type</b>: {self.project.kind.capitalize()} &nbsp; "
            f"<b>Modalities</b>: {n_modalities}<br>"
            f"<b>Project</b>: {project_dir}<br>"
            f"<b>Output</b>: {output_dir}"
        )


def _path_to_project_dir(path: Path) -> Path:
    """Return a project directory for a dropped project path."""
    return path.parent if path.is_file() else path


def _preferred_project_kinds(path: Path) -> tuple[ProjectKind, ...]:
    """Return the preferred load order for a registration project path."""
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".valis" or name in {"valis.config.json"} or ".valis." in name:
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


class ImageRunnerWindow(Window):
    """Queue runner for Elastix and Valis registration projects."""

    APP_NAME = "runner"
    CONFIG: RunnerConfig

    def __init__(
        self,
        parent: QWidget | None,
        run_check_version: bool = True,
        project_dir: str | Path | None = None,
        **_kwargs: ty.Any,
    ):
        self.CONFIG = get_runner_config()
        self.projects: dict[Path, RunnerProject] = {}
        self.cards: dict[Path, QtRunnerProjectCard] = {}
        self.task_to_project: dict[str, Path] = {}
        self._dialogs: list[QWidget] = []
        self.queue_popup: QueuePopup | None = None
        _q.N_PARALLEL = self.CONFIG.n_parallel
        QUEUE.n_parallel = self.CONFIG.n_parallel
        super().__init__(
            parent,
            f"image2image: Elastix/Valis Runner (v{__version__})",
            run_check_version=run_check_version,
        )
        QUEUE.add_action("Open in viewer", self.on_open_task_in_viewer, "viewer")
        if project_dir:
            self.on_add_project_paths([Path(project_dir)])

    @staticmethod
    def _setup_config() -> None:
        """Setup configuration."""

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self.evt_dropped, self.on_drop_projects, state=state)
        connect(QUEUE.evt_queued, self.on_task_queued, state=state)
        connect(QUEUE.evt_started, self.on_task_started, state=state)
        connect(QUEUE.evt_next, self.on_task_progress, state=state)
        connect(QUEUE.evt_progress, self.on_task_progress, state=state)
        connect(QUEUE.evt_finished, self.on_task_finished, state=state)
        connect(QUEUE.evt_errored, self.on_task_failed, state=state)
        connect(QUEUE.evt_cancelled, self.on_task_cancelled, state=state)
        connect(QUEUE.evt_remove_task, self.on_task_removed, state=state)
        connect(QUEUE.evt_empty, self.on_queue_empty, state=state)

    def on_add_project_files(self) -> None:
        """Select registration project files."""
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Elastix/Valis registration projects",
            self.CONFIG.last_dir,
            PROJECT_FILE_FILTER,
        )
        if paths:
            self.CONFIG.last_dir = str(Path(paths[0]).parent)
            self.on_add_project_paths([Path(path) for path in paths])

    def on_add_project_directory(self) -> None:
        """Select a registration project directory."""
        path = QFileDialog.getExistingDirectory(self, "Select Elastix/Valis registration project", self.CONFIG.last_dir)
        if path:
            self.CONFIG.last_dir = str(Path(path).parent)
            self.on_add_project_paths([Path(path)])

    def on_drop_projects(self, event: QDropEvent) -> None:
        """Load projects from a drag-and-drop event."""
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self.on_add_project_paths(paths)

    def on_add_project_paths(self, paths: list[Path]) -> None:
        """Load registration projects from paths."""
        loaded = 0
        for path in paths:
            try:
                project = load_registration_project(path)
            except (FileNotFoundError, ImportError, ValueError) as exc:
                logger.exception(f"Could not load registration project from {path}: {exc}")
                hp.toast(self, "Load failed", f"Could not load {hp.hyper(path)}.", icon="error", position="top_left")
                continue
            if project.project_dir in self.projects:
                self._update_project_status(project.project_dir, "Already loaded")
                continue
            self.projects[project.project_dir] = project
            self._add_project_card(project)
            loaded += 1
        if loaded:
            hp.toast(self, "Loaded projects", f"Loaded {loaded} registration project(s).", icon="success")
            self._refresh_progress_report()

    def on_queue_all_projects(self) -> None:
        """Add all loaded projects to the queue as pending tasks."""
        self._queue_projects(list(self.projects))

    def on_start_queue(self) -> None:
        """Start pending queue tasks."""
        if not QUEUE.pending_queue:
            hp.toast(self, "Queue empty", "There are no pending tasks to start.", icon="warning", position="top_left")
            return
        QUEUE.run_queued()
        self._refresh_progress_report()

    def on_show_project_images(self, path: Path) -> None:
        """Show the image list for a loaded project."""
        card = self.cards.get(path)
        if card is None:
            logger.warning(f"Could not find project card for {path}")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Images: {card.project.project.name}")
        dlg.setMinimumSize(700, 400)
        image_list = QListWidget(dlg)
        image_list.addItems(card.image_lines())
        layout = QVBoxLayout(dlg)
        layout.addWidget(image_list)
        self._dialogs.append(dlg)
        dlg.show()

    def on_show_project_network(self, path: Path) -> None:
        """Show the registration network for an Elastix project."""
        card = self.cards.get(path)
        if card is None:
            logger.warning(f"Could not find project card for {path}")
            return
        if card.project.kind != "elastix":
            hp.toast(
                self,
                "Network unavailable",
                "Registration network preview is currently available for Elastix projects.",
                icon="warning",
                position="top_left",
            )
            return
        from image2image.qt._wsi._network import NetworkViewer

        dlg = NetworkViewer(card)
        self._dialogs.append(dlg)
        dlg.show()

    def on_open_project_in_viewer(self, path: Path) -> None:
        """Open completed registration images for a loaded project."""
        project = self.projects.get(path)
        if project is None:
            logger.warning(f"Could not find loaded registration project for {path}")
            return
        self._open_registration_in_viewer(project.project.project_dir)

    def on_open_task_in_viewer(self, task: Task) -> None:
        """Open completed registration images for a queue task."""
        metadata = task.metadata or {}
        project_dir = metadata.get("project_dir")
        if project_dir is None:
            logger.warning(f"Could not find project directory in task metadata for {task.task_id}")
            return
        self._open_registration_in_viewer(Path(project_dir))

    def _open_registration_in_viewer(self, project_dir: Path) -> None:
        """Open registration output images in the viewer."""
        path = Path(project_dir) / "Images"
        if path.exists():
            self.on_open_viewer("--file_dir", str(path))
            hp.toast(
                self,
                "Opening viewer...",
                f"Opening viewer for {hp.hyper(path, path.name)}.",
                icon="info",
                position="top_left",
            )
            logger.trace(f"Opening viewer for {path}.")
            return
        hp.toast(
            self,
            "Viewer unavailable",
            f"Could not find completed registration images in {hp.hyper(path, path.name)}.",
            icon="warning",
            position="top_left",
        )

    def _queue_projects(self, paths: list[Path]) -> None:
        """Add projects to the shared queue."""
        if not paths:
            hp.toast(
                self,
                "No projects",
                "No registration projects were selected.",
                icon="warning",
                position="top_left",
            )
            return
        queued = 0
        for path in paths:
            project = self.projects.get(path)
            if project is None:
                logger.warning(f"Could not find loaded registration project for {path}")
                continue
            if not self._validate_project(project):
                continue
            task = self._make_registration_task(project)
            self.task_to_project[task.task_id] = path
            if QUEUE.is_queued(task.task_id):
                self._update_project_status(path, "Already queued", "This valid project is already in the queue.")
                continue
            if QUEUE.is_finished(task.task_id) and self.queue_popup is not None:
                self.queue_popup.queue_list.on_remove_task(task.task_id)
                logger.info(f"Removed finished task {task.task_id}.")
            QUEUE.add_task(task, add_delayed=True)
            self._update_project_status(path, "Queued", "Project validated and queued. Press 'Start queue' to run.")
            queued += 1
        if queued:
            hp.toast(self, "Queued projects", f"Added {queued} registration project(s) to the queue.", icon="success")
        self._refresh_progress_report()

    def _validate_project(self, project: RunnerProject) -> bool:
        """Validate a project before adding it to the queue."""
        is_valid, errors = project.project.validate(require_paths=True)
        if is_valid:
            self._update_project_status(project.project_dir, "Valid", "Project validation passed.")
            hp.toast(
                self,
                "Project valid",
                f"{project.project.name} is ready to queue.",
                icon="success",
                position="top_left",
            )
            return True

        from image2image.qt._dialogs._errors import ErrorsDialog

        self._update_project_status(project.project_dir, "Invalid", "Project validation failed.")
        dlg = ErrorsDialog(self, errors)
        dlg.show()
        return False

    def _make_registration_task(self, project: RunnerProject) -> Task:
        """Create a qtextra task for a loaded registration project."""
        if project.kind == "elastix":
            from image2image.qt.dialog_elastix import make_registration_task

            return make_registration_task(
                project=project.project,
                write_not_registered=self.CONFIG.write_not_registered,
                write_transformed=self.CONFIG.write_registered,
                write_attached=self.CONFIG.write_attached,
                write_merged=self.CONFIG.write_merged,
                remove_merged=self.CONFIG.remove_merged,
                as_uint8=self.CONFIG.as_uint8,
                rename=self.CONFIG.rename,
                clip=self.CONFIG.clip,
            )
        from image2image.qt.dialog_valis import make_registration_task

        return make_registration_task(
            project=project.project,
            write_not_registered=self.CONFIG.write_not_registered,
            write_transformed=self.CONFIG.write_registered,
            write_attached=self.CONFIG.write_attached,
            write_merged=self.CONFIG.write_merged,
            remove_merged=self.CONFIG.remove_merged,
            as_uint8=self.CONFIG.as_uint8,
            rename=self.CONFIG.rename,
            clip=self.CONFIG.clip,
            with_i2reg=STATE.allow_valis_run,
        )

    def on_clear_projects(self) -> None:
        """Clear loaded projects."""
        self.projects.clear()
        self.cards.clear()
        self.task_to_project.clear()
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._refresh_progress_report()

    def on_task_queued(self, task: Task) -> None:
        """Update project state when a task is queued."""
        path = self._project_path_from_task(task)
        if path:
            self._update_project_status(path, "Queued", "Project is waiting in the queue.")
            self._refresh_progress_report()

    def on_task_started(self, task: Task) -> None:
        """Update project state when a task starts."""
        path = self._project_path_from_task(task)
        if path:
            self._update_project_status(path, "Running", f"Running {task.pretty_info}.")
            self._refresh_progress_report()

    def on_task_progress(self, task: Task) -> None:
        """Update project state on queue progress."""
        path = self._project_path_from_task(task)
        if path:
            self._update_project_status(path, "In progress", f"Running {task.pretty_info}.")
            self._refresh_progress_report()

    def on_task_finished(self, task: Task) -> None:
        """Update project state when a task finishes."""
        path = self._project_path_from_task(task)
        if path:
            self._update_project_status(path, "Finished", "Registration task finished.")
            self._refresh_progress_report()

    def on_task_failed(self, task: Task, _exc: object) -> None:
        """Update project state when a task fails."""
        path = self._project_path_from_task(task)
        if path:
            self._update_project_status(path, "Failed", "Registration task failed. Open the queue for details.")
            self._refresh_progress_report()

    def on_task_cancelled(self, task: Task) -> None:
        """Update project state when a task is cancelled."""
        path = self._project_path_from_task(task)
        if path:
            self._update_project_status(path, "Cancelled", "Registration task was cancelled.")
            self._refresh_progress_report()

    def on_task_removed(self, task_id: str) -> None:
        """Update project state when a task is removed from the queue."""
        path = self.task_to_project.pop(task_id, None)
        if path:
            self._update_project_status(path, "Ready", "Task was removed from the queue.")
            self._refresh_progress_report()

    def on_queue_empty(self) -> None:
        """Update progress when the queue becomes empty."""
        self._refresh_progress_report()

    def on_update_config(self, *_args: ty.Any) -> None:
        """Update runner configuration."""
        self.CONFIG.n_parallel = self.n_parallel.value()
        self.CONFIG.write_registered = self.write_registered_check.isChecked()
        self.CONFIG.write_not_registered = self.write_not_registered_check.isChecked()
        self.CONFIG.write_attached = self.write_attached_check.isChecked()
        self.CONFIG.write_merged = self.write_merged_check.isChecked()
        self.CONFIG.remove_merged = self.remove_merged_check.isChecked()
        self.CONFIG.rename = self.rename_check.isChecked()
        self.CONFIG.as_uint8 = self.as_uint8_check.isChecked()
        self.CONFIG.clip = self.clip_combo.currentText()
        _q.N_PARALLEL = self.CONFIG.n_parallel
        QUEUE.n_parallel = self.CONFIG.n_parallel
        self.on_set_write_warning()

    def on_toggle_write(self) -> None:
        """Toggle all write options."""
        write = self.write_check.isChecked()
        self.write_registered_check.setChecked(write)
        self.write_not_registered_check.setChecked(write)
        self.write_attached_check.setChecked(write)
        self.write_merged_check.setChecked(write)
        self.on_update_config()

    def on_set_write_warning(self) -> None:
        """Update export warning state."""
        tooltip = []
        if not any(
            [
                self.CONFIG.write_not_registered,
                self.CONFIG.write_registered,
                self.CONFIG.write_attached,
                self.CONFIG.write_merged,
            ]
        ):
            tooltip.append("- Current settings will not export any images as all write options are disabled.")
        if self.CONFIG.as_uint8:
            tooltip.append("- Images will be converted to uint8 to reduce file size. This can lead to data loss.")
        self.hidden_settings.warning_label.setToolTip("<br>".join(tooltip))
        self.hidden_settings.set_warning_visible(len(tooltip) > 0)

    def _add_project_card(self, project: RunnerProject) -> None:
        """Add a loaded project card."""
        card = QtRunnerProjectCard(project, self)
        card.evt_queue.connect(lambda path: self._queue_projects([Path(path)]))
        card.evt_images.connect(lambda path: self.on_show_project_images(Path(path)))
        card.evt_network.connect(lambda path: self.on_show_project_network(Path(path)))
        card.evt_viewer.connect(lambda path: self.on_open_project_in_viewer(Path(path)))
        self.cards[project.project_dir] = card
        self.cards_layout.insertWidget(max(0, self.cards_layout.count() - 1), card)

    def _update_project_status(self, path: Path, status: str, progress: str = "") -> None:
        """Update a project card status."""
        card = self.cards.get(path)
        if card:
            card.set_status(status, progress)

    def _project_path_from_task(self, task: Task) -> Path | None:
        """Return the loaded project path associated with a queue task."""
        path = self.task_to_project.get(task.task_id)
        if path:
            return path
        metadata = task.metadata or {}
        project_dir = metadata.get("project_dir")
        if project_dir is None:
            return None
        path = Path(project_dir)
        if path in self.cards:
            self.task_to_project[task.task_id] = path
            return path
        return None

    def _refresh_progress_report(self) -> None:
        """Refresh the visible progress report."""
        statuses = [card.status for card in self.cards.values()]
        loaded = len(statuses)
        valid = statuses.count("Valid")
        queued = statuses.count("Queued") + statuses.count("Already queued")
        running = statuses.count("Running") + statuses.count("In progress")
        finished = statuses.count("Finished")
        failed = statuses.count("Failed") + statuses.count("Invalid")
        cancelled = statuses.count("Cancelled")
        self.progress_report.setText(
            "<b>Progress</b>: "
            f"{loaded} loaded | {valid} valid | {queued} queued | {running} running | "
            f"{finished} finished | {failed} failed/invalid | {cancelled} cancelled"
        )

    def _make_export_options(self, parent: QWidget):
        """Create export option controls."""
        self.write_check = hp.make_checkbox(
            parent,
            "",
            tooltip="Quickly toggle between writing or not writing images.",
            value=False,
            func=self.on_toggle_write,
        )
        self.write_registered_check = hp.make_checkbox(
            parent,
            "",
            tooltip="Write registered images.",
            value=self.CONFIG.write_registered,
            func=self.on_update_config,
        )
        self.write_not_registered_check = hp.make_checkbox(
            parent,
            "",
            tooltip="Write original, not-registered images.",
            value=self.CONFIG.write_not_registered,
            func=self.on_update_config,
        )
        self.write_attached_check = hp.make_checkbox(
            parent,
            "",
            tooltip="Write attached modalities.",
            value=self.CONFIG.write_attached,
            func=self.on_update_config,
        )
        self.write_merged_check = hp.make_checkbox(
            parent,
            "",
            tooltip="Merge non-transformed and transformed images into a single image.",
            value=self.CONFIG.write_merged,
            func=self.on_update_config,
        )
        self.remove_merged_check = hp.make_checkbox(
            parent,
            "",
            tooltip="Remove intermediate merged images after writing output.",
            value=self.CONFIG.remove_merged,
            func=self.on_update_config,
        )
        self.rename_check = hp.make_checkbox(
            parent,
            "",
            tooltip="Rename images during writing.",
            value=self.CONFIG.rename,
            func=self.on_update_config,
        )
        self.clip_combo = hp.make_combobox(
            parent,
            ["ignore", "clip", "remove", "part-remove"],
            value=self.CONFIG.clip,
            tooltip="What to do about points/shapes outside of the image when using non-linear transformation.",
        )
        self.clip_combo.currentIndexChanged.connect(self.on_update_config)
        self.as_uint8_check = hp.make_checkbox(
            parent,
            "",
            tooltip=C.UINT8_TOOLTIP,
            value=self.CONFIG.as_uint8,
            func=self.on_update_config,
        )

        self.hidden_settings = hp.make_advanced_collapsible(
            parent,
            "Export options",
            allow_checkbox=False,
            allow_icon=False,
            warning_icon=("warning", {"color": THEMES.get_theme_color("error")}),
        )
        self.hidden_settings.addRow(hp.make_label(parent, "Write/don't write"), self.write_check)
        self.hidden_settings.addRow(hp.make_label(parent, "Write registered images"), self.write_registered_check)
        self.hidden_settings.addRow(hp.make_label(parent, "Write unregistered images"), self.write_not_registered_check)
        self.hidden_settings.addRow(hp.make_label(parent, "Write attached modalities"), self.write_attached_check)
        self.hidden_settings.addRow(hp.make_label(parent, "Write merged images"), self.write_merged_check)
        self.hidden_settings.addRow(hp.make_label(parent, "Remove merged images"), self.remove_merged_check)
        self.hidden_settings.addRow(hp.make_label(parent, "Rename images"), self.rename_check)
        self.hidden_settings.addRow(hp.make_label(parent, "Clip"), self.clip_combo)
        self.hidden_settings.addRow(
            hp.make_label(parent, "Reduce data size"),
            hp.make_h_layout(
                self.as_uint8_check,
                hp.make_warning_label(
                    parent,
                    C.UINT8_WARNING,
                    size_preset="normal",
                    icon_name=("warning", {"color": THEMES.get_theme_color("warning")}),
                ),
                spacing=2,
                stretch_id=(0,),
            ),
        )
        return self.hidden_settings

    def _setup_ui(self) -> None:
        """Create panel."""
        self.add_files_btn = hp.make_btn(
            self,
            "Add project files...",
            tooltip="Select Elastix or Valis registration project files.",
            func=self.on_add_project_files,
        )
        self.add_directory_btn = hp.make_btn(
            self,
            "Add project directory...",
            tooltip="Select an Elastix or Valis registration project directory.",
            func=self.on_add_project_directory,
        )
        self.start_queue_btn = hp.make_btn(
            self,
            "Start queue",
            tooltip="Start pending queue tasks.",
            func=self.on_start_queue,
        )
        self.queue_all_btn = hp.make_btn(
            self,
            "Queue all",
            tooltip="Add all loaded projects to the queue as pending tasks.",
            func=self.on_queue_all_projects,
        )
        self.clear_btn = hp.make_btn(self, "Clear", tooltip="Clear loaded projects.", func=self.on_clear_projects)
        self.n_parallel = hp.make_int_spin_box(
            self,
            value=self.CONFIG.n_parallel,
            minimum=1,
            maximum=8,
            tooltip="Number of parallel queue processes.",
            func=self.on_update_config,
        )

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_widget = QWidget(self.scroll_area)
        self.scroll_area.setWidget(scroll_widget)
        self.cards_layout = QVBoxLayout(scroll_widget)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_layout.setSpacing(4)
        self.cards_layout.addStretch(1)

        self.progress_report = hp.make_label(
            self,
            "<b>Progress</b>: 0 loaded | 0 valid | 0 queued | 0 running | 0 finished | 0 failed/invalid",
            enable_url=True,
        )

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout(widget)
        layout.addWidget(
            hp.make_label(
                self,
                "Drop Elastix/Valis registration projects here, or load them with the buttons below. "
                "Queued jobs are added as pending tasks.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="large_text",
                wrap=True,
            )
        )
        layout.addWidget(hp.make_h_line(self))
        layout.addLayout(
            hp.make_h_layout(
                self.add_files_btn,
                self.add_directory_btn,
                self.queue_all_btn,
                self.start_queue_btn,
                self.clear_btn,
                spacing=2,
                stretch_after=True,
            )
        )
        layout.addWidget(self.scroll_area, stretch=1)
        layout.addWidget(self.progress_report)
        layout.addWidget(self._make_export_options(widget))
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Parallel jobs"),
                self.n_parallel,
                spacing=2,
                stretch_after=True,
            )
        )

        self._make_menu()
        self._make_icon()
        self._make_statusbar()
        self.on_set_write_warning()
        self._refresh_progress_report()

    def _make_menu(self) -> None:
        """Make menu items."""
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(self, "Add project files...", "Ctrl+I", menu=menu_file, func=self.on_add_project_files)
        hp.make_menu_item(
            self, "Add project directory...", "Ctrl+D", menu=menu_file, func=self.on_add_project_directory
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Queue all", menu=menu_file, func=self.on_queue_all_projects, icon="queue")
        hp.make_menu_item(self, "Start queue", menu=menu_file, func=self.on_start_queue, icon="run")
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(self._make_tools_menu().menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        super()._make_statusbar()
        self.queue_popup = QueuePopup(self)
        self.queue_btn = hp.make_qta_btn(
            self, "queue", tooltip="Open queue popup.", func=self.queue_popup.show, size_preset="small"
        )
        self.statusbar.insertPermanentWidget(0, self.queue_btn)

    def _get_console_variables(self) -> dict[str, ty.Any]:
        variables = super()._get_console_variables()
        variables.update({"projects": self.projects, "queue": QUEUE})
        return variables

    def closeEvent(self, evt) -> None:
        """Close."""
        if self._console:
            self._console.close()
        self.CONFIG.save()
        evt.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        """Drop event."""
        super().dropEvent(event)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="runner", level=0)
