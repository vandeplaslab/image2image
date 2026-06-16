"""Queue runner for Elastix and Valis registration projects."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
import qtextra.queue.cli_queue as _q
from loguru import logger
from qtextra.config import THEMES
from qtextra.queue.popup import QUEUE, QueuePopup
from qtextra.queue.task import Task
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QListWidget,
    QMenuBar,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import image2image.constants as C
import image2image.qt.helpers as ih
from image2image import __version__
from image2image.config import STATE, RunnerConfig, get_runner_config
from image2image.qt._dialog_base import Window
from image2image.qt._runner._card import QtRunnerProjectCard
from image2image.qt._runner._constants import (
    PROJECT_FILE_FILTER,
    REVIEW_STATE_FILTER,
    REVIEW_STATES,
    RUN_STATE_FILTER,
    ReviewState,
    RunnerProject,
)
from image2image.qt._runner.utilities import (
    discover_overlap_images,
    has_registration_images,
    load_registration_project,
    project_matches_filters,
    write_review_state,
)

if ty.TYPE_CHECKING:
    pass


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

    def on_queue_bad_projects(self) -> None:
        """Add all loaded bad projects to the queue as pending tasks."""
        paths = [path for path, card in self.cards.items() if card.review_state == "bad"]
        if not paths:
            hp.toast(self, "No bad projects", "There are no loaded projects marked as bad.", icon="warning")
            return
        self._queue_projects(paths)

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

    def on_show_overlap_previews(self, path: Path) -> None:
        """Show existing overlap preview images for a loaded project."""
        from image2image.qt._runner._overlap import OverlapPreviewDialog

        project = self.projects.get(path)
        if project is None:
            logger.warning(f"Could not find loaded registration project for {path}")
            return
        image_paths = discover_overlap_images(path)
        if not image_paths:
            hp.toast(
                self,
                "No overlap previews",
                f"Could not find overlap PNG files in {hp.hyper(path / 'Overlap', 'Overlap')}.",
                icon="warning",
                position="top_left",
            )
            return
        card = self.cards[path]
        dlg = OverlapPreviewDialog(project, image_paths, card.review_state, self)
        dlg.evt_review.connect(lambda project_path, state: self.on_set_project_review(Path(project_path), state))
        self._dialogs.append(dlg)
        dlg.show()

    def on_open_project_in_viewer(self, path: Path) -> None:
        """Open completed registration images for a loaded project."""
        project = self.projects.get(path)
        if project is None:
            logger.warning(f"Could not find loaded registration project for {path}")
            return
        self._open_registration_in_viewer(project.project.project_dir)

    def on_open_project_for_edits(self, path: Path) -> None:
        """Open a bad project in its registration app."""
        project = self.projects.get(path)
        if project is None:
            logger.warning(f"Could not find loaded registration project for {path}")
            return
        args = ("--project_dir", str(project.project_dir))
        if project.kind == "elastix":
            self.on_open_elastix(*args)
        else:
            self.on_open_valis(*args)

    def on_set_project_review(self, path: Path, state: ReviewState) -> None:
        """Persist and display a project review state."""
        if state not in REVIEW_STATES:
            logger.warning(f"Ignoring invalid review state {state!r} for {path}")
            return
        write_review_state(path, state)
        card = self.cards.get(path)
        if card:
            card.set_review_state(state)
        self._apply_project_filters()
        self._refresh_progress_report()

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
        if has_registration_images(Path(project_dir)):
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
        ih.warn_if_uint8(self)
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

    def _make_registration_task(self, project: RunnerProject, cli_command_func: ty.Callable | None = None) -> Task:
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
                cli_command_func=cli_command_func,
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
            cli_command_func=cli_command_func,
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

    def on_update_project_filters(self, *_args: ty.Any) -> None:
        """Apply project card filters."""
        self._apply_project_filters()

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
        card.evt_overlap.connect(lambda path: self.on_show_overlap_previews(Path(path)))
        card.evt_review.connect(lambda path, state: self.on_set_project_review(Path(path), state))
        card.evt_edit.connect(lambda path: self.on_open_project_for_edits(Path(path)))
        self.cards[project.project_dir] = card
        self.cards_layout.insertWidget(max(0, self.cards_layout.count() - 1), card)
        self._apply_project_filters()

    def _update_project_status(self, path: Path, status: str, progress: str = "") -> None:
        """Update a project card status."""
        card = self.cards.get(path)
        if card:
            card.set_status(status, progress)
        self._apply_project_filters()

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
        self.queue_bad_btn.setEnabled(any(card.review_state == "bad" for card in self.cards.values()))

    def _apply_project_filters(self) -> None:
        """Apply project card filters to the loaded cards."""
        if not hasattr(self, "filter_by_name"):
            return
        name_filter = self.filter_by_name.text()
        run_filter = ty.cast(RUN_STATE_FILTER, self.filter_by_status.currentText())
        review_filter = ty.cast(REVIEW_STATE_FILTER, self.filter_by_review.currentText())
        for card in self.cards.values():
            visible = project_matches_filters(
                card.project.project.name,
                card.status,
                card.review_state,
                name_filter,
                run_filter,
                review_filter,
            )
            card.setVisible(visible)

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
        self.queue_bad_btn = hp.make_btn(
            self,
            "Queue bad",
            tooltip="Add all loaded projects marked as bad to the queue.",
            func=self.on_queue_bad_projects,
            disabled=True,
        )
        self.clear_btn = hp.make_btn(self, "Clear", tooltip="Clear loaded projects.", func=self.on_clear_projects)
        self.filter_by_name = hp.make_line_edit(
            self,
            placeholder="Filter by name...",
            tooltip="Filter loaded projects by project name.",
            func_changed=self.on_update_project_filters,
        )
        self.filter_by_status = hp.make_combobox(
            self,
            ["All", "Finished", "Running", "Queued", "Failed"],
            value="All",
            tooltip="Filter loaded projects by run state.",
            func=self.on_update_project_filters,
        )
        self.filter_by_review = hp.make_combobox(
            self,
            ["All", "Unknown", "Good", "Bad"],
            value="All",
            tooltip="Filter loaded projects by review state.",
            func=self.on_update_project_filters,
        )
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
                self.queue_bad_btn,
                self.start_queue_btn,
                self.clear_btn,
                spacing=2,
                stretch_after=True,
            )
        )
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Filter"),
                self.filter_by_name,
                hp.make_label(self, "State"),
                self.filter_by_status,
                hp.make_label(self, "Review"),
                self.filter_by_review,
                spacing=2,
                stretch_id=(1,),
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
