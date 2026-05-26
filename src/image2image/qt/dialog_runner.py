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
from qtextra.utils.table_config import TableConfig
from qtpy.QtCore import Qt
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHeaderView,
    QMenuBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import image2image.constants as C
from image2image import __version__
from image2image.config import RunnerConfig, STATE, get_runner_config
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


def _path_to_project_dir(path: Path) -> Path:
    """Return a project directory for a dropped project path."""
    return path.parent if path.is_file() else path


def _preferred_project_kinds(path: Path) -> tuple[ProjectKind, ...]:
    """Return the preferred load order for a registration project path."""
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix == ".valis" or name in {"valis.config.json"} or ".valis." in name:
        return ("valis", "elastix")
    if suffix in {".wsireg", ".i2reg"} or ".i2wsireg." in name or ".i2reg." in name:
        return ("elastix", "valis")
    if name.endswith(".config.json"):
        return ("elastix", "valis")
    return ("elastix", "valis")


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
    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("kind", "kind", "str", 0)
        .add("project", "project", "str", 0)
        .add("status", "status", "str", 0)
    )

    def __init__(
        self,
        parent: QWidget | None,
        run_check_version: bool = True,
        project_dir: str | Path | None = None,
        **_kwargs: ty.Any,
    ):
        self.CONFIG = get_runner_config()
        self.projects: dict[Path, RunnerProject] = {}
        self.queue_popup: QueuePopup | None = None
        _q.N_PARALLEL = self.CONFIG.n_parallel
        super().__init__(
            parent,
            f"image2image: Elastix/Valis Runner (v{__version__})",
            run_check_version=run_check_version,
        )
        if project_dir:
            self.on_add_project_paths([Path(project_dir)])

    @staticmethod
    def _setup_config() -> None:
        """Setup configuration."""

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        if state:
            self.evt_dropped.connect(self.on_drop_projects)
        else:
            self.evt_dropped.disconnect(self.on_drop_projects)

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
            self._add_project_row(project)
            loaded += 1
        if loaded:
            hp.toast(self, "Loaded projects", f"Loaded {loaded} registration project(s).", icon="success")

    def on_queue_all_projects(self) -> None:
        """Add all loaded projects to the queue as pending tasks."""
        self._queue_projects(list(self.projects))

    def on_queue_selected_projects(self) -> None:
        """Add selected projects to the queue as pending tasks."""
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        paths = []
        for row in rows:
            item = self.table.item(row, self.TABLE_CONFIG.project)
            if item:
                paths.append(Path(item.data(Qt.ItemDataRole.UserRole)))
        self._queue_projects(paths)

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
            task = self._make_registration_task(project)
            if QUEUE.is_queued(task.task_id):
                self._update_project_status(path, "Already queued")
                continue
            if QUEUE.is_finished(task.task_id) and self.queue_popup is not None:
                self.queue_popup.queue_list.on_remove_task(task.task_id)
                logger.info(f"Removed finished task {task.task_id}.")
            QUEUE.add_task(task, add_delayed=True)
            self._update_project_status(path, "Queued")
            queued += 1
        if queued:
            hp.toast(self, "Queued projects", f"Added {queued} registration project(s) to the queue.", icon="success")

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
        self.table.setRowCount(0)

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

    def _add_project_row(self, project: RunnerProject) -> None:
        """Add a loaded project to the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        kind_item = QTableWidgetItem(project.kind.capitalize())
        kind_item.setFlags(kind_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        kind_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.table.setItem(row, self.TABLE_CONFIG.kind, kind_item)

        project_item = QTableWidgetItem(str(project.project_dir))
        project_item.setData(Qt.ItemDataRole.UserRole, str(project.project_dir))
        project_item.setFlags(project_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        project_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.table.setItem(row, self.TABLE_CONFIG.project, project_item)

        status_item = QTableWidgetItem("Ready")
        status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        status_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.table.setItem(row, self.TABLE_CONFIG.status, status_item)

    def _update_project_status(self, path: Path, status: str) -> None:
        """Update a project row status."""
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.TABLE_CONFIG.project)
            if item and Path(item.data(Qt.ItemDataRole.UserRole)) == path:
                self.table.item(row, self.TABLE_CONFIG.status).setText(status)
                return

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
        columns = self.TABLE_CONFIG.to_columns()
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setCornerButtonEnabled(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setTextElideMode(Qt.TextElideMode.ElideLeft)
        self.table.setWordWrap(True)

        horizontal_header = self.table.horizontalHeader()
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.kind, QHeaderView.ResizeMode.ResizeToContents)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.project, QHeaderView.ResizeMode.Stretch)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.status, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

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
        self.queue_selected_btn = hp.make_btn(
            self,
            "Queue selected",
            tooltip="Add selected projects to the queue as pending tasks.",
            func=self.on_queue_selected_projects,
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
                self.queue_selected_btn,
                self.queue_all_btn,
                self.clear_btn,
                spacing=2,
                stretch_after=True,
            )
        )
        layout.addWidget(self.table)
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

    def _make_menu(self) -> None:
        """Make menu items."""
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(self, "Add project files...", "Ctrl+I", menu=menu_file, func=self.on_add_project_files)
        hp.make_menu_item(
            self, "Add project directory...", "Ctrl+D", menu=menu_file, func=self.on_add_project_directory
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Queue selected", menu=menu_file, func=self.on_queue_selected_projects, icon="queue")
        hp.make_menu_item(self, "Queue all", menu=menu_file, func=self.on_queue_all_projects, icon="queue")
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
