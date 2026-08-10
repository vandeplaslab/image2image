"""Card describing a loaded registration project."""

from __future__ import annotations

import typing as ty
from pathlib import Path

from image2image_reg.workflows import ElastixReg, ValisReg
from qtextra import helpers as hp
from qtextra.widgets.qt_label_image import ImageViewer
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

from image2image.qt._runner._constants import ReviewState, RunnerProject
from image2image.qt._runner.utilities import discover_overlap_images, has_registration_images, read_review_state

RUNNING_STATUSES = {"Running", "In progress"}
QUEUED_STATUSES = {"Queued", "Already queued"}


class QtRunnerOverlapViewer(ImageViewer):
    """ImageViewer that notifies the card when its zoom level changes."""

    evt_zoom_changed = Signal()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom using the native ImageViewer wheel behavior and notify the card."""
        super().wheelEvent(event)
        self.evt_zoom_changed.emit()

    def mouseDoubleClickEvent(self, event: ty.Any) -> None:
        """Reset zoom using the native ImageViewer behavior and notify the card."""
        super().mouseDoubleClickEvent(event)
        self.evt_zoom_changed.emit()


class QtRunnerProjectCard(QFrame):
    """Card describing a loaded registration project."""

    evt_queue = Signal(object)
    evt_images = Signal(object)
    evt_network = Signal(object)
    evt_viewer = Signal(object)
    evt_review = Signal(object, object)
    evt_edit = Signal(object)
    evt_remove = Signal(object)

    def __init__(self, project: RunnerProject, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.status = "Ready"
        self.review_state = read_review_state(project.project_dir)
        self.overlap_paths: list[Path] = []
        self.overlap_index = 0
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)
        self.setProperty("card", True)

        self.name_label = hp.make_label(
            self,
            f"<b>{project.project.name}</b>",
            enable_url=True,
            object_name="large_text",
        )
        self.summary_label = hp.make_label(self, self._summarize_project(), enable_url=True, wrap=True)
        self.status_label = hp.make_label(self, "Ready")
        self.progress_label = hp.make_label(self, "Waiting to be queued.", wrap=True)

        self.queue_btn = hp.make_qta_btn(
            self,
            "queue",
            tooltip="Validate and add this project to the queue.",
            func=lambda: self.evt_queue.emit(self.project.project_dir),
            size_preset="normal",
        )
        self.images_btn = hp.make_qta_btn(
            self,
            "folder",
            tooltip="Show the project image list.",
            func=lambda: self.evt_images.emit(self.project.project_dir),
            size_preset="normal",
        )
        self.network_btn = hp.make_qta_btn(
            self,
            "network",
            tooltip="Show the Elastix registration network."
            if project.kind == "elastix"
            else "Registration network preview is currently available for Elastix projects.",
            func=lambda: self.evt_network.emit(self.project.project_dir),
            disabled=project.kind != "elastix",
            size_preset="normal",
        )
        self.viewer_btn = hp.make_qta_btn(
            self,
            "viewer",
            tooltip="Open completed registration images in the viewer.",
            func=lambda: self.evt_viewer.emit(self.project.project_dir),
            disabled=not has_registration_images(self.project.project_dir),
            size_preset="normal",
        )

        self.review_toggle = hp.make_toggle(
            self,
            "Good",
            "Bad",
            func=self.on_review_state,
            tooltip="Mark this project as good or bad.",
        )
        edit_app_name = "Elastix" if project.kind == "elastix" else "Valis"
        self.edit_btn = hp.make_qta_btn(
            self,
            "edit",
            tooltip=f"Open this bad project in the {edit_app_name} app for edits.",
            func=lambda: self.evt_edit.emit(self.project.project_dir),
            disabled=self.review_state != "bad",
            size_preset="normal",
        )
        self.remove_btn = hp.make_qta_btn(
            self,
            "delete",
            tooltip="Remove this project from the runner. Project files are kept.",
            func=lambda: self.evt_remove.emit(self.project.project_dir),
            size_preset="normal",
        )

        self.overlap_label = hp.make_label(self, "Overlap preview")
        self.overlap_index_label = hp.make_label(self, "No previews", object_name="tip_label")
        self.prev_overlap_btn = hp.make_qta_btn(
            self,
            "chevron_left_circle",
            tooltip="Show the previous overlap preview.",
            func=lambda: self.move_overlap_preview(-1),
            size_preset="normal",
            standout=True,
        )
        self.next_overlap_btn = hp.make_qta_btn(
            self,
            "chevron_right_circle",
            tooltip="Show the next overlap preview.",
            func=lambda: self.move_overlap_preview(1),
            size_preset="normal",
            standout=True,
        )
        self.zoom_out_btn = hp.make_qta_btn(
            self,
            "remove",
            tooltip="Zoom out of the overlap preview.",
            func=lambda: self.zoom_overlap_preview(-1),
            size_preset="normal",
        )
        self.zoom_label = hp.make_label(self, "100%", object_name="tip_label")
        self.zoom_in_btn = hp.make_qta_btn(
            self,
            "add_one",
            tooltip="Zoom into the overlap preview.",
            func=lambda: self.zoom_overlap_preview(1),
            size_preset="normal",
        )
        self.overlap_viewer = QtRunnerOverlapViewer(parent=self)
        self.overlap_viewer.setMinimumSize(440, 300)
        self.overlap_viewer.setMaximumHeight(340)
        self.overlap_viewer.setToolTip("Use the mouse wheel to zoom, drag to pan, or double-click to reset zoom.")
        self.overlap_viewer.evt_zoom_changed.connect(self._update_overlap_zoom_controls)
        self.overlap_unavailable_label = hp.make_label(
            self,
            "No overlap previews available.",
            object_name="tip_label",
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        layout.addLayout(
            hp.make_h_layout(
                self.name_label,
                self.review_toggle,
                spacing=2,
                stretch_id=(0,),
            )
        )
        details_layout = QVBoxLayout()
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)
        details_layout.addWidget(self.summary_label)
        details_layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Status"),
                self.status_label,
                spacing=2,
                stretch_id=(1,),
            )
        )
        details_layout.addWidget(self.progress_label)
        details_layout.addLayout(
            hp.make_h_layout(
                self.queue_btn,
                self.images_btn,
                self.network_btn,
                self.viewer_btn,
                self.edit_btn,
                self.remove_btn,
                spacing=2,
                stretch_after=True,
            )
        )

        preview_layout = QVBoxLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(4)
        preview_layout.addLayout(
            hp.make_h_layout(
                self.overlap_label,
                self.overlap_index_label,
                self.prev_overlap_btn,
                self.next_overlap_btn,
                self.zoom_out_btn,
                self.zoom_label,
                self.zoom_in_btn,
                spacing=2,
                stretch_id=(1,),
            )
        )
        preview_layout.addWidget(self.overlap_viewer)
        preview_layout.addWidget(self.overlap_unavailable_label)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        content_layout.addLayout(details_layout, stretch=1)
        content_layout.addLayout(preview_layout, stretch=2)
        layout.addLayout(content_layout)
        self.refresh_overlap_previews()
        self.refresh_actions()

    def on_review_state(self, state: str) -> None:
        """Review."""
        self.evt_review.emit(self.project.project_dir, self.review_toggle.value.lower())

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
        if progress:
            self.progress_label.setText(progress)
        self.refresh_overlap_previews()
        self.refresh_actions()

    def set_review_state(self, state: ReviewState) -> None:
        """Update the visible project review state."""
        self.review_state = state
        self.refresh_actions()

    def refresh_actions(self) -> None:
        """Refresh action button availability."""
        is_running = self.status in RUNNING_STATUSES
        is_queued = self.status in QUEUED_STATUSES
        hp.disable_widgets(self.queue_btn, disabled=is_running or is_queued)
        hp.disable_widgets(self.viewer_btn, disabled=not has_registration_images(self.project.project_dir))
        hp.disable_widgets(self.edit_btn, disabled=self.review_state != "bad")
        hp.disable_widgets(self.remove_btn, disabled=is_running)
        self.review_toggle.value = self._review_text()

    def refresh_overlap_previews(self) -> None:
        """Refresh overlap previews available on disk for this project."""
        current_path = self.current_overlap_path
        self.overlap_paths = discover_overlap_images(self.project.project_dir)
        if current_path in self.overlap_paths:
            self.overlap_index = self.overlap_paths.index(current_path)
        else:
            self.overlap_index = 0
        self._show_overlap_preview()

    def move_overlap_preview(self, delta: int) -> None:
        """Move to an adjacent overlap preview without wrapping."""
        if not self.overlap_paths:
            return
        index = max(0, min(len(self.overlap_paths) - 1, self.overlap_index + delta))
        if index == self.overlap_index:
            return
        self.overlap_index = index
        self._show_overlap_preview()

    def zoom_overlap_preview(self, direction: int) -> None:
        """Adjust the ImageViewer zoom using the center of the visible preview."""
        if not self.overlap_paths:
            return
        center = self.overlap_viewer.mapToScene(self.overlap_viewer.viewport().rect().center())
        self.overlap_viewer.zoom(direction, center)
        self._update_overlap_zoom_controls()

    @property
    def current_overlap_path(self) -> Path | None:
        """Return the currently selected overlap preview path, when available."""
        if 0 <= self.overlap_index < len(self.overlap_paths):
            return self.overlap_paths[self.overlap_index]
        return None

    def resizeEvent(self, event: ty.Any) -> None:
        """Keep an unzoomed preview fitted after the card changes size."""
        super().resizeEvent(event)
        if self.overlap_viewer.zoom_level == 0 and not self.overlap_viewer.pixmap_item.pixmap().isNull():
            self.overlap_viewer.reset_zoom()

    def image_lines(self) -> list[str]:
        """Return a simple image list for the project."""
        lines = []
        for index, modality in enumerate(self.modalities, start=1):
            lines.append(f"{index}. {modality.name}: {modality.path}")
        return lines

    def _show_overlap_preview(self) -> None:
        """Display the selected overlap preview or its unavailable state."""
        path = self.current_overlap_path
        has_preview = path is not None
        self.overlap_viewer.setVisible(has_preview)
        self.overlap_unavailable_label.setVisible(not has_preview)
        hp.disable_widgets(self.prev_overlap_btn, disabled=not has_preview or self.overlap_index == 0)
        hp.disable_widgets(
            self.next_overlap_btn,
            disabled=not has_preview or self.overlap_index == len(self.overlap_paths) - 1,
        )
        if path is None:
            self.overlap_index_label.setText("No previews")
            self._update_overlap_zoom_controls()
            return

        self.overlap_index_label.setText(f"{self.overlap_index + 1}/{len(self.overlap_paths)}: {path.name}")
        self.overlap_viewer.set_image(str(path))
        self._update_overlap_zoom_controls()

    def _update_overlap_zoom_controls(self) -> None:
        """Refresh overlap zoom control labels and availability."""
        has_preview = self.current_overlap_path is not None
        zoom = self.overlap_viewer.base_zoom_factor**self.overlap_viewer.zoom_level
        self.zoom_label.setText(f"{round(zoom * 100)}%")
        hp.disable_widgets(self.zoom_out_btn, disabled=not has_preview or self.overlap_viewer.zoom_level == 0)
        hp.disable_widgets(self.zoom_in_btn, disabled=not has_preview)

    def _summarize_project(self) -> str:
        """Return a short project summary."""
        project = self.project.project
        n_modalities = len(project.modalities)
        output_dir = hp.hyper(project.output_dir, value=self._summarize_path(project.output_dir))
        project_dir = hp.hyper(project.project_dir, value=self._summarize_path(project.project_dir))
        return (
            f"<b>Type</b>: {self.project.kind.capitalize()} &nbsp; "
            f"<b>Modalities</b>: {n_modalities}<br>"
            f"<b>Project</b>: {project_dir}<br>"
            f"<b>Output</b>: {output_dir}"
        )

    @staticmethod
    def _summarize_path(path: Path) -> str:
        """Return a compact, clickable path label with its final components visible."""
        parts = Path(path).parts
        if len(parts) <= 3:
            return str(path)
        return f".../{Path(*parts[-3:])}"

    def _review_text(self) -> str:
        """Return review label text."""
        return self.review_state.capitalize()
