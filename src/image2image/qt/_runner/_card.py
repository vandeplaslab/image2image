"""Card describing a loaded registration project."""

from __future__ import annotations

import typing as ty

from image2image_reg.workflows import ElastixReg, ValisReg
from qtextra import helpers as hp
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFrame, QVBoxLayout, QWidget

from image2image.qt._runner._constants import ReviewState, RunnerProject
from image2image.qt._runner.utilities import has_registration_images, read_review_state


class QtRunnerProjectCard(QFrame):
    """Card describing a loaded registration project."""

    evt_queue = Signal(object)
    evt_images = Signal(object)
    evt_network = Signal(object)
    evt_viewer = Signal(object)
    evt_overlap = Signal(object)
    evt_review = Signal(object, object)
    evt_edit = Signal(object)

    def __init__(self, project: RunnerProject, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.status = "Ready"
        self.review_state = read_review_state(project.project_dir)
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
        self.status_label = hp.make_label(self, "Ready", object_name="tip_label")
        self.review_label = hp.make_label(self, self._review_text(), object_name="tip_label")
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
        self.overlap_btn = hp.make_qta_btn(
            self,
            "overlap",
            tooltip="Show existing overlap preview images.",
            func=lambda: self.evt_overlap.emit(self.project.project_dir),
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
        layout.addWidget(self.summary_label)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Status"),
                self.status_label,
                spacing=2,
                stretch_id=(1,),
            )
        )
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Review"),
                self.review_label,
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
                self.overlap_btn,
                self.viewer_btn,
                self.edit_btn,
                spacing=2,
                stretch_after=True,
            )
        )
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
        self.refresh_actions()

    def set_review_state(self, state: ReviewState) -> None:
        """Update the visible project review state."""
        self.review_state = state
        self.review_label.setText(self._review_text())
        self.refresh_actions()

    def refresh_actions(self) -> None:
        """Refresh action button availability."""
        hp.disable_widgets(self.viewer_btn, disabled=not has_registration_images(self.project.project_dir))
        hp.disable_widgets(self.edit_btn, disabled=self.review_state != "bad")
        self.review_toggle.value = self.review_state

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

    def _review_text(self) -> str:
        """Return review label text."""
        return self.review_state.capitalize()
