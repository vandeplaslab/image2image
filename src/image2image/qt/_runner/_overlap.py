from __future__ import annotations

import typing as ty
from pathlib import Path

from qtextra import helpers as hp
from qtpy.QtCore import QSize, Qt, Signal
from qtpy.QtGui import QKeyEvent, QPixmap
from qtpy.QtWidgets import QDialog, QLabel, QListWidget, QVBoxLayout, QWidget

from image2image.qt._runner._constants import ReviewState, RunnerProject


class OverlapPreviewDialog(QDialog):
    """Dialog for reviewing overlap preview PNG images."""

    evt_review = Signal(object, object)

    def __init__(
        self,
        project: RunnerProject,
        image_paths: list[Path],
        review_state: ReviewState,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.project = project
        self.image_paths = image_paths
        self.review_state = review_state
        self.setWindowTitle(f"Overlap previews: {project.project.name}")
        self.setMinimumSize(800, 500)

        self.image_list = QListWidget(self)
        for path in image_paths:
            self.image_list.addItem(path.name)
        self.image_list.currentRowChanged.connect(self.on_select_image)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(500, 400)
        self.image_label.setText("No preview selected.")

        self.review_label = hp.make_label(self, self._review_text(), object_name="tip_label")
        self.good_btn = hp.make_btn(
            self,
            "Good",
            tooltip="Mark this project result as good.",
            func=lambda: self.set_review_state("good"),
        )
        self.bad_btn = hp.make_btn(
            self,
            "Bad",
            tooltip="Mark this project result as bad.",
            func=lambda: self.set_review_state("bad"),
        )

        layout = QVBoxLayout(self)
        layout.addLayout(
            hp.make_h_layout(
                self.image_list,
                self.image_label,
                spacing=4,
                stretch_id=(1,),
            )
        )
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Review"),
                self.review_label,
                self.good_btn,
                self.bad_btn,
                spacing=2,
                stretch_id=(1,),
            )
        )
        self._sync_review_buttons()
        if self.image_paths:
            self.image_list.setCurrentRow(0)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle arrow key navigation."""
        if event.key() in {
            Qt.Key.Key_Left,
            Qt.Key.Key_Up,
        }:
            self._move_selection(-1)
            return
        if event.key() in {
            Qt.Key.Key_Right,
            Qt.Key.Key_Down,
        }:
            self._move_selection(1)
            return
        super().keyPressEvent(event)

    def resizeEvent(self, event: ty.Any) -> None:
        """Refresh the selected image when the dialog is resized."""
        super().resizeEvent(event)
        self.on_select_image(self.image_list.currentRow())

    def on_select_image(self, row: int) -> None:
        """Display the selected overlap image."""
        if row < 0 or row >= len(self.image_paths):
            self.image_label.setText("No preview selected.")
            return
        pixmap = QPixmap(str(self.image_paths[row]))
        if pixmap.isNull():
            self.image_label.setText("Could not load preview image.")
            return
        size = self.image_label.size()
        if not size.isValid():
            size = QSize(500, 400)
        self.image_label.setPixmap(
            pixmap.scaled(
                size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def set_review_state(self, state: ReviewState) -> None:
        """Update and emit the project review state."""
        self.review_state = state
        self.review_label.setText(self._review_text())
        self._sync_review_buttons()
        self.evt_review.emit(self.project.project_dir, state)

    def _move_selection(self, delta: int) -> None:
        """Move preview selection by one item."""
        if not self.image_paths:
            return
        row = self.image_list.currentRow()
        if row < 0:
            row = 0
        row = max(0, min(len(self.image_paths) - 1, row + delta))
        self.image_list.setCurrentRow(row)

    def _review_text(self) -> str:
        """Return review text for the current state."""
        return self.review_state.capitalize()

    def _sync_review_buttons(self) -> None:
        """Refresh review button enabled states."""
        self.good_btn.setEnabled(self.review_state != "good")
        self.bad_btn.setEnabled(self.review_state != "bad")
