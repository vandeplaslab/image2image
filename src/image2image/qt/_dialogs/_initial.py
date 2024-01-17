"""Initial transformation dialog."""
from __future__ import annotations

import typing as ty

import numpy as np
from loguru import logger
from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtCore import QModelIndex, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout

if ty.TYPE_CHECKING:
    from image2image.qt.dialog_register import ImageRegistrationWindow


logger = logger.bind(src="InitialTransformDialog")


class InitialTransformModel:
    """Initial transform model."""

    rotate: float = 0
    translate_x: int = 0
    translate_y: int = 0
    scale: float = 1
    flip_lr: bool = False

    def apply_rotate(self, which: str) -> None:
        """Apply rotation."""
        if which == "left":
            self.rotate += 5
        else:
            self.rotate -= 5
        if self.rotate > 360:
            self.rotate -= 360
        elif self.rotate < 0:
            self.rotate += 360
        if self.rotate == 360:
            self.rotate = 0

    def apply_translate(self, which: str) -> None:
        """Apply rotation."""
        if which == "up":
            self.translate_y -= 250
        elif which == "down":
            self.translate_y += 250
        elif which == "left":
            self.translate_x -= 250
        else:
            self.translate_x += 250

    def affine(self, shape: tuple[int, int], scale: ty.Optional[tuple[float, float]] = None) -> np.ndarray | None:
        """Calculate affine transformation."""
        from image2image.utils.transform import combined_transform, scale_transform

        if scale is None:
            scale = (self.scale, self.scale)
        matrix = combined_transform(
            shape,
            scale,
            self.rotate,
            (self.translate_y, self.translate_x),
            self.flip_lr,
        ) @ scale_transform(scale)
        if np.array_equal(matrix, np.eye(3)):
            return None
        return matrix


class InitialTransformDialog(QtFramelessTool):
    """Dialog to transform initial image."""

    def __init__(self, parent: ImageRegistrationWindow):
        super().__init__(parent)
        self.initial_model = InitialTransformModel()
        self.on_update()

    def on_rotate(self, which: str) -> None:
        """Rotate image."""
        self.initial_model.apply_rotate(which)
        self.on_update()
        logger.trace(f"Rotated image {which}")

    def on_translate(self, which: str) -> None:
        """Translate image."""
        self.initial_model.apply_translate(which)
        self.on_update()
        logger.trace(f"Translated image {which}")

    def on_flip_lr(self) -> None:
        """Flip image."""
        self.initial_model.flip_lr = not self.initial_model.flip_lr
        self.on_update()
        logger.trace("Flipped image")

    def on_update(self):
        """Update image."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        if parent.transformed_moving_image_layer:
            shape = parent.transformed_moving_image_layer.data.shape
            scale = parent.moving_model.min_resolution / parent.fixed_model.min_resolution
            affine = self.initial_model.affine(shape, (scale, scale))
            parent.transformed_moving_image_layer.affine = affine if affine is not None else np.eye(3)

    def accept(self):
        """Accept changes."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        if parent.transformed_moving_image_layer:
            shape = parent.transformed_moving_image_layer.data.shape
            scale = 1  # parent.moving_model.min_resolution / parent.fixed_model.min_resolution
            parent.transform_model.moving_initial_affine = self.initial_model.affine(shape, (scale, scale))
            logger.trace(f"Initial affine: {parent.transform_model.moving_initial_affine.tolist()}")
        return super().accept()

    def reject(self):
        """Reject changes."""
        parent: ImageRegistrationWindow = self.parent()
        parent.transform_model.moving_initial_affine = None
        logger.trace("Initial affine reset.")
        return super().reject()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_close_handle(title="Initial transformation")

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(
            "Rotate",
            hp.make_h_layout(
                hp.make_qta_btn(self, "rotate_left", func=lambda: self.on_rotate("left")),
                hp.make_qta_btn(self, "rotate_right", func=lambda: self.on_rotate("right")),
            ),
        )
        layout.addRow(
            "Translate",
            hp.make_h_layout(
                hp.make_qta_btn(self, "translate_left", func=lambda: self.on_translate("left")),
                hp.make_qta_btn(self, "translate_right", func=lambda: self.on_translate("right")),
                hp.make_qta_btn(self, "translate_up", func=lambda: self.on_translate("up")),
                hp.make_qta_btn(self, "translate_down", func=lambda: self.on_translate("down")),
            ),
        )
        layout.addRow("Flip left-right", hp.make_qta_btn(self, "flip_lr", func=self.on_flip_lr))
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Use shortcuts to quickly rotate, move or flip moving image.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def keyPressEvent(self, evt: ty.Any) -> None:
        """Key press event."""
        if hasattr(evt, "native"):
            evt = evt.native
        key = evt.key()
        if key == Qt.Key.Key_Escape:
            evt.ignore()
        # rotate
        elif key == Qt.Key.Key_Q:
            self.on_rotate("left")
        elif key == Qt.Key.Key_E:
            self.on_rotate("right")
        # translate
        elif key == Qt.Key.Key_A:
            self.on_translate("left")
        elif key == Qt.Key.Key_D:
            self.on_translate("right")
        elif key == Qt.Key.Key_W:
            self.on_translate("up")
        elif key == Qt.Key.Key_S:
            self.on_translate("down")
        # flip left-right
        elif key == Qt.Key.Key_F:
            self.on_flip_lr()
