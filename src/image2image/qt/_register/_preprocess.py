"""Pre-process moving dialog."""

from __future__ import annotations

import typing as ty
from functools import partial

import numpy as np
from loguru import logger
from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout

from image2image.utils.utilities import open_docs

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
            self.rotate += 15
        else:
            self.rotate -= 15
        if self.rotate > 360:
            self.rotate -= 360
        elif self.rotate < 0:
            self.rotate += 360
        if self.rotate == 360:
            self.rotate = 0

    def affine(self, shape: tuple[int, int], scale: tuple[float, float] | None = None) -> np.ndarray | None:
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


class PreprocessMovingDialog(QtFramelessTool):
    """Dialog to pre-process moving image."""

    def __init__(self, parent: ImageRegistrationWindow):
        super().__init__(parent)
        self.initial_model = InitialTransformModel()
        self.on_update()

    def on_rotate(self, value: int | str) -> None:
        """Rotate image."""
        if isinstance(value, str):
            value = 15 if value == "right" else -15
        self.initial_model.rotate = self.rotate_spin.value() + value
        with hp.qt_signals_blocked(self.rotate_spin):
            self.rotate_spin.setValue(self.initial_model.rotate)
        self.on_update()

    def on_set_rotate(self, value: int) -> None:
        """Set rotation."""
        self.initial_model.rotate = value
        self.on_update()

    def on_flip_lr(self) -> None:
        """Flip image."""
        self.initial_model.flip_lr = not self.initial_model.flip_lr
        self.on_update()

    def on_reset(self) -> None:
        """Reset."""
        self.initial_model.rotate = 0
        self.initial_model.flip_lr = False
        with hp.qt_signals_blocked(self.rotate_spin, self.flip_lr):
            self.rotate_spin.setValue(0)
            self.flip_lr.setChecked(False)
        self.on_update()

    def on_update(self) -> None:
        """Update image."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        if parent.moving_image_layer:
            for layer in parent.moving_image_layer:
                shape = layer.data[0].shape
                affine = self.initial_model.affine(shape, (1, 1))
                layer.affine = affine if affine is not None else np.eye(3)
                logger.info(
                    f"Initial affine (rot={self.initial_model.rotate}; flip={self.initial_model.flip_lr}):"
                    f" {parent.transform_model.about('; ', transform=layer.affine.affine_matrix)}"
                )

    def accept(self) -> None:
        """Accept changes."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        if parent.moving_image_layer:
            shape = parent.moving_image_layer[0].data[0].shape
            affine = self.initial_model.affine(shape, (1, 1))
            parent.transform_model.moving_initial_affine = affine
            if affine is not None:
                logger.info(f"Initial affine set - {parent.transform_model.about('; ', transform=affine)}")
            else:
                logger.info("Initial affine set - it was an identity matrix.")
        hp.disable_widgets(parent.initial_btn, disabled=False)
        return super().accept()

    def reject(self) -> None:
        """Reject changes."""
        parent: ImageRegistrationWindow = self.parent()
        try:
            parent.transform_model.moving_initial_affine = None
            for layer in parent.moving_image_layer:
                layer.affine = np.eye(3)
        except TypeError:
            pass
        hp.disable_widgets(parent.initial_btn, disabled=False)
        logger.trace("Initial affine reset.")
        return super().reject()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_close_handle("Initial transformation")

        self.rotate_bck = hp.make_qta_btn(
            self, "rotate_right", tooltip="Rotate (clockwise)", func=partial(self.on_rotate, value=-90)
        )
        self.rotate_fwd = hp.make_qta_btn(
            self, "rotate_left", tooltip="Rotate (counter-clockwise)", func=partial(self.on_rotate, value=90)
        )
        self.rotate_spin = hp.make_double_spin_box(
            self,
            value=0,
            minimum=-360,
            maximum=360,
            step_size=15,
            suffix="°",
            tooltip="Rotate",
            func=self.on_set_rotate,
        )
        self.flip_lr = hp.make_checkbox(self, "", func=self.on_flip_lr)

        layout = hp.make_form_layout(parent=self, margin=6)
        layout.addRow(header_layout)
        layout.addRow(
            "Rotate (counter-clockwise)",
            hp.make_h_layout(self.rotate_spin, self.rotate_bck, self.rotate_fwd, spacing=1, margin=0, stretch_id=(0,)),
        )
        layout.addRow("Flip left-right", self.flip_lr)
        layout.addRow(hp.make_h_line(self))
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
                hp.make_btn(self, "Reset", func=self.on_reset),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_url_btn(self, func=lambda: open_docs(dialog="initial-transform")),
                stretch_before=True,
                spacing=2,
                margin=2,
                alignment=Qt.AlignmentFlag.AlignVCenter,
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
        # flip left-right
        elif key == Qt.Key.Key_F:
            self.on_flip_lr()
