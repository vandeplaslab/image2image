"""Other dialogs."""

from __future__ import annotations

from typing import Optional

from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout


class ImportSelectDialog(QtDialog):
    """Dialog that lets you select what should be imported."""

    def __init__(self, parent, disable: tuple[str, ...] = ()):
        self.disable = disable
        super().__init__(parent)
        self.config = self.get_config()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.all_btn = hp.make_btn(self, "Check all", func=self.on_check_all)
        self.only_fiducial_btn = hp.make_btn(self, "Only fiducials", func=self.on_check_fiducials)
        self.only_images_btn = hp.make_btn(self, "Only images", func=self.on_check_images)

        self.fixed_image_check = hp.make_checkbox(self, "Fixed images (if exist)", value=True, func=self.on_apply)
        self.fixed_image_check.setHidden("fixed_image" in self.disable)
        self.moving_image_check = hp.make_checkbox(self, "Moving images (if exist)", value=True, func=self.on_apply)
        self.moving_image_check.setHidden("moving_image" in self.disable)
        self.fixed_fiducial_check = hp.make_checkbox(self, "Fixed fiducials", value=True, func=self.on_apply)
        self.fixed_fiducial_check.setHidden("fixed_points" in self.disable)
        self.moving_fiducial_check = hp.make_checkbox(self, "Moving fiducials", value=True, func=self.on_apply)
        self.moving_fiducial_check.setHidden("moving_points" in self.disable)

        layout = hp.make_form_layout()
        layout.addRow(
            hp.make_label(
                self, "Please select what should be imported.", alignment=Qt.AlignmentFlag.AlignHCenter, bold=True
            )
        )
        layout.addRow(hp.make_h_layout(self.all_btn, self.only_fiducial_btn, self.only_images_btn, spacing=20))
        layout.addRow(hp.make_h_line())
        layout.addRow(self.fixed_image_check)
        layout.addRow(self.moving_image_check)
        layout.addRow(self.fixed_fiducial_check)
        layout.addRow(self.moving_fiducial_check)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def on_check_all(self, _state: bool | None = None) -> None:
        """Check all."""
        self.fixed_image_check.setChecked(True)
        self.moving_image_check.setChecked(True)
        self.fixed_fiducial_check.setChecked(True)
        self.moving_fiducial_check.setChecked(True)

    def on_check_fiducials(self, _state: bool | None = None) -> None:
        """Check fiducials."""
        self.fixed_image_check.setChecked(False)
        self.moving_image_check.setChecked(False)
        self.fixed_fiducial_check.setChecked(True)
        self.moving_fiducial_check.setChecked(True)

    def on_check_images(self, _state: bool | None = None) -> None:
        """Check images."""
        self.fixed_image_check.setChecked(True)
        self.moving_image_check.setChecked(True)
        self.fixed_fiducial_check.setChecked(False)
        self.moving_fiducial_check.setChecked(False)

    def on_apply(self) -> None:
        """Apply."""
        self.config = self.get_config()

    def get_config(self) -> dict[str, bool]:
        """Return state."""
        return {
            "fixed_image": self.fixed_image_check.isChecked() and not self.fixed_image_check.isHidden(),
            "moving_image": self.moving_image_check.isChecked() and not self.moving_image_check.isHidden(),
            "fixed_points": self.fixed_fiducial_check.isChecked() and not self.fixed_fiducial_check.isHidden(),
            "moving_points": self.moving_fiducial_check.isChecked() and not self.moving_fiducial_check.isHidden(),
        }
