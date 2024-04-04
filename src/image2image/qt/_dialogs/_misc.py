"""Other dialogs."""
from __future__ import annotations

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
        self.all_check = hp.make_checkbox(self, "Check all", clicked=self.on_check_all, value=True)
        self.micro_check = hp.make_checkbox(self, "Fixed images (if exist)", value=True, func=self.on_apply)
        self.micro_check.setHidden("fixed_image" in self.disable)
        self.ims_check = hp.make_checkbox(self, "Moving images (if exist)", value=True, func=self.on_apply)
        self.ims_check.setHidden("moving_image" in self.disable)
        self.fixed_check = hp.make_checkbox(self, "Fixed fiducials", value=True, func=self.on_apply)
        self.fixed_check.setHidden("fixed_points" in self.disable)
        self.moving_check = hp.make_checkbox(self, "Moving fiducials", value=True, func=self.on_apply)
        self.moving_check.setHidden("moving_points" in self.disable)

        layout = hp.make_form_layout()
        hp.style_form_layout(layout)
        layout.addRow(
            hp.make_label(
                self, "Please select what should be imported.", alignment=Qt.AlignmentFlag.AlignHCenter, bold=True
            )
        )
        layout.addRow(self.all_check)
        layout.addRow(hp.make_h_line())
        layout.addRow(self.micro_check)
        layout.addRow(self.ims_check)
        layout.addRow(self.fixed_check)
        layout.addRow(self.moving_check)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def on_check_all(self, state: bool):
        """Check all."""
        self.micro_check.setChecked(state)
        self.ims_check.setChecked(state)
        self.fixed_check.setChecked(state)
        self.moving_check.setChecked(state)

    def on_apply(self):
        """Apply."""
        self.config = self.get_config()
        all_checked = all(self.config.values())
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)

    def get_config(self) -> dict[str, bool]:
        """Return state."""
        return {
            "fixed_image": self.micro_check.isChecked() and not self.micro_check.isHidden(),
            "moving_image": self.ims_check.isChecked() and not self.ims_check.isHidden(),
            "fixed_points": self.fixed_check.isChecked() and not self.fixed_check.isHidden(),
            "moving_points": self.moving_check.isChecked() and not self.moving_check.isHidden(),
        }
