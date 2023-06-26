"""Various dialogs."""
import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout

from ims2micro.utilities import style_form_layout

if ty.TYPE_CHECKING:
    from ims2micro.models import DataModel


class CloseDatasetDialog(QtDialog):
    """Dialog where user can select which path(s) should be removed."""

    def __init__(self, parent, model: "DataModel"):
        self.model = model

        super().__init__(parent)
        self.config = self.get_config()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.all_check = hp.make_checkbox(
            self,
            "Check all",
            clicked=self.on_check_all,
            value=True,
        )
        # iterate over all available paths
        self.checkboxes = []
        for path in self.model.paths:
            # make checkbox for each path
            checkbox = hp.make_checkbox(self, str(path), value=True, clicked=self.on_apply)
            self.checkboxes.append(checkbox)

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(
            hp.make_label(
                self,
                "Please select which images should be <b>removed</b> from the project."
                " Fiducial markers will be unaffected.",
                alignment=Qt.AlignHCenter,
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(self.all_check)
        layout.addRow(hp.make_h_line())
        for checkbox in self.checkboxes:
            layout.addRow(checkbox)
        layout.addRow(hp.make_btn(self, "OK", func=self.accept), hp.make_btn(self, "Cancel", func=self.reject))
        return layout

    def on_check_all(self, state: bool):
        """Check all."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(state)

    def on_apply(self):
        """Apply."""
        self.config = self.get_config()
        all_checked = len(self.config) == len(self.checkboxes)
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)

    def get_config(self) -> ty.List[Path]:
        """Return state."""
        config = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                config.append(Path(checkbox.text()))
        return config


class ImportSelectDialog(QtDialog):
    """Dialog that lets you select what should be imported."""

    def __init__(self, parent):
        super().__init__(parent)
        self.config = self.get_config()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.all_check = hp.make_checkbox(self, "Check all", clicked=self.on_check_all, value=True)
        self.micro_check = hp.make_checkbox(self, "Microscopy images (if exist)", value=True, func=self.on_apply)
        self.ims_check = hp.make_checkbox(self, "IMS images (if exist)", value=True, func=self.on_apply)
        self.fixed_check = hp.make_checkbox(self, "Microscopy fiducials", value=True, func=self.on_apply)
        self.moving_check = hp.make_checkbox(self, "Imaging fiducials", value=True, func=self.on_apply)

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(
            hp.make_label(self, "Please select what should be imported.", alignment=Qt.AlignHCenter, bold=True)
        )
        layout.addRow(self.all_check)
        layout.addRow(hp.make_h_line())
        layout.addRow(self.micro_check)
        layout.addRow(self.ims_check)
        layout.addRow(self.fixed_check)
        layout.addRow(self.moving_check)
        layout.addRow(hp.make_btn(self, "OK", func=self.accept), hp.make_btn(self, "Cancel", func=self.reject))
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

    def get_config(self) -> ty.Dict[str, bool]:
        """Return state."""
        return {
            "micro": self.micro_check.isChecked(),
            "ims": self.ims_check.isChecked(),
            "fixed": self.fixed_check.isChecked(),
            "moving": self.moving_check.isChecked(),
        }
