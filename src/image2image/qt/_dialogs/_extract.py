"""Extract."""

from __future__ import annotations

import typing as ty

from koyo.utilities import pluralize
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_table_view_check import QtCheckableTableView
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout

from image2image.config import get_register_config

if ty.TYPE_CHECKING:
    from image2image.qt._dialogs import DatasetDialog


class ExtractChannelsDialog(QtDialog):
    """Dialog to extract ion images."""

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 0, no_sort=True, hidden=True)
        .add("m/z", "mz", "float", 100)
    )

    def __init__(self, parent: DatasetDialog, key_to_extract: str):
        super().__init__(parent, title="Extract Ion Images")
        self.setFocus()
        self.key_to_extract = key_to_extract
        self.mzs = None
        self.ppm = None

    def on_open_peaklist(self) -> None:
        """Open peaklist."""
        import pandas as pd

        path = hp.get_filename(
            self,
            title="Select peak list...",
            base_dir=get_register_config().fixed_dir,
            file_filter="CSV files (*.csv);;",
            multiple=False,
        )
        if path:
            df = pd.read_csv(path, sep=",")
            if "mz" in df.columns:
                mzs = df.mz.values
            elif "m/z" in df.columns:
                mzs = df["m/z"].values
            else:
                hp.warn_pretty(self, "The file does not contain a column named 'mz' or 'm/z'.")
                return
            data = [[True, mz] for mz in mzs]
            self.table.add_data(data)

    def on_accept(self) -> None:
        """Accept."""
        if not self.mzs:
            hp.warn_pretty(self, "Please add at least one m/z value to extract.")
            return
        n = len(self.mzs)
        if n > 0 and hp.confirm(self, f"Would you like to extract <b>{n}</b> ion {pluralize('image', n)}?"):
            logger.trace(f"Extracting {n} ion images...")
            self.accept()

    def on_add(self) -> None:
        """Add peak."""
        value = self.mz_edit.value()
        values = self.table.get_col_data(self.TABLE_CONFIG.mz)
        if value is not None and value not in values:
            self.table.add_data([[True, value]])
        self.mzs = self.table.get_col_data(self.TABLE_CONFIG.mz)
        self.ppm = self.ppm_edit.value()

    def on_delete_row(self) -> None:
        """Delete row."""
        sel_model = self.table.selectionModel()
        if sel_model.hasSelection():
            indices = [index.row() for index in sel_model.selectedRows()]
            indices = sorted(indices, reverse=True)
            for index in indices:
                self.table.remove_row(index)
                logger.trace(f"Deleted '{index}' from m/z table")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.mz_edit = hp.make_double_spin_box(self, minimum=0, maximum=2500, step=0.1, n_decimals=3)
        self.ppm_edit = hp.make_double_spin_box(
            self, minimum=0.5, maximum=500, value=10, step=2.5, n_decimals=1, suffix=" ppm"
        )

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        self.table.setup_model_from_config(self.TABLE_CONFIG)
        hp.set_font(self.table)

        layout = hp.make_form_layout(parent=self)
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "m/z"),
                self.mz_edit,
                hp.make_btn(self, "Add m/z", tooltip="Add peak", func=self.on_add),
                stretch_id=1,
            )
        )
        layout.addRow(hp.make_h_layout(hp.make_label(self, "ppm"), self.ppm_edit, stretch_id=1))
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Press <b>Delete</b> or <b>Backspace</b> to delete a peak.<br>"
                "<b>Tip.</b> It is a lot more efficient to extra many peaks at once than one at a time.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Open peaklist", func=self.on_open_peaklist),
                hp.make_btn(self, "Extract", func=self.on_accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        if key == Qt.Key_Escape:  # type: ignore[attr-defined]
            evt.ignore()
        elif key == Qt.Key_Backspace or key == Qt.Key_Delete:  # type: ignore[attr-defined]
            self.on_delete_row()
            evt.accept()
        elif key == Qt.Key_Plus or key == Qt.Key_A:  # type: ignore[attr-defined]
            self.on_add()
            evt.accept()
        else:
            super().keyPressEvent(evt)
