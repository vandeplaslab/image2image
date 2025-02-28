"""Modality list."""

from __future__ import annotations

import typing as ty

import qtextra.helpers as hp
from qtextra.widgets.qt_list_widget import QtListItem, QtListWidget
from qtpy.QtCore import QRegularExpression, Qt, Signal, Slot  # type: ignore[attr-defined]
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QHBoxLayout, QListWidgetItem, QWidget

from image2image.qt._wsi._widgets import QtModalityLabel

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader


class QtDatasetItem(QtListItem):
    """Widget used to nicely display information about RGBImageChannel.

    This widget permits display of label information as well as modification of color.
    """

    evt_delete = Signal(str)
    evt_resolution = Signal(str)

    _mode: bool = False

    item_model: str

    def __init__(self, item: QListWidgetItem, key: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.item = item
        self.item_model = key

        self.name_label = hp.make_label(self, "", tooltip="Name of the modality.")
        self.resolution_label = hp.make_line_edit(
            self,
            tooltip="Resolution of the modality.",
            func=self._on_update_resolution,
            validator=QRegularExpressionValidator(QRegularExpression(r"^[0-9]+(\.[0-9]{1,3})?$")),
        )

        self.modality_icon = QtModalityLabel(self)
        self.open_dir_btn = hp.make_qta_btn(
            self, "folder", tooltip="Open directory containing the image.", normal=True, func=self.on_open_directory
        )
        self.remove_btn = hp.make_qta_btn(
            self,
            "delete",
            tooltip="Remove modality from the list.",
            normal=True,
            func=self.on_remove,
        )

        layout = hp.make_form_layout()
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addRow(hp.make_label(self, "Name"), self.name_label)
        layout.addRow(hp.make_label(self, "Pixel size"), self.resolution_label)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(1, 1, 1, 1)
        main_layout.setSpacing(1)
        main_layout.addLayout(
            hp.make_v_layout(
                self.modality_icon,
                self.open_dir_btn,
                self.remove_btn,
                stretch_after=True,
                widget_alignment=Qt.AlignmentFlag.AlignCenter,
            ),
        )
        main_layout.addLayout(layout, stretch=True)

        self.mode = False
        self._set_from_model()

    def get_model(self) -> BaseReader:
        """Get model."""

    def _set_from_model(self, _: ty.Any = None) -> None:
        """Update UI elements."""
        model = self.get_model()
        self.name_label.setText(model.name)
        self.modality_icon.state = model.reader_type
        self.resolution_label.setText(f"{model.resolution:.3f}")
        self.setToolTip(f"<b>Modality</b>: {model.name}<br><b>Path</b>: {model.path}")

    def _on_update_resolution(self) -> None:
        """Update resolution."""
        resolution = self.resolution_label.text()
        if not resolution:
            return
        model = self.get_model()
        model.resolution = float(resolution)
        self.evt_resolution.emit(self.item_model)

    def on_remove(self) -> None:
        """Remove image/modality from the list."""
        model = self.get_model()
        if hp.confirm(self, f"Are you sure you want to remove <b>{model.name}</b> from the list?", "Please confirm."):
            self.evt_delete.emit(self.item_model)

    def on_open_directory(self) -> None:
        """Open directory where the image is located."""
        from koyo.path import open_directory_alt

        model = self.get_model()
        open_directory_alt(model.path)
