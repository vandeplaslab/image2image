"""Scalebar overrides."""

import typing as ty

import qtextra.helpers as hp
from qtextraplot._napari.common.component_controls.qt_scalebar_controls import QtScaleBarControls as _QtScaleBarControls
from qtpy.QtCore import Qt


class QtScaleBarControls(_QtScaleBarControls):
    """Scalebar controls."""

    def __init__(self, *args: ty.Any, **kwargs: ty.Any):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)  # type: ignore[attr-defined]

        self.HIDE_WHEN_CLOSE = False

        close_btn = hp.make_qta_btn(self, "cross", tooltip="Click here to close the popup window", normal=True)
        close_btn.clicked.connect(self.close)
        self._title_layout.addWidget(close_btn)

    def set_px_size(self, px_size: float):
        """Set pixel size based on the smallest resolution."""
        self.units_combobox.setCurrentText("Micrometers")
