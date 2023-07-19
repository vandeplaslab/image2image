"""Scalebar overrides."""
from qtpy.QtCore import Qt

import qtextra.helpers as hp
from qtextra._napari.image.component_controls.qt_scalebar_controls import QtScaleBarControls as _QtScaleBarControls


class QtScaleBarControls(_QtScaleBarControls):
    """Scalebar controls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)

        self.HIDE_WHEN_CLOSE = False

        close_btn = hp.make_qta_btn(self, "cross", tooltip="Click here to close the popup window", normal=True)
        close_btn.clicked.connect(self.close)
        self._title_layout.insertWidget(3, close_btn)
