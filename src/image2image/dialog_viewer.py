"""Viewer dialog."""
import typing as ty
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image
from napari.layers.points.points import Mode, Points
from napari.layers.utils._link_layers import link_layers
from qtextra._napari.mixins import ImageViewMixin
from qtextra.mixins import IndicatorMixin
from qtextra.utils.utilities import connect
from qtpy.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget
from superqt import ensure_main_thread

# need to load to ensure all assets are loaded properly
import image2image.assets  # noqa: F401
from image2image import __version__
from image2image._select import LoadWidget
from image2image.models import DataModel, Transformation
from image2image.utilities import (
    _get_text_data,
    _get_text_format,
    get_colormap,
    init_points_layer,
    log_exception,
    style_form_layout,
)

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


class ImageViewerWindow(QMainWindow, IndicatorMixin, ImageViewMixin):
    """Image viewer dialog."""

    # noinspection PyAttributeOutsideInit
    def _setup_ui(self):
        """Create panel."""
        self.load_btn = hp.make_btn(
            self,
            "Import transformation file",
            tooltip="Import previously computed transformation.",
            func=self.on_load,
        )

        self._view_widget = LoadWidget(self, self.view)
        self._layer_controls = None

        side_layout = hp.make_form_layout()
        style_form_layout(side_layout)
        side_layout.addRow(self.load_btn)
        side_layout.addRow(self._view_widget)
        side_layout.addRow(hp.make_h_line_with_text("Layer controls"))

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QHBoxLayout()
        layout.addWidget(self._make_view(), stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addLayout(side_layout)
        main_layout = QVBoxLayout(widget)
        main_layout.addLayout(layout)

        # extra settings
        # self._make_menu()
        # self._make_icon()

    def _make_view(self):
        self.view = self._make_image_view(self, add_toolbars=False)
        return self.view.widget
