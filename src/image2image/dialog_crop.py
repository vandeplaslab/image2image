"""Viewer dialog."""
import typing as ty
from contextlib import contextmanager
from math import ceil, floor

import numpy as np
import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Shapes
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QMenuBar, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from image2image import __version__
from image2image._select import LoadWidget
from image2image.config import CONFIG
from image2image.dialog_base import Window
from image2image.utilities import init_shapes_layer, style_form_layout

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


class ImageCropWindow(Window):
    """Image viewer dialog."""

    image_layer: ty.Optional[ty.List["Image"]] = None
    _console = None
    _editing = False

    def __init__(self, parent):
        super().__init__(parent, f"image2crop: Crop and export microscopy data app (v{__version__})")

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closed, self.on_close_image, state=state)

    @ensure_main_thread
    def on_load_image(self, model: "DataModel", channel_list: ty.List[str]) -> None:
        """Load fixed image."""
        if model and model.n_paths:
            self._on_load_image(model, channel_list)
            hp.toast(
                self, "Loaded data", f"Loaded model with {model.n_paths} paths.", icon="success", position="top_left"
            )
        else:
            logger.warning(f"Failed to load data - model={model}")
        # self.on_indicator("fixed", False)

    def _on_load_image(self, model: "DataModel", channel_list: ty.Optional[ty.List[str]] = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading fixed data with {model.n_paths} paths...")
            self.plot_image_layers(channel_list)
            self.view.viewer.reset_view()
        logger.info(f"Loaded data in {timer()}")

    def plot_image_layers(self, channel_list: ty.Optional[ty.List[str]] = None) -> None:
        """Plot image layers."""
        self.image_layer = self._plot_image_layers(self.data_model, self.view, channel_list, "view", True)

    def on_close_image(self, model: "DataModel") -> None:
        """Close fixed image."""
        self._close_model(model, self.view, "view")

    def on_load(self) -> None:
        """Load previous data."""

    def on_save(self) -> None:
        """Save data."""

    def on_crop_rect(self):
        """Update crop rect."""
        if self.crop_layer.data:
            self.crop_layer.data = []
        left = self.left_edit.text()
        left = int(left or 0)  # type: ignore
        top = self.top_edit.text()
        top = int(top or 0)  # type: ignore
        right = self.right_edit.text()
        right = int(right or 0)  # type: ignore
        bottom = self.bottom_edit.text()
        bottom = int(bottom or 0)  # type: ignore
        rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
        with self._editing_crop():
            self.crop_layer.data = [(rect, "rectangle")]
        logger.trace("Updated rectangle (from edit).")

    def on_update_crop(self, _evt=None):
        """Update crop values."""
        if self._editing:
            return
        n = len(self.crop_layer.data)
        if n == 0:
            return
        if n > 1:
            hp.toast(
                self,
                "Multiple rectangles detected!",
                "There are more than one crop rectangles. Only the first one will be used.",
                icon="error",
            )
        rect = np.asarray(self.crop_layer.data[0])
        xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
        xmin = max(0, xmin)
        ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
        ymin = max(0, ymin)
        with hp.qt_signals_blocked(self.left_edit, self.right_edit, self.top_edit):
            self.left_edit.setText(str(floor(xmin)))
            self.right_edit.setText(str(ceil(xmax)))
            self.top_edit.setText(str(floor(ymin)))
            self.bottom_edit.setText(str(ceil(ymax)))
        logger.trace("Updated rectangle (from canvas).")

    @property
    def crop_layer(self) -> Shapes:
        """Crop layer."""
        if "Crop rectangle" not in self.view.layers:
            layer = self.view.viewer.add_shapes(
                None,
                edge_width=5,
                name="Crop rectangle",
                face_color="green",
                edge_color="white",
            )
            visual = self.view.widget.layer_to_visual[layer]
            init_shapes_layer(layer, visual)
            connect(self.crop_layer.events.data, self.on_update_crop, state=True)
            # connect(self.crop_layer.events.set_data, self.on_update_crop, state=True)
        return self.view.layers["Crop rectangle"]

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(self, add_toolbars=False, allow_extraction=False, disable_controls=True)
        self._image_widget = LoadWidget(self, self.view, n_max=1)

        self.left_edit = hp.make_line_edit(
            self, placeholder="Left", validator=QIntValidator(0, 75_000), func=self.on_crop_rect
        )
        self.left_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.top_edit = hp.make_line_edit(
            self, placeholder="Top", validator=QIntValidator(0, 75_000), func=self.on_crop_rect
        )
        self.top_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.right_edit = hp.make_line_edit(
            self, placeholder="Right", validator=QIntValidator(0, 75_000), func=self.on_crop_rect
        )
        self.right_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.bottom_edit = hp.make_line_edit(
            self, placeholder="Bottom", validator=QIntValidator(0, 75_000), func=self.on_crop_rect
        )
        self.bottom_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]

        crop_layout = QGridLayout()
        crop_layout.addWidget(self.left_edit, 1, 0, 1, 1)
        crop_layout.addWidget(self.right_edit, 1, 2, 1, 1)
        crop_layout.addWidget(self.top_edit, 0, 1, 1, 1)
        crop_layout.addWidget(self.bottom_edit, 2, 1, 1, 1)

        side_layout = hp.make_form_layout()
        style_form_layout(side_layout)
        side_layout.addRow(hp.make_btn(self, "Import project...", tooltip="Load previous project", func=self.on_load))
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line_with_text("Image crop position"))
        side_layout.addRow(crop_layout)
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(
            hp.make_btn(
                self,
                "Export project...",
                tooltip="Export configuration to a project file. Information such as image path and crop"
                " information are saved.",
                func=self.on_save,
            )
        )
        side_layout.addRow(hp.make_h_line_with_text("Layer controls"))
        side_layout.addRow(self.view.widget.controls)
        side_layout.addRow(self.view.widget.layerButtons)
        side_layout.addRow(self.view.widget.layers)
        side_layout.addRow(self.view.widget.viewerButtons)

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QHBoxLayout()
        layout.addWidget(self.view.widget, stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addLayout(side_layout)
        main_layout = QVBoxLayout(widget)
        main_layout.addLayout(layout)

        # extra settings
        self._make_menu()
        self._make_icon()

    def _make_menu(self) -> None:
        """Make menu items."""
        # File menu
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(
            self,
            "Add image (.tiff, .png, .jpg, .imzML, .tdf, .tsf, + others)...",
            "Ctrl+I",
            menu=menu_file,
            func=self._image_widget.on_select_dataset,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(self, "Show scale bar controls...", "Ctrl+S", menu=menu_tools, func=self.on_show_scalebar)
        menu_tools.addSeparator()
        hp.make_menu_item(self, "Show IPython console...", "Ctrl+T", menu=menu_tools, func=self.on_show_console)

        # Help menu
        menu_help = self._make_help_menu()

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(menu_help.menuAction())
        self.setMenuBar(self.menubar)

    def on_show_scalebar(self):
        """Show scale bar controls for the viewer."""
        from qtextra._napari.image.component_controls.qt_scalebar_controls import QtScaleBarControls

        dlg = QtScaleBarControls(self.view.viewer, self.view.widget)
        dlg.show_below_widget(self._image_widget)

    @property
    def data_model(self) -> "DataModel":
        """Return transform model."""
        return self._image_widget.model

    def _get_console_variables(self) -> ty.Dict:
        return {
            "viewer": self.view.viewer,
            "data_model": self.data_model,
        }

    # noinspection PyAttributeOutsideInit
    @contextmanager
    def _editing_crop(self):
        self._editing = True
        yield
        self._editing = False

    def closeEvent(self, evt):
        """Close."""
        if self._console:
            self._console.close()
        CONFIG.save()
        if self.data_model.is_valid():
            if hp.confirm(self, "There might be unsaved changes. Would you like to save them?"):
                self.on_save()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="crop", level=0)
