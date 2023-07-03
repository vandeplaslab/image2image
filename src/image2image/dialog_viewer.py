"""Viewer dialog."""
from pathlib import Path

from functools import partial

import typing as ty

import numpy as np
import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image

from image2image.enums import ALLOWED_PROJECT_FORMATS
from qtextra._napari.mixins import ImageViewMixin
from qtextra.mixins import IndicatorMixin
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QMainWindow, QVBoxLayout, QWidget, QMenuBar
from superqt import ensure_main_thread

import image2image.assets  # noqa: F401
from image2image import __version__
from image2image._select import LoadWithTransformWidget

# need to load to ensure all assets are loaded properly
from image2image.config import CONFIG
from image2image.models import DataModel
from image2image.utilities import (
    get_colormap,
    log_exception,
    style_form_layout,
)


class ImageViewerWindow(QMainWindow, IndicatorMixin, ImageViewMixin):
    """Image viewer dialog."""

    _console = None

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)  # noqa
        self.setWindowTitle(f"image2viewer: Simple viewer app (v{__version__})")
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setMouseTracking(True)
        self.setMinimumSize(1200, 800)

        # load configuration
        CONFIG.load()

        self._setup_ui()
        self.setup_events()
        # delay asking for telemetry opt-in by 10s
        # hp.call_later(self, install_error_monitor, 5_000)

    def setup_events(self, state: bool = True):
        """Setup events."""
        # connect(self._fixed_widget.evt_loading, self.on_indicator, state=state)
        connect(self._image_widget.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.evt_closed, self.on_close_image, state=state)
        connect(self._image_widget.evt_transform_changed, self.on_update_transform, state=state)

    @ensure_main_thread
    def on_load_image(self, model: DataModel, channel_list: ty.List[str]):
        """Load fixed image."""
        if model and model.n_paths:
            self._on_load_image(model, channel_list)
            hp.toast(self, "Loaded data", f"Loaded model with {model.n_paths} paths.")
        else:
            logger.warning(f"Failed to load data - model={model}")
        # self.on_indicator("fixed", False)

    def _on_load_image(self, model: DataModel, channel_list: ty.Optional[ty.List[str]] = None):
        with MeasureTimer() as timer:
            logger.info(f"Loading fixed data with {model.n_paths} paths...")
            self._plot_image_layers(channel_list)
            self.view.viewer.reset_view()
        logger.info(f"Loaded data in {timer()}")

    def _plot_image_layers(self, channel_list: ty.Optional[ty.List[str]] = None):
        wrapper = self._image_widget.model.get_wrapper()
        if channel_list is None:
            channel_list = wrapper.channel_names()
        fixed_image_layer = []
        used = [layer.colormap for layer in self.view.layers if isinstance(layer, Image)]
        for index, (name, array) in enumerate(wrapper.channel_image_iter()):
            logger.trace(f"Adding '{name}' to view...")
            with MeasureTimer() as timer:
                if name in self.view.layers:
                    fixed_image_layer.append(self.view.layers[name])
                    continue
                fixed_image_layer.append(
                    self.view.viewer.add_image(
                        array,
                        name=name,
                        blending="additive",
                        colormap=get_colormap(index, used),
                        visible=name in channel_list,
                        affine=np.eye(3),  # TODO: need to fix this
                    )
                )
                logger.trace(f"Added '{name}' to view in {timer()}.")
        self.fixed_image_layer = fixed_image_layer

    def on_close_image(self, model: DataModel):
        """Close fixed image."""
        try:
            channel_names = model.channel_names()
            layer_names = [layer.name for layer in self.view.layers if isinstance(layer, Image)]
            for name in layer_names:
                if name not in channel_names:
                    del self.view.layers[name]
                    logger.trace(f"Removed '{name}' from view.")
        except Exception as e:
            log_exception(e)

    def on_update_transform(self, path):
        """Update affine transformation."""
        wrapper = self._image_widget.model.get_wrapper()
        reader = self._image_widget.model.get_reader(path)
        if wrapper and reader:
            channel_names = wrapper.channel_names_for_names([path])
            for name in channel_names:
                layer = self.view.layers[name]
                layer.affine = reader.transform
                logger.trace(f"Updated affine for '{name}' to {reader.transform}.")

    def on_show_console(self):
        """View console."""
        if self._console is None:
            from image2image._console import QtConsoleDialog

            self._console = QtConsoleDialog(self)
            self._console.push_variables(
                {
                    "transforms_model": self._image_widget.table_dlg.transform_model,
                    "viewer": self.view.viewer,
                    "image_model": self._image_widget.model,
                }
            )
        self._console.show()

    def on_load(self):
        """Load a previous project."""
        path = hp.get_filename(
            self, "Load i2v project", base_dir=CONFIG.output_dir, file_filter=ALLOWED_PROJECT_FORMATS
        )
        if path:
            path = Path(path)
            CONFIG.output_dir = str(path.parent)

    def on_save(self):
        """Export project."""
        model = self._image_widget.model
        if model.n_paths == 0:
            logger.warning("Cannot save project - there are no images loaded.")
            return
        # get filename which is based on the moving dataset
        filename = model.get_filename() + ".i2v.json"
        path = hp.get_save_filename(
            self,
            "Save i2v project",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_PROJECT_FORMATS,
            base_filename=filename,
        )
        if path:
            path = Path(path)
            CONFIG.output_dir = str(path.parent)
            model.to_file(path)
            hp.toast(self, "Exported i2v project", f"Saved project to <br><b>{path}</b>")

    # noinspection PyAttributeOutsideInit
    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(self, add_toolbars=False, allow_extraction=False, disable_controls=True)
        self._image_widget = LoadWithTransformWidget(self, self.view)

        side_layout = hp.make_form_layout()
        style_form_layout(side_layout)
        side_layout.addRow(hp.make_btn(self, "Import project...", tooltip="Load previous project", func=self.on_load))
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(
            hp.make_btn(
                self,
                "Export project...",
                tooltip="Export configuration to a project file. Information such as image path and transformation"
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

    def _make_icon(self):
        """Make icon."""
        from image2image.assets import ICON_ICO

        self.setWindowIcon(hp.get_icon_from_img(ICON_ICO))

    def _make_menu(self):
        """Make menu items."""
        from image2image._dialogs import open_about
        from image2image._sentry import ask_opt_in, send_feedback
        from image2image.utilities import open_bug_report, open_docs, open_github, open_request

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
        hp.make_menu_item(self, "Show IPython console...", "Ctrl+T", menu=menu_tools, func=self.on_show_console)

        # Help menu
        menu_help = hp.make_menu(self, "Help")
        hp.make_menu_item(self, "Documentation (in browser)", menu=menu_help, icon="web", func=open_docs)
        hp.make_menu_item(
            self,
            "GitHub (online)",
            menu=menu_help,
            status_tip="Open project's GitHub page.",
            icon="github",
            func=open_github,
        )
        hp.make_menu_item(
            self,
            "Request Feature (online)",
            menu=menu_help,
            status_tip="Open project's GitHub feature request page.",
            icon="request",
            func=open_request,
        )
        hp.make_menu_item(
            self,
            "Report Bug (online)",
            menu=menu_help,
            status_tip="Open project's GitHub bug report page.",
            icon="bug",
            func=open_bug_report,
        )
        menu_help.addSeparator()
        hp.make_menu_item(
            self, "Send feedback...", menu=menu_help, func=partial(send_feedback, parent=self), icon="feedback"
        )
        hp.make_menu_item(self, "Telemetry...", menu=menu_help, func=partial(ask_opt_in, parent=self), icon="telemetry")
        hp.make_menu_item(self, "About...", menu=menu_help, func=partial(open_about, parent=self), icon="info")

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(menu_help.menuAction())
        self.setMenuBar(self.menubar)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="viewer", level=0)
