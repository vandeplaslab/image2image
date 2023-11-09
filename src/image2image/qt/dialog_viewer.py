"""Viewer dialog."""
from __future__ import annotations

import typing as ty
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_image_button import QtThemeButton
from qtpy.QtWidgets import QHBoxLayout, QMenuBar, QStatusBar, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_VIEWER_FORMATS
from image2image.qt._select import LoadWithTransformWidget
from image2image.qt.dialog_base import Window
from image2image.utils.utilities import style_form_layout

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel
    from image2image.models.transform import TransformModel


class ImageViewerWindow(Window):
    """Image viewer dialog."""

    image_layer: list[Image] | None = None
    shape_layer: list[Image] | None = None
    _console = None

    def __init__(self, parent: QWidget | None):
        super().__init__(parent, f"image2viewer: Simple viewer app (v{__version__})")

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        # connect(self._fixed_widget.evt_loading, self.on_indicator, state=state)
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closed, self.on_close_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_resolution, self.on_update_transform, state=state)
        connect(self._image_widget.transform_dlg.evt_transform, self.on_update_transform, state=state)

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    @property
    def transform_model(self) -> TransformModel:
        """Return transform model."""
        return self._image_widget.transform_model

    @ensure_main_thread
    def on_load_image(self, model: DataModel, channel_list: list[str]) -> None:
        """Load fixed image."""
        if model and model.n_paths:
            self._on_load_image(model, channel_list)
            hp.toast(
                self, "Loaded data", f"Loaded model with {model.n_paths} paths.", icon="success", position="top_left"
            )
        else:
            logger.warning(f"Failed to load data - model={model}")

    def _on_load_image(self, model: DataModel, channel_list: list[str] | None = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading fixed data with {model.n_paths} paths...")
            self.plot_image_layers(channel_list)
            self.view.viewer.reset_view()
        logger.info(f"Loaded data in {timer()}")

    def plot_image_layers(self, channel_list: list[str] | None = None) -> None:
        """Plot image layers."""
        self.image_layer, self.shape_layer = self._plot_image_layers(
            self.data_model, self.view, channel_list, "view", True
        )

    def on_close_image(self, model: DataModel) -> None:
        """Close fixed image."""
        self._close_model(model, self.view, "view")

    def on_update_transform(self, path: Path) -> None:
        """Update affine transformation."""
        wrapper = self.data_model.get_wrapper()
        reader = self.data_model.get_reader(path)
        if wrapper and reader:
            channel_names = wrapper.channel_names_for_names([path])
            for name in channel_names:
                if name not in self.view.layers:
                    continue
                layer = self.view.layers[name]
                layer.scale = reader.scale
                layer.affine = wrapper.get_affine(reader, reader.resolution)
                logger.trace(f"Updated affine for '{name}' with resolution={reader.resolution}.")

    def on_show_scalebar(self) -> None:
        """Show scale bar controls for the viewer."""
        from image2image.qt._dialogs._scalebar import QtScaleBarControls

        dlg = QtScaleBarControls(self.view.viewer, self.view.widget)
        dlg.show()

    def on_load_from_project(self, _evt=None):
        """Load a previous project."""
        path = hp.get_filename(self, "Load i2v project", base_dir=CONFIG.output_dir, file_filter=ALLOWED_VIEWER_FORMATS)
        if path:
            from image2image.models.data import load_viewer_setup_from_file
            from image2image.models.utilities import _remove_missing_from_dict

            path_ = Path(path)
            CONFIG.output_dir = str(path_.parent)

            # load data from config file
            try:
                paths, paths_missing, transform_data, resolution = load_viewer_setup_from_file(path_)
            except ValueError as e:
                hp.warn(self, f"Failed to load transformation from {path_}\n{e}", "Failed to load transformation")
                return

            # locate paths that are missing
            if paths_missing:
                from image2image.qt._dialogs import LocateFilesDialog

                locate_dlg = LocateFilesDialog(self, paths_missing)
                if locate_dlg.exec_():  # noqa
                    paths = locate_dlg.fix_missing_paths(paths_missing, paths)

            # clean-up affine matrices
            transform_data = _remove_missing_from_dict(transform_data, paths)
            resolution = _remove_missing_from_dict(resolution, paths)
            # add paths
            if paths:
                self._image_widget.on_set_path(paths, transform_data, resolution)

            # add affine matrices to transform object
            for name, matrix in transform_data.items():
                self.transform_model.add_transform(name, matrix)

    def on_save_to_project(self) -> None:
        """Export project."""
        model = self.data_model
        if model.n_paths == 0:
            logger.warning("Cannot save project - there are no images loaded.")
            return
        # get filename which is based on the moving dataset
        filename = model.get_filename() + ".i2v.json"
        path = hp.get_save_filename(
            self,
            "Save i2v project",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_VIEWER_FORMATS,
            base_filename=filename,
        )
        if path:
            path = Path(path)
            CONFIG.output_dir = str(path.parent)
            model.to_file(path)
            hp.toast(
                self, "Exported i2v project", f"Saved project to <br><b>{path}</b>", icon="success", position="top_left"
            )

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(self, add_toolbars=False, allow_extraction=False, disable_controls=True)
        self._image_widget = LoadWithTransformWidget(self, self.view, allow_geojson=True)

        side_layout = hp.make_form_layout()
        style_form_layout(side_layout)
        side_layout.addRow(
            hp.make_btn(self, "Import project...", tooltip="Load previous project", func=self.on_load_from_project)
        )
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(
            hp.make_btn(
                self,
                "Export project...",
                tooltip="Export configuration to a project file. Information such as image path and transformation"
                " information are saved.",
                func=self.on_save_to_project,
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
        main_layout.addLayout(layout, stretch=True)

        # extra settings
        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        from image2image.qt._sentry import send_feedback

        self.statusbar = QStatusBar()
        self.statusbar.setSizeGripEnabled(False)

        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "save",
                tooltip="Save snapshot of the canvas to file.",
                func=self.view.widget.on_save_figure,
                small=True,
            )
        )
        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "screenshot",
                tooltip="Take a snapshot of the canvas and copy it into your clipboard.",
                func=self.view.widget.clipboard,
                small=True,
            )
        )
        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "ruler",
                tooltip="Show scalebar.",
                func=self.on_show_scalebar,
                small=True,
            )
        )

        self.tutorial_btn = hp.make_qta_btn(
            self, "help", tooltip="Give me a quick tutorial!", func=self.on_show_tutorial, small=True
        )
        self.statusbar.addPermanentWidget(self.tutorial_btn)

        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "feedback",
                tooltip="Send feedback or suggestion.",
                func=partial(send_feedback, parent=self),
                small=True,
            )
        )

        self.theme_btn = QtThemeButton(self)
        self.theme_btn.auto_connect()
        with hp.qt_signals_blocked(self.theme_btn):
            self.theme_btn.dark = CONFIG.theme == "dark"
        self.theme_btn.clicked.connect(self.on_toggle_theme)
        self.theme_btn.set_small()

        self.statusbar.addPermanentWidget(self.theme_btn)
        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "ipython",
                tooltip="Open IPython console",
                small=True,
                func=self.on_show_console,
            )
        )

        self.update_status_btn = hp.make_btn(
            self,
            "Update available - click here to download!",
            tooltip="Show information about available updates.",
            func=self.on_show_update_info,
        )
        self.update_status_btn.setObjectName("update_btn")
        self.update_status_btn.hide()
        self.statusbar.addPermanentWidget(self.update_status_btn)
        self.setStatusBar(self.statusbar)

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
        hp.make_menu_item(
            self,
            "Clear data",
            menu=menu_file,
            func=self._image_widget.on_clear_data,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(
            self, "Show scale bar controls...", "Ctrl+S", menu=menu_tools, icon="ruler", func=self.on_show_scalebar
        )
        menu_tools.addSeparator()
        hp.make_menu_item(
            self, "Show IPython console...", "Ctrl+T", menu=menu_tools, icon="ipython", func=self.on_show_console
        )

        # Help menu
        menu_help = self._make_help_menu()

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(menu_help.menuAction())
        self.setMenuBar(self.menubar)

    def _get_console_variables(self) -> dict:
        return {
            "transforms_model": self.transform_model,
            "viewer": self.view.viewer,
            "data_model": self.data_model,
        }

    def closeEvent(self, evt):
        """Close."""
        if self._console:
            self._console.close()
        CONFIG.save()
        if self.data_model.is_valid():
            if hp.confirm(self, "There might be unsaved changes. Would you like to save them?"):
                self.on_save_to_project()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="viewer", level=0)
