"""Viewer dialog."""
from __future__ import annotations

import typing as ty
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Shapes
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtextra.widgets.qt_image_button import QtThemeButton
from qtpy.QtWidgets import QDialog, QHBoxLayout, QMenuBar, QStatusBar, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_VIEWER_FORMATS
from image2image.qt._select import LoadWithTransformWidget
from image2image.qt.dialog_base import Window
from image2image.utils.utilities import ensure_extension

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel
    from image2image.models.transform import TransformModel


class ImageViewerWindow(Window):
    """Image viewer dialog."""

    image_layer: list[Image] | None = None
    shape_layer: list[Shapes] | None = None
    _console = None

    def __init__(self, parent: QWidget | None):
        super().__init__(parent, f"image2viewer: Simple viewer app (v{__version__})")
        READER_CONFIG.view_type = "overlay"
        READER_CONFIG.only_last_pyramid = False
        READER_CONFIG.init_pyramid = True
        if CONFIG.first_time_viewer:
            hp.call_later(self, self.on_show_tutorial, 10_000)

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closed, self.on_close_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_resolution, self.on_update_transform, state=state)
        connect(self._image_widget.transform_dlg.evt_transform, self.on_update_transform, state=state)
        connect(self._image_widget.evt_toggle_channel, self.on_toggle_channel, state=state)
        connect(self._image_widget.evt_toggle_all_channels, self.on_toggle_all_channels, state=state)

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

    def on_toggle_channel(self, name: str, state: bool) -> None:
        """Toggle channel."""
        self._toggle_channel(self.data_model, self.view, name, state, "view")

    def on_toggle_all_channels(self, state: bool, channel_names: list[str] | None = None) -> None:
        """Toggle channel."""
        self._toggle_all_channels(self.data_model, self.view, state, "view", channel_names)

    def on_update_transform(self, key: str) -> None:
        """Update affine transformation."""
        wrapper = self.data_model.wrapper
        reader = self.data_model.get_reader_for_key(key)
        if wrapper and reader:
            channel_names = wrapper.channel_names_for_names([reader.key])
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
        dlg.show_above_widget(self.scalebar_btn)

    def on_show_save_figure(self) -> None:
        """Show scale bar controls for the viewer."""
        from image2image.qt._dialogs._screenshot import QtScreenshotDialog

        dlg = QtScreenshotDialog(self.view, self)
        dlg.show_above_widget(self.clipboard_btn)

    def on_load_from_project(self, _evt=None):
        """Load a previous project."""
        path_ = hp.get_filename(
            self, "Load i2v project", base_dir=CONFIG.output_dir, file_filter=ALLOWED_VIEWER_FORMATS
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: str) -> None:
        if path_:
            from image2image.models.data import load_viewer_setup_from_file
            from image2image.models.utilities import _remove_missing_from_dict

            path = Path(path_)
            CONFIG.output_dir = str(path.parent)

            # load data from config file
            try:
                paths, paths_missing, transform_data, resolution = load_viewer_setup_from_file(path)
            except ValueError as e:
                hp.warn(self, f"Failed to load transformation from {path}\n{e}", "Failed to load transformation")
                return

            # locate paths that are missing
            if paths_missing:
                from image2image.qt._dialogs import LocateFilesDialog

                locate_dlg = LocateFilesDialog(self, paths_missing)
                if locate_dlg.exec_():  # type: ignore[attr-defined]
                    paths = locate_dlg.fix_missing_paths(paths_missing, paths)

            # clean-up affine matrices
            transform_data = _remove_missing_from_dict(transform_data, paths)
            resolution = _remove_missing_from_dict(resolution, paths)
            # add paths
            if paths:
                self._image_widget.on_set_path(paths, transform_data, resolution)

            # add affine matrices to the transform model
            for name, matrix in transform_data.items():
                self.transform_model.add_transform(name, matrix)

    def on_save_to_project(self) -> None:
        """Export project."""
        model = self.data_model
        if model.n_paths == 0:
            logger.warning("Cannot save project - there are no images loaded.")
            hp.toast(
                self,
                "Cannot save project",
                "Cannot save project - there are no images loaded.",
                icon="warning",
                position="top_left",
            )
            return
        # get filename which is based on the moving dataset
        filename = model.get_filename() + ".i2v.json"
        path_ = hp.get_save_filename(
            self,
            "Save i2v project",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_VIEWER_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2v")
            CONFIG.output_dir = str(path.parent)
            model.to_file(path)
            hp.toast(
                self,
                "Exported i2v project",
                f"Saved project to<br><b>{hp.hyper(path)}</b>",
                icon="success",
                position="top_left",
            )

    def on_save_masks(self) -> None:
        """Export masks."""
        # Ask user which layer(s) to export (select layer(s) from list) - only shapes
        # Specify output dimensions (select layer(s) from list) - only images
        from image2image.qt._dialogs._mask import MasksDialog

        dlg = MasksDialog(self)
        dlg.show()

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(
            self, add_toolbars=False, allow_extraction=False, disable_controls=True, disable_new_layers=True
        )
        self._image_widget = LoadWithTransformWidget(
            self,
            self.view,
            allow_geojson=True,
            project_extension=[".i2v.json", ".i2v.toml"],
        )

        self.import_project_btn = hp.make_btn(
            self, "Import project...", tooltip="Load previous project", func=self.on_load_from_project
        )
        self.export_mask_btn = hp.make_btn(
            self,
            "Export GeoJSON as masks...",
            tooltip="Export masks/regions in a AutoIMS compatible format",
            func=self.on_save_masks,
        )
        self.export_project_btn = hp.make_btn(
            self,
            "Export project...",
            tooltip="Export configuration to a project file. Information such as image path and transformation"
            " information are saved.",
            func=self.on_save_to_project,
        )

        side_layout = hp.make_form_layout()
        hp.style_form_layout(side_layout)
        side_layout.addRow(self.import_project_btn)
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line_with_text("Export"))
        side_layout.addRow(self.export_mask_btn)
        side_layout.addRow(self.export_project_btn)
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

    # def _make_focus_layout(self) -> QFormLayout:
    #     self.lock_btn = hp.make_lock_btn(
    #         self,
    #         func=self.on_lock,
    #         normal=True,
    #         tooltip="Lock the area of interest. Press <b>L</b> on your keyboard to lock.",
    #     )
    #     # self.set_current_focus_btn = hp.make_btn(self, "Set current range", func=self.on_set_focus)
    #     self.x_center = hp.make_double_spin_box(
    #         self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the x-axis."
    #     )
    #     self.y_center = hp.make_double_spin_box(
    #         self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the y-axis."
    #     )
    #     self.zoom = hp.make_double_spin_box(self, -1e5, 1e5, step_size=0.5, n_decimals=4, tooltip="Zoom factor.")
    #     self.use_focus_btn = hp.make_btn(
    #         self,
    #         "Zoom-in",
    #         func=self.on_apply_focus,
    #         tooltip="Zoom-in to an area of interest. Press <b>Z</b> on your keyboard to zoom-in.",
    #     )
    #
    #     layout = hp.make_form_layout()
    #     hp.style_form_layout(layout)
    #     layout.addRow(hp.make_label(self, "Center (x)"), self.x_center)
    #     layout.addRow(hp.make_label(self, "Center (y)"), self.y_center)
    #     layout.addRow(hp.make_label(self, "Zoom"), self.zoom)
    #     layout.addRow(hp.make_h_layout(self.lock_btn, self.use_focus_btn, stretch_id=1))
    #     return layout

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        from image2image.qt._sentry import send_feedback

        self.statusbar = QStatusBar()
        self.statusbar.setSizeGripEnabled(False)

        self.screenshot_btn = hp.make_qta_btn(
            self,
            "save",
            tooltip="Save snapshot of the canvas to file. Right-click to show dialog with more options.",
            func=self.view.widget.on_save_figure,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.screenshot_btn)

        self.clipboard_btn = hp.make_qta_btn(
            self,
            "screenshot",
            tooltip="Take a snapshot of the canvas and copy it into your clipboard. Right-click to show dialog with"
            " more options.",
            func=self.view.widget.clipboard,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.clipboard_btn)
        self.scalebar_btn = hp.make_qta_btn(
            self,
            "ruler",
            tooltip="Show scalebar.",
            func=self.on_show_scalebar,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.scalebar_btn)

        self.feedback_btn = hp.make_qta_btn(
            self,
            "feedback",
            tooltip="Refresh task list ahead of schedule.",
            func=partial(send_feedback, parent=self),
            small=True,
        )
        self.statusbar.addPermanentWidget(self.feedback_btn)

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
        self.tutorial_btn = hp.make_qta_btn(
            self, "help", tooltip="Give me a quick tutorial!", func=self.on_show_tutorial, small=True
        )
        self.statusbar.addPermanentWidget(self.tutorial_btn)

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
            func=self._image_widget.on_close_dataset,
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

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update(
            {
                "transforms_model": self.transform_model,
                "viewer": self.view.viewer,
                "data_model": self.data_model,
                "wrapper": self.data_model.wrapper,
            }
        )
        return variables

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not CONFIG.confirm_close_viewer
            or QtConfirmCloseDialog(
                self, "confirm_close_viewer", self.on_save_to_project, CONFIG
            ).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and CONFIG.confirm_close_viewer
            and self.data_model.is_valid()
            and QtConfirmCloseDialog(
                self, "confirm_close_viewer", self.on_save_to_project, CONFIG
            ).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        CONFIG.save()
        READER_CONFIG.save()
        evt.accept()

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_viewer_tutorial

        show_viewer_tutorial(self)
        CONFIG.first_time_viewer = False


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="viewer", level=0)
