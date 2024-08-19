"""Mixin classes."""

from __future__ import annotations

import typing as ty
from pathlib import Path
from superqt import ensure_main_thread

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from napari.layers import Image, Points, Shapes
from napari.utils.events import Event
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtpy.QtWidgets import QDialog, QMenuBar

from image2image.qt.dialog_base import Window
from image2image.utils.utilities import calculate_zoom, init_shapes_layer

if ty.TYPE_CHECKING:
    from qtextra._napari.image.wrapper import NapariImageView
    from qtextra.widgets.qt_click_label import QtClickLabel

    from image2image.models.data import DataModel
    from image2image.qt._dialogs._select import LoadWidget


TMP_ZOOM = "Temporary (zoom)"


class SingleViewerMixin(Window):
    """Mixin class for single viewer."""

    # Mixin arguments
    WINDOW_CONSOLE_ARGS: tuple[str, ...] = ()

    image_layer: list[Image] | None = None
    shape_layer: list[Shapes] | None = None
    points_layer: list[Points] | None = None

    # Type hints
    view: NapariImageView
    _output_dir = None
    _image_widget: LoadWidget
    output_dir_label: QtClickLabel

    def on_set_output_dir(self) -> None:
        """Set output directory."""
        self.output_dir = hp.get_directory(self, "Select output directory", self.CONFIG.output_dir)

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if self.CONFIG.output_dir is None:
                return Path.cwd()
            return Path(self.CONFIG.output_dir)
        return Path(self._output_dir)

    @output_dir.setter
    def output_dir(self, directory: PathLike) -> None:
        if directory:
            self._output_dir = Path(directory)
            self.CONFIG.update(output_dir=directory)
            formatted_output_dir = f".{self._output_dir.parent}/{self._output_dir.name}"
            self.output_dir_label.setText(hp.hyper(self._output_dir, value=formatted_output_dir))
            self.output_dir_label.setToolTip(str(self._output_dir))
            logger.debug(f"Output directory set to {self._output_dir}")

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def _make_menu(self) -> None:
        """Make menu items."""
        # File menu
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(
            self,
            "Add image (.tiff, .czi, + others)...",
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

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(self._make_tools_menu(scalebar=True).menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        super()._make_statusbar()
        self._make_scalebar_statusbar()
        self._make_export_statusbar()

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not self.CONFIG.confirm_close
            or QtConfirmCloseDialog(
                self,
                "confirm_close",
                self.on_save_to_project,
                self.CONFIG,
            ).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and self.CONFIG.confirm_close
            and QtConfirmCloseDialog(
                self,
                "confirm_close",
                self.on_save_to_project,
                self.CONFIG,
            ).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return
        if self._console:
            self._console.close()
        self.CONFIG.save()
        READER_CONFIG.save()
        evt.accept()

    def _get_console_variables(self) -> dict:
        """Get variables for the console."""

        def _get_nester_arg(args: tuple[str, ...]) -> ty.Any:
            """Get nested argument."""
            obj = self
            for a in args:
                obj = getattr(obj, a)
            return obj

        variables = super()._get_console_variables()
        for arg in self.WINDOW_CONSOLE_ARGS:
            if isinstance(arg, tuple):
                variables[arg[-1]] = _get_nester_arg(arg)
            else:
                variables[arg] = getattr(self, arg)
        return variables

    @property
    def temporary_zoom_layer(self) -> Shapes:
        """Fixed points layer."""
        if TMP_ZOOM not in self.view.layers:
            layer = self.view.viewer.add_shapes(  # noqa
                None,
                # size=self.moving_point_size.value(),
                name=TMP_ZOOM,
                face_color="#00ff0000",
                edge_color="white",
            )
            visual = self.view.widget.layer_to_visual[layer]
            init_shapes_layer(layer, visual)
            layer.events.data.connect(self.on_zoom_finished)
        return self.view.layers[TMP_ZOOM]

    def on_zoom_finished(self, event: Event) -> None:
        """Zoom finished."""
        if len(self.temporary_zoom_layer.data) == 0:
            return
        last_shape = self.temporary_zoom_layer.data[-1]
        zoom, y, x = calculate_zoom(last_shape, self.view, None)
        self.view.viewer.camera.center = (0.0, y, x)
        self.view.viewer.camera.zoom = zoom
        self.view.remove_layer(TMP_ZOOM)

    def on_toggle_zoom(self) -> None:
        """Toggle zoom."""
        if TMP_ZOOM in self.view.layers:
            self.view.remove_layer(TMP_ZOOM)
        else:
            layer = self.temporary_zoom_layer
            self._move_layer(self.view, layer)
            layer.mode = "add_rectangle"

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
            need_reset = len(self.view.layers) == 0
            self.plot_image_layers(channel_list)
            if need_reset:
                self.view.viewer.reset_view()
        logger.info(f"Loaded data in {timer()}")

    def plot_image_layers(self, channel_list: list[str] | None = None) -> None:
        """Plot image layers."""
        self.image_layer, self.shape_layer, self.points_layer = self._plot_image_layers(
            self.data_model, self.view, channel_list, "view", True
        )

    def on_closing_image(self, model: DataModel, channel_names: list[str], keys: list[str]) -> None:
        """Close fixed image."""
        self._closing_model(model, channel_names, self.view, "view")

    def on_close_image(self, model: DataModel) -> None:
        """Close fixed image."""
        self._close_model(model, self.view, "view")

    def on_toggle_channel(self, name: str, state: bool) -> None:
        """Toggle channel."""
        self._toggle_channel(self.data_model, self.view, name, state, "view")

    def on_toggle_all_channels(self, state: bool, channel_names: list[str] | None = None) -> None:
        """Toggle channel."""
        self._toggle_all_channels(self.data_model, self.view, state, "view", channel_names=channel_names)

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

    def on_plot_temporary(self, res: tuple[str, int]) -> None:
        """Plot temporary layer."""
        key, channel_index = res
        with MeasureTimer() as timer:
            self._plot_temporary_layer(self.data_model, self.view, key, channel_index, True)
        logger.trace(f"Plotted temporary layer for '{key}' in {timer()}.")

    def on_remove_temporary(self, res: tuple[str, int]) -> None:
        """Remove temporary layer."""
        key, _ = res
        layer_name = self._get_reader_for_key(self.data_model, key)
        if layer_name:
            self.view.remove_layer(layer_name)

    def on_add_temporary_to_viewer(self, res: tuple[str, int]) -> None:
        """Add temporary layer to viewer."""
        key, channel_index = res
        indices = self._image_widget.channel_dlg.table.find_indices_of(
            self._image_widget.channel_dlg.TABLE_CONFIG.dataset, key
        )
        index = self._image_widget.channel_dlg.table.find_index_of_value_with_indices(
            self._image_widget.channel_dlg.TABLE_CONFIG.index, channel_index, indices
        )
        if index != -1:
            self._image_widget.channel_dlg.table.set_value(
                self._image_widget.channel_dlg.TABLE_CONFIG.check, index, True
            )
            logger.trace(f"Added image {channel_index} for '{key}' to viewer.")
