"""Mixin classes."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from napari.layers import Image, Layer, Points, Shapes
from qtextra.dialogs.qt_close_window import QtConfirmCloseDialog
from qtextra.utils.utilities import connect
from qtpy.QtCore import QModelIndex
from qtpy.QtWidgets import QDialog, QLabel, QMenuBar, QTableWidget
from superqt import ensure_main_thread

from image2image.qt._dialog_base import Window

if ty.TYPE_CHECKING:
    from qtextra.utils.table_config import TableConfig
    from qtextra.widgets.qt_label_click import QtClickLabel
    from qtextraplot._napari.image.wrapper import NapariImageView

    from image2image.models.data import DataModel
    from image2image.qt._dialogs._select import LoadWidget


class SingleViewerMixin(Window):
    """Mixin class for single viewer."""

    # Mixin arguments
    WINDOW_CONSOLE_ARGS: tuple[str, ...] = ()

    image_layer: list[Image] | None = None
    shape_layer: list[Shapes] | None = None
    points_layer: list[Points] | None = None
    _temporary_selection: list[Layer] | None = None

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

    def on_toggle_grid(self) -> None:
        """Toggle grid on/off in the viewer."""
        self.view.viewer.grid.enabled = not self.view.viewer.grid.enabled

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
        self.image_layer, self.shape_layer, self.points_layer = self._plot_reader_layers(
            self.data_model, self.view, channel_list, "view", True
        )

    def on_closing_image(self, model: DataModel, channel_names: list[str], keys: list[str]) -> None:
        """Close fixed image."""
        self._closing_model(model, channel_names, self.view, "view")

    def on_close_image(self, model: DataModel) -> None:
        """Close fixed image."""
        self._close_model(model, self.view, "view")

    def on_toggle_channel(self, state: bool, name: str) -> None:
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

    def on_plot_temporary(self, key: str, channel_index: int) -> None:
        """Plot temporary layer."""
        self._plot_temporary_layer(self.data_model, self.view, key, channel_index, True)

    def on_remove_temporary(self, _key: str, _channel_index: int) -> None:
        """Remove temporary layer."""
        for layer in self.view.layers:
            if layer.name.startswith("temporary"):
                self.view.remove_layer(layer.name)
                logger.trace(f"Removed temporary layer '{layer.name}'.")

    def on_add_temporary_to_viewer(self, key: str, channel_index: int) -> None:
        """Add temporary layer to viewer."""
        reader = self.data_model.get_reader_for_key(key)
        layer = self.temporary_layers.get(key, None)
        if layer and reader:
            channel_name = reader.channel_names[channel_index]
            layer_name = f"{channel_name} | {key}"
            if layer_name in self.view.layers:
                logger.warning(f"Temporary layer '{key}' is already added to viewer.")
                return
            layer = self.temporary_layers.pop(key, None)
            layer.name = layer_name
            logger.trace(f"Added image {channel_index} for '{key}' to viewer.")


class NoViewerMixin(Window):
    """Mixin class for no viewer."""

    TABLE_CONFIG: TableConfig
    # UI Elements
    output_dir_label: QLabel
    _image_widget: LoadWidget
    table: QTableWidget

    # Attributes
    reader_metadata: dict[Path, dict[int, dict[str, bool | dict | list[bool | int | str]]]]
    _output_dir: Path | None = None

    # Methods
    _get_metadata: ty.Callable

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dset_dlg.evt_closed, self.on_remove_image, state=state)

    @ensure_main_thread  # type: ignore[misc]
    def on_load_image(self, model: DataModel, _channel_list: list[str]) -> None:
        """Load fixed image."""
        if model and model.n_paths:
            hp.toast(
                self, "Loaded data", f"Loaded model with {model.n_paths} paths.", icon="success", position="top_left"
            )
            self.on_populate_table()
        else:
            logger.warning(f"Failed to load data - model={model}")

    def on_remove_image(self, model: DataModel) -> None:
        """Remove image."""
        if model:
            self.on_depopulate_table()
        else:
            logger.warning(f"Failed to remove data - model={model}")

    def on_populate_table(self) -> None:
        """Load data."""
        raise NotImplementedError("Must implement method")

    def on_depopulate_table(self) -> None:
        """Remove items that are not present in the model."""
        to_remove = []
        for index in range(self.table.rowCount()):
            key = self.table.item(index, self.TABLE_CONFIG.key).text()
            if not self.data_model.has_key(key):
                to_remove.append((index, key))
        for index, key in reversed(to_remove):
            self.table.removeRow(index)

            for path, reader_metadata in self.reader_metadata.items():
                for scene_index, scene_metadata in reader_metadata.items():
                    key_ = scene_metadata.get("key", None)
                    if key_ == key:
                        self.reader_metadata[path].pop(scene_index)
                        break

    def on_update_reader_metadata(self) -> None:
        """Update reader metadata."""
        from image2image.utils.utilities import format_reader_metadata_alt

        reader_metadata = self._get_metadata(self.reader_metadata)
        for path, reader_metadata_ in reader_metadata.items():
            for scene_index, scene_metadata in reader_metadata_.items():
                key = self.reader_metadata[path][scene_index].get("key", path.name)
                row = hp.find_in_table(self.table, self.TABLE_CONFIG.key, key)
                if row is None:
                    continue
                self.table.item(row, self.TABLE_CONFIG.metadata).setText(
                    format_reader_metadata_alt(scene_index, scene_metadata)
                )

    def on_select(self, evt: QModelIndex) -> None:
        """Select channels."""
        from image2image.qt._dialogs._rename import ChannelRenameDialog

        row = evt.row()
        column = evt.column()
        name = self.table.item(row, self.TABLE_CONFIG.key).text()
        reader = self.data_model.get_reader_for_key(name)
        if column == self.TABLE_CONFIG.metadata:
            scene_metadata = self.reader_metadata[reader.path][reader.scene_index]
            dlg = ChannelRenameDialog(self, reader.scene_index, scene_metadata)
            result = dlg.exec_()
            if result == QDialog.DialogCode.Accepted:
                self.reader_metadata[reader.path][reader.scene_index] = dlg.scene_metadata
                self.on_update_reader_metadata()
                logger.trace(f"Updated metadata for {name}")
        # elif column == self.TABLE_CONFIG.resolution:
        #     new_resolution = hp.get_double(
        #         self,
        #         value=reader.resolution,
        #         label="Specify resolution (um) - don't do this unless you know what you are doing!",
        #         title="Specify image resolution",
        #         n_decimals=3,
        #         minimum=0.001,
        #         maximum=10000,
        #     )
        #     if not new_resolution or new_resolution == reader.resolution:
        #         return
        #     if hp.confirm(self, "Changing resolution may cause issues with the image data. Proceed with caution!"):
        #         reader.resolution = new_resolution
        #         self.table.item(row, self.TABLE_CONFIG.resolution).setText(f"{new_resolution:.2f}")

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if self.CONFIG.output_dir is None:
                return Path.cwd()
            return Path(self.CONFIG.output_dir)
        return Path(self._output_dir)

    def on_set_output_dir(self) -> None:
        """Set output directory."""
        directory = hp.get_directory(self, "Select output directory", self.CONFIG.output_dir)
        if directory:
            self._output_dir = directory
            self.CONFIG.update(output_dir=directory)
            self.output_dir_label.setText(hp.hyper(self.output_dir))
            logger.debug(f"Output directory set to {self._output_dir}")
