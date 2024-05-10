"""Viewer dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_io.readers import CziSceneImageReader, get_simple_reader
from loguru import logger
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtpy.QtCore import Qt
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import QDialog, QHeaderView, QMenuBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import GeneratorWorker, create_worker

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_IMAGE_FORMATS_MICROSCOPY_ONLY
from image2image.qt._dialogs._select import LoadWidget
from image2image.qt.dialog_base import Window
from image2image.utils.utilities import log_exception_or_error

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


def get_metadata(
    readers_metadata: dict[Path, dict[int, dict[str, list[bool | int | str]]]],
) -> dict[Path, dict[int, dict[str, list[int | str]]]]:
    """Cleanup metadata."""
    metadata = {}
    for path, reader_metadata in readers_metadata.items():
        metadata_ = {}
        for scene_index, scene_metadata in reader_metadata.items():
            channel_name_to_ids = {}
            # iterate over channels and merge if necessary
            for index, channel_id in enumerate(scene_metadata["channel_ids"]):
                if channel_id in scene_metadata["channel_id_to_merge"]:
                    merge_channel_name = scene_metadata["channel_id_to_merge"][channel_id]
                    channel_name = scene_metadata["channel_names"][index]
                    if merge_channel_name not in channel_name_to_ids:
                        channel_name_to_ids[merge_channel_name] = []
                    channel_name_to_ids[merge_channel_name].append(channel_id)
                    if not scene_metadata["merge_and_keep"] and scene_metadata["keep"][index]:
                        scene_metadata[channel_name] = channel_id
                if scene_metadata["keep"][index]:
                    channel_name_to_ids[scene_metadata["channel_names"][index]] = channel_id

            # cleanup by removing any duplicates and sorting indices
            for channel_name in channel_name_to_ids:
                if isinstance(channel_name_to_ids[channel_name], list):
                    channel_name_to_ids[channel_name] = sorted(set(channel_name_to_ids[channel_name]))
            channel_names = list(channel_name_to_ids.keys())
            channel_ids = list(channel_name_to_ids.values())
            metadata_[scene_index] = {"channel_ids": channel_ids, "channel_names": channel_names}
        metadata[path] = metadata_
    return metadata


class ImageConvertWindow(Window):
    """Image viewer dialog."""

    _console = None
    _editing = False
    _output_dir = None
    worker: GeneratorWorker | None = None

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("name", "name", "str", 0)
        .add("pixel size (um)", "resolution", "str", 0)
        .add("scenes & channels", "metadata", "str", 0)
        .add("progress", "progress", "str", 0)
    )

    def __init__(self, parent: QWidget | None, run_check_version: bool = True):
        super().__init__(
            parent,
            f"image2image: Convert image to OME-TIFF (v{__version__})",
            run_check_version=run_check_version,
        )
        if CONFIG.first_time_convert:
            hp.call_later(self, self.on_show_tutorial, 10_000)
        self.reader_metadata: dict[Path, dict[int, dict[str, list[bool | int | str]]]] = {}
        self._setup_config()

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.auto_pyramid = False
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False
        READER_CONFIG.split_rgb = True
        READER_CONFIG.only_last_pyramid = False
        logger.trace("Setup reader config for image2tiff.")

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closed, self.on_remove_image, state=state)

    @ensure_main_thread
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

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if CONFIG.output_dir is None:
                return Path.cwd()
            return Path(CONFIG.output_dir)
        return Path(self._output_dir)

    def on_depopulate_table(self) -> None:
        """Remove items that are not present in the model."""
        to_remove = []
        for index in range(self.table.rowCount()):
            key = self.table.item(index, self.TABLE_CONFIG.name).text()
            if not self.data_model.has_key(key):
                to_remove.append(index)
        for index in reversed(to_remove):
            self.table.removeRow(index)

    def on_populate_table(self) -> None:
        """Load data."""
        self.on_depopulate_table()
        wrapper = self.data_model.wrapper
        if wrapper:
            for reader in wrapper.reader_iter():
                index = hp.find_in_table(self.table, self.TABLE_CONFIG.name, reader.key)
                if index is not None:
                    continue

                # get model information
                index = self.table.rowCount()

                self.table.insertRow(index)
                # add name item
                table_item = QTableWidgetItem(reader.key)
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                self.table.setItem(index, self.TABLE_CONFIG.name, table_item)

                table_item = QTableWidgetItem(f"{reader.resolution:.2f}")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(index, self.TABLE_CONFIG.resolution, table_item)

                table_item = QTableWidgetItem("")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(index, self.TABLE_CONFIG.metadata, table_item)

                table_item = QTableWidgetItem("Ready!")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(index, self.TABLE_CONFIG.progress, table_item)
                reader_metadata = self.reader_metadata.get(reader.path, {})
                if reader_metadata:
                    self.reader_metadata[reader.path] = reader_metadata
                else:
                    self.reader_metadata[reader.path] = {}
                    for scene_index in range(reader.n_scenes):
                        reader_ = (
                            CziSceneImageReader(reader.path, scene_index=scene_index)
                            if reader.path.suffix == ".czi"
                            else get_simple_reader(reader.path, auto_pyramid=False, init_pyramid=False)
                        )
                        self.reader_metadata[reader.path][scene_index] = {
                            "keep": [True] * reader_.n_channels,
                            "channel_ids": reader_.channel_ids,
                            "channel_names": reader_.channel_names,
                            "channel_id_to_merge": {},  # dict of int: str
                            "merge_and_keep": False,  # bool
                        }
        self.on_update_reader_metadata()

    def on_select(self, evt) -> None:
        """Select channels."""
        from image2image.qt._dialogs._rename import ChannelRenameDialog

        row = evt.row()
        column = evt.column()
        name = self.table.item(row, self.TABLE_CONFIG.name).text()
        reader = self.data_model.get_reader_for_key(name)
        if column == self.TABLE_CONFIG.metadata:
            reader_metadata = self.reader_metadata[reader.path]
            dlg = ChannelRenameDialog(self, reader_metadata)
            result = dlg.exec_()
            if result == QDialog.DialogCode.Accepted:
                self.reader_metadata[reader.path] = dlg.reader_metadata
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

    def on_update_reader_metadata(self):
        """Update reader metadata."""
        reader_metadata = get_metadata(self.reader_metadata)
        for path, reader_metadata_ in reader_metadata.items():
            key = path.name
            row = hp.find_in_table(self.table, self.TABLE_CONFIG.name, key)
            if row is None:
                continue
            metadata = []
            has_scenes = len(reader_metadata_) > 1
            for index, (scene_index, scene_metadata) in enumerate(reader_metadata_.items()):
                channel_ids = scene_metadata["channel_ids"]
                channel_names = scene_metadata["channel_names"]
                if has_scenes and channel_ids:
                    metadata.append(f"scene {scene_index}")
                for channel_index, channel_name in zip(channel_ids, channel_names):
                    metadata.append(f"- {channel_name}: {channel_index}")
            self.table.item(row, self.TABLE_CONFIG.metadata).setText("\n".join(metadata))

    def on_convert(self):
        """Process data."""
        from image2image_io.writers import images_to_ome_tiff

        if self.output_dir is None:
            hp.warn_pretty(self, "No output directory was selected. Please select directory where to save data.")
            return

        paths = []
        for row in range(self.table.rowCount()):
            key = self.table.item(row, self.TABLE_CONFIG.name).text()
            reader = self.data_model.get_reader_for_key(key)
            if reader is None:
                logger.warning(f"Could not find path for {key}")
                continue
            if reader:
                paths.append(reader.path)

        output_dir = self.output_dir
        CONFIG.as_uint8 = self.as_uint8.isChecked()
        CONFIG.overwrite = self.overwrite.isChecked()
        CONFIG.tile_size = int(self.tile_size.currentText())
        if paths:
            self.worker = create_worker(
                images_to_ome_tiff,
                paths=paths,
                output_dir=output_dir,
                as_uint8=CONFIG.as_uint8,
                tile_size=CONFIG.tile_size,
                metadata=get_metadata(self.reader_metadata),
                overwrite=CONFIG.overwrite,
                _start_thread=True,
                _connect={
                    "aborted": self._on_export_aborted,
                    "yielded": self._on_export_yield,
                    "finished": self._on_export_finished,
                    "errored": self._on_export_error,
                },
                _worker_class=GeneratorWorker,
            )
            hp.disable_widgets(self.export_btn.active_btn, disabled=True)
            self.export_btn.active = True

    def on_cancel(self):
        """Cancel processing."""
        if self.worker:
            self.worker.quit()
            logger.trace("Requested aborting of the export process.")

    @ensure_main_thread()
    def _on_export_aborted(self) -> None:
        """Update CSV."""
        self.worker = None
        self.on_toggle_export_btn(force=True)
        for row in range(self.table.rowCount()):
            value = self.table.item(row, self.TABLE_CONFIG.progress).text()
            if value not in ["Exported!", "Ready!"]:
                item = self.table.item(row, self.TABLE_CONFIG.progress)
                item.setText("Aborted!")
                logger.trace("Aborted export process.")

    @ensure_main_thread()
    def _on_export_yield(self, args: tuple[str, int, int, int, int]) -> None:
        """Update CSV."""
        self.__on_export_yield(args)

    def __on_export_yield(self, args: tuple[str, int, int, int, int]) -> None:
        # with suppress(ValueError):
        key, current_scene, total_scene, current, total_in_files = args
        self.export_btn.setRange(0, total_in_files)
        self.export_btn.setValue(current)
        row = hp.find_in_table(self.table, self.TABLE_CONFIG.name, key)
        if row is not None:
            item = self.table.item(row, self.TABLE_CONFIG.progress)
            item.setText(f"{current_scene}/{total_scene} scenes exported...")
            if current_scene == total_scene:
                item.setText("Exported!")
                self.on_toggle_export_btn()

    @ensure_main_thread()
    def _on_export_error(self, exc: Exception) -> None:
        """Failed exporting of the CSV."""
        self.on_toggle_export_btn(force=True)
        log_exception_or_error(exc)
        self.worker = None

    @ensure_main_thread()
    def _on_export_finished(self):
        """Failed exporting of the CSV."""
        self.on_toggle_export_btn(force=True)
        self.worker = None

    def on_toggle_export_btn(self, force: bool = False, disabled: bool = False) -> None:
        """Toggle export button."""
        if not force:
            for row in range(self.table.rowCount()):
                text = self.table.item(row, self.TABLE_CONFIG.progress).text()
                if text not in ["Exported!", "Ready"]:
                    disabled = True
        hp.disable_widgets(self.export_btn.active_btn, disabled=disabled)
        self.export_btn.active = disabled

    def on_set_output_dir(self):
        """Set output directory."""
        directory = hp.get_directory(self, "Select output directory", CONFIG.output_dir)
        if directory:
            self._output_dir = directory
            CONFIG.output_dir = directory
            self.output_dir_label.setText(f"Output directory: {hp.hyper(self.output_dir)}")
            logger.debug(f"Output directory set to {self._output_dir}")

    def _setup_ui(self):
        """Create panel."""
        self.output_dir_label = hp.make_label(self, f"Output directory: {hp.hyper(self.output_dir)}", enable_url=True)

        self._image_widget = LoadWidget(
            self, None, select_channels=False, available_formats=ALLOWED_IMAGE_FORMATS_MICROSCOPY_ONLY
        )
        self._image_widget.info_text.setVisible(False)

        columns = self.TABLE_CONFIG.to_columns()
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(columns))  # name, scenes, progress, key
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setCornerButtonEnabled(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideLeft)
        self.table.setWordWrap(True)
        self.table.doubleClicked.connect(self.on_select)

        horizontal_header = self.table.horizontalHeader()
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.name, QHeaderView.ResizeMode.Stretch)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.resolution, QHeaderView.ResizeMode.ResizeToContents)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.metadata, QHeaderView.ResizeMode.ResizeToContents)
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.progress, QHeaderView.ResizeMode.ResizeToContents)
        vertical_header = self.table.verticalHeader()
        vertical_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        self.directory_btn = hp.make_btn(
            self,
            "Set output directory...",
            tooltip="Specify output directory for images...",
            func=self.on_set_output_dir,
        )

        self.tile_size = hp.make_combobox(
            self,
            ["256", "512", "1024", "2048", "4096"],
            tooltip="Specify size of the tile. Default is 512",
            default="512",
            value=f"{CONFIG.tile_size}",
        )
        self.as_uint8 = hp.make_checkbox(
            self,
            "Reduce data size (uint8 - dynamic range 0-255)",
            tooltip="Convert to uint8 to reduce file size with minimal data loss.",
            checked=True,
            value=CONFIG.as_uint8,
        )
        self.overwrite = hp.make_checkbox(
            self,
            "Overwrite existing files",
            tooltip="Overwrite existing files without having to delete them (e.g. if adding merged channels).",
            checked=True,
            value=CONFIG.overwrite,
        )
        self.export_btn = hp.make_active_progress_btn(
            self,
            "Convert to OME-TIFF",
            tooltip="Convert to OME-TIFF...",
            func=self.on_convert,
            cancel_func=self.on_cancel,
        )

        side_layout = hp.make_v_layout()
        side_layout.addWidget(
            hp.make_label(
                self,
                "This app will <b>convert</b> many image types to <b>pyramidal OME-TIFF</b>.<br>"
                "You can edit select which channels should be retained, edit their names, and merge channels together.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="large_text",
                enable_url=True,
                wrap=True,
            )
        )
        side_layout.addWidget(hp.make_h_line())
        side_layout.addWidget(self._image_widget)
        side_layout.addWidget(hp.make_h_line())
        side_layout.addWidget(self.table, stretch=True)
        side_layout.addWidget(
            hp.make_label(
                self,
                "<b>Tip.</b> Double-click on the <b>scenes & channels</b> field to select/deselect"
                " scenes/channels, rename channel names or merge multiple channels together.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
                wrap=True,
            )
        )
        side_layout.addWidget(hp.make_h_line(self))
        side_layout.addWidget(self.directory_btn)
        side_layout.addWidget(self.output_dir_label)
        side_layout.addLayout(
            hp.make_h_layout(
                hp.make_label(self, "Tile size:"),
                self.tile_size,
                hp.make_v_line(),
                self.as_uint8,
                hp.make_v_line(),
                self.overwrite,
                stretch_after=True,
            )
        )
        side_layout.addWidget(self.export_btn)

        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout(widget)
        main_layout.addLayout(side_layout)

        # extra settings
        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def _make_menu(self) -> None:
        """Make menu items."""
        # File menu
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(
            self,
            "Add image (.czi, .ome.tiff, .tiff, .scn, and others)...",
            "Ctrl+I",
            menu=menu_file,
            func=self._image_widget.on_select_dataset,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(self, "Show Logger...", "Ctrl+L", menu=menu_tools, func=self.on_show_logger)
        hp.make_menu_item(self, "Show IPython console...", "Ctrl+T", menu=menu_tools, func=self.on_show_console)

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update(
            {
                "data_model": self.data_model,
                "wrapper": self.data_model.wrapper,
            }
        )
        return variables

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_convert_tutorial

        show_convert_tutorial(self)
        CONFIG.first_time_convert = False

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not CONFIG.confirm_close_convert
            or QtConfirmCloseDialog(self, "confirm_close_convert", config=CONFIG).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and CONFIG.confirm_close_convert
            and self.data_model.is_valid()
            and QtConfirmCloseDialog(self, "confirm_close_convert", config=CONFIG).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        CONFIG.save()
        evt.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        """Drop event."""
        self._setup_config()
        super().dropEvent(event)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="convert", level=0)
