"""Viewer dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from loguru import logger
from qtextra.dialogs.qt_close_window import QtConfirmCloseDialog
from qtextra.utils.table_config import TableConfig
from qtpy.QtCore import Qt
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import QDialog, QHeaderView, QMenuBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import GeneratorWorker, create_worker

import image2image.constants as C
from image2image import __version__
from image2image.config import STATE, get_convert_config
from image2image.enums import ALLOWED_IMAGE_FORMATS_MICROSCOPY_ONLY
from image2image.qt._dialog_mixins import NoViewerMixin
from image2image.qt._dialogs._select import LoadWidget
from image2image.utils.utilities import log_exception_or_error

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


def get_metadata(
    readers_metadata: dict[Path, dict[int, dict[str, bool | dict | list[bool | int | str]]]],
) -> dict[Path, dict[int, dict[str, list[int | str]]]]:
    """Cleanup metadata."""
    metadata = {}
    for path, reader_metadata in readers_metadata.items():
        metadata_ = {}
        for scene_index, scene_metadata in reader_metadata.items():
            channel_name_to_ids = {}  # type: ignore[var-annotated]
            # iterate over channels and merge if necessary
            for index, channel_id in enumerate(scene_metadata["channel_ids"]):  # type: ignore[arg-type]
                if channel_id in scene_metadata["channel_id_to_merge"]:  # type: ignore[operator]
                    merge_channel_name = scene_metadata["channel_id_to_merge"][channel_id]  # type: ignore[index]
                    channel_name = scene_metadata["channel_names"][index]  # type: ignore[index]
                    if merge_channel_name not in channel_name_to_ids:
                        channel_name_to_ids[merge_channel_name] = []
                    channel_name_to_ids[merge_channel_name].append(channel_id)
                    if not scene_metadata["merge_and_keep"] and scene_metadata["keep"][index]:  # type: ignore[index]
                        scene_metadata[channel_name] = channel_id  # type: ignore[index]
                if scene_metadata["keep"][index]:  # type: ignore[index]
                    channel_name_to_ids[scene_metadata["channel_names"][index]] = channel_id  # type: ignore[index]

            # cleanup by removing any duplicates and sorting indices
            for channel_name in channel_name_to_ids:
                if isinstance(channel_name_to_ids[channel_name], list):
                    channel_name_to_ids[channel_name] = sorted(set(channel_name_to_ids[channel_name]))
            channel_names = list(channel_name_to_ids.keys())
            channel_ids = list(channel_name_to_ids.values())
            metadata_[scene_index] = {"channel_ids": channel_ids, "channel_names": channel_names}
        metadata[path] = metadata_
    return metadata


class ImageConvertWindow(NoViewerMixin):
    """Image viewer dialog."""

    APP_NAME = "convert"

    worker: GeneratorWorker | None = None

    TABLE_CONFIG = (
        TableConfig()
        .add("key", "key", "str", 0)
        .add("pixel size (um)", "resolution", "str", 0)
        .add("scenes & channels", "metadata", "str", 0)
        .add("progress", "progress", "str", 0)
    )

    _get_metadata = staticmethod(get_metadata)

    def __init__(self, parent: QWidget | None, run_check_version: bool = True, **kwargs: ty.Any):
        self.CONFIG = get_convert_config()
        super().__init__(
            parent,
            f"image2image: Convert image to OME-TIFF (v{__version__})",
            run_check_version=run_check_version,
        )
        if self.CONFIG.first_time:
            hp.call_later(self, self.on_show_tutorial, 10_000)
        self.reader_metadata: dict[Path, dict[int, dict[str, bool | dict | list[bool | int | str]]]] = {}
        self._setup_config()

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.auto_pyramid = False
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = True
        READER_CONFIG.split_rgb = True
        READER_CONFIG.only_last_pyramid = False

    def on_depopulate_table(self) -> None:
        """Remove items that are not present in the model."""
        to_remove = []
        for index in range(self.table.rowCount()):
            key = self.table.item(index, self.TABLE_CONFIG.key).text()
            if not self.data_model.has_key(key):
                to_remove.append(index)
        for index in reversed(to_remove):
            self.table.removeRow(index)

    def on_cleanup_reader_metadata(self, model: DataModel) -> None:
        """Cleanup metadata."""
        if not model:
            return
        paths = model.paths
        to_remove = []
        for path in self.reader_metadata:
            if path not in paths:
                to_remove.append(path)
        for path in to_remove:
            del self.reader_metadata[path]
            logger.trace(f"Removed metadata for {path}")

    def on_populate_table(self) -> None:
        """Load data."""
        self.on_depopulate_table()

        wrapper = self.data_model.wrapper
        if wrapper:
            for reader in wrapper.reader_iter():
                index = hp.find_in_table(self.table, self.TABLE_CONFIG.key, reader.key)
                if index is not None:
                    continue

                # get model information
                index = self.table.rowCount()

                self.table.insertRow(index)
                # add name item
                table_item = QTableWidgetItem(reader.key)
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.key, table_item)

                table_item = QTableWidgetItem(f"{reader.resolution:.2f}")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.resolution, table_item)

                table_item = QTableWidgetItem("")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.metadata, table_item)

                table_item = QTableWidgetItem("Ready!")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
                self.table.setItem(index, self.TABLE_CONFIG.progress, table_item)
                path_metadata = self.reader_metadata.get(reader.path, {})
                reader_metadata = path_metadata.get(reader.scene_index, {})
                if reader_metadata:
                    self.reader_metadata[reader.path][reader.scene_index] = reader_metadata
                else:
                    if reader.path not in self.reader_metadata:
                        self.reader_metadata[reader.path] = {}
                    self.reader_metadata[reader.path][reader.scene_index] = {
                        "key": reader.key,
                        "keep": [True] * reader.n_channels,
                        "channel_ids": reader.channel_ids,
                        "channel_names": reader.channel_names,
                        "channel_id_to_merge": {},  # dict of int: str
                        "merge_and_keep": False,  # bool
                    }
        self.on_update_reader_metadata()

    def on_open_convert(self):
        """Process data."""
        from image2image_io.writers import images_to_ome_tiff

        if self.output_dir is None:
            hp.warn_pretty(self, "No output directory was selected. Please select directory where to save data.")
            return

        paths, scenes = set(), {}
        for row in range(self.table.rowCount()):
            key = self.table.item(row, self.TABLE_CONFIG.key).text()
            reader = self.data_model.get_reader_for_key(key)
            if reader is None:
                logger.warning(f"Could not find path for {key}")
                continue
            if reader:
                paths.add(reader.path)
                if reader.path not in scenes:
                    scenes[reader.path] = set()
                scenes[reader.path].add(reader.scene_index)

        output_dir = self.output_dir

        if paths:
            if STATE.is_mac_arm_pyinstaller:
                logger.warning("Conversion process is running in the UI thread, meaning that the app will freeze!")
                for args in images_to_ome_tiff(
                    paths=paths,
                    output_dir=output_dir,
                    as_uint8=self.CONFIG.as_uint8,
                    tile_size=self.CONFIG.tile_size,
                    metadata=get_metadata(self.reader_metadata),
                    path_to_scene=scenes,
                    overwrite=self.CONFIG.overwrite,
                ):
                    self.__on_export_yield(args)
            else:
                self.worker = create_worker(
                    images_to_ome_tiff,
                    paths=paths,
                    output_dir=output_dir,
                    as_uint8=self.CONFIG.as_uint8,
                    tile_size=self.CONFIG.tile_size,
                    metadata=get_metadata(self.reader_metadata),
                    path_to_scene=scenes,
                    overwrite=self.CONFIG.overwrite,
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

    def on_cancel(self) -> None:
        """Cancel processing."""
        if self.worker:
            self.worker.quit()
            logger.trace("Requested aborting of the export process.")

    @ensure_main_thread()  # type: ignore[misc]
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

    @ensure_main_thread()  # type: ignore[misc]
    def _on_export_yield(self, args: tuple[str, int, int, int, int, bool]) -> None:
        """Update CSV."""
        self.__on_export_yield(args)

    def __on_export_yield(self, args: tuple[str, int, int, int, int, bool]) -> None:
        # with suppress(ValueError):
        key, current_scene, total_scene, current, total_in_files, is_exported = args
        reader = self.data_model.get_reader_for_key(key)
        self.export_btn.setRange(0, total_in_files)
        self.export_btn.setValue(current)
        row = hp.find_in_table(self.table, self.TABLE_CONFIG.key, key)
        if row is not None and reader is not None:
            item = self.table.item(row, self.TABLE_CONFIG.progress)
            item.setText("Exported!" if is_exported else "Exporting...")
            if current_scene == total_scene:
                self.on_toggle_export_btn()

    @ensure_main_thread()  # type: ignore[misc]
    def _on_export_error(self, exc: Exception) -> None:
        """Failed exporting of the CSV."""
        self.on_toggle_export_btn(force=True)
        log_exception_or_error(exc)
        self.worker = None

    @ensure_main_thread()  # type: ignore[misc]
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

    def on_update_config(self, _: ty.Any = None) -> None:
        """Update configuration file."""
        self.CONFIG.update(
            as_uint8=self.as_uint8.isChecked(),
            overwrite=self.overwrite.isChecked(),
            tile_size=int(self.tile_size.currentText()),
        )
        READER_CONFIG.split_czi = self.split_czi.isChecked()

    def _setup_ui(self) -> None:
        """Create panel."""
        self.output_dir_label = hp.make_label(self, hp.hyper(self.output_dir), enable_url=True)

        self._image_widget = LoadWidget(
            self,
            None,
            self.CONFIG,
            allow_channels=False,
            available_formats=ALLOWED_IMAGE_FORMATS_MICROSCOPY_ONLY,
            show_split_czi=False,
        )
        self._image_widget.dset_dlg.evt_closed.connect(self.on_cleanup_reader_metadata)

        columns = self.TABLE_CONFIG.to_columns()
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(columns))  # name, scenes, progress, key
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setCornerButtonEnabled(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideLeft)
        self.table.setWordWrap(True)
        self.table.doubleClicked.connect(self.on_select)

        horizontal_header = self.table.horizontalHeader()
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.key, QHeaderView.ResizeMode.Stretch)
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
            value=f"{self.CONFIG.tile_size}",
            func=self.on_update_config,
        )
        self.split_czi = hp.make_checkbox(
            self,
            "",
            tooltip="Split CZI into multiple scenes when exporting"
            "<br><b>option is disabled - please contact us if you would like to make this option available again</b>.",
            checked=True,
            value=READER_CONFIG.split_czi,
            func=self.on_update_config,
        )
        hp.disable_widgets(self.split_czi, disabled=True)
        self.as_uint8 = hp.make_checkbox(
            self, "", tooltip=C.UINT8_TIP, checked=True, value=self.CONFIG.as_uint8, func=self.on_update_config
        )
        self.overwrite = hp.make_checkbox(
            self,
            "",
            tooltip="Overwrite existing files without having to delete them (e.g. if adding merged channels).",
            checked=True,
            value=self.CONFIG.overwrite,
            func=self.on_update_config,
        )
        self.export_btn = hp.make_active_progress_btn(
            self,
            "Convert to OME-TIFF",
            tooltip="Convert to OME-TIFF...",
            func=self.on_open_convert,
            cancel_func=self.on_cancel,
        )

        side_layout = hp.make_form_layout()
        side_layout.addRow(
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
        if STATE.is_mac_arm_pyinstaller:
            side_layout.addRow(
                hp.make_h_layout(
                    hp.make_warning_label(self, "", average=True),
                    hp.make_label(
                        self,
                        "Warning: On Apple Silicon, the conversion process happens in the UI thread, meaning that"
                        " the app freezes!",
                        warp=True,
                        object_name="warning_label",
                    ),
                    stretch_id=(1,),
                )
            )
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(self.table)  # , stretch=True)
        side_layout.addRow(
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
        side_layout.addRow(hp.make_h_line(self))
        side_layout.addRow(self.directory_btn)
        side_layout.addRow("Output directory", self.output_dir_label)
        side_layout.addRow("Split CZI", hp.make_h_layout(self.split_czi, stretch_after=True))
        side_layout.addRow("Tile size", hp.make_h_layout(self.tile_size, stretch_after=True))
        side_layout.addRow(
            "Reduce file size",
            hp.make_h_layout(
                self.as_uint8,
                hp.make_warning_label(
                    self,
                    "While this option reduces the amount of space an image takes on your disk, it can lead to data"
                    " loss and should be used with caution.",
                    normal=True,
                ),
                spacing=2,
                stretch_after=True,
            ),
        )
        side_layout.addRow("Overwrite", hp.make_h_layout(self.overwrite, stretch_after=True))
        side_layout.addRow(self.export_btn)

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

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(self._make_tools_menu().menuAction())
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

        if show_convert_tutorial(self):
            self.CONFIG.update(first_time=False)

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not self.CONFIG.confirm_close
            or QtConfirmCloseDialog(self, "confirm_close", config=self.CONFIG).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and self.CONFIG.confirm_close
            and self.data_model.is_valid()
            and QtConfirmCloseDialog(self, "confirm_close", config=self.CONFIG).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        self.CONFIG.save()
        evt.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        """Drop event."""
        self._setup_config()
        super().dropEvent(event)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="convert", level=0)
