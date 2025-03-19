"""Viewer dialog."""

from __future__ import annotations

import typing as ty
from contextlib import suppress
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from loguru import logger
from qtextra.dialogs.qt_close_window import QtConfirmCloseDialog
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import QDialog, QHeaderView, QMenuBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import GeneratorWorker, create_worker

from image2image import __version__
from image2image.config import get_fusion_config
from image2image.qt._dialog_base import Window
from image2image.qt._dialogs._select import LoadWidget
from image2image.utils.utilities import log_exception_or_error

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


class ImageFusionWindow(Window):
    """Image viewer dialog."""

    APP_NAME = "fusion"

    _editing = False
    _output_dir = None
    worker: GeneratorWorker | None = None

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("name", "name", "str", 0)
        .add("channels", "metadata", "str", 0)
        .add("progress", "progress", "str", 0)
    )

    def __init__(self, parent: QWidget | None, run_check_version: bool = True, **kwargs: ty.Any):
        self.CONFIG = get_fusion_config()
        super().__init__(
            parent,
            f"image2image: Export images for MATLAB fusion (v{__version__})",
            run_check_version=run_check_version,
        )
        if self.CONFIG.first_time:
            hp.call_later(self, self.on_show_tutorial, 10_000)
        self.reader_metadata: dict[Path, dict[int, dict[str, list[bool | int | str]]]] = {}

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.auto_pyramid = False
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False
        READER_CONFIG.split_roi = True
        READER_CONFIG.split_rgb = False
        logger.trace("Setup reader config for image2fusion.")

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dset_dlg.evt_closed, self.on_remove_image, state=state)

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
            if self.CONFIG.output_dir is None:
                return Path.cwd()
            return Path(self.CONFIG.output_dir)
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
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)  # type: ignore[attr-defined]
                table_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                self.table.setItem(index, self.TABLE_CONFIG.name, table_item)

                table_item = QTableWidgetItem("")
                table_item.setFlags(table_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(index, self.TABLE_CONFIG.metadata, table_item)

                table_item = QTableWidgetItem("Ready!")
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)  # type: ignore[attr-defined]
                self.table.setItem(index, self.TABLE_CONFIG.progress, table_item)
                reader_metadata = self.reader_metadata.get(reader.path, {})
                if reader_metadata:
                    self.reader_metadata[reader.path] = reader_metadata
                else:
                    self.reader_metadata[reader.path] = {}
                    for scene_index in range(reader.n_scenes):
                        self.reader_metadata[reader.path][scene_index] = {
                            "keep": [True] * reader.n_channels,
                            "channel_ids": reader.channel_ids,
                            "channel_names": reader.channel_names,
                        }
        self.on_update_reader_metadata()

    def on_open_fusion(self):
        """Process data."""
        from image2image_io.writers import images_to_fusion

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
        if paths:
            self.worker = create_worker(
                images_to_fusion,
                paths=paths,
                output_dir=output_dir,
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

    @ensure_main_thread()
    def _on_export_yield(self, args: tuple[str, int, int, str]) -> None:
        """Update CSV."""
        self.__on_export_yield(args)

    def __on_export_yield(self, args: tuple[str, int, int, str]) -> None:
        with suppress(ValueError):
            key, current, total, remaining = args
            self.export_btn.setRange(0, total)
            self.export_btn.setValue(current)
            row = hp.find_in_table(self.table, self.TABLE_CONFIG.name, key)
            if row is not None:
                item = self.table.item(row, self.TABLE_CONFIG.progress)
                item.setText(f"{current / total:.1%} {remaining}")
                if current == total:
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

    def on_toggle_export_btn(self, force: bool = False) -> None:
        """Toggle export button."""
        disabled = False
        if not force:
            for row in range(self.table.rowCount()):
                text = self.table.item(row, self.TABLE_CONFIG.progress).text()
                if text not in ["Exported!", "Ready"]:
                    disabled = True
        hp.disable_widgets(self.export_btn.active_btn, disabled=disabled)
        self.export_btn.active = disabled

    def on_set_output_dir(self):
        """Set output directory."""
        directory = hp.get_directory(self, "Select output directory", self.CONFIG.output_dir)
        if directory:
            self._output_dir = directory
            self.CONFIG.update(output_dir=directory)
            self.output_dir_label.setText(f"<b>Output directory</b>: {hp.hyper(self.output_dir)}")
            logger.debug(f"Output directory set to {self._output_dir}")

    def on_select(self, row: int) -> None:
        """Select channels."""
        from image2image.qt._dialogs._rename import ChannelRenameDialog

        name = self.table.item(row, self.TABLE_CONFIG.name).text()

        reader = self.data_model.get_reader_for_key(name)
        reader_metadata = self.reader_metadata[reader.path]
        dlg = ChannelRenameDialog(self, reader_metadata)
        if dlg.exec_() == QDialog.DialogCode.Accepted:
            self.reader_metadata[reader.path] = dlg.reader_metadata
            self.on_update_reader_metadata()

    def on_update_reader_metadata(self):
        """Update reader metadata."""
        for path, reader_metadata in self.reader_metadata.items():
            key = path.name
            row = hp.find_in_table(self.table, self.TABLE_CONFIG.name, key)
            if row is None:
                continue
            metadata = []
            for scene_index, scene_metadata in reader_metadata.items():
                channel_ids = [x for x, keep in zip(scene_metadata["channel_ids"], scene_metadata["keep"]) if keep]
                metadata.append(f"{scene_index}: {channel_ids}")
            self.table.item(row, self.TABLE_CONFIG.metadata).setText("\n".join(metadata))

    def _setup_ui(self):
        """Create panel."""
        self._image_widget = LoadWidget(self, None, self.CONFIG, allow_channels=False)

        columns = self.TABLE_CONFIG.to_columns()
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(columns))  # name, progress, key
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setCornerButtonEnabled(False)
        self.table.setTextElideMode(Qt.TextElideMode.ElideLeft)
        self.table.setWordWrap(True)
        # self.table.doubleClicked.connect(lambda index: self.on_select(index.row()))

        horizontal_header = self.table.horizontalHeader()
        horizontal_header.setSectionResizeMode(self.TABLE_CONFIG.name, QHeaderView.ResizeMode.Stretch)
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
        self.output_dir_label = hp.make_label(
            self, f"<b>Output directory</b>: {hp.hyper(self.output_dir)}", enable_url=True
        )
        self.export_btn = hp.make_active_progress_btn(
            self, "Export to CSV", tooltip="Export to csv file...", func=self.on_open_fusion, cancel_func=self.on_cancel
        )

        side_layout = hp.make_v_layout()
        side_layout.addWidget(
            hp.make_label(
                self,
                "This app will <b>convert</b> many image types to <b>comma-delimited</b> file compatible with"
                " image fusion.",
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
        side_layout.addWidget(hp.make_h_line(self))
        side_layout.addWidget(self.directory_btn)
        side_layout.addWidget(self.output_dir_label)
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
            "Add image (.tiff, .png, .jpg, .imzML, .tdf, .tsf, + others)...",
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
        variables.update({"data_model": self.data_model, "wrapper": self.data_model.wrapper})
        return variables

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
            and QtConfirmCloseDialog(self, "confirm_close", config=self.CONFIG).exec_() != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        self.CONFIG.save()
        evt.accept()

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_fusion_tutorial

        if show_fusion_tutorial(self):
            self.CONFIG.update(first_time=False)

    def dropEvent(self, event: QDropEvent) -> None:
        """Drop event."""
        self._setup_config()
        super().dropEvent(event)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="fusion", level=0)
