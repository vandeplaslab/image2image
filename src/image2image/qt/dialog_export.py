"""Viewer dialog."""
from __future__ import annotations

import typing as ty
from contextlib import suppress
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from loguru import logger
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHeaderView, QLineEdit, QMenuBar, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import GeneratorWorker, create_worker

from image2image import __version__
from image2image.config import CONFIG
from image2image.qt._select import LoadWidget
from image2image.qt.dialog_base import Window
from image2image.utils.utilities import log_exception_or_error

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


class ImageExportWindow(Window):
    """Image viewer dialog."""

    _console = None
    _editing = False
    _output_dir = None

    TABLE_CONFIG = (
        TableConfig().add("name", "name", "str", 0).add("path", "path", "str", 0).add("progress", "progress", "str", 0)
    )

    def __init__(self, parent: QWidget | None):
        super().__init__(parent, f"image2export: Export images in MATLAB fusion format (v{__version__})")

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
            name = self.table.item(index, self.TABLE_CONFIG.name).text()
            if not self.data_model.has_path(name):
                to_remove.append(index)
        for index in reversed(to_remove):
            self.table.removeRow(index)

    def on_populate_table(self) -> None:
        """Load data."""
        self.on_depopulate_table()
        wrapper = self.data_model.get_wrapper()
        if wrapper:
            for _path, reader in wrapper.path_reader_iter():
                name = reader.name
                index = hp.find_in_table(self.table, self.TABLE_CONFIG.name, name)
                if index is not None:
                    continue

                # get model information
                index = self.table.rowCount()

                self.table.insertRow(index)
                # add name item
                table_item = QTableWidgetItem(name)
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)  # type: ignore[attr-defined]
                table_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                self.table.setItem(index, self.TABLE_CONFIG.name, table_item)

                # add resolution item
                item = QLineEdit()
                item.setText(f"{reader.stem}.txt".replace(".ome", ""))
                item.setObjectName("table_cell")
                item.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                self.table.setCellWidget(index, self.TABLE_CONFIG.path, item)

                table_item = QTableWidgetItem("Ready!")
                table_item.setFlags(table_item.flags() & ~Qt.ItemIsEditable)  # type: ignore[attr-defined]
                self.table.setItem(index, self.TABLE_CONFIG.progress, table_item)

    def on_export(self):
        """Process data."""
        from image2image.utils.utilities import write_reader_to_txt, write_reader_to_xml

        if self.output_dir is None:
            hp.warn(self, "No output directory was selected. Please select directory where to save data.")
            return

        for row in range(self.table.rowCount()):
            name = self.table.item(row, self.TABLE_CONFIG.name).text()
            is_exported = self.table.item(row, self.TABLE_CONFIG.progress).text() == "Exported!"
            if is_exported:
                logger.info(f"Skipping {name} as it is already exported.")
            #     continue
            path = self.data_model.get_path(name)
            if path is None:
                logger.warning(f"Could not find path for {name}")
                continue
            item: QLineEdit = self.table.cellWidget(row, self.TABLE_CONFIG.path)  # type: ignore[assignment]
            output_path = self.output_dir / item.text()
            reader = self.data_model.get_reader(path)
            if reader:
                logger.info(f"Exporting {name} to {output_path}...")
                item = self.table.item(row, self.TABLE_CONFIG.progress)
                item.setText("Exporting...")
                write_reader_to_xml(reader, output_path.with_suffix(".xml"))
                create_worker(
                    write_reader_to_txt,
                    reader=reader,
                    path=output_path.with_suffix(".txt"),
                    _start_thread=True,
                    _connect={
                        "started": lambda: hp.toast(
                            self, "Exporting image...", f"Started export of {path}...", icon="success"
                        ),
                        "yielded": partial(self._on_export_yield, name),
                        "errored": partial(self._on_export_error, name),
                    },
                    _worker_class=GeneratorWorker,
                )
                hp.disable_widgets(self.export_btn, disabled=True)

    @ensure_main_thread()
    def _on_export_yield(self, *args) -> None:
        """Update CSV."""
        self.__on_export_yield(*args)

    def __on_export_yield(self, *args: ty.Any) -> None:
        with suppress(ValueError):
            name, (current, total, remaining) = args
            row = hp.find_in_table(self.table, self.TABLE_CONFIG.name, name)
            if row is not None:
                item = self.table.item(row, self.TABLE_CONFIG.progress)
                item.setText(f"{current/total:.1%} {remaining}")
                if current == total:
                    item.setText("Exported!")
                    self.on_toggle_export_btn()

    @ensure_main_thread()
    def _on_export_error(self, *args):
        """Failed exporting of the CSV."""
        name, exc = args
        row = hp.find_in_table(self.table, self.TABLE_CONFIG.name, name)
        if row is not None:
            item = self.table.item(row, self.TABLE_CONFIG.progress)
            item.setText("Export failed!")
        self.on_toggle_export_btn(force=True)
        log_exception_or_error(exc)

    def on_toggle_export_btn(self, force: bool = False) -> None:
        """Toggle export button."""
        disabled = False
        if not force:
            for row in range(self.table.rowCount()):
                text = self.table.item(row, self.TABLE_CONFIG.progress).text()
                if text not in ["Exported!", "Ready"]:
                    disabled = True
        hp.disable_widgets(self.export_btn, disabled=disabled)

    def on_set_output_dir(self):
        """Set output directory."""
        directory = hp.get_directory(self, "Select output directory", CONFIG.output_dir)
        if directory:
            self._output_dir = directory
            CONFIG.output_dir = directory
            logger.debug(f"Output directory set to {self._output_dir}")

    def _setup_ui(self):
        """Create panel."""
        self.output_dir_label = hp.make_label(self, f"Output directory: {self.output_dir}")

        self._image_widget = LoadWidget(self, None, select_channels=False)
        self._image_widget.info_text.setVisible(False)

        self.table = QTableWidget(self)
        self.table.setColumnCount(3)  # name, path, progress
        self.table.setHorizontalHeaderLabels(["name", "path (click to edit)", "progress"])
        self.table.setCornerButtonEnabled(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.TABLE_CONFIG.name, QHeaderView.Stretch)  # type: ignore[attr-defined]
        header.setSectionResizeMode(self.TABLE_CONFIG.path, QHeaderView.Stretch)  # type: ignore[attr-defined]
        header.setSectionResizeMode(
            self.TABLE_CONFIG.progress,
            QHeaderView.ResizeToContents,  # type: ignore[attr-defined]
        )

        side_layout = hp.make_form_layout()
        hp.style_form_layout(side_layout)
        side_layout.addRow(
            hp.make_btn(
                self,
                "Set output directory...",
                tooltip="Specify output directory for images...",
                func=self.on_set_output_dir,
            )
        )
        side_layout.addRow(self.output_dir_label)
        side_layout.addRow(hp.make_h_line(self))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(self.table)

        self.export_btn = hp.make_btn(self, "Export to CSV", tooltip="Export to csv file...", func=self.on_export)
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
        menu_help = self._make_help_menu()

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(menu_help.menuAction())
        self.setMenuBar(self.menubar)

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def _get_console_variables(self) -> dict:
        return {"data_model": self.data_model}

    def closeEvent(self, evt):
        """Close."""
        if self._console:
            self._console.close()
        CONFIG.save()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="export", level=0)
