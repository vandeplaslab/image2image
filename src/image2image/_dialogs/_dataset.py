"""Windows for dataset management."""
import typing as ty
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import numpy as np
from koyo.typing import PathLike
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog, QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import QFormLayout, QHeaderView, QLineEdit, QTableWidget, QTableWidgetItem
from superqt.utils import thread_worker

from image2image.config import CONFIG
from image2image.enums import ALLOWED_FORMATS
from image2image.utilities import log_exception, style_form_layout

if ty.TYPE_CHECKING:
    from image2image.models import DataModel


class CloseDatasetDialog(QtDialog):
    """Dialog where user can select which path(s) should be removed."""

    def __init__(self, parent, model: "DataModel"):
        self.model = model

        super().__init__(parent)
        self.paths = self.get_paths()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.all_check = hp.make_checkbox(
            self,
            "Check all",
            clicked=self.on_check_all,
            value=True,
        )
        # iterate over all available paths
        self.checkboxes = []
        for path in self.model.paths:
            # make checkbox for each path
            checkbox = hp.make_checkbox(self, str(path), value=True, clicked=self.on_apply)
            self.checkboxes.append(checkbox)

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(
            hp.make_label(
                self,
                "Please select which images should be <b>removed</b> from the project."
                " Fiducial markers will be unaffected.",
                alignment=Qt.AlignHCenter,  # noqa
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(self.all_check)
        layout.addRow(hp.make_h_line())
        for checkbox in self.checkboxes:
            layout.addRow(checkbox)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def on_check_all(self, state: bool):
        """Check all."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(state)

    def on_apply(self):
        """Apply."""
        self.paths = self.get_paths()
        all_checked = len(self.paths) == len(self.checkboxes)
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)  # noqa

    def get_paths(self) -> ty.List[Path]:
        """Return state."""
        paths = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                paths.append(Path(checkbox.text()))
        return paths


class SelectChannelsToLoadDialog(QtDialog):
    """Dialog to enable creation of overlays."""

    TABLE_CONFIG = (
        TableConfig()
        .add("", "check", "bool", 25, no_sort=True)
        .add("channel name", "channel_name", "str", 200)
        .add("channel name (full)", "channel_name_full", "str", 0, hidden=True)
    )

    def __init__(self, parent: "SelectImagesDialog", model: "DataModel"):
        super().__init__(parent, title="Select Channels to Load")
        self.model = model

        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.on_load()
        self.channels = self.get_channels()

    def connect_events(self, state: bool = True):
        """Connect events."""
        connect(self.table.evt_checked, self.on_select_channel, state=state)

    def on_select_channel(self, _index: int, _state: bool):
        """Toggle channel."""
        self.channels = self.get_channels()

    def get_channels(self):
        """Select all channels."""
        channels = []
        for index in self.table.get_all_checked():
            channels.append(self.table.get_value(self.TABLE_CONFIG.channel_name_full, index))
        return channels

    def on_load(self):
        """On load."""
        data = []
        wrapper = self.model.get_wrapper()
        if wrapper:
            for name in wrapper.channel_names_for_names(self.model.just_added):
                channel_name, _ = name.split(" | ")
                data.append([True, channel_name, name])
        else:
            logger.warning(f"Wrapper was not specified - {wrapper}")
        self.table.add_data(data)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.table = QtCheckableTableView(
            self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True, double_click_to_check=True
        )
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> You can quickly check/uncheck row by double-clicking on a row.<br>"
                "<b>Tip.</b> Check/uncheck a row to select which channels should be immediately loaded.<br>"
                "<b>Tip.</b> You can quickly check/uncheck all rows by clicking on the first column header.",
                alignment=Qt.AlignHCenter,  # noqa
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout


class ExtractChannelsDialog(QtDialog):
    """Dialog to extract ion images."""

    TABLE_CONFIG = TableConfig().add("", "check", "bool", 0, no_sort=True, hidden=True).add("m/z", "mz", "float", 100)

    def __init__(self, parent: "SelectImagesDialog", path_to_extract):
        super().__init__(parent, title="Extract Ion Images")
        self.setFocus()
        self.path_to_extract = path_to_extract
        self.mzs = None
        self.ppm = None

    def on_add(self):
        """Add peak."""
        value = self.mz_edit.value()
        values = self.table.get_col_data(self.TABLE_CONFIG.mz)
        if value is not None and value not in values:
            self.table.add_data([[True, value]])
        self.mzs = self.table.get_col_data(self.TABLE_CONFIG.mz)
        self.ppm = self.ppm_edit.value()

    def on_delete_row(self):
        """Delete row."""
        sel_model = self.table.selectionModel()
        if sel_model.hasSelection():
            indices = [index.row() for index in sel_model.selectedRows()]
            indices = sorted(indices, reverse=True)
            for index in indices:
                self.table.remove_row(index)
                logger.trace(f"Deleted '{index}' from m/z table")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.mz_edit = hp.make_double_spin_box(self, minimum=0, maximum=2500, step=0.1, n_decimals=3)
        self.ppm_edit = hp.make_double_spin_box(
            self, minimum=0.5, maximum=500, value=10, step=2.5, n_decimals=1, suffix=" ppm"
        )

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "m/z"),
                self.mz_edit,
                hp.make_qta_btn(self, "add", tooltip="Add peak", func=self.on_add, normal=True),
                stretch_id=1,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "ppm"),
                self.ppm_edit,
                stretch_id=1,
            )
        )
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Press <b>Delete</b> or <b>Backspace</b> to delete a peak.",
                alignment=Qt.AlignHCenter,  # noqa
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        if key == Qt.Key_Escape:  # noqa
            evt.ignore()
        elif key == Qt.Key_Backspace or key == Qt.Key_Delete:  # noqa
            self.on_delete_row()
            evt.accept()
        elif key == Qt.Key_Plus or key == Qt.Key_A:  # noqa
            self.on_add()
            evt.accept()
        else:
            super().keyPressEvent(evt)


class SelectImagesDialog(QtFramelessTool):
    """Dialog window to select images and specify some parameters."""

    HIDE_WHEN_CLOSE = True

    _editing = False
    evt_loading = Signal()
    evt_loaded = Signal(object, object)
    evt_closed = Signal(object)
    evt_resolution = Signal(Path)

    TABLE_CONFIG = (
        TableConfig()
        .add("name", "name", "str", 0)
        .add("resolution", "resolution", "str", 0)
        .add("extract", "extract", "str", 0)
        # .add("remove", "remove", "str", 0)
    )

    def __init__(self, parent, model: "DataModel", is_fixed: bool = False, n_max: int = 0):
        self.is_fixed = is_fixed
        super().__init__(parent)
        self.n_max = n_max
        self.model = model
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self.on_populate_table()

    def on_remove(self, name: str, path: str):
        """Remove path."""
        row = None
        for row in range(self.table.rowCount()):
            item = self.table.item(row, self.TABLE_CONFIG.name)
            if item.text() == name:
                break

        if row is None:
            logger.warning("Could not find row to remove.")
            return
        if not hp.confirm(self, f"Are  you sure you wish to remove <b>{name}</b>?", title="Remove?"):
            return

        # remove from table
        self.table.removeRow(row)

        # remove from model
        self.model.remove_paths(Path(path))
        self.evt_closed.emit(self.model)  # noqa

    def on_resolution(self, item, path: Path):
        """Table item changed."""
        if self._editing:
            return
        value = float(item.text())
        reader = self.model.get_reader(path)
        if reader:
            reader.resolution = value
            self.evt_resolution.emit(reader.path)
            logger.trace(f"Updated resolution of '{path.name}' to {value:.2f}.")

    def _clear_table(self):
        """Remove all rows."""
        with self._editing_table():
            while self.table.rowCount() > 0:
                self.table.removeRow(0)

    def on_populate_table(self):
        """Load data."""
        self._clear_table()

        wrapper = self.model.get_wrapper()
        if wrapper:
            with self._editing_table():
                for path, reader in wrapper.path_reader_iter():
                    name = reader.name
                    index = hp.find_in_table(self.table, self.TABLE_CONFIG.name, name)
                    if index is not None:
                        continue

                    # get model information
                    index = self.table.rowCount()
                    resolution = reader.resolution

                    self.table.insertRow(index)
                    # add name item
                    item = QTableWidgetItem(name)
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # noqa
                    item.setTextAlignment(Qt.AlignCenter)  # noqa
                    self.table.setItem(index, self.TABLE_CONFIG.name, item)
                    # add resolution item
                    item = QLineEdit(f"{resolution:.2f}")
                    item.setObjectName("table_cell")
                    item.setAlignment(Qt.AlignCenter)  # noqa
                    item.setValidator(QDoubleValidator(0, 100, 2))
                    item.editingFinished.connect(partial(self.on_resolution, path=path, item=item))
                    self.table.setCellWidget(index, self.TABLE_CONFIG.resolution, item)
                    # # add extract button
                    if reader.allow_extraction:
                        self.table.setCellWidget(
                            index,
                            self.TABLE_CONFIG.extract,
                            hp.make_qta_btn(
                                self, "add", normal=True, func=partial(self._on_extract_channels, path=path)
                            ),
                        )
                    else:
                        item = QTableWidgetItem("N/A")
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # noqa
                        item.setTextAlignment(Qt.AlignCenter)  # noqa
                        self.table.setItem(index, self.TABLE_CONFIG.extract, item)
                    # # add remove button
                    # self.table.setCellWidget(
                    #     index,
                    #     self.TABLE_CONFIG.remove,
                    #     hp.make_qta_btn(
                    #         self, "remove_all", normal=True, func=partial(self.on_remove, name=name, path=path)
                    #     ),
                    # )

    def on_select_dataset(self):
        """Load path."""
        path = hp.get_filename(
            self,
            title="Select data...",
            base_dir=CONFIG.fixed_dir if self.is_fixed else CONFIG.moving_dir,
            file_filter=ALLOWED_FORMATS,
        )
        if path:
            if self.is_fixed:
                CONFIG.fixed_dir = str(Path(path).parent)
            else:
                CONFIG.moving_dir = str(Path(path).parent)

            if self.n_max and self.model.n_paths >= self.n_max:
                verb = "image" if self.n_max == 1 else "images"
                hp.warn(
                    self,
                    f"Maximum number of images reached. You can only have {self.n_max} {verb} loaded at at time. Please"
                    f" remove other images first.",
                )
                return
            self._on_load_dataset(path)

    def _on_close_dataset(self, force: bool = False) -> bool:
        """Close dataset."""
        from image2image._dialogs import CloseDatasetDialog

        if self.model.n_paths:
            paths = None
            if not force:  # only ask user if not forced
                dlg = CloseDatasetDialog(self, self.model)
                if dlg.exec_():  # noqa
                    paths = dlg.paths
            else:
                paths = self.model.paths
            if paths:
                self.model.remove_paths(paths)
                self.evt_closed.emit(self.model)  # noqa
            self.on_populate_table()
            return True
        return False

    def _on_load_dataset(
        self,
        path_or_paths: ty.Union[PathLike, ty.List[PathLike]],
        affine: ty.Optional[ty.Dict[str, np.ndarray]] = None,
        resolution: ty.Optional[ty.Dict[str, float]] = None,
    ):
        """Load data."""
        self.evt_loading.emit()  # noqa
        if not isinstance(path_or_paths, list):
            path_or_paths = [path_or_paths]
        self.model.add_paths(path_or_paths)
        func = thread_worker(
            partial(self.model.load, affine=affine, resolution=resolution),
            start_thread=True,
            connect={"returned": self._on_loaded_dataset, "errored": self._on_failed_dataset},
        )
        func()
        logger.info(f"Started loading dataset - '{self.model.paths}'")

    def _on_loaded_dataset(self, model: "DataModel"):
        """Finished loading data."""
        # select what should be loaded
        dlg = SelectChannelsToLoadDialog(self, model)
        channel_list = []
        if dlg.exec_():  # noqa
            channel_list = dlg.channels
        logger.info(f"Selected channels: {channel_list}")
        if not channel_list:
            model.remove_paths(model.just_added)
            model, channel_list = None, None
            logger.warning("No channels selected - dataset not loaded")
        # load data into an image
        self.evt_loaded.emit(model, channel_list)  # noqa
        self.on_populate_table()

    def _on_failed_dataset(self, exception: Exception):
        """Failed to load dataset."""
        logger.error("Error occurred while loading dataset.")
        log_exception(exception)
        self.evt_loaded.emit(None, None)  # noqa

    def _on_extract_channels(self, path: PathLike):
        """Extract channels from the list."""
        if not self.model.get_extractable_paths():
            logger.warning("No paths to extract data from.")
            hp.warn(
                self,
                "No paths to extract data from. Only <b>.imzML</b>, <b>.tdf</b> and <b>.tsf</b> files support data"
                " extraction.",
            )
            return

        dlg = ExtractChannelsDialog(self, path)
        path, mzs, ppm = None, None, None
        if dlg.exec_():  # noqa
            path = dlg.path_to_extract
            mzs = dlg.mzs
            ppm = dlg.ppm

        if path and mzs and ppm:
            reader: "CoordinateReader" = self.model.get_reader(path)  # noqa
            if reader:
                self.evt_loading.emit()  # noqa
                func = thread_worker(
                    partial(reader.extract, mzs=mzs, ppm=ppm),  # noqa
                    start_thread=True,
                    connect={"returned": self._on_update_dataset, "errored": self._on_failed_update_dataset},
                )
                func()

    def _on_update_dataset(self, result: ty.Tuple[Path, ty.List[str]]):
        """Finished loading data."""
        path, channel_list = result
        # load data into an image
        self.evt_loaded.emit(self.model, channel_list)  # noqa

    @staticmethod
    def _on_failed_update_dataset(exception: Exception):
        """Failed to load dataset."""
        logger.error("Error occurred while extracting images.", exception)
        log_exception(exception)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Images")

        self.table = QTableWidget(self)
        self.table.setColumnCount(3)  # name, resolution, extract, delete
        self.table.setHorizontalHeaderLabels(["name", "pixel size (μm)", "", ""])
        self.table.setCornerButtonEnabled(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.TABLE_CONFIG.name, QHeaderView.Stretch)  # noqa
        header.setSectionResizeMode(self.TABLE_CONFIG.resolution, QHeaderView.ResizeToContents)  # noqa
        header.setSectionResizeMode(self.TABLE_CONFIG.extract, QHeaderView.ResizeToContents)  # noqa
        # header.setSectionResizeMode(self.TABLE_CONFIG.remove, QHeaderView.ResizeToContents)  # noqa

        # add delegate
        # self.table.setItemDelegateForColumn(self.TABLE_CONFIG.resolution, NumericDelegate(self.table))

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Add image...", func=self.on_select_dataset),
                hp.make_btn(self, "Remove image...", func=self._on_close_dataset),
                stretch_id=0,
            )
        )
        layout.addRow(hp.make_h_line())
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> You can edit pixel size by double-clicking on the cell.",
                alignment=Qt.AlignHCenter,  # noqa
                object_name="tip_label",
                enable_url=True,
            )
        )
        return layout

    # noinspection PyAttributeOutsideInit
    @contextmanager
    def _editing_table(self):
        self._editing = True
        yield
        self._editing = False

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        if key == Qt.Key_Escape:  # noqa
            evt.ignore()
        else:
            super().keyPressEvent(evt)
