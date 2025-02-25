"""Windows for dataset management."""

from __future__ import annotations

import typing as ty
from collections import Counter
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path

import numpy as np
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.path import open_directory_alt
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import pluralize
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog, QtFramelessTool
from qtextra.widgets.qt_table_view_check import MultiColumnSingleValueProxyModel, QtCheckableTableView
from qtpy.QtCore import QModelIndex, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QDoubleValidator, QDropEvent
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QLineEdit,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from image2image.config import STATE, SingleAppConfig, get_register_config
from image2image.enums import ALLOWED_IMAGE_FORMATS, ALLOWED_IMAGE_FORMATS_WITH_GEOJSON
from image2image.exceptions import MultiSceneCziError, UnsupportedFileFormatError
from image2image.models.transform import TransformData
from image2image.utils.utilities import extract_extension, format_shape, log_exception_or_error, open_docs

if ty.TYPE_CHECKING:
    from image2image_io.readers.coordinate_reader import CoordinateImageReader

    from image2image.models.data import DataModel


class CloseDatasetDialog(QtDialog):
    """Dialog where user can select which path(s) should be removed."""

    def __init__(self, parent: QWidget, model: DataModel):
        self.model = model

        super().__init__(parent)
        self.keys = self.get_keys()
        self.setMinimumWidth(800)
        self.setMaximumHeight(800)
        self.on_apply()

    def accept(self):
        """Accept."""
        self.keys = self.get_keys()
        return super().accept()

    def on_check_all(self, state: bool) -> None:
        """Check all."""
        for checkbox in self.checkboxes:
            if not checkbox.isHidden():
                checkbox.setChecked(state)
        self.on_apply()

    def on_apply(self):
        """Apply."""
        self.keys = self.get_keys()
        all_checked = len(self.keys) == len(self.checkboxes)
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)  # type: ignore[attr-defined]
        hp.disable_widgets(self.ok_btn, disabled=len(self.keys) == 0)

    def get_keys(self) -> list[str]:
        """Return state."""
        keys = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                text = checkbox.text()
                key = text.split("\n")[0]
                keys.append(key)
        return keys

    def on_filter(self):
        """Filter."""
        text = self.filter_by_name.text()
        for checkbox in self.checkboxes:
            checkbox.show() if text in checkbox.text() else checkbox.hide()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QVBoxLayout:
        """Make panel."""
        self.filter_by_name = hp.make_line_edit(
            self, placeholder="Type in name or path to filter...", func_changed=self.on_filter
        )

        self.all_check = hp.make_checkbox(
            self,
            "Check all",
            func=self.on_check_all,
            tooltip="Check all datasets that are currently visible. Only visible datasets will be removed.",
            value=False,
        )

        # iterate over all available paths
        scroll_area, scroll_widget = hp.make_scroll_area(self)
        scroll_layout = hp.make_form_layout(parent=scroll_area)
        wrapper = self.model.wrapper
        self.checkboxes = []
        if wrapper:
            for reader in wrapper.reader_iter():
                # make checkbox for each path
                checkbox = hp.make_checkbox(
                    scroll_area, f"{reader.key}\n{reader.path}\n", value=False, clicked=self.on_apply
                )
                scroll_layout.addRow(checkbox)
                self.checkboxes.append(checkbox)

        self.ok_btn = hp.make_btn(self, "OK", func=self.accept)

        layout = hp.make_v_layout()
        layout.addWidget(
            hp.make_label(
                self,
                "Please select which images should be <b>removed</b> from the project."
                " Fiducial markers will be unaffected.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                enable_url=True,
                wrap=True,
            )
        )
        layout.addWidget(scroll_widget, stretch=True)
        layout.addWidget(hp.make_h_line())
        layout.addWidget(self.filter_by_name)
        layout.addWidget(self.all_check)
        layout.addLayout(
            hp.make_h_layout(
                self.ok_btn,
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout


class SelectChannelsToLoadDialog(QtDialog):
    """Dialog to enable creation of overlays."""

    TABLE_CONFIG = (
        TableConfig()  # type: ignore
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("index", "index", "int", 50, sizing="contents")
        .add("channel name (full)", "channel_name_full", "str", 400)
    )

    def __init__(self, parent: SelectDataDialog, model: DataModel):
        super().__init__(parent, title="Select Channels to Load")
        self.model = model

        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.on_load()
        self.channels = self.get_channels()

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        connect(self.table.evt_checked, self.on_select_channel, state=state)

    def on_select_channel(self, _index: int, _state: bool | None) -> None:
        """Toggle channel."""
        self.channels = self.get_channels()
        hp.disable_widgets(self.ok_btn, disabled=len(self.channels) == 0)

    def get_channels(self) -> list[str]:
        """Select all channels."""
        channels = []
        for index in self.table.get_all_checked():
            channels.append(self.table.get_value(self.TABLE_CONFIG.channel_name_full, index))
        return channels

    def on_load(self) -> None:
        """On load."""
        data = []
        wrapper = self.model.wrapper
        if wrapper:
            channel_list = list(wrapper.channel_names_for_names(self.model.just_added_keys))
            auto_check = len(channel_list) < 10
            if len(channel_list) > 10:
                self.warning_label.show()
            if not channel_list:
                self.warning_no_channels_label.show()
            counter = Counter()
            for _i, channel_name in enumerate(channel_list):
                check = auto_check
                _, dataset = channel_name.split(" | ")
                data.append([check, counter[dataset], channel_name])
                counter[dataset] += 1
        else:
            logger.warning(f"Wrapper was not specified - {wrapper}")
            self.warning_no_channels_label.show()
        self.table.add_data(data)
        self.on_select_channel(-1, None)

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
        if STATE.allow_filters:
            self.table_proxy = MultiColumnSingleValueProxyModel(self)
            self.table_proxy.setSourceModel(self.table.model())
            self.table.model().table_proxy = self.table_proxy
            self.table.setModel(self.table_proxy)
            self.filter_by_name = hp.make_line_edit(
                self,
                placeholder="Type in channel name...",
                func_changed=lambda text, col=self.TABLE_CONFIG.channel_name_full: self.table_proxy.setFilterByColumn(
                    text, col
                ),
            )

        self.warning_label = hp.make_label(
            self,
            "Warning: There are more than <b>10</b> channels to load which can result in a slow loading time. You"
            " should probably load <b>some</b> of the channels now and can always add the others later.",
            wrap=True,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self.warning_label.hide()

        self.warning_no_channels_label = hp.make_label(
            self,
            "Warning: There are <b>no channels</b> to load. This most likely happened because we failed to read the"
            " input image. Please check your image and if the issue persists, please report this as a bug.",
            wrap=True,
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self.warning_no_channels_label.hide()

        layout = hp.make_form_layout(parent=self)
        layout.addRow(self.warning_no_channels_label)
        layout.addRow(self.warning_label)
        layout.addRow(self.table)
        if STATE.allow_filters:
            layout.addRow(self.filter_by_name)
            self.filter_by_name.setFocus()
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> You can quickly check/uncheck row by <b>double-clicking</b> on a row.<br>"
                "<b>Tip.</b> Check/uncheck a row to select which channels should be immediately loaded.<br>"
                "<b>Tip.</b> You can quickly check/uncheck <b>all</b> rows by clicking on the first column header.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        self.ok_btn = hp.make_btn(self, "OK", func=self.accept)
        layout.addRow(
            hp.make_h_layout(
                self.ok_btn,
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout


class ExtractChannelsDialog(QtDialog):
    """Dialog to extract ion images."""

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 0, no_sort=True, hidden=True)
        .add("m/z", "mz", "float", 100)
    )

    def __init__(self, parent: SelectDataDialog, key_to_extract: str):
        super().__init__(parent, title="Extract Ion Images")
        self.setFocus()
        self.key_to_extract = key_to_extract
        self.mzs = None
        self.ppm = None

    def on_open_peaklist(self) -> None:
        """Open peaklist."""
        import pandas as pd

        path = hp.get_filename(
            self,
            title="Select peak list...",
            base_dir=get_register_config().fixed_dir,
            file_filter="CSV files (*.csv);;",
            multiple=False,
        )
        if path:
            df = pd.read_csv(path, sep=",")
            if "mz" in df.columns:
                mzs = df.mz.values
            elif "m/z" in df.columns:
                mzs = df["m/z"].values
            else:
                hp.warn_pretty(self, "The file does not contain a column named 'mz' or 'm/z'.")
                return
            data = [[True, mz] for mz in mzs]
            self.table.add_data(data)

    def on_accept(self) -> None:
        """Accept."""
        if not self.mzs:
            hp.warn_pretty(self, "Please add at least one m/z value to extract.")
            return
        n = len(self.mzs)
        if n > 0 and hp.confirm(self, f"Would you like to extract <b>{n}</b> ion {pluralize('image', n)}?"):
            logger.trace(f"Extracting {n} ion images...")
            self.accept()

    def on_add(self) -> None:
        """Add peak."""
        value = self.mz_edit.value()
        values = self.table.get_col_data(self.TABLE_CONFIG.mz)
        if value is not None and value not in values:
            self.table.add_data([[True, value]])
        self.mzs = self.table.get_col_data(self.TABLE_CONFIG.mz)
        self.ppm = self.ppm_edit.value()

    def on_delete_row(self) -> None:
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

        layout = hp.make_form_layout(parent=self)
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "m/z"),
                self.mz_edit,
                hp.make_btn(self, "Add m/z", tooltip="Add peak", func=self.on_add),
                stretch_id=1,
            )
        )
        layout.addRow(hp.make_h_layout(hp.make_label(self, "ppm"), self.ppm_edit, stretch_id=1))
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Press <b>Delete</b> or <b>Backspace</b> to delete a peak.<br>"
                "<b>Tip.</b> It is a lot more efficient to extra many peaks at once than one at a time.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Open peaklist", func=self.on_open_peaklist),
                hp.make_btn(self, "Extract", func=self.on_accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        if key == Qt.Key_Escape:  # type: ignore[attr-defined]
            evt.ignore()
        elif key == Qt.Key_Backspace or key == Qt.Key_Delete:  # type: ignore[attr-defined]
            self.on_delete_row()
            evt.accept()
        elif key == Qt.Key_Plus or key == Qt.Key_A:  # type: ignore[attr-defined]
            self.on_add()
            evt.accept()
        else:
            super().keyPressEvent(evt)


class SelectDataDialog(QtFramelessTool):
    """Dialog window to select images and specify some parameters."""

    HIDE_WHEN_CLOSE = True

    _editing = False
    evt_loading = Signal()
    evt_loaded = Signal(object, object)
    evt_loaded_keys = Signal(list)
    evt_closing = Signal(object, list, list)
    evt_closed = Signal(object)
    evt_resolution = Signal(str)
    evt_import_project = Signal(str)
    evt_export_project = Signal()
    evt_files = Signal(list)
    evt_rejected_files = Signal(list)
    evt_swap = Signal(str, str)

    TABLE_CONFIG = (
        TableConfig()  # type: ignore
        .add("name", "key", "str", 0, sizing="stretch")
        .add("pixel size (um)", "resolution", "str", 0, sizing="contents")
        .add("shape", "image_size", "str", 0, sizing="contents")
        .add("type", "type", "str", 0, sizing="contents")
        .add("rotation", "rotation", "str", 0, hidden=True, sizing="contents")
        .add("flip", "flip", "button", 0, hidden=True, sizing="contents")
        .add("swap", "swap", "button", 0, hidden=True, sizing="contents")
        .add("", "extract", "button", 0, sizing="contents")
        .add("", "save", "button", 0, sizing="contents")
        .add("", "remove", "button", 0, sizing="contents")
    )

    def __init__(
        self,
        parent: QWidget,
        model: DataModel,
        config: SingleAppConfig,
        is_fixed: bool = False,
        n_max: int = 0,
        allow_geojson: bool = False,
        select_channels: bool = True,
        available_formats: str | None = None,
        project_extension: list[str] | None = None,
        allow_flip_rotation: bool = False,
        allow_swap: bool = False,
        show_split_czi: bool = True,
    ):
        # update table config
        self.TABLE_CONFIG.update_attribute("rotation", "hidden", not allow_flip_rotation)
        self.TABLE_CONFIG.update_attribute("flip", "hidden", not allow_flip_rotation)
        self.TABLE_CONFIG.update_attribute("swap", "hidden", not allow_swap)

        self.is_fixed = is_fixed
        self.allow_geojson = allow_geojson
        self.select_channels = select_channels
        self.available_formats = available_formats
        self.project_extension = project_extension
        self.show_split_czi = show_split_czi
        super().__init__(parent)
        self.n_max = n_max
        self.model = model
        self.CONFIG = config

        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.on_populate_table()

    def _clear_table(self) -> None:
        """Remove all rows."""
        with self._editing_table(), suppress(RuntimeError):
            while self.table.rowCount() > 0:
                self.table.removeRow(0)

    def on_populate_table(self) -> None:
        """Load data."""
        self._clear_table()
        wrapper = self.model.wrapper
        if wrapper:
            with self._editing_table(), MeasureTimer() as timer:
                for _path, reader in wrapper.path_reader_iter():
                    index = hp.find_in_table(self.table, self.TABLE_CONFIG.key, reader.key)
                    if index is not None:
                        continue

                    # get model information
                    index = self.table.rowCount()
                    self.table.insertRow(index)

                    # add name item
                    name_item = QTableWidgetItem(reader.key)
                    name_item.setToolTip(str(reader.path))
                    name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    name_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(index, self.TABLE_CONFIG.key, name_item)

                    # add type item
                    type_item = QTableWidgetItem(format_shape(reader.shape))
                    type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(index, self.TABLE_CONFIG.image_size, type_item)

                    # add type item
                    type_item = QTableWidgetItem(reader.reader_type)
                    type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(index, self.TABLE_CONFIG.type, type_item)

                    # add resolution item
                    res_item = QLineEdit(f"{reader.resolution:.3f}")
                    res_item.setObjectName("table_cell")
                    res_item.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    res_item.setValidator(QDoubleValidator(0, 1000, 4))
                    res_item.editingFinished.connect(partial(self.on_resolution, key=reader.key, item=res_item))
                    self.table.setCellWidget(index, self.TABLE_CONFIG.resolution, res_item)

                    # add extract button
                    if reader.allow_extraction:
                        self.table.setCellWidget(
                            index,
                            self.TABLE_CONFIG.extract,
                            hp.make_qta_btn(
                                self,
                                "add",
                                func=partial(self.on_extract_channels, key=reader.key),
                                tooltip="Extract ion images...",
                            ),
                        )
                    else:
                        item = QTableWidgetItem("N/A")
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.table.setItem(index, self.TABLE_CONFIG.extract, item)

                    if reader.reader_type == "image":
                        self.table.setCellWidget(
                            index,
                            self.TABLE_CONFIG.save,
                            hp.make_qta_btn(
                                self,
                                "save",
                                normal=True,
                                func=partial(self.on_export_settings, key=reader.key),
                                tooltip="Save image as OME-TIFF...",
                            ),
                        )
                    else:
                        item = QTableWidgetItem("N/A")
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        self.table.setItem(index, self.TABLE_CONFIG.save, item)

                    # add swap button
                    self.table.setCellWidget(
                        index,
                        self.TABLE_CONFIG.swap,
                        hp.make_qta_btn(self, "swap", normal=True, func=partial(self.on_swap, key=reader.key)),
                    )
                    # remove button
                    self.table.setCellWidget(
                        index,
                        self.TABLE_CONFIG.remove,
                        hp.make_qta_btn(
                            self,
                            "delete",
                            func=partial(self.on_remove_dataset, key=reader.key),
                            tooltip="Remove image from project. You will <b>not</b> be asked to confirm removal..",
                        ),
                    )
            logger.trace(f"Populated table with {self.table.rowCount()} rows in {timer}")

    @property
    def available_formats_filter(self) -> str:
        """Return string of available formats."""
        return self.available_formats or (
            ALLOWED_IMAGE_FORMATS if not self.allow_geojson else ALLOWED_IMAGE_FORMATS_WITH_GEOJSON
        )

    @property
    def allowed_extensions(self) -> list[str]:
        """Return list of available extensions based on the specified filter."""
        return extract_extension(self.available_formats_filter)

    def on_import_project(self) -> None:
        """Open project."""
        if self.project_extension:
            project_ext = " ".join(self.project_extension)
            project_extensions = f"Project files ({project_ext});;"

            path_ = hp.get_filename(
                self,
                "Select project...",
                base_dir=self.CONFIG.output_dir,
                file_filter=project_extensions,
            )
            if path_:
                self.evt_import_project.emit(path_)

    def on_export_project(self) -> None:
        """Export project."""
        self.evt_export_project.emit()

    def on_select_dataset(self) -> None:
        """Load path."""
        paths = hp.get_filename(
            self,
            title="Select data...",
            base_dir=get_register_config().fixed_dir if self.is_fixed else get_register_config().moving_dir,
            file_filter=self.available_formats_filter,
            multiple=True,
        )
        if paths:
            for path in paths:
                if self.is_fixed:
                    get_register_config().fixed_dir = str(Path(path).parent)
                else:
                    get_register_config().moving_dir = str(Path(path).parent)

                if self.n_max and self.model.n_paths >= self.n_max:
                    verb = "image" if self.n_max == 1 else "images"
                    hp.warn_pretty(
                        self,
                        f"Maximum number of images reached. You can only have {self.n_max} {verb} loaded at at"
                        f" time. Please remove other images first.",
                    )
                    return
            self._on_load_dataset(paths)

    def on_swap(self, key: str) -> None:
        """Swap from fixed to moving or vice versa."""
        reader = self.model.get_reader_for_key(key)
        if reader:
            source = "fixed" if self.is_fixed else "moving"
            target = "moving" if self.is_fixed else "fixed"
            if not hp.confirm(
                self,
                f"Are you sure you wish to swap this image from <b>{source}</b> to <b>{target}</b>?",
                title="Swap?",
            ):
                logger.trace(f"User cancelled swap of '{reader.key}'")
                return
            logger.trace(f"Swapping '{reader.key}' from '{source}' to '{target}'")
            self.evt_swap.emit(key, source)

    def on_set_resolution(self, key: str, resolution: float) -> None:
        """Set resolution."""
        reader = self.model.get_reader_for_key(key)
        if reader and reader.resolution != resolution:
            reader.resolution = resolution
            self.evt_resolution.emit(reader.key)
            self.on_populate_table()
            logger.trace(f"Updated pixel size of '{reader.key}' to {resolution:.2f}.")

    def on_resolution(self, item: QTableWidgetItem, key: str) -> None:
        """Table item changed."""
        if self._editing:
            return

        value = float(item.text())
        if value <= 0:
            hp.toast(
                self,
                "Pixel size cannot be 0.",
                "Pixel size cannot be 0. Please change to something reasonable.",
                icon="error",
            )
            return
        reader = self.model.get_reader_for_key(key)
        if reader:
            reader.resolution = value
            self.evt_resolution.emit(reader.key)
            logger.trace(f"Updated pixel size of '{reader.key}' to {value:.2f}.")

    def on_drop(self, event: QDropEvent) -> None:
        """Handle drop event."""
        allowed_extensions = tuple(self.allowed_extensions)
        filenames = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                # directories get a trailing "/", Path conversion removes it
                filenames.append(str(Path(url.toLocalFile())))
            else:
                filenames.append(url.toString())
        # clear filenames by removing those that might not be permitted
        filenames_ = []
        other_files_ = []
        for filename in filenames:
            if self.project_extension and any(filename.endswith(ext) for ext in self.project_extension):
                self.evt_import_project.emit(filename)
            elif filename.endswith(allowed_extensions):
                filenames_.append(filename)
            else:
                other_files_.append(filename)
                # logger.warning(
                #     f"File '{filename}' is not in a supported format. Permitted: {', '.join(allowed_extensions)}"
                # )
        if filenames_:
            logger.trace(f"Dropped {filenames_} file(s)...")
            self.evt_files.emit(filenames_)
            self._on_load_dataset(filenames_)
        if other_files_:
            self.evt_rejected_files.emit(other_files_)

    def on_close_dataset(self, force: bool = False) -> bool:
        """Close dataset."""
        if self.model.n_paths:
            keys = None
            if not force:  # only ask user if not forced
                dlg = CloseDatasetDialog(self, self.model)
                dlg.show_in_center_of_screen()
                if dlg.exec_():  # type: ignore[attr-defined]
                    keys = dlg.keys
            else:
                wrapper = self.model.wrapper
                keys = [reader.key for reader in wrapper.reader_iter()] if wrapper else self.model.keys
            logger.trace(f"Closing {keys} keys...")
            if keys:
                self.evt_closing.emit(self.model, self.model.get_channel_names_for_keys(keys), keys)  # noqa
                self.model.remove_keys(keys)
                self.evt_closed.emit(self.model)  # noqa
            self.on_populate_table()
            return True
        logger.warning("There are no dataset to close.")
        return False

    def _on_load_dataset(
        self,
        path_or_paths: PathLike | ty.Sequence[PathLike],
        transform_data: dict[str, TransformData] | None = None,
        resolution: dict[str, float] | None = None,
        reader_kws: dict[str, dict] | None = None,
    ) -> None:
        """Load data."""
        self.evt_loading.emit()  # noqa
        if not isinstance(path_or_paths, list):
            path_or_paths = [path_or_paths]

        create_worker(
            self.model.load,
            paths=path_or_paths,
            transform_data=transform_data,
            resolution=resolution,
            reader_kws=reader_kws,
            _start_thread=True,
            _connect={
                "returned": self._on_loaded_dataset,
                "errored": self._on_failed_dataset,
            },
        )

    def _on_loaded_dataset(self, model: DataModel, select: bool = True, keys: list[str] | None = None) -> None:
        """Finished loading data."""
        channel_list = []
        wrapper = model.wrapper
        if not keys:
            keys = model.just_added_keys

        if not self.select_channels or not select:
            if wrapper:
                channel_list = wrapper.channel_names_for_names(keys)
        else:
            if wrapper:
                channel_list_ = list(wrapper.channel_names_for_names(keys))
                if channel_list_:
                    dlg = SelectChannelsToLoadDialog(self, model)
                    dlg.show_in_center_of_screen()
                    if dlg.exec_():  # type: ignore
                        channel_list = dlg.channels
        logger.trace(f"Loaded {len(channel_list)} channels")
        if not channel_list:
            model.remove_keys(keys)
            model, channel_list = None, None
            logger.warning("No channels selected - dataset not loaded")
        # load data into an image
        self.evt_loaded.emit(model, channel_list)  # noqa
        self.on_populate_table()
        if model:
            self.evt_loaded_keys.emit(model.just_added_keys)  # noqa

    def _on_loaded_dataset_with_preselection(self, model: DataModel, select: bool = True) -> None:
        """Finished loading data."""
        from natsort import natsorted

        channel_list = []
        remove_keys = []
        wrapper = model.wrapper

        if not self.select_channels or not select:
            if wrapper:
                channel_list = wrapper.channel_names_for_names(model.just_added_keys)
        else:
            just_added = model.just_added_keys
            options = {k: k for k in natsorted(just_added)}
            if len(options) == 1:
                which = next(iter(options.keys()))
            elif len(options) > 1:
                from qtextra.widgets.qt_select_one import QtScrollablePickOption

                if not self.is_fixed:
                    options = {"each image": "each image", **options}

                dlg = QtScrollablePickOption(
                    self,
                    "Please select which image(s) would you like to register?",
                    options=options,
                    orientation="vertical",
                )
                which = None
                hp.show_in_center_of_screen(dlg)
                if dlg.exec_() == QDialog.DialogCode.Accepted:
                    which = dlg.option
            else:
                logger.warning("No images to select from.")
                which = None

            if which == "each image":
                remove_keys = []
                just_added = [k for k in just_added if k not in remove_keys]
            else:
                remove_keys = [k for k in just_added if k != which]
                just_added = [which] if which else None
            if wrapper and just_added:
                model.just_added_keys = just_added
                channel_list_ = list(wrapper.channel_names_for_names(just_added))
                if channel_list_:
                    dlg = SelectChannelsToLoadDialog(self, model)
                    dlg.show_in_center_of_screen()
                    if dlg.exec_():  # type: ignore
                        channel_list = dlg.channels
        logger.trace(f"Loaded {len(channel_list)} channels")
        if remove_keys:
            model.remove_keys(remove_keys)
        if not channel_list:
            model.remove_keys(model.just_added_keys)
            model, channel_list = None, None
            logger.warning("No channels selected - dataset not loaded")
        # load data into an image
        self.evt_loaded.emit(model, channel_list)  # noqa
        self.on_populate_table()

    def _on_failed_dataset(self, exception: Exception) -> None:
        """Failed to load dataset."""
        logger.error("Error occurred while loading dataset.")
        if isinstance(exception, UnsupportedFileFormatError):
            hp.toast(self.parent(), "Unsupported file format", str(exception), icon="error")
        elif isinstance(exception, MultiSceneCziError):
            hp.toast(self.parent(), "Multi-scene CZI", str(exception), icon="error")
        else:
            log_exception_or_error(exception)
        self.evt_loaded.emit(None, None)  # noqa

    def on_remove_dataset(self, key: str) -> None:
        """Remove dataset."""
        self.evt_closing.emit(self.model, self.model.get_channel_names_for_keys([key]), [key])  # noqa
        self.model.remove_keys([key])
        self.evt_closed.emit(self.model)  # noqa
        self.on_populate_table()

    def on_export_settings(self, key: str) -> None:
        """Save image as OME-TIFF."""
        from image2image.qt._dialogs._save import ExportImageDialog

        dlg = ExportImageDialog(self, self.model, key, self.CONFIG)
        dlg.exec()

    def on_extract_channels(self, key: str) -> None:
        """Extract channels from the list."""
        if not self.model.get_extractable_paths():
            logger.warning("No paths to extract data from.")
            hp.warn_pretty(
                self,
                "No paths to extract data from. Only <b>.imzML</b>, <b>.tdf</b> and <b>.tsf</b> files support data"
                " extraction.",
            )
            return

        dlg = ExtractChannelsDialog(self, key)
        key, mzs, ppm = None, None, None
        if dlg.exec_() == QDialog.DialogCode.Accepted:  # type: ignore[attr-defined]
            key = dlg.key_to_extract
            mzs = dlg.mzs
            ppm = dlg.ppm

        logger.trace(f"Extracting data for {key} ({mzs}, {ppm})")
        if key and mzs and ppm:
            reader: CoordinateImageReader = self.model.get_reader_for_key(key)
            logger.trace(f"Extracting data for {key} ({reader}")
            if reader:
                self.evt_loading.emit()  # noqa
                create_worker(
                    reader.extract,
                    mzs=mzs,
                    ppm=ppm,
                    _start_thread=True,
                    _connect={"returned": self._on_update_dataset, "errored": self._on_failed_update_dataset},
                )

    def _on_update_dataset(self, result: tuple[Path, list[str]]) -> None:
        """Finished loading data."""
        path, channel_list = result
        # load data into an image
        self.evt_loaded.emit(self.model, channel_list)  # noqa

    @staticmethod
    def _on_failed_update_dataset(exception: Exception) -> None:
        """Failed to load dataset."""
        logger.error("Error occurred while extracting images.", exception)
        log_exception_or_error(exception)

    def on_double_click(self, index: QModelIndex) -> None:
        """Double clicked."""
        row = index.row()
        name = self.table.item(row, self.TABLE_CONFIG.key).text()
        reader = self.model.get_reader_for_key(name)
        if reader:
            open_directory_alt(reader.path)
            logger.trace(f"Opening directory for '{name}'")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle(title="Modalities")

        self.table = hp.make_table(self, self.TABLE_CONFIG)
        self.table.doubleClicked.connect(self.on_double_click)

        self.split_czi_check = hp.make_checkbox(
            self,
            value=READER_CONFIG.split_czi,
            tooltip="When a CZI image contains multiple scenes, they should be split into individual datasets.",
            func=self.on_update_config,
        )
        if not self.show_split_czi:
            self.split_czi_check.hide()
        self.split_rgb_check = hp.make_checkbox(
            self,
            value=READER_CONFIG.split_rgb,
            tooltip="When loading RGB images (e.g. PAS or H&E), split those into individual <b>R</b>, <b>G</b> and"
            " <b>B</b> channels.",
            func=self.on_update_config,
        )
        self.split_roi_check = hp.make_checkbox(
            self,
            value=READER_CONFIG.split_roi,
            tooltip="When loading Bruker .d image(s), slit them by the region of interest.",
            func=self.on_update_config,
        )

        self.shapes_combo = hp.make_combobox(
            self,
            ["polygon", "path", "polygon or path", "points"],
            value=READER_CONFIG.shape_display,
            tooltip="Decide how shapes should be displayed when loading from GeoJSON."
            "<br><b>polygon</b> - filled polygons (can be slow)"
            "<br><b>path</b> - only outlines of polygons (much faster)"
            "<br><b>polygon</b> or path - use polygons if number of shapes is not too high, otherwise use paths"
            "<br><b>points</b> - display points as points (much faster but no shape information is retained)",
            func=self.on_update_config,
        )
        self.subsample_check = hp.make_checkbox(
            self,
            tooltip="Subsample shapes to speed-up rendering. Subsampling only happens if there are more than 10,000"
            " shapes.",
            func=self.on_update_config,
            value=READER_CONFIG.subsample,
        )
        self.subsample_ratio = hp.make_double_spin_box(
            self,
            minimum=1,
            maximum=100,
            value=READER_CONFIG.subsample_ratio * 100,
            step_size=1,
            n_decimals=1,
            tooltip="Ratio of samples.",
            func=self.on_update_config,
            suffix="%",
        )
        self.subsample_random = hp.make_int_spin_box(
            self,
            minimum=-1,
            maximum=np.iinfo(np.int32).max - 1,  # maximum of np.int32
            value=READER_CONFIG.subsample_random_seed,
            tooltip="Random seed for sub-selecting points.",
            func=self.on_update_config,
        )

        layout = hp.make_form_layout(margin=6)
        layout.addRow(header_layout)

        layout.addRow(hp.make_label(self, "How to load image data", bold=True))
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "Split CZI (recommended)", hide=not self.show_split_czi),
                self.split_czi_check,
                hp.make_v_line(),
                hp.make_label(self, "Split Bruker .d (recommended)"),
                self.split_roi_check,
                hp.make_v_line(),
                hp.make_label(self, "Split RGB (not recommended)"),
                self.split_rgb_check,
                stretch_after=True,
            )
        )
        layout.addRow(hp.make_label(self, "How to load shape and scatter data", bold=True))
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "Shape display"),
                self.shapes_combo,
                hp.make_v_line(),
                hp.make_label(self, "Subsample shapes"),
                self.subsample_check,
                hp.make_label(self, "Ratio"),
                self.subsample_ratio,
                hp.make_label(self, "Seed"),
                self.subsample_random,
                stretch_after=True,
            )
        )
        layout.addRow(hp.make_h_line())
        layout.addRow(self.table)
        layout.addRow(hp.make_h_line())
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(
                    self,
                    "<b>Tip.</b> You can edit pixel size by double-clicking on the cell.",
                    alignment=Qt.AlignmentFlag.AlignHCenter,
                    object_name="tip_label",
                    enable_url=True,
                ),
                hp.make_url_btn(self, func=lambda: open_docs(dialog="dataset-metadata")),
                stretch_id=(0,),
                spacing=2,
                margin=2,
                alignment=Qt.AlignmentFlag.AlignVCenter,
            )
        )
        return layout

    def on_update_config(self, _=None) -> None:
        """Update configuration."""
        READER_CONFIG.update(
            shape_display=self.shapes_combo.currentText(),
            subsample=self.subsample_check.isChecked(),
            subsample_ratio=self.subsample_ratio.value() / 100,
            subsample_random_seed=self.subsample_random.value(),
            split_rgb=self.split_rgb_check.isChecked(),
            split_czi=self.split_czi_check.isChecked(),
            split_roi=self.split_roi_check.isChecked(),
        )

    # noinspection PyAttributeOutsideInit
    @contextmanager
    def _editing_table(self):
        self._editing = True
        yield
        self._editing = False

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        if key == Qt.Key_Escape:  # type: ignore[attr-defined]
            evt.ignore()
        else:
            super().keyPressEvent(evt)
