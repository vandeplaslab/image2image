"""Windows for dataset management."""
import typing as ty
from contextlib import contextmanager
from functools import partial
from pathlib import Path

from koyo.typing import PathLike
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog, QtFramelessTool
from qtextra.widgets.qt_table_view import FilterProxyModel, QtCheckableTableView
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QDoubleValidator, QDropEvent
from qtpy.QtWidgets import QFormLayout, QHeaderView, QLineEdit, QTableWidget, QTableWidgetItem, QWidget
from superqt.utils import create_worker

from image2image.config import CONFIG
from image2image.enums import ALLOWED_FORMATS, ALLOWED_FORMATS_WITH_GEOJSON
from image2image.exceptions import MultiSceneCziError, UnsupportedFileFormatError
from image2image.models.transform import TransformData
from image2image.utils.utilities import log_exception_or_error

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel
    from image2image.readers.coordinate_reader import CoordinateImageReader


class CloseDatasetDialog(QtDialog):
    """Dialog where user can select which path(s) should be removed."""

    def __init__(self, parent: QWidget, model: "DataModel"):
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
        hp.style_form_layout(layout)
        layout.addRow(
            hp.make_label(
                self,
                "Please select which images should be <b>removed</b> from the project."
                " Fiducial markers will be unaffected.",
                alignment=Qt.AlignHCenter,  # type: ignore[attr-defined]
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

    def on_check_all(self, state: bool) -> None:
        """Check all."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(state)

    def on_apply(self):
        """Apply."""
        self.paths = self.get_paths()
        all_checked = len(self.paths) == len(self.checkboxes)
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)  # type: ignore[attr-defined]

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
        TableConfig()  # type: ignore
        .add("", "check", "bool", 25, no_sort=True)
        .add("channel name", "channel_name", "str", 200)
        .add("channel name (full)", "channel_name_full", "str", 0, hidden=True)
    )

    def __init__(self, parent: "SelectDataDialog", model: "DataModel"):
        super().__init__(parent, title="Select Channels to Load")
        self.model = model

        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.on_load()
        self.channels = self.get_channels()

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        connect(self.table.evt_checked, self.on_select_channel, state=state)

    def on_select_channel(self, _index: int, _state: bool) -> None:
        """Toggle channel."""
        self.channels = self.get_channels()

    def get_channels(self) -> ty.List[str]:
        """Select all channels."""
        channels = []
        for index in self.table.get_all_checked():
            channels.append(self.table.get_value(self.TABLE_CONFIG.channel_name_full, index))
        return channels

    def on_load(self) -> None:
        """On load."""
        data = []
        wrapper = self.model.get_wrapper()
        if wrapper:
            channel_list = list(wrapper.channel_names_for_names(self.model.just_added))
            auto_check = len(channel_list) < 10
            if len(channel_list) > 10:
                self.warning_label.show()
            if not channel_list:
                self.warning_no_channels_label.show()
            for i, name in enumerate(channel_list):
                check = True if auto_check else i < 5
                channel_name, _ = name.split(" | ")
                data.append([check, channel_name, name])
        else:
            logger.warning(f"Wrapper was not specified - {wrapper}")
            self.warning_no_channels_label.show()
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
        self.table_proxy = FilterProxyModel(self)
        self.table_proxy.setSourceModel(self.table.model())
        self.table.setModel(self.table_proxy)

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

        self.filter_by_name = hp.make_line_edit(
            self,
            placeholder="Filter by channel name...",
            func_changed=lambda text, col=self.TABLE_CONFIG.channel_name: self.table_proxy.setFilterByColumn(text, col),
        )

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
        layout.addRow(self.warning_no_channels_label)
        layout.addRow(self.warning_label)
        layout.addRow(hp.make_label(self, "Filter by channel name:"), self.filter_by_name)
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> You can quickly check/uncheck row by <b>double-clicking</b> on a row.<br>"
                "<b>Tip.</b> Check/uncheck a row to select which channels should be immediately loaded.<br>"
                "<b>Tip.</b> You can quickly check/uncheck <b>all</b> rows by clicking on the first column header.",
                alignment=Qt.AlignHCenter,  # type: ignore[attr-defined]
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

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 0, no_sort=True, hidden=True)
        .add("m/z", "mz", "float", 100)
    )

    def __init__(self, parent: "SelectDataDialog", path_to_extract: PathLike):
        super().__init__(parent, title="Extract Ion Images")
        self.setFocus()
        self.path_to_extract = path_to_extract
        self.mzs = None
        self.ppm = None

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

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
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
                alignment=Qt.AlignHCenter,  # type: ignore[attr-defined]
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
    evt_closed = Signal(object)
    evt_resolution = Signal(Path)

    TABLE_CONFIG = (
        TableConfig()  # type: ignore
        .add("name", "name", "str", 0)
        .add("resolution", "resolution", "str", 0)
        .add("type", "type", "str", 0)
        .add("extract", "extract", "str", 0)
    )

    def __init__(
        self,
        parent: QWidget,
        model: "DataModel",
        is_fixed: bool = False,
        n_max: int = 0,
        allow_geojson: bool = False,
        select_channels: bool = True,
    ):
        self.is_fixed = is_fixed
        super().__init__(parent)
        self.n_max = n_max
        self.model = model
        self.allow_geojson = allow_geojson
        self.select_channels = select_channels
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self.on_populate_table()

    def on_remove(self, name: str, path: str) -> None:
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

    def on_resolution(self, item, path: Path) -> None:
        """Table item changed."""
        if self._editing:
            return
        value = float(item.text())
        reader = self.model.get_reader(path)
        if reader:
            reader.resolution = value
            self.evt_resolution.emit(reader.path)
            logger.trace(f"Updated resolution of '{path.name}' to {value:.2f}.")

    def _clear_table(self) -> None:
        """Remove all rows."""
        with self._editing_table():
            while self.table.rowCount() > 0:
                self.table.removeRow(0)

    def on_populate_table(self) -> None:
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
                    name_item = QTableWidgetItem(name)
                    name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)  # type: ignore[attr-defined]
                    name_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                    self.table.setItem(index, self.TABLE_CONFIG.name, name_item)

                    # add type item
                    type_item = QTableWidgetItem(reader.reader_type)
                    type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)  # type: ignore[attr-defined]
                    type_item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                    self.table.setItem(index, self.TABLE_CONFIG.type, type_item)

                    # add resolution item
                    res_item = QLineEdit(f"{resolution:.2f}")
                    res_item.setObjectName("table_cell")
                    res_item.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                    res_item.setValidator(QDoubleValidator(0, 100, 2))
                    res_item.editingFinished.connect(partial(self.on_resolution, path=path, item=res_item))
                    self.table.setCellWidget(index, self.TABLE_CONFIG.resolution, res_item)
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
                        item.setFlags(item.flags() & ~Qt.ItemIsEditable)  # type: ignore[attr-defined]
                        item.setTextAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
                        self.table.setItem(index, self.TABLE_CONFIG.extract, item)

    def on_select_dataset(self) -> None:
        """Load path."""
        paths = hp.get_filename(
            self,
            title="Select data...",
            base_dir=CONFIG.fixed_dir if self.is_fixed else CONFIG.moving_dir,
            file_filter=ALLOWED_FORMATS if not self.allow_geojson else ALLOWED_FORMATS_WITH_GEOJSON,
            multiple=True,
        )
        if paths:
            for path in paths:
                if self.is_fixed:
                    CONFIG.fixed_dir = str(Path(path).parent)
                else:
                    CONFIG.moving_dir = str(Path(path).parent)

                if self.n_max and self.model.n_paths >= self.n_max:
                    verb = "image" if self.n_max == 1 else "images"
                    hp.warn(
                        self,
                        f"Maximum number of images reached. You can only have {self.n_max} {verb} loaded at at"
                        f" time. Please remove other images first.",
                    )
                    return
            self._on_load_dataset(paths)

    def on_drop(self, event: QDropEvent) -> None:
        """Handle drop event."""
        filenames = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                # directories get a trailing "/", Path conversion removes it
                filenames.append(str(Path(url.toLocalFile())))
            else:
                filenames.append(url.toString())
        if filenames:
            self._on_load_dataset(filenames)

    def on_close_dataset(self, force: bool = False) -> bool:
        """Close dataset."""
        from image2image.qt._dialogs import CloseDatasetDialog

        if self.model.n_paths:
            paths = None
            if not force:  # only ask user if not forced
                dlg = CloseDatasetDialog(self, self.model)
                if dlg.exec_():  # type: ignore[attr-defined]
                    paths = dlg.paths
            else:
                paths = self.model.paths
            if paths:
                self.model.remove_paths(paths)
                self.evt_closed.emit(self.model)  # noqa
            self.on_populate_table()
            return True
        else:
            logger.warning("There are no dataset to close.")
        return False

    def _on_load_dataset(
        self,
        path_or_paths: ty.Union[PathLike, ty.Sequence[PathLike]],
        transform_data: ty.Optional[ty.Dict[str, TransformData]] = None,
        resolution: ty.Optional[ty.Dict[str, float]] = None,
    ) -> None:
        """Load data."""
        self.evt_loading.emit()  # noqa
        if not isinstance(path_or_paths, list):
            path_or_paths = [path_or_paths]
        self.model.add_paths(path_or_paths)
        create_worker(
            self.model.load,
            transform_data=transform_data,
            resolution=resolution,
            _start_thread=True,
            _connect={
                "returned": self._on_loaded_dataset,
                "errored": self._on_failed_dataset,
            },
        )
        logger.info(f"Started loading dataset - '{self.model.paths}'")

    def _on_loaded_dataset(self, model: "DataModel") -> None:
        """Finished loading data."""
        channel_list = []
        if not self.select_channels:
            wrapper = model.get_wrapper()
            if wrapper:
                channel_list = wrapper.channel_names_for_names(model.just_added)
        else:
            dlg = SelectChannelsToLoadDialog(self, model)
            if dlg.exec_():  # type: ignore
                channel_list = dlg.channels
        logger.info(f"Selected channels: {channel_list}")
        if not channel_list:
            model.remove_paths(model.just_added)
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
        log_exception_or_error(exception)
        self.evt_loaded.emit(None, None)  # noqa

    def _on_extract_channels(self, path: PathLike) -> None:
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
        if dlg.exec_():  # type: ignore[attr-defined]
            path = dlg.path_to_extract
            mzs = dlg.mzs
            ppm = dlg.ppm

        if path and mzs and ppm:
            reader: "CoordinateImageReader" = self.model.get_reader(path)  # noqa
            if reader:
                self.evt_loading.emit()  # noqa
                create_worker(
                    reader.extract,
                    mzs=mzs,
                    ppm=ppm,
                    _start_thread=True,
                    _connect={"returned": self._on_update_dataset, "errored": self._on_failed_update_dataset},
                )

    def _on_update_dataset(self, result: ty.Tuple[Path, ty.List[str]]) -> None:
        """Finished loading data."""
        path, channel_list = result
        # load data into an image
        self.evt_loaded.emit(self.model, channel_list)  # noqa

    @staticmethod
    def _on_failed_update_dataset(exception: Exception) -> None:
        """Failed to load dataset."""
        logger.error("Error occurred while extracting images.", exception)
        log_exception_or_error(exception)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Images")
        column_names = ["name", "pixel size (μm)", "type", "extract"]
        self.table = QTableWidget(self)
        self.table.setColumnCount(len(column_names))  # name, resolution, layer type, extract
        self.table.setHorizontalHeaderLabels(column_names)
        self.table.setCornerButtonEnabled(False)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.TABLE_CONFIG.name, QHeaderView.Stretch)  # type: ignore[attr-defined]
        header.setSectionResizeMode(
            self.TABLE_CONFIG.resolution,
            QHeaderView.ResizeToContents,  # type: ignore[attr-defined]
        )
        header.setSectionResizeMode(
            self.TABLE_CONFIG.type,
            QHeaderView.ResizeToContents,  # type: ignore[attr-defined]
        )
        header.setSectionResizeMode(
            self.TABLE_CONFIG.extract,
            QHeaderView.ResizeToContents,  # type: ignore[attr-defined]
        )

        # add delegate
        # self.table.setItemDelegateForColumn(self.TABLE_CONFIG.resolution, NumericDelegate(self.table))

        layout = hp.make_form_layout()
        hp.style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Add image...", func=self.on_select_dataset),
                hp.make_btn(self, "Remove image...", func=self.on_close_dataset),
                stretch_id=0,
            )
        )
        layout.addRow(hp.make_h_line())
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> You can edit pixel size by double-clicking on the cell.",
                alignment=Qt.AlignHCenter,  # type: ignore[attr-defined]
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
        if key == Qt.Key_Escape:  # type: ignore[attr-defined]
            evt.ignore()
        else:
            super().keyPressEvent(evt)
