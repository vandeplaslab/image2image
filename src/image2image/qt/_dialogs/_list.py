"""Modality list."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_list_widget import QtListItem, QtListWidget
from qtextra.widgets.qt_table_view_check import MultiColumnSingleValueProxyModel, QtCheckableTableView
from qtpy.QtCore import QRegularExpression, Qt, Signal, Slot  # type: ignore[attr-defined]
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QGridLayout, QListWidgetItem, QSizePolicy
from superqt.utils import qdebounced

from image2image.models.transform import TransformData, TransformModel
from image2image.qt._wsi._widgets import QtModalityLabel
from image2image.utils.utilities import ensure_list, format_shape, format_size

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader
    from image2image_io.wrapper import ImageWrapper
    from napari.utils.events import Event

    from image2image.qt._dialogs import DatasetDialog


TABLE_CONFIG = (
    TableConfig()  # type: ignore[no-untyped-call]
    .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
    .add("index", "index", "int", 50, sizing="fixed")
    .add("channel name", "channel_name", "str", 125)
    .add("dataset", "dataset", "str", 250)
)


class QtDatasetItem(QtListItem):
    """Widget used to nicely display information about RGBImageChannel.

    This widget permits display of label information as well as modification of color.
    """

    _mode: bool = False
    _editing: bool = False
    item_model: str  # type: ignore[assignment]

    # events
    evt_delete = Signal(str)
    evt_resolution = Signal(str)
    evt_transform = Signal(str)
    evt_channel_all = Signal(bool, list)  # list of channel | dataset
    evt_channel = Signal(bool, str)  # channel | dataset

    evt_iter_add = Signal(str, int)
    evt_iter_remove = Signal(str, int)
    evt_iter_next = Signal(str, int)

    def __init__(
        self,
        item: QListWidgetItem,
        parent: QtDatasetList,
        allow_transform: bool = True,
        allow_iterate: bool = True,
        allow_channels: bool = True,
    ):
        super().__init__(parent)
        self._parent: QtDatasetList = parent
        self.setMouseTracking(True)
        self.item = item
        self.allow_transform = allow_transform
        self.allow_iterate = allow_iterate
        self.allow_channels = allow_channels

        self.name_label = hp.make_label(
            self,
            "",
            tooltip="Name of the modality.",
            object_name="header_label",
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )
        self.resolution_label = hp.make_line_edit(
            self,
            tooltip="Resolution of the modality.",
            func=self._on_update_resolution,
            validator=QRegularExpressionValidator(QRegularExpression(r"^[0-9]+(\.[0-9]{1,3})?$")),
            object_name="discreet_line_edit",
        )
        self.resolution_label.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.shape_label = hp.make_label(self, "", tooltip="Shape of the modality.")
        self.dtype_label = hp.make_label(self, "", tooltip="Data type of the modality.")
        self.size_label = hp.make_label(self, "", tooltip="Uncompressed size of the modality in GB.")

        self.modality_icon = QtModalityLabel(self)
        self.open_dir_btn = hp.make_qta_btn(
            self, "folder", tooltip="Open directory containing the image.", normal=True, func=self.on_open_directory
        )
        self.remove_btn = hp.make_qta_btn(
            self, "delete", tooltip="Remove modality from the list.", normal=True, func=self.on_remove
        )
        self.extract_btn = hp.make_qta_btn(
            self, "extract", tooltip="Extract images for dataset (e.g. from IMS).", normal=True, func=self.on_extract
        )
        self.transform_btn = hp.make_qta_btn(
            self, "transform", tooltip="Transform...", normal=True, func=self.on_transform_menu
        )
        self.iterate_btn = hp.make_qta_btn(
            self, "iterate", tooltip="Activate iteration...", normal=True, func=self.on_iterate
        )
        self.save_btn = hp.make_qta_btn(self, "save", tooltip="Save data as...", normal=True, func=self.on_save)

        self.table = QtCheckableTableView(self, config=TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(TABLE_CONFIG.header, TABLE_CONFIG.no_sort_columns, TABLE_CONFIG.hidden_columns)
        self.table.evt_checked.connect(self.on_toggle_channel)
        self.table.setVisible(self.allow_channels)

        self.table_proxy = MultiColumnSingleValueProxyModel(self)
        self.table_proxy.setSourceModel(self.table.model())
        self.table.model().table_proxy = self.table_proxy
        self.table.setModel(self.table_proxy)

        grid = QGridLayout(self)
        # widget, row, column, rowspan, colspan
        grid.setSpacing(1)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(3, True)
        grid.setRowStretch(2, True)
        # column 1
        layout = hp.make_v_layout(
            self.modality_icon,
            self.open_dir_btn,
            self.remove_btn,
            self.extract_btn,
            self.transform_btn,
            self.iterate_btn,
            self.save_btn,
            stretch_after=True,
            margin=1,
            spacing=1,
        )
        grid.addLayout(layout, 0, 0, 5, 1)
        # column 2
        grid.addWidget(self.name_label, 0, 1, 1, 7)
        grid.addWidget(hp.make_label(self, "Pixel size", bold=True), 1, 1, 1, 1)
        grid.addWidget(self.resolution_label, 1, 2, 1, 1)
        grid.addWidget(hp.make_label(self, "Shape", bold=True), 1, 4, 1, 1)
        grid.addWidget(self.shape_label, 1, 5, 1, 1)
        grid.addWidget(hp.make_label(self, "Size", bold=True), 1, 6, 1, 1)
        grid.addWidget(self.size_label, 1, 7, 1, 1)
        # row 3
        grid.addWidget(self.table, 2, 1, 5, 7)

        # set from model
        self._set_from_model()

    def get_model(self) -> BaseReader:
        """Get model."""
        parent: QtDatasetList = self._parent  # type: ignore[assignment]
        data_model = parent.data_model
        return data_model.get_reader_for_key(self.item_model)  # type: ignore[return-value]

    @property
    def transform_model(self) -> TransformModel:
        """Transform model."""
        return self._parent.transform_model

    @contextmanager
    def editing(self) -> ty.Generator[None, None, None]:
        """Context manager to set editing."""
        self._editing = True
        yield
        self._editing = False

    def _set_from_model(self, _: ty.Any = None) -> None:
        """Update UI elements."""
        reader = self.get_model()
        self.setToolTip(f"<b>Modality</b>: {reader.name}<br><b>Path</b>: {reader.path}")

        # metadata information
        self.modality_icon.state = reader.reader_type
        self.name_label.setText(reader.name)
        self.modality_icon.state = reader.reader_type
        self.resolution_label.setText(f"{reader.resolution:.3f}")
        self.shape_label.setText(format_shape(reader.shape))
        self.size_label.setText(format_size(reader.shape, reader.dtype))
        self.save_btn.setVisible(reader.reader_type == "image")
        self.extract_btn.setVisible(reader.allow_extraction)
        self.iterate_btn.setVisible(reader.reader_type == "image" and self.allow_iterate)
        self.transform_btn.setVisible(reader.reader_type == "image" and self.allow_transform)
        # channel information
        self._update_channel_list(reader)

    def _update_channel_list(self, reader: BaseReader) -> None:
        """On load."""
        with self.editing():
            data = []
            # existing_data = self.table.get_data()
            for index, channel_name in enumerate(reader.channel_names):
                # checked, channel_id, channel_name, dataset, key
                data.append([False, index, channel_name, reader.key])
            self.table.append_data(data)
            self.table.enable_all_check = self.table.row_count() < 20
        logger.trace(f"Updated channel table - {len(data)} rows for {reader.name}.")

    def select_channel(self, channel_name: str, state: bool) -> None:
        """Select channel in table."""
        with self.editing(), hp.qt_signals_blocked(self.table):
            index = self.table.get_row_id(TABLE_CONFIG.channel_name, channel_name)
            if index != -1:
                self.table.set_value(TABLE_CONFIG.check, index, state)

    def _on_update_resolution(self) -> None:
        """Update resolution."""
        resolution = self.resolution_label.text()
        if not resolution:
            return
        model = self.get_model()
        model.resolution = float(resolution)
        self.evt_resolution.emit(self.item_model)

    def set_resolution(self, resolution: float) -> None:
        """Update resolution."""
        reader = self.get_model()
        if reader and reader.resolution != resolution:
            self.resolution_label.setText(f"{reader.resolution:.3f}")
            logger.trace(f"Updated pixel size of '{reader.key}' to {resolution:.2f}.")

    def on_remove(self) -> None:
        """Remove image/modality from the list."""
        model = self.get_model()
        if not model or hp.confirm(
            self,
            f"Are you sure you want to remove <b>{model.name}</b> from the list?",
            "Please confirm.",
        ):
            self.evt_delete.emit(self.item_model)

    def on_toggle_channel(self, index: int, state: bool) -> None:
        """Toggle channel."""
        if index == -1:
            reader = self.get_model()
            channel_names = [f"{channel_name} | {reader.key}" for channel_name in reader.channel_names]
            self.evt_channel_all.emit(state, channel_names)
        else:
            channel_name = f"{self.table.get_value(TABLE_CONFIG.channel_name, index)} | {self.item_model}"
            self.evt_channel.emit(state, channel_name)

    def on_extract(self) -> None:
        """Extract data."""

    def on_save(self) -> None:
        """Save data."""

    def on_iterate(self) -> None:
        """Activate iteration data."""

    def on_transform_menu(self) -> None:
        """Open transform menu."""
        reader = self.get_model()
        menu = hp.make_menu(self)
        hp.make_menu_item(self, "Add transform...", menu=menu, icon="add", func=self.on_add_transform)
        hp.make_menu_item(self, "Remove transform...", menu=menu, icon="remove")
        menu.addSeparator()
        for transform in self.transform_model.transform_names:
            hp.make_menu_item(
                self,
                transform,
                menu=menu,
                func=partial(self.on_select_transform, transform),
                checkable=True,
                checked=reader.transform_name == transform,
            )
        hp.show_right_of_mouse(menu)

    def on_add_transform(self) -> None:
        """Add transform from file."""
        from image2image.config import get_viewer_config
        from image2image.enums import ALLOWED_PROJECT_EXPORT_REGISTER_FORMATS

        path = hp.get_filename(
            self,
            "Load transformation",
            base_dir=get_viewer_config().output_dir,
            file_filter=ALLOWED_PROJECT_EXPORT_REGISTER_FORMATS,
        )

        if path:
            # load transformation
            path_ = Path(path)
            get_viewer_config().output_dir = str(path_.parent)

            # load data from config file
            try:
                with MeasureTimer() as timer:
                    transform_data = TransformData.from_i2r(path_, validate_paths=False)
                logger.trace(f"Loaded transform data in {timer()}")
            except ValueError as e:
                hp.warn_pretty(
                    self, f"Failed to load transformation from {path_}\n{e}", "Failed to load transformation"
                )
                logger.exception(f"Failed to load transformation from {path_}")
                return
            self.transform_model.add_transform(path_, transform_data)

    def on_remove_transform(self) -> None:
        """Add transform from file."""
        transforms = self.transform_model.transform_names
        choices = hp.choose_from_list(self, transforms, title="Select transforms to remove")
        if choices:
            for transform_name in choices:
                if transform_name == "Identity matrix":
                    continue
                self.transform_model.remove_transform(transform_name)

    def on_select_transform(self, transform_name: str) -> None:
        """Select and apply transform."""
        transform_data = self.transform_model.get_matrix(transform_name)
        reader = self.get_model()
        reader.transform_name = transform_name
        reader.transform_data = deepcopy(transform_data)
        self.evt_transform.emit(self.item_model)
        logger.trace(f"Updated transformation matrix for '{self.item_model}'")

    def on_open_directory(self) -> None:
        """Open directory where the image is located."""
        from koyo.path import open_directory_alt

        model = self.get_model()
        open_directory_alt(model.path)


class QtDatasetList(QtListWidget):
    """List of notifications."""

    evt_delete = Signal(str)
    evt_resolution = Signal(str)
    evt_transform = Signal(str)
    evt_channel = Signal(bool, str)  # channel | dataset
    evt_channel_all = Signal(bool, list)  # list of channel | dataset

    evt_iter_add = Signal(str, int)
    evt_iter_remove = Signal(str, int)
    evt_iter_next = Signal(str, int)

    def __init__(self, parent: DatasetDialog, allow_channels: bool, allow_transform: bool, allow_iterate: bool):
        self.allow_channels = allow_channels
        self.allow_transform = allow_transform
        self.allow_iterate = allow_iterate

        super().__init__(parent)
        self.setSpacing(1)
        # self.setSelectionsMode(QListWidget.SingleSelection)
        self.setMinimumHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.setUniformItemSizes(True)

        # set parent attributes
        self._parent = parent
        self.view = parent.view
        self.transform_model = parent.transform_model
        self.data_model = parent.model
        self.config = parent.CONFIG

        self._reader_type_filters: list[str] = []
        self._dataset_filters: list[str] = []
        self._channel_filters: list[str] = []

    @property
    def wrapper(self) -> ImageWrapper:
        """Get ImageWrapper."""
        return self.data_model.wrapper

    def _make_widget(self, item: QListWidgetItem) -> QtDatasetItem:
        widget = QtDatasetItem(
            item,
            parent=self,
            allow_iterate=self.allow_iterate,
            allow_transform=self.allow_transform,
            allow_channels=self.allow_channels,
        )  # type: ignore[attr-defined]
        widget.evt_delete.connect(self.on_remove)
        widget.evt_resolution.connect(self.evt_resolution.emit)
        widget.evt_transform.connect(self.evt_transform.emit)
        widget.evt_channel_all.connect(self.evt_channel_all.emit)
        widget.evt_channel.connect(self.evt_channel.emit)
        widget.evt_iter_add.connect(self.evt_iter_add.emit)
        widget.evt_iter_remove.connect(self.evt_iter_remove.emit)
        widget.evt_iter_next.connect(self.evt_iter_next.emit)
        widget.evt_remove.connect(self.remove_item)
        return widget

    def _check_existing(self, item_model: str) -> bool:  # type: ignore[override]
        """Check whether model already exists."""
        return any(item_model_ == item_model for item_model_ in self.model_iter())

    def on_remove(self, item_model: str) -> None:
        """Remove model."""
        self.remove_by_item_model(item_model, force=True)
        self.evt_delete.emit(item_model)

    def set_resolution(self, key: str, resolution: float) -> None:
        """Update resolution"""
        widget: QtDatasetItem = self.get_widget_for_item_model(key)
        if widget:
            widget.set_resolution(resolution)

    def populate(self) -> None:
        """Create list of items."""
        wrapper = self.wrapper
        if wrapper:
            for _path, reader in wrapper.path_reader_iter():
                if not self._check_existing(reader.key):
                    self.append_item(reader.key)
            for item_model in self.model_iter():  # type: ignore[var-annotated]
                reader = wrapper.get_reader_for_key(item_model)
                if not reader:
                    self.remove_by_item_model(item_model, force=True)
        logger.debug("Populated modality list.")

    def validate(self) -> None:
        """Validate visibilities."""
        self.on_filter_by_reader_type(self._reader_type_filters)
        self.on_filter_by_dataset_name(self._dataset_filters)
        self.on_filter_by_channel_name(self._channel_filters)

    def on_filter_by_reader_type(self, filters: str | list[str]) -> None:
        """Filter by reader type."""
        self._reader_type_filters = ensure_list(filters)
        self._filter_by_reader_type_and_name()

    def on_filter_by_dataset_name(self, filters: str | list[str]) -> None:
        """Filter by dataset name."""
        if filters == "":
            filters = []
        self._dataset_filters = ensure_list(filters)
        self._filter_by_reader_type_and_name()

    def _filter_by_reader_type_and_name(self) -> None:
        widget: QtDatasetItem
        for widget in self.widget_iter():
            visible = True
            reader = widget.get_model()
            if not reader:
                continue
            if reader.reader_type not in self._reader_type_filters:
                visible = False
            if visible and any(filter_ in reader.name for filter_ in self._dataset_filters):
                visible = False
            widget.setHidden(not visible)
        self.refresh()

    def on_filter_by_channel_name(self, filters: list[str]):
        """Filter by dataset name."""
        self._channel_filters = filters
        widget: QtDatasetItem
        for widget in self.widget_iter():
            widget.table_proxy.setFilterByColumn(filters, TABLE_CONFIG.channel_name)

    def sync_layers(self) -> None:
        """Manually synchronize layers."""
        if not self.view:
            return
        for layer in self.view.layers:
            name = layer.name
            if " | " not in name:
                continue
            channel_name, dataset = name.split(" | ")
            widget = self.get_widget_for_item_model(dataset)
            if widget:
                widget.select_channel(channel_name, False)

    @qdebounced(timeout=500, leading=False)
    def on_sync_layers(self, event: Event) -> None:
        """Synchronize layers."""
        self._sync_layers(event)

    def _sync_layers(self, event: Event) -> None:
        if event.type in ["visible", "inserted"]:
            self._sync_layer_visibility(event)
        elif event.type == "removed":
            self._sync_layer_presence(event)

    def _sync_layer_presence(self, event: Event) -> None:
        layer = event.value
        name = layer.name
        if " | " not in name:
            return
        channel_name, dataset = name.split(" | ")
        widget = self.get_widget_for_item_model(dataset)
        if widget:
            widget.select_channel(channel_name, False)

    def _sync_layer_visibility(self, _event: Event) -> None:
        for layer in self.view.layers:
            name = layer.name
            if " | " not in name:
                continue
            channel_name, dataset = name.split(" | ")
            widget = self.get_widget_for_item_model(dataset)
            if widget:
                widget.select_channel(channel_name, layer.visible)
