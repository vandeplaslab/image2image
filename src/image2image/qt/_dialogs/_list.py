"""Modality list."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from koyo.utilities import find_nearest_value_in_dict
from loguru import logger
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_table_view_check import MultiColumnSingleValueProxyModel, QtCheckableTableView
from qtextra.widgets.qt_toolbar_mini import QtMiniToolbar
from qtpy.QtCore import QEvent, QRegularExpression, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QDialog, QFrame, QGridLayout, QScrollArea, QSizePolicy, QWidget
from superqt.utils import create_worker, qdebounced

from image2image.models.transform import TransformData, TransformModel
from image2image.qt._wsi._widgets import QtModalityLabel
from image2image.utils.utilities import ensure_list, format_shape_with_pyramid, format_size, get_resolution_options

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader
    from image2image_io.wrapper import ImageWrapper
    from napari.utils.events import Event

    from image2image.config import SingleAppConfig
    from image2image.qt._dialogs import DatasetDialog

TABLE_CONFIG = (
    TableConfig()  # type: ignore[no-untyped-call]
    .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
    .add("index", "index", "int", 50, sizing="fixed")
    .add("channel name", "channel_name", "str", 125)
)


class QtDatasetItem(QFrame):
    """Widget used to nicely display information about RGBImageChannel.

    This widget permits display of label information as well as modification of color.
    """

    _mode: bool = False
    _editing: bool = False
    item_model: str  # type: ignore[assignment]

    # events
    evt_delete = Signal(str, bool)
    evt_resolution = Signal(str)
    evt_transform = Signal(str)
    evt_channel_all = Signal(bool, list)  # list of channel | dataset
    evt_channel = Signal(bool, str)  # channel | dataset
    evt_refresh = Signal(str)

    evt_iter_add = Signal(str, int)
    evt_iter_remove = Signal(str, int)
    evt_iter_next = Signal(str, int)

    def __init__(
        self,
        key: str,
        parent: QtDatasetList,
        allow_transform: bool = True,
        allow_iterate: bool = True,
        allow_channels: bool = True,
        allow_save: bool = True,
    ):
        super().__init__(parent)
        self._parent: QtDatasetList = parent
        self.key = key
        self.setMouseTracking(True)
        self.allow_transform = allow_transform
        self.allow_iterate = allow_iterate
        self.allow_channels = allow_channels

        self.name_label = hp.make_label(
            self,
            "",
            tooltip="Name of the modality.",
            object_name="header_label",
            alignment=Qt.AlignmentFlag.AlignHCenter,
            elide_mode=Qt.TextElideMode.ElideLeft,
        )
        self.resolution_label = hp.make_line_edit(
            self,
            tooltip="Resolution of the modality.",
            func_changed=self.on_update_resolution,
            validator=QRegularExpressionValidator(QRegularExpression(r"^[0-9]+(\.[0-9]{1,5})?$")),
            object_name="discreet_line_edit",
        )
        self.resolution_label.setFixedWidth(100)

        self.shape_label = hp.make_label(
            self, "", tooltip="Shape of the modality (shape, (number of images in pyramid))."
        )
        self.dtype_label = hp.make_label(self, "", tooltip="Data type of the modality.")
        self.size_label = hp.make_label(self, "", tooltip="Uncompressed size of the modality in GB.")

        self.modality_icon = QtModalityLabel(self)
        self.modality_icon.set_average()
        self.open_dir_btn = hp.make_qta_btn(
            self, "folder", tooltip="Open directory containing the image.", normal=True, func=self.on_open_directory
        )
        self.remove_btn = hp.make_qta_btn(
            self,
            "delete",
            tooltip="Remove modality from the list.<br>Right-click to remove without confirmation.",
            normal=True,
            func=self.on_remove,
            func_menu=self.on_force_remove,
        )
        self.extract_btn = hp.make_qta_btn(
            self, "extract", tooltip="Extract images for dataset (e.g. from IMS).", normal=True, func=self.on_extract
        )
        self.transform_btn = hp.make_qta_btn(
            self, "transform", tooltip="Apply transform...", normal=True, func=self.on_transform_menu
        )
        self.iterate_btn = hp.make_qta_btn(
            self, "iterate", tooltip="Activate iteration...", normal=True, func=self.on_iterate
        )
        self.save_btn = hp.make_qta_btn(
            self, "save", tooltip="Save data as...", normal=True, func=self.on_save, hide=not allow_save
        )

        self.table = QtCheckableTableView(self, config=TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(TABLE_CONFIG.header, TABLE_CONFIG.no_sort_columns, TABLE_CONFIG.hidden_columns)
        self.table.evt_checked.connect(self.on_toggle_channel)

        self.table_proxy = MultiColumnSingleValueProxyModel(self)
        self.table_proxy.setSourceModel(self.table.model())
        self.table.model().table_proxy = self.table_proxy
        self.table.setModel(self.table_proxy)

        grid = QGridLayout(self)
        # widget, row, column, rowspan, colspan
        grid.setSpacing(1)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(7, True)
        grid.setRowStretch(2 if allow_channels else 3, True)
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
        grid.addWidget(hp.make_label(self, "Shape", bold=True), 1, 3, 1, 1)
        grid.addWidget(self.shape_label, 1, 4, 1, 1)
        grid.addWidget(hp.make_label(self, "Size", bold=True), 1, 5, 1, 1)
        grid.addWidget(self.size_label, 1, 6, 1, 1)
        # row 3
        grid.addWidget(self.table, 2, 1, 5, 7)
        if not allow_channels:
            self.table.setVisible(allow_channels)
            grid.addWidget(
                hp.make_label(
                    self,
                    "Channels cannot be selected in this app",
                    object_name="header_label",
                    alignment=Qt.AlignmentFlag.AlignCenter,
                ),
                3,
                1,
                5,
                7,
            )

        # set from model
        self._set_from_model()
        self.resolution_label.installEventFilter(self)

    def eventFilter(self, recv, event):
        """Event filter."""
        if event.type() == QEvent.Type.FocusOut and not self.resolution_label.hasFocus():
            if self.resolution_label.text() == "":
                reader = self.get_model()
                self.resolution_label.setText(f"{reader.resolution:.5f}")
            else:
                with hp.qt_signals_blocked(self.resolution_label):
                    value = float(self.resolution_label.text())
                    self.resolution_label.setText(f"{value:.5f}")
        return super().eventFilter(recv, event)

    def get_model(self) -> BaseReader:
        """Get model."""
        parent: QtDatasetList = self._parent  # type: ignore[assignment]
        data_model = parent.data_model
        return data_model.get_reader_for_key(self.key)  # type: ignore[return-value]

    @property
    def transform_model(self) -> TransformModel:
        """Transform model."""
        return self._parent.transform_model

    @property
    def config(self) -> SingleAppConfig:
        """Transform model."""
        return self._parent.config

    @contextmanager
    def editing(self) -> ty.Generator[None, None, None]:
        """Context manager to set editing."""
        self._editing = True
        yield
        self._editing = False

    def _set_from_model(self, _: ty.Any = None) -> None:
        """Update UI elements."""
        reader = self.get_model()
        self.setToolTip(f"<b>Modality</b>: {reader.clean_key}<br><b>Path</b>: {reader.path}")
        # metadata information
        self.modality_icon.state = reader.reader_type
        self.name_label.setText(reader.clean_key)
        self.modality_icon.state = reader.reader_type
        self.resolution_label.setText(f"{reader.resolution:.5f}")
        self.shape_label.setText(format_shape_with_pyramid(reader.shape, reader.n_in_pyramid_quick))
        self.size_label.setText(format_size(reader.shape, reader.dtype))
        self.save_btn.setVisible(reader.reader_type == "image")
        self.extract_btn.setVisible(reader.allow_extraction)
        self.iterate_btn.setVisible(
            self.allow_iterate and reader.reader_type == "image" and len(reader.channel_names) > 1
        )
        self._update_channel_list(reader)

    def _update_channel_list(self, reader: BaseReader) -> None:
        """On load."""
        with self.editing():
            data = []
            for index, channel_name in enumerate(reader.channel_names):
                # checked, channel_id, channel_name, dataset
                data.append([False, index, channel_name])
            self.table.append_data(data)
            self.table.enable_all_check = self.table.row_count() < 20
            self.setMinimumHeight(
                find_nearest_value_in_dict({1: 150, 4: 200, 10: 300}, len(data)) if self.allow_channels else 100,
            )
        logger.trace(f"Updated channel table - {len(data)} rows for {reader.name}.")

    def select_channel(self, channel_name: str, state: bool) -> None:
        """Select channel in table."""
        with self.editing(), hp.qt_signals_blocked(self.table):
            index = self.table.get_row_id(TABLE_CONFIG.channel_name, channel_name)
            if index != -1:
                self.table.set_value(TABLE_CONFIG.check, index, state)

    def channel_list(self) -> list[str]:
        """Channel list."""
        channel_list = []
        for index in self.table.get_all_checked():
            channel_list.append(f"{self.table.get_value(TABLE_CONFIG.channel_name, index)} | {self.key}")
        return channel_list

    @qdebounced(timeout=500, leading=False)
    def on_update_resolution(self) -> None:
        """Update resolution."""
        resolution = self.resolution_label.text()
        if not resolution:
            return
        reader = self.get_model()
        reader.resolution = float(resolution)
        self.evt_resolution.emit(self.key)

    def set_resolution(self, resolution: float) -> None:
        """Update resolution."""
        reader = self.get_model()
        if reader and reader.resolution != resolution:
            reader.resolution = resolution
            self.resolution_label.setText(f"{reader.resolution:.5f}")
            logger.trace(f"Updated pixel size of '{reader.key}' to {resolution:.2f}.")

    def on_remove(self) -> None:
        """Remove image/modality from the list."""
        self.evt_delete.emit(self.key, False)

    def on_force_remove(self) -> None:
        """Remove image/modality from the list."""
        self.evt_delete.emit(self.key, True)

    def on_toggle_channel(self, index: int, state: bool) -> None:
        """Toggle channel."""
        if index == -1:
            reader = self.get_model()
            indices = self.table.get_all_checked()
            channel_names = [f"{reader.channel_names[index]} | {reader.key}" for index in indices]
            self.evt_channel_all.emit(state, channel_names)
        else:
            channel_name = f"{self.table.get_value(TABLE_CONFIG.channel_name, index)} | {self.key}"
            self.evt_channel.emit(state, channel_name)

    def remove_all_channels(self) -> None:
        """Remove all channels."""

    def on_save(self) -> None:
        """Save data."""

        reader = self.get_model()
        if reader:
            from image2image.qt._dialogs._save import ExportImageDialog

            # export images as OME-TIFF
            if reader.reader_type == "image":
                dlg = ExportImageDialog(self, reader, self.config)
                dlg.exec()
            # # export shapes as GeoJSON
            # elif reader.reader_type == "shapes":

    def on_extract(self) -> None:
        """Extract data."""
        from image2image.qt._dialogs import ExtractChannelsDialog

        reader = self.get_model()
        if reader and reader.allow_extraction:
            dlg = ExtractChannelsDialog(self, reader.key)
            key, mzs, ppm = None, None, None
            if dlg.exec_() == QDialog.DialogCode.Accepted:  # type: ignore[attr-defined]
                key = dlg.key_to_extract
                mzs = dlg.mzs
                ppm = dlg.ppm

            logger.trace(f"Extracting data for {key} ({mzs}, {ppm})")
            if key and mzs and ppm:
                logger.trace(f"Extracting data for {key} ({reader}")
                if reader:
                    self.evt_loading.emit()  # noqa
                    create_worker(
                        reader.extract,
                        mzs=mzs,
                        ppm=ppm,
                        _start_thread=True,
                        _connect={
                            # "returned": self._on_update_dataset,
                            "errored": lambda _: hp.toast(
                                hp.get_main_window(), "Failed to extract data.", "Failed to extract data."
                            ),
                        },
                    )

    def on_iterate(self) -> None:
        """Activate iteration data."""
        from image2image.qt._dialogs._iterate import IterateDialog

        dlg = IterateDialog(self)
        dlg.evt_iter_add.connect(self.evt_iter_add.emit)
        dlg.evt_iter_remove.connect(self.evt_iter_remove.emit)
        dlg.evt_iter_next.connect(self.evt_iter_next.emit)
        dlg.show_left_of_mouse()

    def on_transform_menu(self) -> None:
        """Open transform menu."""
        reader = self.get_model()
        if not reader:
            return
        if reader.reader_type == "image":
            menu = hp.make_menu(self)
            hp.make_menu_item(self, "Add transform...", menu=menu, icon="add", func=self._parent.on_add_transform)
            hp.make_menu_item(
                self, "Remove transform...", menu=menu, icon="remove", func=self._parent.on_remove_transform
            )
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
        else:
            options = get_resolution_options(self._parent.wrapper)
            if len(options) > 1:
                menu = hp.make_menu(self)
                for resolution in options.values():
                    hp.make_menu_item(
                        self,
                        f"{resolution:.5f}",
                        menu=menu,
                        func=partial(self.on_select_resolution, resolution),
                        checkable=True,
                        checked=reader.resolution == resolution,
                    )
                hp.show_right_of_mouse(menu)

    def on_select_transform(self, transform_name: str) -> None:
        """Select and apply transform."""
        transform_data = self.transform_model.get_matrix(transform_name)
        reader = self.get_model()
        reader.transform_name = transform_name
        reader.transform_data = deepcopy(transform_data)
        self.evt_transform.emit(self.key)
        logger.trace(f"Updated transformation matrix for '{self.key}'")

    def on_select_resolution(self, resolution: float) -> None:
        """Set resolution."""
        self.resolution_label.setText(f"{resolution:.5f}")

    def on_open_directory(self) -> None:
        """Open directory where the image is located."""
        from koyo.path import open_directory_alt

        model = self.get_model()
        open_directory_alt(model.path)


class QtDatasetList(QScrollArea):
    """List of notifications."""

    evt_delete = Signal(str)
    evt_resolution = Signal(str)
    evt_transform = Signal(str)
    evt_channel = Signal(bool, str)  # channel | dataset
    evt_channel_all = Signal(bool, list)  # list of channel | dataset
    evt_refresh = Signal(str)

    evt_iter_add = Signal(str, int)
    evt_iter_remove = Signal(str, int)
    evt_iter_next = Signal(str, int)

    def __init__(
        self, parent: DatasetDialog, allow_channels: bool, allow_transform: bool, allow_iterate: bool, allow_save: bool
    ):
        self.allow_channels = allow_channels
        self.allow_transform = allow_transform
        self.allow_iterate = allow_iterate
        self.allow_save = allow_save
        # filters
        self._reader_type_filters: list[str] = ["image", "shapes", "points"]
        self._dataset_filters: list[str] = []
        self._channel_filters: str = ""
        self.widgets: dict[str, QtDatasetItem] = {}

        # set parent attributes
        self._parent = parent
        self.view = parent.view
        self.transform_model = parent.transform_model
        self.data_model = parent.model
        self.config = parent.CONFIG

        super().__init__(parent)

        # setup UI
        scroll_widget = QWidget()
        self.setWidget(scroll_widget)
        self._layout = hp.make_v_layout(parent=scroll_widget, spacing=2, margin=1, stretch_after=True)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]

    @property
    def wrapper(self) -> ImageWrapper:
        """Get ImageWrapper."""
        return self.data_model.wrapper

    def widget_iter(self, with_widgets: bool = True) -> ty.Iterable[QtDatasetItem]:
        """Iterate over widgets."""
        yield from self.widgets.values()

    def model_iter(self) -> ty.Iterable[str]:
        """Iterate over models."""
        yield from self.widgets.keys()

    def get_widget_for_key(self, key: str) -> QtDatasetItem:
        """Get widget for specified item model."""
        return self.widgets.get(key)

    def on_make_dataset_item(self, key: str) -> QtDatasetItem:
        """Create item."""
        widget = QtDatasetItem(
            key,
            parent=self,
            allow_iterate=self.allow_iterate,
            allow_transform=self.allow_transform,
            allow_channels=self.allow_channels,
        )  # type: ignore[attr-defined]
        widget.evt_resolution.connect(self.evt_resolution.emit)
        widget.evt_transform.connect(self.evt_transform.emit)
        widget.evt_channel_all.connect(self.evt_channel_all.emit)
        widget.evt_channel.connect(self.evt_channel.emit)
        widget.evt_iter_add.connect(self.evt_iter_add.emit)
        widget.evt_iter_remove.connect(self.evt_iter_remove.emit)
        widget.evt_iter_next.connect(self.evt_iter_next.emit)
        widget.evt_refresh.connect(self.evt_refresh.emit)
        widget.evt_delete.connect(self.on_remove)
        self.widgets[key] = widget
        self._layout.insertWidget(0, widget)
        self.validate()
        return widget

    def on_remove(self, key: str, force: bool = False) -> None:
        """Remove image/modality from the list."""
        model = self.get_widget_for_key(key).get_model()
        if (
            force
            or not model
            or hp.confirm(
                self,
                f"Are you sure you want to remove <b>{model.name}</b> from the list?",
                "Please confirm.",
            )
        ):
            self.evt_delete.emit(key)

    def _check_existing(self, key: str) -> bool:  # type: ignore[override]
        """Check whether model already exists."""
        return any(key_ == key for key_ in self.model_iter())

    def remove_by_key(self, key: str) -> None:
        """Remove model."""
        self._remove_by_key(key)
        self.evt_delete.emit(key)

    def _remove_by_key(self, key: str) -> None:
        widget = self.get_widget_for_key(key)
        if widget:
            self._layout.removeWidget(widget)
        if widget:
            widget.deleteLater()
        self.widgets.pop(key, None)
        del widget

    def set_resolution(self, key: str, resolution: float) -> None:
        """Update resolution"""
        widget: QtDatasetItem = self.get_widget_for_key(key)
        if widget:
            widget.set_resolution(resolution)

    def populate(self) -> None:
        """Create list of items."""
        wrapper = self.wrapper
        if wrapper:
            for _path, reader in wrapper.path_reader_iter():
                if not self._check_existing(reader.key):
                    self.on_make_dataset_item(reader.key)
            keys = list(self.model_iter())
            for key in keys:  # type: ignore[var-annotated]
                reader = wrapper.get_reader_for_key(key)
                if not reader:
                    self._remove_by_key(key)
        self.validate()
        logger.debug("Populated dataset list.")

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

        check_reader = any(self._reader_type_filters)
        check_dataset_name = any(self._dataset_filters)

        for widget in self.widget_iter():
            visible = True
            reader = widget.get_model()
            if not reader:
                continue
            if check_reader and reader.reader_type not in self._reader_type_filters:
                visible = False
            if (
                check_dataset_name
                and visible
                and not any(filter_ in reader.name.lower() for filter_ in self._dataset_filters)
            ):
                visible = False
            widget.setVisible(visible)

    def on_filter_by_channel_name(self, filters: str) -> None:
        """Filter by dataset name."""
        self._channel_filters = filters
        widget: QtDatasetItem
        for widget in self.widget_iter():
            widget.table_proxy.setFilterByColumn(filters, TABLE_CONFIG.channel_name)
            widget.setVisible(widget.table.row_visible_count() != 0)

    def sync_layers(self) -> None:
        """Manually synchronize layers."""
        if not self.view:
            return
        for layer in self.view.layers:
            name = layer.name
            if " | " not in name:
                continue
            channel_name, dataset = name.split(" | ")
            widget = self.get_widget_for_key(dataset)
            if widget:
                widget.select_channel(channel_name, layer.visible)

    # @qdebounced(timeout=500, leading=False)
    def on_sync_layers(self, event: Event) -> None:
        """Synchronize layers."""
        self._sync_layers(event)

    def _sync_layers(self, event: Event) -> None:
        if event.type in ["visible", "inserted"]:
            self._sync_layer_visibility(event)
        elif event.type == "removed":
            self._sync_layer_presence(event)

    def _sync_layer_visibility(self, _event: Event) -> None:
        for layer in self.view.layers:
            name = layer.name
            if " | " not in name:
                continue
            channel_name, dataset = name.split(" | ")
            widget = self.get_widget_for_key(dataset)
            if widget:
                widget.select_channel(channel_name, layer.visible)

    def _sync_layer_presence(self, event: Event) -> None:
        layer = event.value
        name = layer.name
        if " | " not in name:
            return
        channel_name, dataset = name.split(" | ")
        widget = self.get_widget_for_key(dataset)
        if widget:
            widget.select_channel(channel_name, False)

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
            self._on_add_transform(path)

    def _on_add_transform(self, path: str) -> None:
        from image2image.config import get_viewer_config

        # load transformation
        path_ = Path(path)
        get_viewer_config().output_dir = str(path_.parent)

        # load data from config file
        try:
            with MeasureTimer() as timer:
                transform_data = TransformData.from_i2r(path_, validate_paths=False)
            logger.trace(f"Loaded transform data in {timer()}")
        except ValueError as e:
            hp.warn_pretty(self, f"Failed to load transformation from {path_}\n{e}", "Failed to load transformation")
            logger.exception(f"Failed to load transformation from {path_}")
            return
        self.transform_model.add_transform(path_, transform_data)

    def on_remove_transform(self) -> None:
        """Add transform from file."""
        # get all transforms excluding "Identity matrix"
        transforms = self.transform_model.transform_names
        transforms = [transform for transform in transforms if transform != "Identity matrix"]
        if not transforms:
            hp.notification(
                hp.get_main_window(), icon="warning", title="No transforms", message="No transforms to remove."
            )
            return
        choices = hp.choose_from_list(self, transforms, title="Select transforms to remove")
        if choices:
            for transform_name in choices:
                if transform_name == "Identity matrix":
                    continue
                self.transform_model.remove_transform(transform_name)


class QtDatasetToolbar(QtMiniToolbar):
    """Mini toolbar."""

    def __init__(self, parent: DatasetDialog):
        super().__init__(parent, orientation=Qt.Orientation.Horizontal, add_spacer=False, spacing=2)

        # must place them in reverse
        self.add_qta_tool(
            "toggle_on",
            tooltip="Check all visible channels in each dataset.<br>If a reader has more than 10 channels, you will be"
            " asked to confirm.",
            func=self.on_check_all,
        )
        self.add_qta_tool(
            "toggle_off",
            tooltip="Uncheck all visible channels in each dataset.",
            func=self.on_uncheck_all,
        )
        # self.add_qta_tool(
        #     "clear",
        #     tooltip="Remove all layers for each dataset. <b>This does not delete the dataset!</b>",
        #     func=self.on_remove_all,
        # )
        self.add_separator()
        self.add_widget(
            hp.make_toggle(
                self,
                "image",
                "shapes",
                "points",
                func=self.dataset_list.on_filter_by_reader_type,
                tooltip="Filter by type of reader.",
                exclusive=False,
                value=["image", "shapes", "points"],
            )
        )
        self.add_widget(
            hp.make_line_edit(
                self, placeholder="Type in dataset name...", func_changed=self.dataset_list.on_filter_by_dataset_name
            ),
            stretch=True,
        )
        self.add_widget(
            hp.make_line_edit(
                self,
                placeholder="Type in channel name...",
                func_changed=self.dataset_list.on_filter_by_channel_name,
                hide=not parent.allow_channels,
            ),
            stretch=True,
        )

    @property
    def dataset_list(self) -> QtDatasetList:
        """Dataset list."""
        return self.parent()._list

    def on_check_all(self, event: Event) -> None:
        """Check all visible channels."""
        for widget in self.dataset_list.widget_iter():
            if not widget.isVisible():
                continue
            n = widget.table.row_visible_count()
            if n < 10 or hp.confirm(
                self,
                f"Are you sure you want to check all visible channels?<br>"
                f"There are <b>{n}</b> channels in the table which might take a bit of time to load.",
                "Check all channels",
            ):
                widget.table.check_all_rows()

    def on_uncheck_all(self, event: Event) -> None:
        """Uncheck all visible channels."""
        for widget in self.dataset_list.widget_iter():
            if not widget.isVisible():
                continue
            widget.table.uncheck_all_rows()

    def on_remove_all(self, event: Event) -> None:
        """Uncheck all visible channels."""
        for widget in self.dataset_list.widget_iter():
            if not widget.isVisible():
                continue
            widget.remove_all_channels()
