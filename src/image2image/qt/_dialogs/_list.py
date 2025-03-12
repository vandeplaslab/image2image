"""Modality list."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_list_widget import QtListItem, QtListWidget
from qtextra.widgets.qt_table_view_check import QtCheckableTableView
from qtpy.QtCore import QRegularExpression, Qt, Signal, Slot  # type: ignore[attr-defined]
from qtpy.QtGui import QDropEvent, QRegularExpressionValidator
from qtpy.QtWidgets import QDialog, QFormLayout, QGridLayout, QListWidgetItem, QSizePolicy, QWidget
from superqt import ensure_main_thread
from superqt.utils import create_worker

from image2image.config import get_register_config
from image2image.exceptions import MultiSceneCziError, UnsupportedFileFormatError
from image2image.models.transform import TransformData, TransformModel
from image2image.qt._wsi._widgets import QtModalityLabel
from image2image.utils.utilities import extract_extension, format_shape, format_size, log_exception_or_error, open_docs

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader
    from image2image_io.wrapper import ImageWrapper
    from napari.utils.events import Event
    from qtextraplot._napari.image.wrapper import NapariImageView

    from image2image.config import SingleAppConfig
    from image2image.models.data import DataModel

TABLE_CONFIG = (
    TableConfig()  # type: ignore[no-untyped-call]
    .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
    .add("index", "index", "int", 50, sizing="fixed")
    .add("channel name", "channel_name", "str", 125)
    .add("dataset", "dataset", "str", 250)
    .add("key", "key", "str", 0, hidden=True)
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
        parent: QtDatasetList | None = None,
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
        self.allow_channels = allow_iterate

        self.name_label = hp.make_label(self, "", tooltip="Name of the modality.")
        self.resolution_label = hp.make_line_edit(
            self,
            tooltip="Resolution of the modality.",
            func=self._on_update_resolution,
            validator=QRegularExpressionValidator(QRegularExpression(r"^[0-9]+(\.[0-9]{1,3})?$")),
        )
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

        grid = QGridLayout(self)
        # widget, row, column, rowspan, colspan
        grid.setSpacing(1)
        grid.setContentsMargins(0, 0, 0, 0)
        # grid.setColumnStretch(2, True)
        grid.setRowStretch(2, True)

        # column 1
        layout = hp.make_v_layout(margin=1, spacing=1)
        column = 0
        layout.addWidget(self.modality_icon)
        layout.addWidget(self.open_dir_btn)
        layout.addWidget(self.remove_btn)
        layout.addWidget(self.extract_btn)
        layout.addWidget(self.transform_btn)
        layout.addWidget(self.iterate_btn)
        layout.addWidget(self.save_btn)
        layout.addStretch(True)
        grid.addLayout(layout, 0, column, 5, 1)
        # column 2
        column += 1
        grid.addWidget(hp.make_label(self, "Name", bold=True), 0, column, 1, 1)
        grid.addWidget(self.name_label, 0, column + 1, 1, 7)
        grid.addWidget(hp.make_label(self, "Pixel size", bold=True), 1, column, 1, 1)
        grid.addWidget(self.resolution_label, 1, column + 1, 1, 1)
        grid.addWidget(hp.make_label(self, "Shape", bold=True), 1, column + 2, 1, 1)
        grid.addWidget(self.shape_label, 1, column + 3, 1, 1)
        grid.addWidget(hp.make_label(self, "Size", bold=True), 1, column + 4, 1, 1)
        grid.addWidget(self.size_label, 1, column + 5, 1, 1)
        # row 3
        grid.addWidget(self.table, 2, column, 5, 7)

        # set from model
        self._set_from_model()

    def get_model(self) -> BaseReader:
        """Get model."""
        parent: QtDatasetList = self._parent  # type: ignore[assignment]
        data_model = parent.data_model
        return data_model.get_reader_for_key(self.item_model)  # type: ignore[return-value]

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
                data.append([False, index, channel_name, reader.key, f"{channel_name} | {reader.key}"])
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

    def on_remove(self) -> None:
        """Remove image/modality from the list."""
        model = self.get_model()
        if hp.confirm(
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
            self.evt_channel.emit(state, self.table.get_value(TABLE_CONFIG.key, index))

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
        for transform in self._parent.transform_model.transform_names:
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
            self._parent.transform_model.add_transform(path_, transform_data)

    def on_remove_transform(self) -> None:
        """Add transform from file."""
        transforms = self._parent.transform_model.transform_names
        choices = hp.choose_from_list(self, transforms, title="Select transforms to remove")
        if choices:
            for transform_name in choices:
                if transform_name == "Identity matrix":
                    continue
                self._parent.transform_model.remove_transform(transform_name)

    def on_select_transform(self, transform_name: str) -> None:
        """Select and apply transform."""
        transform_data = self._parent.transform_model.get_matrix(transform_name)
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
        self._parent = parent

    @property
    def transform_model(self) -> TransformModel:
        return self._parent.transform_model

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
        for item_model_ in self.model_iter():  # noqa: SIM110; type: ignore[var-annotated]
            if item_model_ == item_model:
                return True
        return False

    def on_remove(self, item_model: str) -> None:
        """Remove model."""
        self.remove_by_item_model(item_model, force=True)
        self.evt_delete.emit(item_model)

    def populate(self) -> None:
        """Create list of items."""
        wrapper = self.wrapper
        if wrapper:
            for _path, reader in wrapper.path_reader_iter():
                if not self._check_existing(reader.key):
                    self.append_item(reader.key)
            for item_model in self.model_iter():
                reader = wrapper.get_reader_for_key(item_model)
                if not reader:
                    self.remove_by_item_model(item_model, force=True)
        logger.debug("Populated modality list.")

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

    @ensure_main_thread
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

    @property
    def data_model(self) -> DataModel:
        """Get registration model."""
        return self._parent.model

    @property
    def wrapper(self) -> ImageWrapper:
        """Get registration model."""
        return self.data_model.wrapper  # type: ignore[return-value]

    @property
    def view(self) -> NapariImageView | None:
        """Image view."""
        return self._parent.view

    @property
    def config(self) -> SingleAppConfig:
        """Get configuration."""
        return self._parent.CONFIG


class DatasetDialog(QtFramelessTool):
    """Dialog window to select images and specify some parameters."""

    HIDE_WHEN_CLOSE = True

    evt_loading = Signal()
    evt_loaded = Signal(object, object)
    evt_loaded_keys = Signal(list)
    evt_closing = Signal(object, list, list)
    evt_closed = Signal(object)
    evt_import_project = Signal(str)
    evt_export_project = Signal()
    evt_files = Signal(list)
    evt_rejected_files = Signal(list)

    # channels
    evt_channel = Signal(bool, str)  # channel | dataset
    evt_channel_all = Signal(bool, list)  # list of channel | dataset
    evt_transform = Signal(str)
    evt_resolution = Signal(str)

    evt_iter_add = Signal(str, int)
    evt_iter_remove = Signal(str, int)
    evt_iter_next = Signal(str, int)

    def __init__(
        self,
        parent: QWidget,
        model: DataModel,
        view: NapariImageView,
        transform_model: TransformModel,
        config: SingleAppConfig,
        is_fixed: bool = False,
        n_max: int = 0,
        allow_geojson: bool = False,
        allow_iterate: bool = False,
        allow_transform: bool = False,
        allow_channels: bool = True,
        available_formats: str | None = None,
        project_extension: list[str] | None = None,
        show_split_czi: bool = True,
    ):
        self.model = model
        self.view = view
        self.transform_model = transform_model
        self.CONFIG = config

        self.is_fixed = is_fixed
        self.allow_geojson = allow_geojson
        self.allow_iterate = allow_iterate
        self.allow_transform = allow_transform
        self.allow_channels = allow_channels
        self.available_formats = available_formats
        self.project_extension = project_extension
        self.show_split_czi = show_split_czi

        super().__init__(parent)
        self.n_max = n_max

        self.setMinimumWidth(600)
        self.setMinimumHeight(800)

        self._list.evt_channel.connect(self.evt_channel.emit)
        self._list.evt_channel_all.connect(self.evt_channel_all.emit)
        self._list.evt_resolution.connect(self.evt_resolution.emit)
        self._list.evt_transform.connect(self.evt_transform.emit)
        self._list.evt_iter_add.connect(self.evt_iter_add.emit)
        self._list.evt_iter_remove.connect(self.evt_iter_remove.emit)
        self._list.evt_iter_next.connect(self.evt_iter_next.emit)
        self._list.evt_delete.connect(self.on_remove_dataset)
        if self.view:
            connect(self.view.layers.events, self._list.on_sync_layers, state=True)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle(title="Modalities")

        self._list = QtDatasetList(self, self.allow_channels, self.allow_transform, self.allow_iterate)

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
        layout.addRow(self._list)
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

    def on_update_config(self, _: ty.Any = None) -> None:
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

    def on_populate_table(self) -> None:
        """Populate table."""
        self._list.populate()
        self._list.sync_layers()

    def on_set_resolution(self, key: str, resolution: float) -> None:
        """Set resolution."""
        # reader = self.model.get_reader_for_key(key)
        # if reader and reader.resolution != resolution:
        #     reader.resolution = resolution
        #     self.evt_resolution.emit(reader.key)
        #     self.on_populate_table()
        #     logger.trace(f"Updated pixel size of '{reader.key}' to {resolution:.2f}.")

    def on_close_dataset(self, force: bool = False) -> bool:
        """Close dataset."""
        from image2image.qt._dialogs._dataset import CloseDatasetDialog

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

        self._on_loaded_dataset(
            self.model.load(
                paths=path_or_paths,
                transform_data=transform_data,
                resolution=resolution,
                reader_kws=reader_kws,
                # _start_thread=True,
                # _connect={
                #     "returned": self._on_loaded_dataset,
                #     "errored": self._on_failed_dataset,
                # },
            )
        )
        # create_worker(
        #     self.model.load,
        #     paths=path_or_paths,
        #     transform_data=transform_data,
        #     resolution=resolution,
        #     reader_kws=reader_kws,
        #     _start_thread=True,
        #     _connect={
        #         "returned": self._on_loaded_dataset,
        #         "errored": self._on_failed_dataset,
        #     },
        # )

    def _on_loaded_dataset(self, model: DataModel, select: bool = True, keys: list[str] | None = None) -> None:
        """Finished loading data."""
        from image2image.qt._dialogs._dataset import SelectChannelsToLoadDialog

        channel_list = []
        wrapper = model.wrapper
        if not keys:
            keys = model.just_added_keys

        if not self.allow_channels or not select:
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
        from image2image.qt._dialogs._dataset import SelectChannelsToLoadDialog

        channel_list = []
        remove_keys = []
        wrapper = model.wrapper

        if not self.allow_channels or not select:
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

    @property
    def available_formats_filter(self) -> str:
        """Return string of available formats."""
        from image2image.enums import ALLOWED_IMAGE_FORMATS, ALLOWED_IMAGE_FORMATS_WITH_GEOJSON

        return self.available_formats or (
            ALLOWED_IMAGE_FORMATS if not self.allow_geojson else ALLOWED_IMAGE_FORMATS_WITH_GEOJSON
        )

    @property
    def allowed_extensions(self) -> list[str]:
        """Return list of available extensions based on the specified filter."""
        return extract_extension(self.available_formats_filter)

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
