"""Windows for dataset management."""

from __future__ import annotations

import typing as ty
from collections import Counter
from pathlib import Path

import numpy as np
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.typing import PathLike
from loguru import logger
from natsort import natsorted
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_table_view_check import MultiColumnSingleValueProxyModel, QtCheckableTableView
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import QCheckBox, QDialog, QFormLayout, QLabel, QVBoxLayout, QWidget
from superqt.utils import create_worker

from image2image.config import SingleAppConfig, get_register_config
from image2image.enums import ALLOWED_IMAGE_FORMATS, ALLOWED_IMAGE_FORMATS_WITH_GEOJSON
from image2image.exceptions import MultiSceneCziError, UnsupportedFileFormatError
from image2image.qt._dialogs._list import QtDatasetList, QtDatasetToolbar
from image2image.utils.utilities import extract_extension, log_exception_or_error, open_docs

if ty.TYPE_CHECKING:
    from qtextraplot._napari.image.wrapper import NapariImageView

    from image2image.models.data import DataModel
    from image2image.models.transform import TransformData, TransformModel


class CloseDatasetDialog(QtDialog):
    """Dialog where user can select which path(s) should be removed."""

    def __init__(self, parent: QWidget, model: DataModel):
        self.checkboxes: list[QCheckBox] = []
        self.labels: list[QLabel] = []
        self.index_to_path: dict[int, str] = {}
        self.model = model
        self.max_length: int = 0
        super().__init__(parent, title="Remove datasets")
        self.keys = self.get_keys()
        self.setMinimumWidth(self.max_length * 7 + 50)
        self.setMinimumHeight(len(self.checkboxes) * 50 + 120)
        self.setMaximumHeight(800)
        self.on_apply()

    def accept(self) -> bool:
        """Accept."""
        self.keys = self.get_keys()
        return super().accept()

    def on_check_all(self, state: bool) -> None:
        """Check all."""
        for checkbox in self.checkboxes:
            if not checkbox.isHidden():
                checkbox.setChecked(state)
        self.on_apply()

    def on_apply(self) -> None:
        """Apply."""
        self.keys = self.get_keys()
        all_checked = len(self.keys) == len(self.checkboxes)
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)  # type: ignore[attr-defined]
        hp.disable_widgets(self.ok_btn, disabled=len(self.keys) == 0)

    def get_keys(self) -> list[str]:
        """Return state."""
        keys = []
        for index, checkbox in enumerate(self.checkboxes):
            if checkbox.isChecked():
                text = self.index_to_path[index]
                keys.append(text.split("\n")[0])
        return keys

    def on_filter(self) -> None:
        """Filter."""
        text = self.filter_by_name.text().lower()
        for index, (checkbox, label) in enumerate(zip(self.checkboxes, self.labels)):
            visible = text in self.index_to_path[index].lower()
            checkbox.setHidden(not visible)
            label.setHidden(not visible)

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
        max_length = 0
        if wrapper:
            for reader in wrapper.reader_iter():
                # make checkbox for each path
                self.index_to_path[len(self.checkboxes)] = f"{reader.key}\n{reader.path}"
                self.checkboxes.append(hp.make_checkbox(scroll_area, value=False, clicked=self.on_apply))
                self.labels.append(hp.make_label(scroll_area, f"<b>{reader.key}</b><br>{reader.path}", enable_url=True))
                scroll_layout.addRow(self.checkboxes[-1], self.labels[-1])
                max_length = max(max_length, len(str(reader.path)))

        self.max_length = max_length
        self.ok_btn = hp.make_btn(self, "OK", func=self.accept)

        layout = hp.make_v_layout(spacing=2, margin=4)
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
        layout.addWidget(hp.make_h_line())
        layout.addWidget(scroll_widget, stretch=True)
        layout.addWidget(self.all_check)
        layout.addWidget(hp.make_h_line_with_text("Filter"))
        layout.addWidget(self.filter_by_name)
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

    def __init__(self, parent: DatasetDialog, model: DataModel):
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

    def on_show_selected(self) -> None:
        """Show/hide selected."""
        self.table_proxy.setFilterByState(True, self.TABLE_CONFIG.check)

    def on_show_unselected(self) -> None:
        """Show/hide unselected."""
        self.table_proxy.setFilterByState(False, self.TABLE_CONFIG.check)

    def on_show_selected_clear(self) -> None:
        """Show/hide unselected."""
        self.table_proxy.setFilterByState(None, self.TABLE_CONFIG.check)

    def on_select_first_channel(self) -> None:
        """Select the first channel only."""
        self.table.uncheck_all_rows()
        for index in range(self.table.n_rows):
            if self.table.get_value(self.TABLE_CONFIG.index, index) == 0:
                self.table.set_value(self.TABLE_CONFIG.check, index, True)

    def on_select_dapi_channel(self) -> None:
        """Select the first channel only."""
        self.table.uncheck_all_rows()
        for index in range(self.table.n_rows):
            if "dapi" in self.table.get_value(self.TABLE_CONFIG.channel_name_full, index).lower():
                self.table.set_value(self.TABLE_CONFIG.check, index, True)

    def on_select_all_channels(self) -> None:
        """Select all channels."""
        self.table.check_all_rows()

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
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(
                    self,
                    "Select all channels",
                    tooltip="Select all channels.",
                    func=self.on_select_all_channels,
                ),
                hp.make_btn(
                    self,
                    "Select first channel only",
                    tooltip="Select the first channel only, often to preview results or speed-up loading of dataset.",
                    func=self.on_select_first_channel,
                ),
                hp.make_btn(
                    self,
                    "Select DAPI channels only",
                    tooltip="Select the DAPI channels only, often to preview results or speed-up loading of dataset.",
                    func=self.on_select_dapi_channel,
                ),
                stretch_after=True,
            )
        )
        layout.addRow(self.table)
        layout.addRow(
            hp.make_h_layout(
                hp.make_qta_btn(self, "visible_on", func=self.on_show_selected, tooltip="Only show checked items."),
                hp.make_qta_btn(
                    self, "visible_off", func=self.on_show_unselected, tooltip="Only show unchecked items."
                ),
                hp.make_qta_btn(self, "clear", func=self.on_show_selected_clear, tooltip="Clear checked filter."),
                self.filter_by_name,
                stretch_id=(3,),
                spacing=2,
            )
        )
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


class DatasetDialog(QtDialog):
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
        allow_save: bool = True,
        confirm_czi: bool = False,
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
        self.allow_save = allow_save
        self.allow_channels = allow_channels
        self.available_formats = available_formats
        self.project_extension = project_extension
        self.show_split_czi = show_split_czi
        self.confirm_czi = confirm_czi
        self.n_max = n_max
        super().__init__(parent, title="Datasets")

        self.setMinimumWidth(600)
        self.setMinimumHeight(800)
        self.setAcceptDrops(True)

        # add signals
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
        self._list = QtDatasetList(self, self.allow_channels, self.allow_transform, self.allow_iterate, self.allow_save)
        self._toolbar = QtDatasetToolbar(self)

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
        layout.addRow(hp.make_label(self, "How to load image data", bold=True))
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "Split CZI (recommended)", hide=not self.show_split_czi),
                self.split_czi_check,
                hp.make_v_line(hide=not self.show_split_czi),
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
        layout.addRow(hp.make_h_line_with_text("Actions"))
        layout.addRow(self._toolbar)
        layout.addRow(self._list)
        layout.addRow(
            hp.make_h_layout(
                hp.make_url_btn(self, func=lambda: open_docs(dialog="dataset-metadata")),
                spacing=2,
                margin=2,
                alignment=Qt.AlignmentFlag.AlignVCenter,
                stretch_before=True,
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
        self._list.set_resolution(key, resolution)

    def _on_add_transform(self, path: str) -> None:
        """Set resolution."""
        self._list._on_add_transform(path)

    def channel_list(self) -> list[str]:
        """Get list of visible channels."""
        channels = []
        for widget in self._list.widget_iter():
            channels.extend(widget.channel_list())
        return channels

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
        return False

    def _on_load_dataset(
        self,
        path_or_paths: PathLike | ty.Sequence[PathLike] | ty.Sequence[dict[str, ty.Any]],
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

        if not self.allow_channels or not select:
            if wrapper:
                channel_list = wrapper.channel_names_for_names(keys)
        else:
            if wrapper:
                channel_list_ = list(wrapper.channel_names_for_names(keys))
                if channel_list_:
                    dlg = SelectChannelsToLoadDialog(self, model)
                    dlg.show_in_center_of_screen()
                    dlg.raise_()
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
                    dlg = SelectChannelsToLoadDialog(self, model)  # type: ignore[assignment]
                    dlg.show_in_center_of_screen()
                    dlg.raise_()
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
        kept_filenames = []
        confirm_filenames = []
        rejected_filenames = []
        for filename in filenames:
            if self.project_extension and any(filename.endswith(ext) for ext in self.project_extension):
                self.evt_import_project.emit(filename)
            elif filename.endswith(allowed_extensions):
                if self.confirm_czi and filename.endswith(".czi"):
                    confirm_filenames.append(filename)
                else:
                    kept_filenames.append(filename)
            else:
                rejected_filenames.append(filename)

        self._handle_filenames(kept_filenames)
        self._handle_confirm_czi_filenames(confirm_filenames)

        if rejected_filenames:
            self.evt_rejected_files.emit(rejected_filenames)

    def _handle_filenames(self, kept_filenames: list[str]) -> None:
        """Handle filenames."""
        if not kept_filenames:
            return
        logger.trace(f"Dropped {kept_filenames} file(s)...")
        self.evt_files.emit(kept_filenames)
        self._on_load_dataset(kept_filenames)

    def _handle_confirm_czi_filenames(self, confirm_filenames: list[str]) -> None:
        """Handle filenames."""
        from image2image_io.readers import get_czi_metadata
        from qtextra.widgets.qt_select_one import QtScrollablePickOption

        if not confirm_filenames:
            return
        logger.trace(f"Dropped {confirm_filenames} CZI file(s)...")
        kept_filenames: list[dict] = []
        for path in confirm_filenames:
            metadata = get_czi_metadata(path)
            if len(metadata) == 1:
                kept_filenames.append(path)
            else:
                dlg = QtScrollablePickOption(
                    self,
                    "Please select the <b>CZI scene</b> you would like to load.<br>Only a <b>single</b> scene can"
                    " be loaded from a multi-scene CZI file in a single project.",
                    metadata,
                    orientation="vertical",
                    max_width=min(600, self.width() - 100),
                )
                hp.show_in_center_of_screen(dlg)
                if dlg.exec_() == QDialog.DialogCode.Accepted:
                    kept_filenames.append(dlg.option)
        if not kept_filenames:
            return
        self.evt_files.emit(kept_filenames)
        self._on_load_dataset(kept_filenames)

    @property
    def allow_drop(self) -> bool:
        """Return if drop is allowed."""
        return hp.get_main_window().allow_drop

    def dragEnterEvent(self, event):
        """Override Qt method. Provide style updates on event."""
        if self.allow_drop:
            hp.update_property(self, "drag", True)
            hp.call_later(self, lambda: hp.update_property(self, "drag", False), 2000)
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()

    def dragLeaveEvent(self, event):
        """Override Qt method."""
        if self.allow_drop:
            hp.update_property(self, "drag", False)

    def dropEvent(self, event):
        """Override Qt method."""
        if self.allow_drop:
            hp.update_property(self, "drag", False)
            self.on_drop(event)
