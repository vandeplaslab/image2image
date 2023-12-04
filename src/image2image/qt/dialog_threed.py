"""Three-D dialog."""
from __future__ import annotations

import typing as ty
from contextlib import suppress
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from image2image_reader.config import CONFIG as READER_CONFIG
from image2image_reader.readers import BaseReader
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Shapes
from natsort import natsorted
from PyQt6.QtWidgets import QSizePolicy
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtextra.widgets.qt_image_button import QtThemeButton
from qtextra.widgets.qt_mini_toolbar import QtMiniToolbar
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QMenuBar,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleRangeSlider, ensure_main_thread
from superqt.utils import qdebounced
from tqdm import tqdm

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_IMAGE_FORMATS_TIFF_ONLY, ALLOWED_THREED_FORMATS
from image2image.models.threed import Registration, RegistrationImage, as_icon
from image2image.qt._select import LoadWidget
from image2image.qt.dialog_base import Window
from image2image.utils.utilities import (
    ensure_extension,
    format_group_info,
    get_contrast_limits,
    get_groups,
    groups_to_group_id,
    write_project,
)

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


class ImageThreeDWindow(Window):
    """Image viewer dialog."""

    image_layer: list[Image] | None = None
    shape_layer: list[Shapes] | None = None
    _console = None
    _editing = False

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed", checkable=True)
        .add("dataset", "key", "str", 250, sizing="stretch")
        .add(
            "keep",
            "keep",
            "bool",
            55,
            sizing="fixed",
            tooltip="Keep/remove image from the processing pipeline.",
            checkable=True,
        )
        .add(
            "lock",
            "lock",
            "bool",
            55,
            sizing="fixed",
            tooltip="Lock/unlock image. If locked, no changes will be applied.",
            checkable=True,
        )
        .add("ref", "reference", "icon", 50, sizing="fixed", tooltip="Is reference.")
        .add("group`", "group_id", "int", 65, sizing="fixed", tooltip="Group ID for images that should be grouped.")
        .add("order", "image_order", "int", 60, sizing="fixed", tooltip="Order in which images will be processed.")
        .add("rotate", "rotate", "int", 70, sizing="fixed", tooltip="Rotate image by N degrees.")
        .add("translate(x)", "translate_x", "int", 85, sizing="fixed", tooltip="Translate image by N microns.")
        .add("translate(y)", "translate_y", "int", 82, sizing="fixed", tooltip="Translate image by N microns.")
        .add("flip(l-r)", "flip_lr", "bool", 70, sizing="fixed", tooltip="Flip image left-right.", checkable=True)
    )

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, f"image2threed: Co-register 3D data from 2D sections app (v{__version__})")
        self.registration = Registration()  # type: ignore[call-arg]
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closed, self.on_remove_image, state=state)
        connect(self.view.widget.canvas.events.key_press, self.keyPressEvent, state=state)
        connect(self.table.evt_keypress, self.keyPressEvent, state=state)
        connect(self.table.evt_value_checked, self.on_table_updated, state=state)
        connect(self.table.selectionModel().selectionChanged, qdebounced(self.on_highlight, 50), state=state)
        connect(self.contrast_limit.valueChanged, self.on_contrast_limits, state=state)

    def on_remove_image(self, model: DataModel) -> None:
        """Remove image."""
        if model:
            self.on_depopulate_table()
        else:
            logger.warning(f"Failed to remove data - model={model}")

    def on_populate_table(self) -> None:
        """Remove items that are not present in the model."""
        wrapper = self.data_model.wrapper
        if wrapper:
            data = []
            for reader in wrapper.reader_iter():
                if reader.key not in self.registration.images:
                    self.registration.images[reader.key] = RegistrationImage.from_reader(reader)
                data.append(self.registration.images[reader.key].to_table())
            # update table
            model_index = self.table.selectionModel().currentIndex()
            self.table.reset_data()
            self.table.add_data(data)
            if model_index.isValid() and model_index.row() < self.table.n_rows:
                self.table.scrollTo(model_index)
            self.on_plot()

    def on_depopulate_table(self) -> None:
        """Remove items that are not present in the model."""
        to_remove = []
        for index in range(self.table.n_rows):
            key = self.table.get_value(self.TABLE_CONFIG.key, index)
            if not self.data_model.has_key(key):
                to_remove.append(index)
                with suppress(ValueError):
                    self.view.viewer.layers.remove(key)
        for index in reversed(to_remove):
            self.table.remove_row(index)

    def on_contrast_limits(self, _value: bool | tuple[float, float] | None = None) -> None:
        """Set contrast limits."""
        common = self.common_contrast_limit.isChecked()
        contrast_limits: tuple[float, float] = self.contrast_limit.value()  # type: ignore[no-untyped-call]
        if common:
            for layer in self.view.layers:
                layer.contrast_limits = contrast_limits
        else:
            for layer in self.view.layers:
                layer.contrast_limits = layer.metadata["contrast_limits"]

    def on_plot(self) -> None:
        """Add images."""
        # with MeasureTimer() as timer:
        #     wrapper = self.data_model.wrapper
        #     if wrapper:
        #         for key in self.registration.key_iter():
        #             reader = wrapper.data[key]
        #             model = self.registration.images[reader.key]
        #             if reader.key not in self.view.layers:
        #                 image = reader.get_channel(0, -1)
        #                 contrast_limits, _ = get_contrast_limits([image])
        #                 if contrast_limits:
        #                     contrast_limits = (contrast_limits[0], contrast_limits[1] / 2)
        #                 self.view.add_image(
        #                     image,
        #                     name=reader.key,
        #                     scale=model.scale,
        #                     blending="additive",
        #                     contrast_limits=contrast_limits,
        #                     affine=model.affine(image.shape),
        #                 )
        #         self.on_select()
        # logger.trace(f"Plotted images in {timer()}")

    def on_plot_selected(self) -> None:
        """Plot currently selected images."""
        with MeasureTimer():
            wrapper = self.data_model.wrapper
            if not wrapper:
                return
            ref_key = self.registration.reference
            all_contrast_range: list[float] = []
            if ref_key:
                reader = wrapper.data[ref_key]
                model = self.registration.images[reader.key]
                contrast_range = self._plot_model(reader, model)
                if contrast_range:
                    all_contrast_range.extend(contrast_range)
            checked = self.table.get_all_checked()

            if len(checked) < 48 or hp.confirm(
                self,
                f"There is more  than 48 images selected ({len(checked)}). Would you like to continue?",
                "There are many images....",
            ):
                # if len(checked) < 48:
                for index in tqdm(checked, desc="Plotting images..."):
                    key = self.table.get_value(self.TABLE_CONFIG.key, index)
                    reader = wrapper.data[key]
                    model = self.registration.images[reader.key]
                    if not model.keep:
                        logger.info(f"Skipping '{key}' as it is marked for removal.")
                        continue
                    contrast_range = self._plot_model(reader, model)
                    if contrast_range:
                        all_contrast_range.extend(contrast_range)
                if all_contrast_range:
                    self.contrast_limit.setRange(min(all_contrast_range), max(all_contrast_range))
                    self.contrast_limit.setDecimals(2 if max(all_contrast_range) < 1 else 0)
            # else:
            #     hp.toast(self, "Too many images selected", "Please select less than 24 images.", icon="error")
            self.on_contrast_limits()

    def _plot_model(self, reader: BaseReader, model: RegistrationImage) -> tuple[float, float] | None:
        image = reader.get_channel(0, -1)
        contrast_limits, contrast_limits_range = get_contrast_limits([image])
        if contrast_limits:
            contrast_limits = (contrast_limits[0], contrast_limits[1] / 2)
        layer = self.view.add_image(
            image,
            name=reader.key,
            scale=model.scale,
            blending="additive",
            contrast_limits=contrast_limits,
            affine=model.affine(image.shape),  # type: ignore[arg-type]
            metadata={
                "key": reader.key,
                "contrast_limits_range": contrast_limits_range,
                "contrast_limits": contrast_limits,
            },
        )
        if contrast_limits_range:
            layer.contrast_limits_range = contrast_limits_range
        return contrast_limits

    def on_select(self) -> None:
        """Select image."""
        with MeasureTimer() as timer:
            checked = self.table.get_all_checked()
            for index in range(self.table.n_rows):
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                is_reference = key == self.registration.reference
                cmap = "red" if is_reference else ("magenta" if index in checked else "cyan")
                if key in self.view.layers:
                    self.view.layers[key].colormap = cmap
        logger.trace(f"Selected images in {timer()}")

    def on_highlight(self, *args: ty.Any) -> None:
        """Highlight specific image."""
        if self.table.selectionModel().hasSelection():
            self.on_select()
            with MeasureTimer() as timer:
                index = self.table.selectionModel().currentIndex().row()
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                if key in self.view.layers:
                    self.view.layers[key].colormap = "yellow"
            logger.trace(f"Highlighted image in {timer()}")

    def on_table_updated(self, row: int, column: int, value: bool) -> None:
        """State was changed for specified row and column."""
        key = self.table.get_value(self.TABLE_CONFIG.key, row)
        if column == self.TABLE_CONFIG.keep:
            self.registration.images[key].keep = value
        elif column == self.TABLE_CONFIG.lock:
            self.registration.images[key].lock = value
        elif column == self.TABLE_CONFIG.reference:
            self.registration.reference = key
        elif column == self.TABLE_CONFIG.flip_lr:
            self.registration.images[key].flip_lr = value
            self._transform(self.registration.images[key])

    def _transform(self, model: RegistrationImage) -> None:
        """Apply transformation to the specified model."""
        key = model.key
        if key in self.view.layers:
            self.view.layers[key].affine = model.affine(self.view.layers[key].data.shape)

    def on_rotate(self, which: str) -> None:
        """Rotate image."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            # values = self.table.get_col_data(self.TABLE_CONFIG.rotate)
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                if model.is_reference:
                    continue
                model.apply_rotate(which)
                # values[index] = model.rotate
                self.table.set_value(self.TABLE_CONFIG.rotate, index, self.registration.images[key].rotate)
                self._transform(model)
            # with MeasureTimer():
            #     self.table.update_column(self.TABLE_CONFIG.rotate, values)
        logger.trace(f"Rotate {which} '{len(checked)}' images in {timer()}")

    def on_translate(self, which: str) -> None:
        """Translate image."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                if model.is_reference:
                    continue
                model.apply_translate(which)
                if which in ["up", "down"]:
                    self.table.set_value(
                        self.TABLE_CONFIG.translate_y, index, self.registration.images[key].translate_y
                    )
                else:
                    self.table.set_value(
                        self.TABLE_CONFIG.translate_x, index, self.registration.images[key].translate_x
                    )
                self._transform(model)
        logger.trace(f"Translated {which} '{len(checked)}' images in {timer()}")

    def on_flip_lr(self) -> None:
        """Flip image."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                if model.is_reference:
                    continue
                model.flip_lr = not model.flip_lr
                self.table.set_value(self.TABLE_CONFIG.flip_lr, index, model.flip_lr)
                self._transform(model)
        logger.trace(f"Flipped '{len(checked)}' images in {timer()}")

    def on_group(self, group: bool) -> None:
        """Group images."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            groups = self.table.get_col_data(self.TABLE_CONFIG.group_id)
            group_id = max(groups) + 1 if groups else 1
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                model.group_id = group_id if group else 0
                self.table.set_value(self.TABLE_CONFIG.group_id, index, model.group_id)
            self._generate_group_options()
        logger.trace(f"{'Grouped' if group else 'Ungrouped'} '{len(checked)}' images in {timer()}")

    def on_lock(self, lock: bool) -> None:
        """Keep/remove images."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                model.lock = lock
                self.table.set_value(self.TABLE_CONFIG.lock, index, model.lock)
        logger.trace(f"{'Locked' if lock else 'Unlocked'} '{len(checked)}' images in {timer()}")

    def on_keep(self, keep: bool) -> None:
        """Keep/remove images."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                model.keep = keep
                self.table.set_value(self.TABLE_CONFIG.keep, index, model.keep)
        logger.trace(f"{'Kept' if keep else 'Removed'} '{len(checked)}' images in {timer()}")

    def _get_groups(self) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Return mapping between image key and group ID."""
        value = self.group_by.text()
        if value:
            filenames: list[str] = self.table.get_col_data(self.TABLE_CONFIG.key)
            groups = get_groups(filenames, value)
            dataset_to_group_map = groups_to_group_id(groups)
            return groups, dataset_to_group_map
        return {}, {}

    def on_preview_group_by(self):
        """Preview groups specified by user."""
        from qtextra.widgets.qt_info_popup import InfoDialog

        groups, dataset_to_group_map = self._get_groups()
        if dataset_to_group_map:
            ret = format_group_info(groups)
            dlg = InfoDialog(self, ret)
            dlg.setMinimumWidth(600)
            dlg.setMinimumHeight(600)
            dlg.show()

    def on_group_by(self):
        """Group by user specified string."""
        groups, dataset_to_group_map = self._get_groups()
        if (
            groups
            and self.registration.is_grouped()
            and not hp.confirm(
                self,
                "Are you sure you wish to group images? This will overwrite any previous grouping.",
                "Are you sure?",
            )
        ):
            return
        with MeasureTimer() as timer, self.table.block_model():
            values = []
            if dataset_to_group_map:
                for key, group_id in tqdm(
                    dataset_to_group_map.items(), desc="Grouping images...", total=self.table.n_rows
                ):
                    self.registration.images[key].group_id = group_id
                    values.append(group_id)
                self.table.update_column(self.TABLE_CONFIG.group_id, values)
            self._generate_group_options()
        logger.trace(f"Grouped images in {timer()}")

    def _generate_group_options(self) -> None:
        """Update combo box with group options including None and All."""
        labels = []
        for model in self.registration.model_iter():
            if f"Group '{model.group_id}'" not in labels:
                labels.append(f"Group '{model.group_id}'")
        labels = ["None", "All", *natsorted(labels)]

        current = self.groups_choice.currentText()
        with hp.qt_signals_blocked(self.groups_choice):
            self.groups_choice.clear()
            hp.set_combobox_text_data(self.groups_choice, labels, current)

    def on_order_by(self):
        """Reorder images according to the current group identification."""
        if self.registration.is_ordered() and not hp.confirm(
            self,
            "Are you sure you wish to reorder images? This will overwrite any previous ordering.",
            "Are you sure?",
        ):
            return
        with MeasureTimer() as timer, self.table.block_model():
            self.registration.reorder()
            layers, values = [], []
            for key in tqdm(self.registration.key_iter(), desc="Reordering images...", total=self.table.n_rows):
                values.append(self.registration.images[key].image_order)
                if key in self.view.layers:
                    layers.append(self.view.layers.pop(key))
            self.table.update_column(self.TABLE_CONFIG.image_order, values)
            for layer in layers:
                self.view.viewer.add_layer(layer)
            self.view.viewer.reset_view()
        logger.trace(f"Reordered images in {timer()}")

    def _get_for_group(self) -> tuple[list[str], list[bool]]:
        """Get all for group."""
        group = self.groups_choice.currentText()
        in_group, values = [], []
        if group == "None":
            values = [False] * self.table.n_rows
        elif group == "All":
            values = [True] * self.table.n_rows
            in_group = self.table.get_col_data(self.TABLE_CONFIG.key)
        else:
            group_id = int(group.split("'")[1])
            for index in range(self.table.n_rows):
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                values.append(model.group_id == group_id)
                in_group.append(model.key)
        return in_group, values

    def on_select_group(self):
        """Select group."""
        _, values = self._get_for_group()
        self.table.update_column(self.TABLE_CONFIG.check, values)
        # clear canvas and plot
        self.view.viewer.layers.clear()
        self.on_plot_selected()
        self.on_select()
        self.view.viewer.reset_view()

    def on_check_group(self, check: bool) -> None:
        """Show or hide group images."""
        _, values = self._get_for_group()
        if not check:
            values = [False] * self.table.n_rows
        self.table.update_column(self.TABLE_CONFIG.check, values)

    def on_show_group(self, show: bool) -> None:
        """Show or hide group images."""
        checked = self.table.get_all_checked()
        for index in checked:
            key = self.table.get_value(self.TABLE_CONFIG.key, index)
            if key in self.view.layers:
                self.view.layers[key].visible = show

    def on_reference(self) -> None:
        """Set the currently selected row as a reference."""
        if self.table.selectionModel().hasSelection():
            with MeasureTimer() as timer, self.table.block_model():
                index = self.table.selectionModel().currentIndex().row()
                key_ = self.table.get_value(self.TABLE_CONFIG.key, index)
                if not hp.confirm(self, f"Set <b>{key_}</b> as the reference image?", "Set reference"):
                    return
                self.registration.reference = key_
                for row in range(self.table.n_rows):
                    key = self.table.get_value(self.TABLE_CONFIG.key, row)
                    self.registration.images[key].is_reference = row == index
                    self.table.set_value(self.TABLE_CONFIG.reference, row, as_icon(row == index))
                self.on_select()
            logger.trace(f"Set {key_} as reference in {timer()}")

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

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(
            self, add_toolbars=False, allow_extraction=False, disable_controls=True, disable_new_layers=True
        )

        self._image_widget = LoadWidget(
            self, self.view, select_channels=False, available_formats=ALLOWED_IMAGE_FORMATS_TIFF_ONLY
        )
        self.import_project_btn = hp.make_btn(
            self, "Import project...", tooltip="Load previous project", func=self.on_load_from_project
        )
        self.table = QtCheckableTableView(
            self,
            config=self.TABLE_CONFIG,
            enable_all_check=True,
            sortable=True,
            drag=True,
        )
        hp.set_font(self.table)
        self.table.setup_model(
            header=self.TABLE_CONFIG.header,
            no_sort_columns=self.TABLE_CONFIG.no_sort_columns,
            hidden_columns=self.TABLE_CONFIG.hidden_columns,
            checkable_columns=self.TABLE_CONFIG.checkable_columns,
        )

        self.toolbar = QtMiniToolbar(self, orientation=Qt.Orientation.Vertical, add_spacer=False)
        self.toolbar.add_qta_tool(
            "rotate_left", tooltip="Rotate image left (E)", func=lambda: self.on_rotate("left"), small=True
        )
        self.toolbar.add_qta_tool(
            "rotate_right", tooltip="Rotate image right (Q)", func=lambda: self.on_rotate("right"), small=True
        )
        self.toolbar.add_qta_tool(
            "translate_up", tooltip="Translate image up (W)", func=lambda: self.on_translate("up"), small=True
        )
        self.toolbar.add_qta_tool(
            "translate_down", tooltip="Translate image down (S)", func=lambda: self.on_translate("down"), small=True
        )
        self.toolbar.add_qta_tool(
            "translate_left", tooltip="Translate image left (A)", func=lambda: self.on_translate("left"), small=True
        )
        self.toolbar.add_qta_tool(
            "translate_right", tooltip="Translate image right (D)", func=lambda: self.on_translate("right"), small=True
        )
        self.toolbar.add_qta_tool("flip_lr", tooltip="Flip image left-right (F)", func=self.on_flip_lr, small=True)
        self.toolbar.add_qta_tool("group", tooltip="Group images", func=lambda x: self.on_group(True), small=True)
        self.toolbar.add_qta_tool("ungroup", tooltip="Ungroup images", func=lambda x: self.on_group(False), small=True)
        self.toolbar.add_qta_tool("lock_open", tooltip="Lock images (L)", func=lambda x: self.on_lock(True), small=True)
        self.toolbar.add_qta_tool(
            "lock_closed", tooltip="Unlock images (U)", func=lambda x: self.on_lock(False), small=True
        )
        self.toolbar.add_qta_tool(
            "keep_image", tooltip="Keep images (Z)", func=lambda x: self.on_keep(True), small=True
        )
        self.toolbar.add_qta_tool(
            "remove_image", tooltip="Remove images (X)", func=lambda x: self.on_keep(False), small=True
        )
        self.toolbar.append_spacer()

        self.export_project_btn = hp.make_btn(
            self,
            "Export project...",
            tooltip="Export configuration to a project file. Information such as image path and crop"
            " information are saved. (This does not save the cropped image)",
            func=self.on_save_to_project,
        )

        self.group_by = hp.make_line_edit(self, placeholder="Type in part of the filename")
        # self.group_by.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview_group_by_btn = hp.make_btn(self, "Preview", func=self.on_preview_group_by)
        self.group_by_btn = hp.make_btn(self, "Group", func=self.on_group_by)
        self.reorder_btn = hp.make_btn(self, "Reorder", func=self.on_order_by)

        self.groups_choice = hp.make_combobox(self, tooltip="Select group to show", func=self.on_select_group)
        self.show_group_btn = hp.make_btn(
            self,
            "Check group",
            tooltip="Only show the selected group (+ reference image)",
            func=lambda: self.on_check_group(True),
            # func=lambda: self.on_show_group(True),
        )
        self.hide_group_btn = hp.make_btn(
            self,
            "Uncheck group",
            tooltip="Hide the selected image",
            func=lambda: self.on_check_group(False),
            # func=lambda: self.on_show_group(False),
        )

        side_widget = QWidget()  # noqa
        side_widget.setMinimumWidth(300)
        side_layout = hp.make_form_layout(side_widget)
        side_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        hp.style_form_layout(side_layout)
        side_layout.addRow(self.import_project_btn)
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(self.export_project_btn)
        side_layout.addRow(hp.make_h_line_with_text("Group by part of the filename"))
        side_layout.addRow(self.group_by)
        side_layout.addRow(hp.make_h_layout(self.preview_group_by_btn, self.group_by_btn, self.reorder_btn))
        side_layout.addRow(hp.make_h_line_with_text("Select group"))
        side_layout.addRow("Group", self.groups_choice)
        side_layout.addRow(hp.make_h_layout(self.show_group_btn, self.hide_group_btn))

        bottom_widget = QWidget()  # noqa
        bottom_widget.setMinimumHeight(400)
        bottom_widget.setMaximumHeight(600)
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setSpacing(2)
        bottom_layout.addWidget(side_widget)
        bottom_layout.addWidget(self.toolbar)
        bottom_layout.addWidget(self.table, stretch=True)

        layout = QVBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.view.widget, stretch=True)
        layout.addWidget(bottom_widget)

        widget = QWidget()  # noqa
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.addLayout(layout)

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
            "Add image (.tiff, + others)...",
            "Ctrl+I",
            menu=menu_file,
            func=self._image_widget.on_select_dataset,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(self, "Show scale bar controls...", "Ctrl+S", menu=menu_tools, func=self.on_show_scalebar)
        menu_tools.addSeparator()
        hp.make_menu_item(self, "Show IPython console...", "Ctrl+T", menu=menu_tools, func=self.on_show_console)

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def keyPressEvent(self, evt: ty.Any) -> None:
        """Key press event."""
        if hasattr(evt, "native"):
            evt = evt.native
        key = evt.key()
        if key == Qt.Key.Key_Escape:
            evt.ignore()
        # rotate
        elif key == Qt.Key.Key_Q:
            self.on_rotate("left")
        elif key == Qt.Key.Key_E:
            self.on_rotate("right")
        elif key == Qt.Key.Key_A:
            # translate
            self.on_translate("left")
        elif key == Qt.Key.Key_D:
            self.on_translate("right")
        elif key == Qt.Key.Key_W:
            self.on_translate("up")
        elif key == Qt.Key.Key_S:
            self.on_translate("down")
        # flip left-right
        elif key == Qt.Key.Key_F:
            self.on_flip_lr()
        # # group/ungroup
        # elif key == Qt.Key.Key_G:
        #     self.on_group(True)
        # elif key == Qt.Key.Key_U:
        #     self.on_group(False)
        # reference
        elif key == Qt.Key.Key_R:
            self.on_reference()
        # keep/remove
        elif key == Qt.Key.Key_Z:
            self.on_keep(True)
        elif key == Qt.Key.Key_X:
            self.on_keep(False)
        # lock/unlock
        elif key == Qt.Key.Key_L:
            self.on_lock(True)
        elif key == Qt.Key.Key_U:
            self.on_lock(False)
        # viewer
        elif key == Qt.Key.Key_G:
            self.grid_btn.click()
        else:
            super().keyPressEvent(evt)

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update(
            {"viewer": self.view.viewer, "data_model": self.data_model, "wrapper": self.data_model.wrapper}
        )
        return variables

    def on_show_scalebar(self) -> None:
        """Show scale bar controls for the viewer."""
        from image2image.qt._dialogs._scalebar import QtScaleBarControls

        dlg = QtScaleBarControls(self.view.viewer, self.view.widget)
        dlg.show_above_widget(self.scalebar_btn)

    def on_show_save_figure(self) -> None:
        """Show scale bar controls for the viewer."""
        from image2image.qt._dialogs._screenshot import QtScreenshotDialog

        dlg = QtScreenshotDialog(self.view, self)
        dlg.show_above_widget(self.clipboard_btn)

    def on_set_config(self):
        """Update config."""
        CONFIG.rotate_step_size = int(self.rotate_step_size.currentText().split(" ")[-1].split("°")[0])
        CONFIG.translate_step_size = int(self.translate_step_size.currentText().split(" ")[2])

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        from qtextra._napari.image.components._viewer_key_bindings import toggle_grid

        from image2image.qt._sentry import send_feedback

        self.statusbar = QStatusBar()  # noqa
        self.statusbar.setSizeGripEnabled(False)

        self.common_contrast_limit = hp.make_checkbox(
            self, "Common intensity", tooltip="Use common contrast limit for all images", func=self.on_contrast_limits
        )
        hp.set_sizer_policy(self.common_contrast_limit, h_stretch=False)
        self.statusbar.addPermanentWidget(self.common_contrast_limit)
        self.contrast_limit = QLabeledDoubleRangeSlider()
        self.contrast_limit.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.LabelIsValue)
        self.contrast_limit.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        self.contrast_limit.setOrientation(Qt.Orientation.Horizontal)  # type: ignore[no-untyped-call]
        self.contrast_limit.setRange(0, 1)
        self.contrast_limit.setValue((0, 1))
        self.contrast_limit.setDecimals(0)
        self.contrast_limit.layout().setSpacing(1)
        self.statusbar.addPermanentWidget(self.contrast_limit)
        self.statusbar.addPermanentWidget(hp.make_v_line())

        self.rotate_step_size = hp.make_combobox(
            self,
            [
                "Rotate by 5°",
                "Rotate by 10°",
                "Rotate by 15°",
                "Rotate by 30°",
                "Rotate by 45°",
                "Rotate by 60°",
                "Rotate by 90°",
            ],
            func=self.on_set_config,
            value=f"{CONFIG.rotate_step_size}°",
        )
        self.rotate_step_size.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.statusbar.addPermanentWidget(self.rotate_step_size)
        self.statusbar.addPermanentWidget(hp.make_v_line())

        self.translate_step_size = hp.make_combobox(
            self,
            ["Move by 50 µm", "Move by 100 µm°", "Move by 250 µm", "Move by 500 µm", "Move by 1000 µm"],
            func=self.on_set_config,
            value=f"{CONFIG.translate_step_size} µm",
        )

        self.translate_step_size.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.statusbar.addPermanentWidget(self.translate_step_size)
        self.statusbar.addPermanentWidget(hp.make_v_line())

        self.grid_btn = hp.make_qta_btn(
            self,
            "grid_off",
            "Toggle grid view. Right-click on the button to change grid settings.",
            checkable=True,
            checked=self.view.viewer.grid.enabled,
            checked_icon_name="grid_on",
            func=lambda: toggle_grid(self.view.viewer),
        )
        self.statusbar.addPermanentWidget(self.grid_btn)

        self.screenshot_btn = hp.make_qta_btn(
            self,
            "save",
            tooltip="Save snapshot of the canvas to file. Right-click to show dialog with more options.",
            func=self.view.widget.on_save_figure,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.screenshot_btn)
        self.clipboard_btn = hp.make_qta_btn(
            self,
            "screenshot",
            tooltip="Take a snapshot of the canvas and copy it into your clipboard. Right-click to show dialog with"
            " more options.",
            func=self.view.widget.clipboard,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.clipboard_btn)
        self.scalebar_btn = hp.make_qta_btn(
            self,
            "ruler",
            tooltip="Show scalebar.",
            func=self.on_show_scalebar,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.scalebar_btn)

        self.feedback_btn = hp.make_qta_btn(
            self,
            "feedback",
            tooltip="Refresh task list ahead of schedule.",
            func=partial(send_feedback, parent=self),
            small=True,
        )
        self.statusbar.addPermanentWidget(self.feedback_btn)

        self.theme_btn = QtThemeButton(self)
        self.theme_btn.auto_connect()
        with hp.qt_signals_blocked(self.theme_btn):
            self.theme_btn.dark = CONFIG.theme == "dark"
        self.theme_btn.clicked.connect(self.on_toggle_theme)  # noqa
        self.theme_btn.set_small()

        self.statusbar.addPermanentWidget(self.theme_btn)
        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "ipython",
                tooltip="Open IPython console",
                small=True,
                func=self.on_show_console,
            )
        )
        self.tutorial_btn = hp.make_qta_btn(
            self, "help", tooltip="Give me a quick tutorial!", func=self.on_show_tutorial, small=True
        )
        self.statusbar.addPermanentWidget(self.tutorial_btn)

        self.update_status_btn = hp.make_btn(
            self,
            "Update available - click here to download!",
            tooltip="Show information about available updates.",
            func=self.on_show_update_info,
        )
        self.update_status_btn.setObjectName("update_btn")
        self.update_status_btn.hide()
        self.statusbar.addPermanentWidget(self.update_status_btn)
        self.setStatusBar(self.statusbar)

    def on_save_to_project(self) -> None:
        """Save data to config file."""
        filename = "project.i2threed.json"
        path_ = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_THREED_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2threed")
            CONFIG.output_dir = str(path.parent)
            config = self.registration.to_dict()
            write_project(path, config)
            hp.toast(
                self,
                "Exported project",
                f"Saved project to<br><b>{hp.hyper(path)}</b>",
                icon="success",
                position="top_left",
            )

    def on_load_from_project(self) -> None:
        """Load previous data."""
        # path = hp.get_filename(self, "Load i2c project", base_dir=CONFIG.output_dir,
        # file_filter=ALLOWED_THREED_FORMATS)
        # if path:
        #     self.registration

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not CONFIG.confirm_close_threed
            or QtConfirmCloseDialog(
                self, "confirm_close_threed", self.on_save_to_project, CONFIG
            ).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and CONFIG.confirm_close_threed
            and self.data_model.is_valid()
            and QtConfirmCloseDialog(
                self, "confirm_close_threed", self.on_save_to_project, CONFIG
            ).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        CONFIG.save()
        evt.accept()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="threed", level=0)
