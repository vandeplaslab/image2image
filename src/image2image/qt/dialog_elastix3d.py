"""Three-D dialog."""

from __future__ import annotations

import typing as ty
from contextlib import suppress
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_io.readers import BaseReader, get_key
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Shapes
from natsort import natsorted
from qtextra.dialogs.qt_close_window import QtConfirmCloseDialog
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_table_view_check import QtCheckableTableView
from qtextra.widgets.qt_toolbar_mini import QtMiniToolbar
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMenuBar,
    QProgressBar,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)
from skimage.exposure import rescale_intensity
from superqt import QLabeledDoubleRangeSlider, ensure_main_thread
from superqt.utils import qdebounced
from tqdm import tqdm

from image2image import __version__
from image2image.config import Elastix3dConfig, get_elastix3d_config
from image2image.enums import ALLOWED_IMAGE_FORMATS_TIFF_ONLY, ALLOWED_PROJECT_WSIPREP_FORMATS
from image2image.models.wsiprep import (
    Registration,
    RegistrationGroup,
    RegistrationImage,
    as_icon,
    load_from_file,
    remove_if_not_present,
)
from image2image.qt._dialog_base import Window
from image2image.qt._dialogs._select import LoadWidget
from image2image.utils.utilities import ensure_extension, get_contrast_limits, write_project

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel
    from image2image.qt._wsiprep._wsiprep import GroupByDialog, MaskDialog


class ImageElastix3dWindow(Window):
    """Image viewer dialog."""

    image_layer: list[Image] | None = None
    shape_layer: list[Shapes] | None = None
    _console = None
    _editing = False
    group_by_dlg: GroupByDialog | None = None
    mask_dlg: MaskDialog | None = None
    iwsireg_dlg: None = None

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

    def __init__(self, parent: QWidget | None = None, run_check_version: bool = True, **kwargs: ty.Any):
        self.CONFIG: Elastix3dConfig = get_elastix3d_config()
        super().__init__(
            parent,
            f"image2image: Prepare your microscopy data for co-registration (v{__version__})",
            run_check_version=run_check_version,
        )
        self.registration = Registration()
        self._setup_config()

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dset_dlg.evt_import_project, self._on_load_from_project, state=state)
        connect(self._image_widget.dset_dlg.evt_closed, self.on_remove_image, state=state)
        connect(self.view.widget.canvas.events.key_press, self.keyPressEvent, state=state)
        connect(self.view.viewer.events.status, self._status_changed, state=state)
        connect(self.table.evt_keypress, self.keyPressEvent, state=state)
        connect(self.table.evt_value_checked, self.on_table_updated, state=state)
        connect(self.table.doubleClicked, self.on_table_double_clicked, state=state)
        connect(self.table.selectionModel().selectionChanged, qdebounced(self.on_highlight, 50), state=state)
        connect(self.contrast_limit.valueChanged, self.on_contrast_limits, state=state)

    def on_remove_image(self, model: DataModel) -> None:
        """Remove image."""
        if model:
            self.on_depopulate_table()
            self._update_registration_model()
        else:
            logger.warning(f"Failed to remove data - model={model}")

    def _update_registration_model(self) -> None:
        """Update registration model."""
        to_remove: list[str] = []
        for image_model in self.registration.model_iter():
            if image_model.key not in self.data_model.keys:
                to_remove.append(image_model.key)
        for key in to_remove:
            self.registration.remove(key)
        self._generate_group_options()
        self.registration.regroup()

    def on_populate_table(self) -> None:
        """Remove items that are not present in the model."""
        wrapper = self.data_model.wrapper
        if wrapper:
            data = []
            for reader in wrapper.reader_iter():
                if reader.key not in self.registration.images:
                    self.registration.append(reader)
                else:
                    self.registration.images[reader.key].update_from_reader(reader)
                data.append(self.registration.images[reader.key].to_table())
            # update table
            model_index = self.table.selectionModel().currentIndex()
            self.table.reset_data()
            self.table.add_data(data)
            if model_index.isValid() and model_index.row() < self.table.n_rows:
                self.table.scrollTo(model_index)
            self._generate_group_options()

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
        hp.disable_widgets(self.contrast_limit, disabled=not common)
        contrast_limits: tuple[float, float] = self.contrast_limit.value()  # type: ignore[no-untyped-call]
        if common:
            for layer in self.view.layers:
                layer.contrast_limits = contrast_limits
        else:
            for layer in self.view.layers:
                layer.contrast_limits = layer.metadata["contrast_limits"]

    def on_plot_selected(self) -> None:
        """Plot currently selected images."""
        with MeasureTimer():
            wrapper = self.data_model.wrapper
            if not wrapper:
                return
            all_contrast_range: list[float] = []

            group_id: int | None = None
            # plot non-reference images
            checked = self.table.get_all_checked()
            if len(checked) < 48 or hp.confirm(
                self,
                f"There is more  than 48 images selected ({len(checked)}). Would you like to continue?",
                "There are many images....",
            ):
                for index in tqdm(checked, desc="Plotting images..."):
                    key = self.table.get_value(self.TABLE_CONFIG.key, index)
                    reader = wrapper.data[key]
                    model = self.registration.images[reader.key]
                    if not model.keep:
                        logger.info(f"Skipping '{key}' as it is marked for removal.")
                        continue
                    group_id = model.group_id
                    contrast_range = self._plot_model(reader, model)
                    if contrast_range:
                        all_contrast_range.extend(contrast_range)

            # reset group mode in case the first reference is requested
            single_ref = "per list" in self.CONFIG.project_mode
            if single_ref:
                group_id = None

            # plot reference image
            ref_key = self.registration.get_reference_for_group(group_id)
            if ref_key and ref_key in wrapper.data:
                reader = wrapper.data[ref_key]
                model = self.registration.images[reader.key]
                contrast_range = self._plot_model(reader, model)
                if contrast_range:
                    all_contrast_range.extend(contrast_range)
            if all_contrast_range:
                min_val = min(all_contrast_range)
                max_val = max(all_contrast_range)
                self.contrast_limit.setRange(0 if min_val > 0 else min_val, 255 if max_val < 255 else max_val)
                if not self.common_contrast_limit.isChecked():
                    self.contrast_limit.setValue((min_val, max_val))
                self.contrast_limit.setDecimals(2 if max(all_contrast_range) < 1 else 0)
            self.on_contrast_limits()

    def _plot_model(self, reader: BaseReader, model: RegistrationImage) -> tuple[float, float] | None:
        image = reader.get_channel(0, -1)
        image = rescale_intensity(image, out_range=(0, 255))
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
                model = self.registration.images[key]
                is_reference = model.is_reference
                # is_reference = key == self.registration.reference
                cmap = "red" if is_reference else ("magenta" if index in checked else "cyan")
                if key in self.view.layers:
                    self.view.layers[key].colormap = cmap
        logger.trace(f"Selected images in {timer()}")

    def on_highlight(self, *_args: ty.Any) -> None:
        """Highlight specific image."""
        if self.table.selectionModel().hasSelection():
            self.on_select()
            with MeasureTimer() as timer:
                index = self.table.selectionModel().currentIndex().row()
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                if key in self.view.layers:
                    self.view.layers[key].colormap = "gray_r"  # "yellow"
            logger.trace(f"Highlighted image in {timer()}")

    def check_if_can_update(
        self, model: RegistrationImage, silent: bool = False, check_ref: bool = True, check_lock: bool = True
    ) -> bool:
        """Check if the model can be updated."""
        if check_ref and model.is_reference:
            if not silent:
                hp.toast(
                    self,
                    "Cannot modify reference",
                    f"Image <b>{model.key}</b> is a reference and cannot be modified.",
                    icon="warning",
                )
            return False
        if check_lock and model.lock:
            if not silent:
                hp.toast(
                    self,
                    "Image is locked",
                    f"Image <b>{model.key}</b> is locked and cannot be translated.",
                    icon="warning",
                )
            return False
        return True

    def on_table_updated(self, row: int, column: int, value: bool) -> None:
        """State was changed for specified row and column."""

        def _force_previous(col: int, val: bool) -> None:
            with hp.qt_signals_blocked(self.table):
                self.table.set_value(col, row, val)

        key = self.table.get_value(self.TABLE_CONFIG.key, row)
        model = self.registration.images[key]
        if column == self.TABLE_CONFIG.keep:
            if not self.check_if_can_update(model):
                _force_previous(self.TABLE_CONFIG.keep, model.keep)
                return
            model.keep = value
        elif column == self.TABLE_CONFIG.lock:
            if not self.check_if_can_update(model, check_lock=False):
                _force_previous(self.TABLE_CONFIG.lock, model.lock)
                return
            model.lock = value
        elif column == self.TABLE_CONFIG.flip_lr:
            if not self.check_if_can_update(model):
                _force_previous(self.TABLE_CONFIG.flip_lr, model.flip_lr)
                return
            model.flip_lr = value
            self._transform(model)

    def on_table_double_clicked(self, index):
        """Double-clicked."""
        row, column = index.row(), index.column()
        model = self.registration.images[self.table.get_value(self.TABLE_CONFIG.key, row)]
        if not self.check_if_can_update(model):
            return
        if column in [self.TABLE_CONFIG.rotate, self.TABLE_CONFIG.translate_x, self.TABLE_CONFIG.translate_y]:
            initial_value = int(self.table.get_value(column, row))
            verb = {
                self.TABLE_CONFIG.rotate: "Rotate",
                self.TABLE_CONFIG.translate_x: "Move in horizontal axis",
                self.TABLE_CONFIG.translate_y: "Move in vertical axis",
            }[column]
            attr = {
                self.TABLE_CONFIG.rotate: "rotate",
                self.TABLE_CONFIG.translate_x: "translate_x",
                self.TABLE_CONFIG.translate_y: "translate_y",
            }[column]
            min_val, max_val, step = {
                self.TABLE_CONFIG.rotate: (-360, 360, 15),
                self.TABLE_CONFIG.translate_x: (-100000, 100000, 250),
                self.TABLE_CONFIG.translate_y: (-100000, 100000, 250),
            }[column]
            new_value: int = hp.get_integer(
                self, f"Previous value: {initial_value}", f"{verb} image by...", initial_value, min_val, max_val, step
            )
            if new_value is not None and new_value != initial_value:
                self.table.set_value(column, row, new_value)
                setattr(model, attr, new_value)
                self._transform(model)

    def _transform(self, model: RegistrationImage) -> None:
        """Apply transformation to the specified model."""
        key = model.key
        if key in self.view.layers:
            self.view.layers[key].affine = model.affine(self.view.layers[key].data.shape)

    def on_rotate(self, which: str) -> None:
        """Rotate image."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                if not self.check_if_can_update(model, silent=True):
                    continue
                if model.lock:
                    hp.toast(
                        self,
                        "Image is locked",
                        f"Image <b>{key}</b> is locked and cannot be translated.",
                        icon="warning",
                    )
                    continue
                model.apply_rotate(which)
                self.table.set_value(self.TABLE_CONFIG.rotate, index, self.registration.images[key].rotate)
                self._transform(model)
        logger.trace(f"Rotate {which} '{len(checked)}' images in {timer()}")

    def on_translate(self, which: str) -> None:
        """Translate image."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                if not self.check_if_can_update(model, silent=True):
                    continue
                if model.lock:
                    hp.toast(
                        self,
                        "Image is locked",
                        f"Image <b>{key}</b> is locked and cannot be translated.",
                        icon="warning",
                    )
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
                if not self.check_if_can_update(model, silent=True):
                    continue
                if model.lock:
                    hp.toast(
                        self,
                        "Image is locked",
                        f"Image <b>{key}</b> is locked and cannot be translated.",
                        icon="warning",
                    )
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
                model.group_id = group_id if group else -1
                self.table.set_value(self.TABLE_CONFIG.group_id, index, model.group_id)
            self._generate_group_options()
            self.registration.regroup()
        logger.trace(f"{'Grouped' if group else 'Ungrouped'} '{len(checked)}' images in {timer()}")

    def on_lock(self, lock: bool) -> None:
        """Keep/remove images."""
        with MeasureTimer() as timer, self.table.block_model():
            checked = self.table.get_all_checked()
            for index in checked:
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                model = self.registration.images[key]
                if model.is_reference:
                    continue
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
                if model.is_reference:
                    continue
                model.keep = keep
                self.table.set_value(self.TABLE_CONFIG.keep, index, model.keep)
        logger.trace(f"{'Kept' if keep else 'Removed'} '{len(checked)}' images in {timer()}")

    def _generate_group_options(self) -> None:
        """Update combo box with group options including None and All."""
        self.CONFIG.view_mode = "group" if self.group_mode_btn.isChecked() else "slide"

        labels = []
        if self.CONFIG.view_mode == "group":
            for model in self.registration.model_iter():
                if f"Group '{model.group_id}'" not in labels:
                    labels.append(f"Group '{model.group_id}'")
        else:
            for model in self.registration.model_iter():
                labels.append(model.key)
        labels = ["None", "All", *natsorted(labels)]

        current = self.groups_choice.currentText()
        if len(labels) == 2:
            current = "None"
        with hp.qt_signals_blocked(self.groups_choice):
            self.groups_choice.clear()
            hp.set_combobox_text_data(self.groups_choice, labels, current)
            self.progress_bar.setRange(0, len(labels) - 2)
            self.progress_bar.setVisible(self.progress_bar.maximum() > 0)
        if self.mask_dlg is not None and self.CONFIG.view_mode == "group":
            self.mask_dlg.on_update_group_options()

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
            if "Group '" in group:
                group_id = int(group.split("'")[1])
                for index in range(self.table.n_rows):
                    key = self.table.get_value(self.TABLE_CONFIG.key, index)
                    model = self.registration.images[key]
                    values.append(model.group_id == group_id)
                    if model.group_id == group_id:
                        in_group.append(model.key)
            else:
                for index in range(self.table.n_rows):
                    key = self.table.get_value(self.TABLE_CONFIG.key, index)
                    values.append(key == group)
                    if key == group:
                        in_group.append(key)
        return in_group, values

    def on_select_group(self):
        """Select group."""
        current = self.groups_choice.currentText()
        _, values = self._get_for_group()
        self.table.update_column(self.TABLE_CONFIG.check, values, match_to_sort=False)
        self.progress_bar.setValue(self.groups_choice.currentIndex() - 2)

        # clear canvas and plot
        self.view.viewer.layers.clear()
        self.on_plot_selected()
        self.on_select()
        self.view.viewer.reset_view()

        if self.mask_dlg is not None and self.mask_dlg.isVisible() and "Group" in current:
            self.mask_dlg.mask_choice.setCurrentText(current)

    def on_group_increment(self, increment: int = 0) -> None:
        """Increment group."""
        hp.increment_combobox(self.groups_choice, increment, skip=[0, 1])
        self.progress_bar.setValue(self.groups_choice.currentIndex() - 2)
        self.on_scroll()
        if self.skip_if_locked.isChecked():
            checked = self.table.get_all_checked()
            values_for_checked = [self.table.get_value(self.TABLE_CONFIG.lock, index) for index in checked]
            if all(values_for_checked):
                self.on_group_increment(increment)
                logger.trace("Skipped locked group")

    def on_check_group(self, check: bool) -> None:
        """Show or hide group images."""
        _, values = self._get_for_group()
        if not check:
            values = [False] * self.table.n_rows
        self.table.update_column(self.TABLE_CONFIG.check, values, match_to_sort=False)

    def on_show_group(self, show: bool) -> None:
        """Show or hide group images."""
        checked = self.table.get_all_checked()
        for index in checked:
            key = self.table.get_value(self.TABLE_CONFIG.key, index)
            if key in self.view.layers:
                self.view.layers[key].visible = show

    def on_scroll(self) -> None:
        """Scroll to the currently selected group."""
        _, values = self._get_for_group()
        if any(values):
            index = values.index(True)
            if index != -1:
                index = min(index + 3, self.table.n_rows - 1)
                self.table.scrollTo(self.table.create_index(index, 0))

    def on_reference(self) -> None:
        """Set the currently selected row as a reference."""
        project_mode = self.project_mode.currentText()
        single_ref = "per list" in project_mode
        if self.table.selectionModel().hasSelection():
            with MeasureTimer() as timer, self.table.block_model():
                index = self.table.selectionModel().currentIndex().row()
                key_ = self.table.get_value(self.TABLE_CONFIG.key, index)
                if not hp.confirm(self, f"Set <b>{key_}</b> as the reference image?", "Set reference"):
                    return

                group_id = None if single_ref else self.registration.images[key_].group_id
                logger.trace(f"Setting reference image (single_ref={single_ref}; group_id={group_id})")
                for row in range(self.table.n_rows):
                    key = self.table.get_value(self.TABLE_CONFIG.key, row)
                    model = self.registration.images[key]
                    if group_id is None or group_id == model.group_id:
                        model.is_reference = row == index

                    if row == index:
                        model.lock = True
                        self.table.set_value(self.TABLE_CONFIG.lock, row, True)
                    self.table.set_value(self.TABLE_CONFIG.reference, row, as_icon(model.is_reference))
                self.on_select()
            logger.trace(f"Set {key_} as reference in {timer()} (single_ref={single_ref})")

    def on_auto_reference(self) -> None:
        """Automatically select reference for each group."""
        # TODO: check for project mode
        if not hp.confirm(
            self,
            "Set reference for each group? This will override any previous references and there is no going back.",
            "Set reference",
        ):
            return
        for group_id in self.registration.group_iter():
            group = self.registration.groups[group_id]
            if len(group.keys) == 1:
                continue
            # find the best reference
            ref = natsorted(group.keys)[0]
            logger.trace(f"Selected '{ref}' as reference for group '{group_id}'")
            model = self.registration.images[ref]
            old_ref = self.registration.get_reference_for_group(group_id)
            if old_ref:
                self.registration.images[old_ref].is_reference = False
                self.registration.images[old_ref].lock = False
            model.is_reference = True
            model.lock = True
        # update table
        with self.table.block_model():
            for row in range(self.table.n_rows):
                key = self.table.get_value(self.TABLE_CONFIG.key, row)
                model = self.registration.images[key]
                self.table.set_value(self.TABLE_CONFIG.reference, row, as_icon(model.is_reference))
                self.table.set_value(self.TABLE_CONFIG.lock, row, model.lock)

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

    def on_project_mode(self) -> None:
        """Update project mode."""
        prev_mode = self.CONFIG.project_mode
        if (
            "per group" in prev_mode
            and "per group" not in self.project_mode.currentText()
            and not hp.confirm(
                self,
                "Changing the project mode from <b>'per group'</b> where <b>multiple</b> references are permitted to"
                " <b>'per list'</b> where only <b>one</b> reference is permitted will remove all but the first"
                " reference in each group. Are you sure you wish to continue?",
                "Are you sure?",
            )
        ):
            self.project_mode.setCurrentText(prev_mode)
            return

        self.CONFIG.project_mode = self.project_mode.currentText()
        self._generate_group_options()

    def on_open_group_by_popup(self) -> None:
        """Open group-by dialog."""
        if self.group_by_dlg is None:
            from image2image.qt._wsiprep._wsiprep import GroupByDialog

            self.group_by_dlg = GroupByDialog(self)
        self.group_by_dlg.show()

    def on_open_mask_popup(self) -> None:
        """Open group-by dialog."""
        if self.mask_dlg is None:
            from image2image.qt._wsiprep._wsiprep import MaskDialog

            self.mask_dlg = MaskDialog(self)
        self.mask_dlg.show()

    def on_open_iwsireg_popup(self) -> None:
        """Open group-by dialog."""
        from image2image.qt._wsiprep._wsiprep import ConfigDialog

        dlg = ConfigDialog(self)
        dlg.show()

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(
            self, add_toolbars=False, allow_extraction=False, disable_controls=False, disable_new_layers=True
        )

        self._image_widget = LoadWidget(
            self,
            self.view,
            self.CONFIG,
            allow_channels=False,
            available_formats=ALLOWED_IMAGE_FORMATS_TIFF_ONLY,
            project_extension=[".i2wsiprep.json", ".i2wsiprep.toml"],
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
        self.table.verticalHeader().setVisible(True)
        hp.set_font(self.table)
        self.table.setup_model(
            header=self.TABLE_CONFIG.header,
            no_sort_columns=self.TABLE_CONFIG.no_sort_columns,
            hidden_columns=self.TABLE_CONFIG.hidden_columns,
            checkable_columns=self.TABLE_CONFIG.checkable_columns,
        )

        self.toolbar = QtMiniToolbar(self, orientation=Qt.Orientation.Vertical, add_spacer=False)
        self.toolbar.add_qta_tool(
            "rotate_left", tooltip="Rotate image left (E)", func=lambda: self.on_rotate("left"), average=True
        )
        self.toolbar.add_qta_tool(
            "rotate_right", tooltip="Rotate image right (Q)", func=lambda: self.on_rotate("right"), average=True
        )
        self.toolbar.add_qta_tool(
            "translate_up", tooltip="Translate image up (W)", func=lambda: self.on_translate("up"), average=True
        )
        self.toolbar.add_qta_tool(
            "translate_down", tooltip="Translate image down (S)", func=lambda: self.on_translate("down"), average=True
        )
        self.toolbar.add_qta_tool(
            "translate_left", tooltip="Translate image left (A)", func=lambda: self.on_translate("left"), average=True
        )
        self.toolbar.add_qta_tool(
            "translate_right",
            tooltip="Translate image right (D)",
            func=lambda: self.on_translate("right"),
            average=True,
        )
        self.toolbar.add_qta_tool("flip_lr", tooltip="Flip image left-right (F)", func=self.on_flip_lr, average=True)
        self.toolbar.add_qta_tool("group", tooltip="Group images", func=lambda x: self.on_group(True), average=True)
        self.toolbar.add_qta_tool(
            "ungroup", tooltip="Ungroup images", func=lambda x: self.on_group(False), average=True
        )
        self.toolbar.add_qta_tool(
            "lock_open", tooltip="Lock images (L)", func=lambda x: self.on_lock(True), average=True
        )
        self.toolbar.add_qta_tool(
            "lock_closed", tooltip="Unlock images (U)", func=lambda x: self.on_lock(False), average=True
        )
        self.toolbar.add_qta_tool(
            "keep_image", tooltip="Keep images (Z)", func=lambda x: self.on_keep(True), average=True
        )
        self.toolbar.add_qta_tool(
            "remove_image", tooltip="Remove images (X)", func=lambda x: self.on_keep(False), average=True
        )
        self.toolbar.append_spacer()
        self.toolbar.add_qta_tool(
            "layers",
            func=self.view.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )

        self.toolbar.layout().setSpacing(0)
        self.toolbar.layout().setContentsMargins(0, 0, 0, 0)

        self.export_project_btn = hp.make_btn(
            self,
            "Export project...",
            tooltip="Export configuration to a project file. Information such as image path and crop"
            " information are saved. (This does not save the cropped image)",
            func=self.on_save_to_project,
        )
        # options
        self.project_mode = hp.make_combobox(
            self,
            [
                "2D AF/MxIF (one ref per group)",
                "2D AF/MxIF (one ref per list)",
                "3D AF/MxIF (one ref per list)",
                "2D preAF>AF>postAF (one ref per group)",
            ],
            tooltip="Select project mode. This will determine how the images are grouped and registered.",
            func=self.on_project_mode,
            value=self.CONFIG.project_mode,
        )

        # group-by options
        self.group_by_btn = hp.make_btn(
            self,
            "Group by...",
            tooltip="Automatically group images according to filename ruels...",
            func=self.on_open_group_by_popup,
        )
        self.mask_btn = hp.make_btn(
            self, "Mask group...", tooltip="Create masks to aid co-registrations...", func=self.on_open_mask_popup
        )
        self.iwsireg_btn = hp.make_btn(
            self,
            "Generate config...",
            tooltip="Generate iwsireg configuration data...",
            func=self.on_open_iwsireg_popup,
        )
        self.auto_ref_btn = hp.make_btn(
            self,
            "Auto-reference group",
            tooltip="Automatically select a reference image for each group",
            func=self.on_auto_reference,
        )

        # select options
        self.group_mode_btn = hp.make_radio_btn(
            self, "Group mode", tooltip="Group mode", checked=True, func=self._generate_group_options
        )
        self.slide_mode_btn = hp.make_radio_btn(
            self, "Slide mode", tooltip="Slide mode", checked=False, func=self._generate_group_options
        )
        self.group_mode = hp.make_radio_btn_group(self, [self.group_mode_btn, self.slide_mode_btn])

        self.groups_choice = hp.make_combobox(self, tooltip="Select group to show", func=self.on_select_group)
        self.progress_bar = QProgressBar(self)
        hp.set_retain_hidden_size_policy(self.progress_bar)
        hp.set_sizer_policy(self.progress_bar, h_stretch=True, v_stretch=False)
        self.progress_bar.setObjectName("progress_timer")
        self.progress_bar.setTextVisible(False)

        self.check_group_btn = hp.make_btn(
            self,
            "Select group",
            tooltip="Only show the selected group (+ reference image)",
            func=lambda: self.on_check_group(True),
        )
        self.uncheck_group_btn = hp.make_btn(
            self,
            "Unselect group",
            tooltip="Hide the selected image",
            func=lambda: self.on_check_group(False),
        )
        self.scroll_group_btn = hp.make_btn(
            self,
            "Scroll to group",
            tooltip="Scroll to the selected group",
            func=self.on_scroll,
        )
        self.skip_if_locked = hp.make_checkbox(
            self,
            "Skip locked",
            tooltip="Skip locked images when walking through groups or slides",
        )

        side_widget = QWidget()  # noqa
        side_widget.setMinimumWidth(300)
        side_layout = hp.make_form_layout(parent=side_widget, margin=0)
        # project options
        side_layout.addRow(self.import_project_btn)
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(self.export_project_btn)
        # project mode
        side_layout.addRow(hp.make_h_line())
        side_layout.addRow(hp.make_label(self, "Project mode"), self.project_mode)
        side_layout.addRow(self.group_by_btn)
        side_layout.addRow(self.auto_ref_btn)
        side_layout.addRow(self.mask_btn)
        side_layout.addRow(self.iwsireg_btn)
        # select group options
        side_layout.addRow(hp.make_h_line_with_text("Select/view group"))
        side_layout.addRow(hp.make_h_layout(self.group_mode_btn, self.slide_mode_btn))
        side_layout.addRow("Group", self.groups_choice)
        side_layout.addRow(self.progress_bar)
        side_layout.addRow(hp.make_h_layout(self.check_group_btn, self.uncheck_group_btn, self.scroll_group_btn))
        side_layout.addRow(self.skip_if_locked)

        bottom_widget = QWidget()  # noqa
        bottom_widget.setMinimumHeight(400)
        bottom_widget.setMaximumHeight(600)

        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setSpacing(2)
        bottom_layout.addWidget(side_widget)
        bottom_layout.addWidget(hp.make_v_line())
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
        main_layout.setContentsMargins(0, 0, 0, 0)
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

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(self._make_tools_menu(scalebar=True).menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
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
        # translate
        elif key == Qt.Key.Key_A:
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
        # selection
        elif key == Qt.Key.Key_N:
            self.on_group_increment(1)
        elif key == Qt.Key.Key_P:
            self.on_group_increment(-1)
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
            {
                "viewer": self.view.viewer,
                "data_model": self.data_model,
                "wrapper": self.data_model.wrapper,
                "registration": self.registration,
            }
        )
        return variables

    def on_show_grid(self) -> None:
        """Show scale bar controls for the viewer."""
        from qtextraplot._napari.common.component_controls.qt_grid_controls import QtGridControls

        dlg = QtGridControls(self.view.viewer, self.view.widget)
        dlg.show_above_mouse()

    def on_set_config(self):
        """Update config."""
        self.CONFIG.rotate_step_size = int(self.rotate_step_size.currentText().split(" ")[2].split("°")[0])
        self.CONFIG.translate_step_size = int(self.translate_step_size.currentText().split(" ")[2])

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        from qtextraplot._napari.image.components._viewer_key_bindings import toggle_grid

        self.statusbar = QStatusBar()  # noqa
        self.statusbar.setSizeGripEnabled(False)

        # self.statusbar.addPermanentWidget(hp.make_v_line())
        # self.statusbar.addPermanentWidget(hp.make_label(self, "BF"))
        # self.bf_colormap = hp.make_btn(
        #     self,
        #     "",
        #     tooltip="Brightfield colormap",
        #     object_name="colorbar",
        #     # func=self.on_make_colormap,
        # )
        # self.statusbar.addPermanentWidget(self.bf_colormap)
        # self.statusbar.addPermanentWidget(hp.make_label(self, "DF"))
        # self.df_colormap = hp.make_btn(
        #     self,
        #     "",
        #     tooltip="Darkfield colormap",
        #     object_name="colorbar",
        #     # func=self.on_make_colormap,
        # )
        # self.statusbar.addPermanentWidget(self.df_colormap)
        # self.statusbar.addPermanentWidget(hp.make_v_line())

        self.common_contrast_limit = hp.make_checkbox(
            self,
            "Common intensity",
            tooltip="Use common contrast limit for all images",
            func=self.on_contrast_limits,
            value=self.CONFIG.common_intensity,
        )
        hp.set_sizer_policy(self.common_contrast_limit, h_stretch=False)
        self.statusbar.addPermanentWidget(self.common_contrast_limit)
        self.contrast_limit = QLabeledDoubleRangeSlider()
        self.contrast_limit.setEdgeLabelMode(QLabeledDoubleRangeSlider.EdgeLabelMode.LabelIsValue)
        self.contrast_limit.setHandleLabelPosition(QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
        self.contrast_limit.setOrientation(Qt.Orientation.Horizontal)  # type: ignore[no-untyped-call]
        self.contrast_limit.setRange(0, 1)
        self.contrast_limit.setValue((0, 1))
        self.contrast_limit.setDecimals(2)
        hp.disable_widgets(self.contrast_limit, disabled=not self.common_contrast_limit.isChecked())
        self.contrast_limit.layout().setSpacing(1)
        self.statusbar.addPermanentWidget(self.contrast_limit)
        self.statusbar.addPermanentWidget(hp.make_v_line())

        self.rotate_step_size = hp.make_combobox(
            self,
            [
                "Rotate in 1° steps",
                "Rotate in 5° steps",
                "Rotate in 10° steps",
                "Rotate in 15° steps",
                "Rotate in 30° steps",
                "Rotate in 45° steps",
                "Rotate in 60° steps",
                "Rotate in 90° steps",
            ],
            func=self.on_set_config,
            value=f"Rotate in {self.CONFIG.rotate_step_size}° steps",
        )
        self.rotate_step_size.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.statusbar.addPermanentWidget(self.rotate_step_size)
        self.statusbar.addPermanentWidget(hp.make_v_line())

        self.translate_step_size = hp.make_combobox(
            self,
            [
                "Move in 10 µm steps",
                "Move in 25 µm steps",
                "Move in 50 µm steps",
                "Move in 100 µm steps°",
                "Move in 250 µm steps",
                "Move in 500 µm steps",
                "Move in 1000 µm steps",
                "Move in 2500 µm steps",
            ],
            func=self.on_set_config,
            value=f"Move in {self.CONFIG.translate_step_size:d} µm steps",
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
            func_menu=self.on_show_grid,
        )
        self.statusbar.addPermanentWidget(self.grid_btn)

        self._make_export_statusbar(front=False)
        self._make_scalebar_statusbar(front=False)
        self._make_theme_statusbar()
        self._make_feedback_statusbar()
        self._make_theme_statusbar()
        self._make_shortcut_statusbar()
        self._make_tutorial_statusbar()
        self._make_logger_statusbar()
        self._make_ipython_statusbar()
        self._make_update_statusbar()
        self.setStatusBar(self.statusbar)

    def on_show_shortcuts(self):
        """Show shortcuts."""
        from image2image.qt._dialogs._shortcuts import WsiPrepShortcutsDialog

        dlg = WsiPrepShortcutsDialog(self)
        dlg.show()

    def on_save_to_project(self) -> None:
        """Save data to config file."""
        filename = "project.i2wsiprep.json"
        path_ = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=self.CONFIG.output_dir,
            file_filter=ALLOWED_PROJECT_WSIPREP_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2wsiprep")
            self.CONFIG.output_dir = str(path.parent)
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
        path_ = hp.get_filename(
            self, "Load i2c project", base_dir=self.CONFIG.output_dir, file_filter=ALLOWED_PROJECT_WSIPREP_FORMATS
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: str) -> None:
        if path_:
            from image2image.qt._dialogs import LocateFilesDialog

            path = Path(path_)
            self.CONFIG.output_dir = str(path.parent)
            paths, missing_paths, config = load_from_file(path)

            # locate paths that are missing
            if missing_paths:
                locate_dlg = LocateFilesDialog(self, self.CONFIG, missing_paths)
                if locate_dlg.exec_():  # type: ignore[attr-defined]
                    paths = locate_dlg.fix_missing_paths(missing_paths, paths)

            # cleanup paths
            keys_to_keep = [get_key(path) for path in paths]
            config = remove_if_not_present(config, keys_to_keep)

            # add images to registration
            for item in config["images"].values():
                self.registration.images[item["key"]] = RegistrationImage(**item)
            # add groups to registration
            for item in config["groups"].values():
                self.registration.groups[item["group_id"]] = RegistrationGroup(**item)

            # add images
            logger.trace(f"Found {len(paths)} images in project file")
            self._image_widget.on_set_path(paths)

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not self.CONFIG.confirm_close
            or QtConfirmCloseDialog(self, "confirm_close", self.on_save_to_project, self.CONFIG).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and self.CONFIG.confirm_close
            and (self.data_model.is_valid() or self.registration.is_valid())
            and QtConfirmCloseDialog(self, "confirm_close", self.on_save_to_project, self.CONFIG).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        self.CONFIG.save()
        evt.accept()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="wsiprep", level=0)
