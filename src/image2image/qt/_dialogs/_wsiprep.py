"""Dialogs for the wsiprep module."""
from __future__ import annotations

import typing as ty
from contextlib import contextmanager, suppress
from math import ceil, floor
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Shapes
from napari.layers.shapes._shapes_constants import Box
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLayout
from tqdm import tqdm

from image2image.config import CONFIG
from image2image.utils.utilities import format_group_info, get_groups, groups_to_group_id, init_shapes_layer

if ty.TYPE_CHECKING:
    from qtextra._napari.image.wrapper import NapariImageView
    from qtextra.utils.table_config import TableConfig
    from qtextra.widgets.qt_table_view import QtCheckableTableView

    from image2image.models.wsiprep import Registration, RegistrationGroup
    from image2image.qt.dialog_wsiprep import ImageWsiPrepWindow


class WsiPrepMixin(QtFramelessTool):
    """Mixin class for dialogs accessing parts of the WsiPrep App."""

    HIDE_WHEN_CLOSE = True

    parent: ty.Callable[[], ImageWsiPrepWindow]

    @property
    def view(self) -> NapariImageView:
        """Registration model."""
        return self.parent().view

    @property
    def registration(self) -> Registration:
        """Registration model."""
        return self.parent().registration

    @property
    def table(self) -> QtCheckableTableView:
        """Table model."""
        return self.parent().table

    @property
    def TABLE_CONFIG(self) -> TableConfig:
        """Table model."""
        return self.parent().TABLE_CONFIG


class GroupByDialog(WsiPrepMixin):
    """Dialog to group table."""

    def __init__(self, parent: ImageWsiPrepWindow):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(600)

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
        groups, dataset_to_group_map = self._get_groups()
        if dataset_to_group_map:
            ret = format_group_info(groups)
            self.label.setText(ret)
        else:
            self.label.setText("<fill-in group-by pattern above>")

    def on_group_by(self):
        """Group by user specified string."""
        if self.table.n_rows == 0:
            return
        groups, dataset_to_group_map = self._get_groups()
        if (
            groups
            and self.registration.is_grouped()
            and not hp.confirm(
                self,
                "Are you sure you wish to group images?<br>This will overwrite any previous groupings and <b>will"
                " result in loss of any masks</b>!",
                "Are you sure?",
            )
        ):
            return
        with MeasureTimer() as timer, self.table.block_model():
            values = [-1] * self.table.n_rows
            if dataset_to_group_map:
                for key, group_id in tqdm(
                    dataset_to_group_map.items(), desc="Grouping images...", total=self.table.n_rows
                ):
                    self.registration.images[key].group_id = group_id
                    index = self.table.find_index_of(self.TABLE_CONFIG.key, key)
                    values[index] = group_id
                self.registration.regroup()
                self.table.update_column(self.TABLE_CONFIG.group_id, values, match_to_sort=False)
            self.parent()._generate_group_options()
        logger.trace(f"Grouped images in {timer()}")

    def on_order_by(self):
        """Reorder images according to the current group identification."""
        if self.table.n_rows == 0:
            return
        if self.registration.is_ordered() and not hp.confirm(
            self,
            "Are you sure you wish to reorder images? This will overwrite any previous ordering.",
            "Are you sure?",
        ):
            return
        with MeasureTimer() as timer, self.table.block_model():
            self.registration.reorder()
            layers = []
            values = [0] * self.table.n_rows
            for key in tqdm(self.registration.key_iter(), desc="Reordering images...", total=self.table.n_rows):
                index = self.table.find_index_of(self.TABLE_CONFIG.key, key)
                values[index] = self.registration.images[key].image_order
                if key in self.view.layers:
                    layers.append(self.view.layers.pop(key))
            self.table.update_column(self.TABLE_CONFIG.image_order, values, match_to_sort=False)
            for layer in layers:
                self.view.viewer.add_layer(layer)
            self.view.viewer.reset_view()
        logger.trace(f"Reordered images in {timer()}")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QLayout:
        """Make panel."""
        self.group_by = hp.make_line_edit(
            self, placeholder="Type in part of the filename", func_changed=self.on_preview_group_by
        )
        self.group_by_btn = hp.make_btn(self, "Group", func=self.on_group_by)
        self.reorder_btn = hp.make_btn(self, "Reorder", func=self.on_order_by)

        self.label = hp.make_scrollable_label(
            self,
            "<fill-in group-by pattern above>",
            wrap=True,
            enable_url=True,
            selectable=True,
            object_name="title_label",
        )

        layout = hp.make_form_layout()
        layout.addRow(self._make_hide_handle("Group by...")[1])
        layout.addRow(
            hp.make_label(
                self,
                "If you are loading multiple images, you might want to group them together to make the co-registration"
                " better. This is often necessary when co-registering multiple cycles from MxIF or CODEX. You probably"
                " don't need to do this if you are loading a single dataset or whole set of images from a 3D"
                " experiment, although you might want to <b>order</b> your images so we know how to process them.",
                wrap=True,
                enable_url=True,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
        )
        layout.addRow(self.group_by)
        layout.addRow(hp.make_h_layout(self.group_by_btn, self.reorder_btn))
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self.label)
        return layout


class MaskDialog(WsiPrepMixin):
    """Dialog to mask a group."""

    _editing = False

    def __init__(self, parent: ImageWsiPrepWindow):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)

    @contextmanager
    def _editing_crop(self):
        self._editing = True
        yield
        self._editing = False

    @property
    def crop_layer(self) -> Shapes:
        """Crop layer."""
        if "Crop rectangle" not in self.view.layers:
            layer = self.view.viewer.add_shapes(
                None,
                edge_width=5,
                name="Crop rectangle",
                face_color="green",
                edge_color="white",
                opacity=0.5,
            )
            visual = self.view.widget.layer_to_visual[layer]
            init_shapes_layer(layer, visual)
            connect(self.crop_layer.events.set_data, self.on_update_crop_from_canvas, state=True)
        return self.view.layers["Crop rectangle"]

    def _get_default_crop_area(self) -> tuple[int, int, int, int]:
        (_, y, x) = self.view.viewer.camera.center
        top, bottom = y - 4096, y + 4096
        left, right = x - 4096, x + 4096
        return max(0, left), max(0, right), max(0, top), max(0, bottom)

    def on_update_crop_from_canvas(self, _evt: ty.Any = None) -> None:
        """Update crop values."""
        if self._editing:
            return
        n = len(self.crop_layer.data)
        if n > 1:
            logger.warning("More than one crop rectangle found. Using the first one.")
            hp.toast(
                self.parent(),
                "Only one crop rectangle is allowed.",
                "More than one crop rectangle found. We will <b>only</b> use the first region!",
                icon="warning",
            )
        if n == 0:
            return
        self._on_update_crop_from_canvas(0)
        if self.auto_update.isChecked():
            self.on_add_mask_to_group(auto=True)

    def _on_update_crop_from_canvas(self, index: int = 0) -> None:
        n = len(self.crop_layer.data)
        if n == 0 or index > n or index < 0:
            self.horizontal_label.setText("")
            self.vertical_label.setText("")
            return
        left, right, top, bottom = self._get_crop_area_for_index(index)
        self.horizontal_label.setText(f"{left:<10} - {right:>10} ({right - left:>7})")
        self.vertical_label.setText(f"{top:<10} - {bottom:>10} ({bottom - top:>7})")

    def _get_crop_area_for_index(self, index: int = 0) -> tuple[int, int, int, int]:
        """Return crop area."""
        n = len(self.crop_layer.data)
        if index > n or index < 0:
            return 0, 0, 0, 0
        rect = self.crop_layer.interaction_box(index)
        rect = rect[Box.LINE_HANDLE]
        xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
        xmin = max(0, xmin)
        ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
        ymin = max(0, ymin)
        return floor(xmin), ceil(xmax), floor(ymin), ceil(ymax)

    def on_initialize_mask(self) -> None:
        """Make mask for the currently selected group."""
        self.crop_layer.mode = "select"
        if len(self.crop_layer.data) == 0:
            left, right, top, bottom = self._get_default_crop_area()
            rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
            self.crop_layer.data = [(rect, "rectangle")]
        self.crop_layer.selected_data = [0]
        self.crop_layer.mode = "select"

    def _get_current_group_id(self) -> tuple[str, int | None, RegistrationGroup | None]:
        current = self.mask_choice.currentText()
        if "Group" not in current:
            return current, None, None
        group_id = int(current.split("'")[1])
        group = self.registration.groups[group_id]
        return current, group_id, group

    def on_select_mask(self) -> None:
        """Select mask from a list of available options."""
        current, group_id, group = self._get_current_group_id()
        if not group:
            return
        with self._editing_crop():
            if CONFIG.view_mode == "group":
                self.parent().groups_choice.setCurrentText(current)
            if group.mask_bbox is None:
                self.crop_layer.data = []
            else:
                left, top, width, height = group.mask_bbox
                right = left + width
                bottom = top + height
                rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
                self.crop_layer.data = [(rect, "rectangle")]
            self._on_update_crop_from_canvas()
            self.crop_layer.mode = "select"

    def on_add_mask_to_group(self, auto: bool = False) -> None:
        """Make mask for the currently selected group."""
        _, group_id, group = self._get_current_group_id()
        if not group:
            return

        left, right, top, bottom = self._get_crop_area_for_index(0)
        group.mask_bbox = (left, top, (right - left), (bottom - top))
        if not auto:
            logger.trace(f"Updated mask for group {group_id} to {group.mask_bbox}")

    def on_remove_mask_from_group(self) -> None:
        """Make mask for the currently selected group."""
        _, group_id, group = self._get_current_group_id()
        if not group:
            return
        group.mask_bbox = None
        logger.trace(f"Removed mask for group {group_id}")

    def on_show_mask(self) -> None:
        """Show mask."""
        self.parent()._move_layer(self.view, self.crop_layer, select=True)
        with suppress(TypeError):
            self.crop_layer.visible = True

    def on_hide_mask(self) -> None:
        """Hide current mask."""
        self.parent()._move_layer(self.view, self.crop_layer, select=False)
        layers: list[Image] = self.view.get_layers_of_type(Image)
        if layers:
            self.view.select_one_layer(layers[0])
        with suppress(TypeError):
            self.crop_layer.visible = False

    def on_update_group_options(self) -> None:
        """Synchronize group options."""
        choice = self.parent().groups_choice
        options = [choice.itemText(index) for index in range(2, choice.count())]
        current = self.mask_choice.currentText()
        with hp.qt_signals_blocked(self.mask_choice):
            self.mask_choice.clear()
            self.mask_choice.addItems(options)
            self.mask_choice.setCurrentText(current)

    def show(self):
        """Show dialog."""
        self.on_show_mask()
        self.on_update_group_options()
        self.on_select_mask()
        super().show()

    def hide(self):
        """Hide dialog."""
        self.on_hide_mask()
        super().hide()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QLayout:
        """Make panel."""
        self.mask_choice = hp.make_combobox(
            self, tooltip="Select group for which to show/add mask", func=self.on_select_mask
        )
        self.initialize_btn = hp.make_btn(self, "Initialize mask", func=self.on_initialize_mask)
        self.add_btn = hp.make_btn(self, "Associate mask", func=self.on_add_mask_to_group)
        self.remove_btn = hp.make_btn(self, "Dissociate mask", func=self.on_remove_mask_from_group)
        self.auto_update = hp.make_checkbox(self, "Auto-update", tooltip="Auto-update mask when cropping", checked=True)

        self.horizontal_label = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")
        self.vertical_label = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")

        layout = hp.make_form_layout()
        layout.addRow(self._make_hide_handle("Mask...")[1])
        layout.addRow(
            hp.make_label(
                self,
                "This dialog allows you to draw a rectangular mask for each group of images. Masks will be converted to"
                " a bounding box which we will use to define a region of interest during the co-registration. It is"
                " sometimes necessary to define a mask if there are a lot of <b>changes</b> to the tissue"
                " (e.g. tissue damage, folding between washes, etc). If you are unsure, you can skip this step.<br><br>"
                "Aim: You should aim to encompass the appropriate section of the <b>reference</b> image.",
                wrap=True,
                enable_url=True,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
        )
        layout.addRow(hp.make_label(self, "Group:"), self.mask_choice)
        layout.addRow(hp.make_label(self, "Horizontal"), self.horizontal_label)
        layout.addRow(hp.make_label(self, "Vertical"), self.vertical_label)
        layout.addRow(hp.make_h_layout(self.initialize_btn, self.add_btn, self.remove_btn))
        layout.addRow(hp.make_label(self, "Auto-update"), self.auto_update)
        return layout


class ConfigDialog(WsiPrepMixin):
    """Dialog to generate IWsiReg config."""

    _output_dir = None

    def __init__(self, parent: ImageWsiPrepWindow):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if CONFIG.output_dir is None:
                return Path.cwd()
            return Path(CONFIG.output_dir)
        return Path(self._output_dir)

    def on_set_output_dir(self):
        """Set output directory."""
        directory = hp.get_directory(self, "Select output directory", CONFIG.output_dir)
        if directory:
            self._output_dir = directory
            CONFIG.output_dir = directory
            self.output_dir_label.setText(f"<b>Output directory</b>: {hp.hyper(self.output_dir)}")
            logger.debug(f"Output directory set to {self._output_dir}")

    def on_export(self):
        """Export to disk."""
        pass

    def on_cancel(self):
        """Cancel."""
        pass

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QLayout:
        """Make panel."""
        self.directory_btn = hp.make_btn(
            self,
            "Set output directory...",
            tooltip="Specify output directory for images...",
            func=self.on_set_output_dir,
        )
        self.output_dir_label = hp.make_label(self, f"<b>Output directory</b>: {hp.hyper(self.output_dir)}")
        self.export_btn = hp.make_active_progress_btn(
            self, "Export to CSV", tooltip="Export to csv file...", func=self.on_export, cancel_func=self.on_cancel
        )

        self.label = hp.make_scrollable_label(
            self,
            "<fill-in group-by pattern above>",
            wrap=True,
            enable_url=True,
            selectable=True,
            object_name="title_label",
        )

        layout = hp.make_form_layout()
        layout.addRow(self._make_hide_handle("Generate iwsireg config...")[1])
        layout.addRow(
            hp.make_label(
                self,
                "",
                wrap=True,
                enable_url=True,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
        )
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self.directory_btn)
        layout.addRow(self.output_dir_label)
        layout.addRow(self.export_btn)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self.label)
        return layout
