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
from qtextraplot._napari.common.layer_controls.qt_shapes_controls import QtShapesControls
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLayout
from tqdm import tqdm

from image2image.config import get_elastix3d_config
from image2image.utils.utilities import format_group_info, get_groups, groups_to_group_id, init_shapes_layer

if ty.TYPE_CHECKING:
    from qtextra.utils.table_config import TableConfig
    from qtextra.widgets.qt_table_view_check import QtCheckableTableView
    from qtextraplot._napari.image.wrapper import NapariImageView

    from image2image.models.wsiprep import Registration, RegistrationGroup
    from image2image.qt.dialog_elastix3d import ImageElastix3dWindow


class WsiPrepMixin(QtFramelessTool):
    """Mixin class for dialogs accessing parts of the WsiPrep App."""

    HIDE_WHEN_CLOSE = True

    parent: ty.Callable[[], ImageElastix3dWindow]

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

    HIDE_WHEN_CLOSE = True

    def __init__(self, parent: ImageElastix3dWindow):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.on_preview_group_by()

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        connect(self.parent()._image_widget.dset_dlg.evt_loaded, self.on_preview_group_by, state=state)
        connect(self.parent()._image_widget.dset_dlg.evt_closed, self.on_preview_group_by, state=state)

    def _get_groups(self, by_slide: bool = False) -> tuple[dict[str, list[str]], dict[str, int]]:
        """Return mapping between image key and group ID."""
        value = self.group_by.text()
        if value or by_slide:
            filenames: list[str] = self.table.get_col_data(self.TABLE_CONFIG.key)
            groups = get_groups(filenames, value, by_slide=by_slide)
            dataset_to_group_map = groups_to_group_id(groups)
            return groups, dataset_to_group_map
        return {}, {}

    def on_preview_group_by(self) -> None:
        """Preview groups specified by user."""
        groups, dataset_to_group_map = self._get_groups(by_slide=self.group_by.text() == "")
        if dataset_to_group_map:
            ret = format_group_info(groups)
            self.label.setText(ret)
        else:
            self.label.setText("<b>fill-in group-by pattern above</b>")

    def on_add_metadata(self) -> None:
        """Add metadata."""
        if self.table.n_rows == 0:
            logger.trace("No images loaded")
            return
        groups, dataset_to_group_map = self._get_groups()
        if dataset_to_group_map:
            value = self.group_by.text()
            value = value.strip("= ")
            tag = hp.get_text(self, "Enter metadata tag", "Enter metadata tag", value)
            if tag == "auto":
                hp.toast(self, "Invalid metadata tag", "Metadata tag cannot be 'auto'", icon="warning")
                return
            if not tag:
                logger.trace("No metadata tag specified")
                return
            for key, group_id in tqdm(dataset_to_group_map.items(), desc="Grouping images...", total=self.table.n_rows):
                self.registration.images[key].metadata[tag] = group_id
            logger.trace(f"Added metadata tag {tag} to {len(dataset_to_group_map)} images")

    def on_group_by(self) -> None:
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

    def on_group_by_slide(self) -> None:
        """Group by slide. Each slide is given it's own group."""
        if self.table.n_rows == 0:
            return
        groups, dataset_to_group_map = self._get_groups(by_slide=True)
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

    def on_order_by(self) -> None:
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
        self.group_by_slide_btn = hp.make_btn(self, "Group by slide", func=self.on_group_by_slide)
        self.group_by = hp.make_line_edit(
            self, placeholder="Type in part of the filename", func_changed=self.on_preview_group_by
        )
        self.add_metadata_btn = hp.make_btn(self, "Add metadata", func=self.on_add_metadata)
        self.group_by_btn = hp.make_btn(self, "Group", func=self.on_group_by)
        self.reorder_btn = hp.make_btn(self, "Reorder", func=self.on_order_by)

        self.label = hp.make_scrollable_label(
            self,
            "<b>fill-in group-by pattern above</b>",
            wrap=True,
            enable_url=True,
            selectable=True,
            object_name="title_label",
        )

        layout = hp.make_form_layout()
        layout.setContentsMargins(6, 6, 6, 6)
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
        layout.addRow(hp.make_h_line())
        layout.addRow(self.group_by_slide_btn)
        layout.addRow(hp.make_h_line_with_text("or"))
        layout.addRow(self.group_by)
        layout.addRow(self.group_by_btn)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self.reorder_btn)
        layout.addRow(self.add_metadata_btn)
        layout.addRow(hp.make_h_line_with_text("Preview"))
        layout.addRow(self.label)
        return layout


class MaskDialog(WsiPrepMixin):
    """Dialog to mask a group."""

    HIDE_WHEN_CLOSE = True
    _editing = False

    def __init__(self, parent: ImageElastix3dWindow):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self.group_to_mask: dict[int, np.ndarray] = {}

    @contextmanager
    def _editing_crop(self):
        self._editing = True
        yield
        self._editing = False

    @property
    def crop_layer(self) -> Shapes:
        """Crop layer."""
        if "Mask" not in self.view.layers:
            layer = self.view.viewer.add_shapes(
                None,
                edge_width=5,
                name="Mask",
                face_color="green",
                edge_color="white",
                opacity=0.5,
            )
            visual = self.view.widget.canvas.layer_to_visual[layer]
            init_shapes_layer(layer, visual)
            connect(self.crop_layer.events.set_data, self.on_update_crop_from_canvas, state=True)

        layer = self.view.layers["Mask"]
        if hasattr(self, "layer_controls"):
            self.layer_controls.set_layer(layer)
        return layer

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
        left, right, top, bottom, shape_type, data = self._get_crop_area_for_index(index)
        self.horizontal_label.setText(f"{left:<10} - {right:>10} ({right - left:>7})")
        self.vertical_label.setText(f"{top:<10} - {bottom:>10} ({bottom - top:>7})")

    def _get_crop_area_for_index(self, index: int = 0) -> tuple[int, int, int, int, str | None, np.ndarray | None]:
        """Return crop area."""
        n = len(self.crop_layer.data)
        if index > n or index < 0:
            return 0, 0, 0, 0, None, None
        array = self.crop_layer.data[index]
        shape_type = self.crop_layer.shape_type[index]
        rect = self.crop_layer.interaction_box(index)
        rect = rect[Box.LINE_HANDLE]
        xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
        xmin = max(0, xmin)
        ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
        ymin = max(0, ymin)
        return floor(xmin), ceil(xmax), floor(ymin), ceil(ymax), shape_type, array

    def on_initialize_mask(self) -> None:
        """Make mask for the currently selected group."""
        self.crop_layer.mode = "select"
        if len(self.crop_layer.data) == 0:
            left, right, top, bottom = self._get_default_crop_area()
            rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
            self.crop_layer.data = [(rect, "polygon")]
        self.crop_layer.selected_data = [0]
        self.crop_layer.mode = "select"

    def _get_current_group_id(self) -> tuple[str, int | None, RegistrationGroup | None]:
        current = self.mask_choice.currentText()
        if "Group" not in current:
            return current, None, None
        group_id = int(current.split("'")[1])
        try:
            group = self.registration.groups[group_id]
            return current, group_id, group
        except KeyError:
            return current, None, None

    def on_select_mask(self) -> None:
        """Select mask from a list of available options."""
        current, group_id, group = self._get_current_group_id()
        if not group:
            return
        with self._editing_crop():
            if get_elastix3d_config().view_mode == "group":
                self.parent().groups_choice.setCurrentText(current)
            if group.mask_polygon is None and group.mask_bbox is None:
                self.crop_layer.data = []
            else:
                if group.mask_polygon is not None:
                    self.crop_layer.data = [(group.mask_polygon, "polygon")]
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
            logger.trace("Could not update group mask - no group selected")
            return

        left, right, top, bottom, shape_type, data = self._get_crop_area_for_index(0)
        if shape_type == "polygon":
            group.mask_polygon = data
            group.mask_bbox = None
        else:
            group.mask_bbox = (left, top, (right - left), (bottom - top))
            group.mask_polygon = None
        if not auto and group.is_masked():
            logger.trace(
                f"Updated mask for group {group_id} to "
                f"{group.mask_bbox if group.mask_bbox is not None else len(group.mask_polygon)}"
            )

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
        self.layer_controls = QtShapesControls(self.crop_layer)
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
        layout.setContentsMargins(6, 6, 6, 6)
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
        layout.addRow(hp.make_h_line())
        layout.addRow(self.layer_controls)
        layout.addRow(hp.make_h_line())
        layout.addRow(hp.make_label(self, "Group:"), self.mask_choice)
        layout.addRow(hp.make_label(self, "Horizontal"), self.horizontal_label)
        layout.addRow(hp.make_label(self, "Vertical"), self.vertical_label)
        layout.addRow(hp.make_h_layout(self.initialize_btn, self.add_btn, self.remove_btn))
        layout.addRow(hp.make_label(self, "Auto-update"), self.auto_update)
        return layout


class ConfigDialog(WsiPrepMixin):
    """Dialog to generate ElastixReg config."""

    HIDE_WHEN_CLOSE = True
    _output_dir = None

    def __init__(self, parent: ImageElastix3dWindow):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(500)
        self.transformations = list(get_elastix3d_config().transformations)
        self._update_transformation_path()
        self._update_indexing_mode()

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        connect(self.parent()._image_widget.dset_dlg.evt_loaded, self._update_indexing_mode, state=state)
        connect(self.parent()._image_widget.dset_dlg.evt_loaded, self.on_preview, state=state)
        connect(self.parent()._image_widget.dset_dlg.evt_closed, self._update_indexing_mode, state=state)
        connect(self.parent()._image_widget.dset_dlg.evt_closed, self.on_preview, state=state)

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if get_elastix3d_config().output_dir is None:
                return Path.cwd()
            return Path(get_elastix3d_config().output_dir)
        return Path(self._output_dir)

    def on_set_output_dir(self):
        """Set output directory."""
        directory = hp.get_directory(self, "Select output directory", get_elastix3d_config().output_dir)
        if directory:
            self._output_dir = directory
            get_elastix3d_config().output_dir = directory
            self.output_dir_label.setText(f"<b>Output directory</b>: {hp.hyper(self.output_dir)}")
            logger.debug(f"Output directory set to {self._output_dir}")

    def _get_suffix_prefix(self) -> tuple[str, str]:
        project_prefix = self.project_prefix.text()
        if "wsireg" in project_prefix:
            project_prefix = project_prefix.replace("wsireg", "")
        if project_prefix:
            project_prefix = project_prefix.strip("_- ")
            project_prefix = project_prefix + "-"

        project_suffix = self.project_suffix.text()
        if "wsireg" in project_suffix:
            project_suffix = project_suffix.replace("wsireg", "")
        if project_suffix:
            project_suffix = project_suffix.strip("_- ")
            project_suffix = "-" + project_suffix
        return project_prefix, project_suffix

    def _get_project_name(self, group: RegistrationGroup | None) -> str:
        project_prefix, project_suffix = self._get_suffix_prefix()
        get_elastix3d_config().project_tag = project_tag = self.project_tag.text() or "group"
        if group:
            return f"{project_prefix}{project_tag}={group.group_id}{project_suffix}.wsireg"
        return f"{project_prefix}{project_tag}{project_suffix}.wsireg"

    def on_export(self):
        """Export to disk."""
        if list(self.output_dir.glob("*.wsireg")) and not hp.confirm(
            self,
            "Are you sure you wish to overwrite existing files?",
            "Are you sure?",
        ):
            logger.trace("User cancelled export")
            return

        transformations = self.transformations
        if not transformations:
            hp.toast(self, "No registration selected", "Please select at least one registration type", icon="warning")
            return

        get_elastix3d_config().slide_tag = prefix = self.tag_prefix.text()
        get_elastix3d_config().project_prefix_tag = self.project_prefix.text()
        get_elastix3d_config().project_suffix_tag = self.project_suffix.text()
        index_mode = self.index_choice.currentText() or "auto"
        export_mode = self.export_type.currentText()
        first_only = self.first_channel_only.isChecked()
        target_mode = self.target_mode.currentText()
        if self.registration.is_single_project():
            self.registration.to_iwsireg(
                self.output_dir,
                self._get_project_name(None),
                prefix=prefix,
                index_mode=index_mode,
                export_mode=export_mode,
                first_only=first_only,
                direct=target_mode == "reference",
                transformations=transformations,
            )
        else:
            n = len(self.registration.groups)
            logger.trace(f"Generating configs for {n} groups...")
            with MeasureTimer():
                for _group_id, group in tqdm(self.registration.groups.items(), desc="Generating configs...", total=n):
                    path = group.to_iwsireg(
                        self.registration,
                        self.output_dir,
                        self._get_project_name(group),
                        prefix=prefix,
                        index_mode=index_mode,
                        export_mode=export_mode,
                        first_only=first_only,
                        direct=target_mode == "reference",
                        transformations=transformations,
                    )
                    logger.trace(f"Generated config for group '{group.group_id}' at '{path}'")

    def on_preview(self):
        """Preview registration paths."""
        n = len(self.registration.images)
        if n == 0:
            self.label.setText("<b>No images found</b>")
            return
        n = len(self.registration.groups)
        if n == 0:
            self.label.setText("<b>No groups found</b>")
            return
        logger.trace(f"Generating configs for {n} groups...")
        preview = ""
        get_elastix3d_config().slide_tag = prefix = self.tag_prefix.text()
        get_elastix3d_config().project_prefix_tag = self.project_prefix.text()
        get_elastix3d_config().project_suffix_tag = self.project_suffix.text()
        index_mode = self.index_choice.currentText() or "auto"
        target_mode = self.target_mode.currentText()
        for _group_id, group in tqdm(self.registration.groups.items(), desc="Previewing...", total=n):
            preview += (
                f"<b>Group {group.group_id}</b> ({self._get_project_name(group)})<br>"
                + group.to_preview(
                    self.registration,
                    prefix=prefix,
                    index_mode=index_mode,
                    target_mode=target_mode,
                )
                + "<br><br>"
            )
        self.label.setText(preview)

    def on_add_transformation(self) -> None:
        """Add transformation to the list."""
        current = self.transformation_choice.currentText()
        self.transformations.append(current)
        get_elastix3d_config().transformations = tuple(self.transformations)
        self._update_transformation_path()

    def on_reset_transformation(self) -> None:
        """Reset transformation list."""
        self.transformations = []
        get_elastix3d_config().transformations = tuple(self.transformations)
        self._update_transformation_path()

    def _update_transformation_path(self) -> None:
        """Update transformation path."""
        self.transformation_path.setText(
            " » ".join(self.transformations) if self.transformations else "<please select transformations>"
        )

    def _update_indexing_mode(self) -> None:
        """Update indexing mode."""
        keys = self.registration.get_metadata_keys()
        if keys:
            self.index_choice.clear()
            self.index_choice.addItems(["auto", *keys])
            self.index_choice.setCurrentText("auto")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QLayout:
        """Make panel."""
        self.directory_btn = hp.make_btn(
            self,
            "Set output directory...",
            tooltip="Specify output directory for images...",
            func=self.on_set_output_dir,
        )
        self.output_dir_label = hp.make_label(
            self, f"<b>Output directory</b>: {hp.hyper(self.output_dir)}", enable_url=True
        )
        self.transformation_choice = hp.make_combobox(
            self,
            options=[
                "rigid",
                "affine",
                "similarity",
                "nl",
                "fi_correction",
                "nl_reduced",
                "nl_mid",
                "nl2",
                "rigid_expanded",
                "affine_expanded",
                "nl_expanded",
                "rigid_ams",
                "affine_ams",
                "nl_ams",
                "rigid_anc",
                "affine_anc",
                "similarity_anc",
                "nl_anc",
            ],
            default="rigid",
            tooltip="Select registration type(s)...",
        )
        self.index_choice = hp.make_combobox(self, ["auto"], func=self.on_preview)
        self.transformation_path = hp.make_label(self, "<please select transformations>", wrap=True)
        self.tag_prefix = hp.make_line_edit(
            self,
            placeholder="Slide/section prefix",
            default=get_elastix3d_config().slide_tag,
            func_changed=self.on_preview,
        )
        self.project_tag = hp.make_line_edit(
            self,
            placeholder="Group/project tag",
            default=get_elastix3d_config().project_tag,
            func_changed=self.on_preview,
        )
        self.project_prefix = hp.make_line_edit(
            self,
            placeholder="Project prefix",
            default=get_elastix3d_config().project_prefix_tag,
            func_changed=self.on_preview,
        )
        self.project_suffix = hp.make_line_edit(
            self,
            placeholder="Project suffix",
            default=get_elastix3d_config().project_suffix_tag,
            func_changed=self.on_preview,
        )
        self.export_type = hp.make_combobox(
            self,
            [
                "Export with mask + affine initialization",
                "Export with mask + affine(translate) + rotation/flip initialization",
                "Export with mask + translation/rotation/flip initialization",
                "Export with mask + no initialization",
                "Export with no mask + affine initialization",
                "Export with no mask + affine(translate) + rotation/flip initialization",
                "Export with no mask + translation/rotation/flip initialization",
                "Export with no mask + no initialization",
            ],
            default="Export with mask + translation/rotation/flip initialization",
        )
        self.first_channel_only = hp.make_checkbox(
            self,
            "Only use first channel in co-registration",
            checked=False,
            tooltip="Only use the first channel when co-registering. This can be useful if you have a lot of channels"
            " and only want to co-register by e.g. the DAPI channel.",
        )
        self.target_mode = hp.make_combobox(
            self, ["sequential", "reference", "next", "none"], tooltip="Select target type...", func=self.on_preview
        )

        self.preview_btn = hp.make_btn(self, "Preview", tooltip="Preview registration paths...", func=self.on_preview)
        self.export_btn = hp.make_btn(self, "Export", tooltip="Generate IWsiReg configurations...", func=self.on_export)

        self.label = hp.make_scrollable_label(
            self,
            "<fill-in group-by pattern above>",
            wrap=True,
            enable_url=True,
            selectable=True,
            object_name="title_label",
        )
        self.label.setMinimumHeight(300)

        layout = hp.make_form_layout()
        layout.setContentsMargins(6, 6, 6, 6)
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
        layout.addRow(
            hp.make_h_layout(
                self.transformation_choice,
                hp.make_btn(self, "Add", func=self.on_add_transformation),
                hp.make_btn(self, "Reset", func=self.on_reset_transformation),
                stretch_id=(0,),
            )
        )
        layout.addRow(self.transformation_path)
        layout.addRow(self.tag_prefix)
        layout.addRow(
            hp.make_h_layout(self.project_prefix, self.project_tag, self.project_suffix, stretch_id=(0, 1, 2))
        )
        layout.addRow("Indexing mode", self.index_choice)
        layout.addRow(self.preview_btn)
        layout.addRow("Export type", self.export_type)
        layout.addRow("Target type", self.target_mode)
        layout.addRow("", self.first_channel_only)
        layout.addRow(self.export_btn)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self.label)
        return layout
