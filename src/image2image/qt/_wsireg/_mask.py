"""Mask dialog."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager, suppress
from math import ceil, floor

import numpy as np
import qtextra.helpers as hp
from loguru import logger
from napari.layers import Image, Shapes
from napari.layers.shapes._shapes_constants import Box
from qtextra._napari.common.layer_controls.qt_shapes_controls import QtShapesControls
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QLayout

from image2image.utils.utilities import init_shapes_layer

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality
    from qtextra._napari.image.wrapper import NapariImageView

    from image2image.qt.dialog_wsireg import ImageWsiRegWindow


logger = logger.bind(src="MaskDialog")


class MaskDialog(QtFramelessTool):
    """Dialog to mask a group."""

    HIDE_WHEN_CLOSE = True
    _editing = False

    evt_mask = Signal(object)

    def __init__(self, parent: ImageWsiRegWindow | None):
        super().__init__(parent)
        self._parent = parent
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        self.on_update_modality_options()

    def on_update_modality_options(self) -> None:
        """Update list of available modalities."""
        current = self.slide_choice.currentText()
        hp.combobox_setter(self.slide_choice, items=self._parent.registration_model.modalities.keys(), set_item=current)

    @property
    def view(self) -> NapariImageView:
        """Registration model."""
        return self._parent.view

    @contextmanager
    def _editing_crop(self):
        self._editing = True
        yield
        self._editing = False

    @property
    def mask_layer(self) -> Shapes:
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
            visual = self.view.widget.layer_to_visual[layer]
            init_shapes_layer(layer, visual)
            connect(self.mask_layer.events.set_data, self.on_update_crop_from_canvas, state=True)

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
        n = len(self.mask_layer.data)
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
        if self.auto_update.isChecked() and self.is_current_masked():
            self.on_associate_mask_with_modality()

    def is_current_masked(self) -> bool:
        """Check if current modality is masked."""
        name = self.slide_choice.currentText()
        if name:
            modality = self._parent.registration_model.modalities[name]
            return modality.is_masked()  # TODO: Implement is_masked method
        return False

    def _on_update_crop_from_canvas(self, index: int = 0) -> None:
        n = len(self.mask_layer.data)
        if n == 0 or index > n or index < 0:
            self.horizontal_label.setText("")
            self.vertical_label.setText("")
            return
        left, right, top, bottom, shape_type, data = self._get_crop_area_for_index(index)
        self.horizontal_label.setText(f"{left:<10} - {right:>10} ({right - left:>7})")
        self.vertical_label.setText(f"{top:<10} - {bottom:>10} ({bottom - top:>7})")

    def _get_crop_area_for_index(self, index: int = 0) -> tuple[int, int, int, int, str | None, np.ndarray | None]:
        """Return crop area."""
        n = len(self.mask_layer.data)
        if index > n or index < 0:
            return 0, 0, 0, 0, None, None
        array = self.mask_layer.data[index]
        shape_type = self.mask_layer.shape_type[index]
        rect = self.mask_layer.interaction_box(index)
        rect = rect[Box.LINE_HANDLE]
        xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
        xmin = max(0, xmin)
        ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
        ymin = max(0, ymin)
        return floor(xmin), ceil(xmax), floor(ymin), ceil(ymax), shape_type, array

    def on_initialize_mask(self) -> None:
        """Make mask for the currently selected group."""
        self.mask_layer.mode = "select"
        if len(self.mask_layer.data) == 0:
            left, right, top, bottom = self._get_default_crop_area()
            rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
            self.mask_layer.data = [(rect, "polygon")]
        self.mask_layer.selected_data = [0]
        self.mask_layer.mode = "select"

    def on_show_mask(self) -> None:
        """Show mask."""
        self._parent._move_layer(self.view, self.mask_layer, select=True)
        with suppress(TypeError):
            self.mask_layer.visible = True

    def show(self):
        """Show dialog."""
        self.on_update_modality_options()
        self.on_show_mask()
        super().show()

    def on_hide_mask(self) -> None:
        """Hide current mask."""
        self._parent._move_layer(self.view, self.mask_layer, select=False)
        layers: list[Image] = self.view.get_layers_of_type(Image)
        if layers:
            self.view.select_one_layer(layers[0])
        with suppress(TypeError):
            self.mask_layer.visible = False

    def hide(self):
        """Hide dialog."""
        self.on_hide_mask()
        super().hide()

    def _transform_from_preprocessing(self, modality: Modality) -> tuple[list] | None:
        """Transform data from stored data (which is stored in the micron units."""
        wrapper = self._parent.data_model.get_wrapper()
        reader = wrapper.get_reader_for_path(modality.path)
        if modality.mask_polygon is None:
            bbox = modality.mask_bbox
            if bbox is None:
                return []
            left, top, width, height = bbox.x, bbox.y, bbox.width, bbox.height
            right = left + width
            bottom = top + height
            data = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
            data = data * reader.resolution
            return [(data, "rectangle")]
        yx = modality.mask_polygon.xy[:, ::-1]
        yx = yx * reader.resolution
        return [(yx, "polygon")]

    def _transform_to_preprocessing(self, modality: Modality) -> tuple:
        """Transform data from napari layer to preprocessing data."""
        wrapper = self._parent.data_model.get_wrapper()
        reader = wrapper.get_reader_for_path(modality.path)
        left, right, top, bottom, shape_type, yx = self._get_crop_area_for_index(0)
        if shape_type == "polygon":
            # convert to pixel coordinates and round
            yx = yx * reader.inv_resolution
            return np.round(yx).astype(np.int32)[:, ::-1], None
        bbox = np.asarray([left, right, (right - left), (bottom - top)])
        return None, tuple(np.round(bbox * reader.inv_resolution).astype(int))

    def on_select_modality(self, _=None) -> None:
        """Select mask for the currently selected group."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        data = self._transform_from_preprocessing(modality)
        self.mask_layer.data = data or []

        # hide all other image layers
        if self.only_current.isChecked():
            for layer in self.view.get_layers_of_type(Image):
                layer.visible = layer.name == modality.name

    def on_associate_mask_with_modality(self) -> None:
        """Associate mask with modality at the specified location."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        yx, bbox = self._transform_to_preprocessing(modality)
        if yx is not None:
            modality.mask_polygon = yx
            modality.mask_bbox = None
        elif bbox is not None:
            modality.mask_bbox = bbox
            modality.mask_polygon = None
        kind = "polygon" if yx is not None else "bbox"
        logger.trace(f"Added mask for modality {name} to {kind}")
        self.evt_mask.emit(modality)

    def on_dissociate_mask_from_modality(self) -> None:
        """Dissociate mask from modality."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        modality.mask_polygon = None
        modality.mask_bbox = None
        logger.trace(f"Removed mask for modality {name}")
        self.evt_mask.emit(modality)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QLayout:
        """Make panel."""
        self.layer_controls = QtShapesControls(self.mask_layer)
        self.slide_choice = hp.make_combobox(self, tooltip="Select modality", func=self.on_select_modality)
        self.only_current = hp.make_checkbox(
            self, tooltip="Only show currently selected modality.", func=self.on_select_modality
        )
        self.initialize_btn = hp.make_btn(self, "Initialize mask", func=self.on_initialize_mask)
        self.add_btn = hp.make_btn(self, "Associate mask", func=self.on_associate_mask_with_modality)
        self.remove_btn = hp.make_btn(self, "Dissociate mask", func=self.on_dissociate_mask_from_modality)
        self.auto_update = hp.make_checkbox(self, "", tooltip="Auto-associate mask when making changes", checked=True)

        self.horizontal_label = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")
        self.vertical_label = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")

        layout = hp.make_form_layout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addRow(self._make_hide_handle("Mask...")[1])
        layout.addRow(
            hp.make_label(
                self,
                "This dialog allows for you to draw a mask for some (or all) of the images. Masks can be helpful"
                " in registration problems by focusing on a specific region of interest.",
                wrap=True,
                enable_url=True,
                alignment=Qt.AlignmentFlag.AlignCenter,
            )
        )
        layout.addRow(hp.make_h_line())
        layout.addRow(self.layer_controls)
        layout.addRow(hp.make_h_line())
        layout.addRow(hp.make_label(self, "Modality:"), self.slide_choice)
        layout.addRow(hp.make_label(self, "Show current"), self.only_current)
        layout.addRow(hp.make_label(self, "Horizontal"), self.horizontal_label)
        layout.addRow(hp.make_label(self, "Vertical"), self.vertical_label)
        layout.addRow(hp.make_h_layout(self.initialize_btn, self.add_btn, self.remove_btn))
        layout.addRow(hp.make_label(self, "Auto-update"), self.auto_update)
        return layout
