"""Mask dialog."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager, suppress
from functools import partial
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
    from image2image_io.readers import BaseReader
    from image2image_reg.models import Modality, Preprocessing
    from qtextra._napari.image.wrapper import NapariImageView

    from image2image.qt.dialog_wsireg import ImageWsiRegWindow


logger = logger.bind(src="MaskDialog")


def get_transform_mask(modality: Modality) -> bool:
    """Transform mask for modality."""
    pre = modality.preprocessing
    return pre.translate_x == 0 and pre.translate_y == 0 and pre.rotate_counter_clockwise == 0


def get_affine(reader: BaseReader, preprocessing: Preprocessing) -> tuple[np.ndarray, np.ndarray]:
    """Get affine matrix for preprocessing."""
    from image2image.utils.transform import combined_transform

    if preprocessing is None:
        return np.eye(3), np.eye(3)

    shape = reader.image_shape
    scale = reader.scale_for_pyramid(-1)
    affine = combined_transform(
        shape,
        scale,
        preprocessing.rotate_counter_clockwise,
        (preprocessing.translate_y, preprocessing.translate_x),
        flip_lr=preprocessing.flip == "h",
        flip_ud=preprocessing.flip == "v",
    )
    return affine, np.linalg.inv(affine)


class ShapesDialog(QtFramelessTool):
    """Dialog to mask a group."""

    HIDE_WHEN_CLOSE = True
    _editing = False
    TITLE: str
    INFO_LABEL: str
    MASK_OR_CROP: str

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
        options = self._parent.registration_model.modalities.keys()
        hp.combobox_setter(self.slide_choice, items=options, set_item=current)
        current = self.copy_from_choice.currentText()
        hp.combobox_setter(self.copy_from_choice, items=options, set_item=current)

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

    def is_current_masked(self) -> bool:
        """Check if current modality is masked."""
        name = self.slide_choice.currentText()
        if name:
            modality = self._parent.registration_model.modalities[name]
            return modality.preprocessing.is_masked() if self.MASK_OR_CROP == "mask" else modality.is_cropped()
        return False

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
        if n == 0:
            return
        if self.auto_update.isChecked() and self.is_current_masked():
            self.on_associate_mask_with_modality()

    def _get_crop_area_for_index(
        self,
    ) -> ty.Generator[tuple[int, int, int, int, str | None, np.ndarray | None], None, None]:
        """Return crop area."""
        n = len(self.mask_layer.data)
        if n == 0:
            return 0, 0, 0, 0, None, None
        for index in range(n):
            array = self.mask_layer.data[index]
            shape_type = self.mask_layer.shape_type[index]
            rect = self.mask_layer.interaction_box(index)
            rect = rect[Box.LINE_HANDLE]
            xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
            xmin = max(0, xmin)
            ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
            ymin = max(0, ymin)
            yield floor(xmin), ceil(xmax), floor(ymin), ceil(ymax), shape_type, array

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

    def on_remove_mask(self) -> None:
        """Remove mask."""
        self.view.remove_layer(self.mask_layer)

    def hide(self):
        """Hide dialog."""
        self.on_hide_mask()
        super().hide()

    def _transform_from_preprocessing(self, modality: Modality) -> list | None:
        """Transform data from stored data (which is stored in the micron units."""
        wrapper = self._parent.data_model.get_wrapper()
        reader = wrapper.get_reader_for_path(modality.path)
        polygon = getattr(modality.preprocessing, f"{self.MASK_OR_CROP}_polygon")
        bbox = getattr(modality.preprocessing, f"{self.MASK_OR_CROP}_bbox")
        if polygon is None:
            if bbox is None:
                return []
            data = []
            for left, top, width, height in zip(bbox.x, bbox.y, bbox.width, bbox.height):
                right = left + width
                bottom = top + height
                data.append(
                    (
                        np.asarray([(top, left), (top, right), (bottom, right), (bottom, left)]) * reader.resolution,
                        "rectangle",
                    )
                )
            return data
        data = []
        yx = polygon.xy
        if isinstance(yx, list):
            for xy in yx:
                data.append((np.asarray(xy[:, ::-1]) * reader.resolution, "polygon"))
        return data

    def _transform_to_preprocessing(self, modality: Modality) -> tuple[np.ndarray | None, tuple[int] | None]:
        """Transform data from napari layer to preprocessing data."""
        wrapper = self._parent.data_model.get_wrapper()
        reader = wrapper.get_reader_for_path(modality.path)
        # _, inv_affine = get_affine(reader, modality.preprocessing)
        inv_affine = np.eye(3)
        yx_, bbox_ = [], []
        for left, right, top, bottom, shape_type, yx in self._get_crop_area_for_index():
            if shape_type == "polygon":
                # convert to pixel coordinates and round
                yx = np.dot(inv_affine[:2, :2], yx.T).T + inv_affine[:2, 2]
                yx = yx * reader.inv_resolution
                yx_.append(np.round(yx).astype(np.int32)[:, ::-1])
            else:
                bbox = np.asarray([left, top, (right - left), (bottom - top)])
                bbox_.append(tuple(np.round(bbox * reader.inv_resolution).astype(int)))
        return yx_ or None, bbox_ or None

    def on_select_modality(self, _=None) -> None:
        """Select mask for the currently selected group."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        data = self._transform_from_preprocessing(modality)
        self.mask_layer.data = data or []

        # hide all other image layers
        if self.only_current.isChecked():
            self._parent.on_hide_modalities(modality)

    def on_increment_modality(self, increment_by: int) -> None:
        """Increment modality."""
        count = self.slide_choice.count()
        index = self.slide_choice.currentIndex()
        index = (index + increment_by) % count
        self.slide_choice.setCurrentIndex(index)
        # self.on_select_modality()
        logger.trace(f"Changed modality to {index}")

    def _check_if_has_mask(self, modality: Modality) -> bool:
        """Associate mask with modality at the specified location."""
        raise NotImplementedError("Must implement method")

    def on_associate_mask_with_modality(self) -> None:
        """Associate mask with modality at the specified location."""
        raise NotImplementedError("Must implement method")

    def on_dissociate_mask_from_modality(self) -> None:
        """Dissociate mask from modality."""
        raise NotImplementedError("Must implement method")

    def on_copy(self) -> None:
        """Copy mask from another modality."""
        copy_to = self.slide_choice.currentText()
        copy_from = self.copy_from_choice.currentText()
        if copy_to == copy_from:
            hp.toast(self, "Cannot copy mask", "Cannot copy from and to the same modality", icon="warning")
            return
        if copy_from:
            copy_from_modality = self._parent.registration_model.modalities[copy_from]
            if not self._check_if_has_mask(copy_from_modality):
                return

            # copy_to_modality = self._parent.registration_model.modalities[copy_to]
            data = self._transform_from_preprocessing(copy_from_modality)
            if data:
                self.mask_layer.data = data or []
                logger.trace(f"Copied mask from {copy_from}")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QLayout:
        """Make panel."""
        self.layer_controls = QtShapesControls(self.mask_layer)
        # disable certain modes
        hp.disable_widgets(
            self.layer_controls.line_button,
            self.layer_controls.path_button,
            self.layer_controls.ellipse_button,
            self.layer_controls.editable_checkbox,
            disabled=True,
        )

        self.slide_choice = hp.make_combobox(self, tooltip="Select modality", func=self.on_select_modality)
        self.previous_btn = hp.make_qta_btn(
            self,
            "chevron_left_circle",
            func=partial(self.on_increment_modality, increment_by=-1),
            tooltip="Go to previous modality.",
        )
        self.next_btn = hp.make_qta_btn(
            self,
            "chevron_right_circle",
            func=partial(self.on_increment_modality, increment_by=1),
            tooltip="Go to next modality.",
        )
        self.copy_from_choice = hp.make_combobox(self, tooltip="Copy mask from another modality")

        self.only_current = hp.make_checkbox(
            self, tooltip="Hide other modalities when drawing a mask.", func=self.on_select_modality
        )
        self.add_btn = hp.make_btn(self, f"Add {self.MASK_OR_CROP}", func=self.on_associate_mask_with_modality)
        self.remove_btn = hp.make_btn(self, f"Remove {self.MASK_OR_CROP}", func=self.on_dissociate_mask_from_modality)
        self.auto_update = hp.make_checkbox(self, "", tooltip="Auto-associate mask when making changes", value=True)

        layout = hp.make_form_layout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addRow(self._make_hide_handle(self.TITLE)[1])
        layout.addRow(
            hp.make_label(self, self.INFO_LABEL, wrap=True, enable_url=True, alignment=Qt.AlignmentFlag.AlignCenter)
        )
        layout.addRow(hp.make_h_line())
        layout.addRow(self.layer_controls)
        layout.addRow(hp.make_h_line())
        layout.addRow(
            hp.make_label(self, "Modality"),
            hp.make_h_layout(self.slide_choice, self.previous_btn, self.next_btn, stretch_id=(0,), spacing=2),
        )
        layout.addRow(
            hp.make_label(self, "Copy from"),
            hp.make_h_layout(
                self.copy_from_choice, hp.make_btn(self, "Copy", func=self.on_copy), spacing=2, stretch_id=(0,)
            ),
        )
        layout.addRow(hp.make_label(self, "Hide other"), self.only_current)
        layout.addRow(hp.make_h_layout(self.add_btn, self.remove_btn))
        layout.addRow(hp.make_label(self, "Auto-update"), self.auto_update)
        return layout


class MaskDialog(ShapesDialog):
    """Mask dialog."""

    TITLE = "Mask"
    INFO_LABEL = (
        "This dialog allows for you to draw a mask for some (or all) of the images. Masks can be helpful"
        " in registration problems by focusing on a specific region of interest."  # <br>"
        # "<b>Please remember that masks should be created on the original image (not translated or rotated)."
        # "<br>Transformations will be applied during the registration process!</b>"
    )
    MASK_OR_CROP = "mask"

    def _check_if_has_mask(self, modality: Modality) -> bool:
        """Check if modality has mask."""
        if not modality.preprocessing.is_masked():
            hp.toast(self, "Cannot copy mask", "No mask associated with this modality", icon="warning")
            return False
        return True

    def on_associate_mask_with_modality(self) -> None:
        """Associate mask with modality at the specified location."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        yx, bbox = self._transform_to_preprocessing(modality)
        if yx is not None:
            modality.preprocessing.use_mask = True
            modality.preprocessing.mask_polygon = yx
            modality.preprocessing.mask_bbox = None
        elif bbox is not None:
            modality.preprocessing.use_mask = True
            modality.preprocessing.mask_bbox = bbox
            modality.preprocessing.mask_polygon = None
        else:
            modality.preprocessing.use_mask = False
        modality.preprocessing.transform_mask = False
        kind = "polygon" if yx is not None else "bbox"
        logger.trace(f"Added mask for modality {name} to {kind}")
        self.evt_mask.emit(modality)

    def on_dissociate_mask_from_modality(self) -> None:
        """Dissociate mask from modality."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        modality.preprocessing.mask_polygon = None
        modality.preprocessing.mask_bbox = None
        logger.trace(f"Removed mask for modality {name}")
        self.evt_mask.emit(modality)


class CropDialog(ShapesDialog):
    """Mask dialog."""

    TITLE = "Crop"
    INFO_LABEL = (
        "This dialog allows for you draw a shape for some (or all) of the images. This can be helpful"
        " by cropping the image ahead of registration."  # <br>"
        # "<b>Please remember that masks should be created on the original image (not translated or rotated)."
        # "<br>Transformations will be applied during the registration process!</b>"
    )
    MASK_OR_CROP = "crop"

    def _check_if_has_mask(self, modality: Modality) -> bool:
        """Check if modality has mask."""
        if not modality.preprocessing.is_cropped():
            hp.toast(self, "Cannot copy mask", "No crop associated with this modality", icon="warning")
            return False
        return True

    def on_associate_mask_with_modality(self) -> None:
        """Associate mask with modality at the specified location."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        yx, bbox = self._transform_to_preprocessing(modality)
        if yx is not None and bbox is not None:
            hp.warn("Cannot have both polygon and bbox for crop")
            return
        if yx is not None:
            modality.preprocessing.use_crop = True
            modality.preprocessing.crop_polygon = yx
            modality.preprocessing.crop_bbox = None
        elif bbox is not None:
            modality.preprocessing.use_crop = True
            modality.preprocessing.crop_bbox = bbox
            modality.preprocessing.crop_polygon = None
        else:
            modality.preprocessing.use_crop = False
        modality.transform_mask = False
        kind = "polygon" if yx is not None else "bbox"
        logger.trace(f"Added crop for modality {name} to {kind}")
        self.evt_mask.emit(modality)

    def on_dissociate_mask_from_modality(self) -> None:
        """Dissociate mask from modality."""
        name = self.slide_choice.currentText()
        modality = self._parent.registration_model.modalities[name]
        modality.preprocessing.crop_polygon = None
        modality.preprocessing.crop_bbox = None
        logger.trace(f"Removed crop for modality {name}")
        self.evt_mask.emit(modality)
