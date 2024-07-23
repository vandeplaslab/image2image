"""Item list."""

from __future__ import annotations

import typing as ty
from contextlib import suppress
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_reg.models import Modality, Preprocessing
from loguru import logger
from qtextra.widgets.qt_image_button import QtLockButton, QtVisibleButton
from qtextra.widgets.qt_list_widget import QtListItem, QtListWidget
from qtpy.QtCore import QRegularExpression, Qt, Signal, Slot  # type: ignore[attr-defined]
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QHBoxLayout, QListWidgetItem, QSizePolicy, QWidget

from image2image.config import CONFIG
from image2image.qt._wsireg._widgets import QtModalityLabel

if ty.TYPE_CHECKING:
    from image2image_reg.workflows import IWsiReg, ValisReg

    from image2image.qt._wsireg._preprocessing import PreprocessingDialog


logger = logger.bind(src="QtModalityList")


class QtModalityItem(QtListItem):
    """Widget used to nicely display information about RGBImageChannel.

    This widget permits display of label information as well as modification of color.
    """

    evt_delete = Signal(Modality)
    evt_show = Signal(Modality, bool)
    evt_rename = Signal(object, str)
    evt_resolution = Signal(Modality)
    evt_set_preprocessing = Signal(Modality)
    evt_hide_others = Signal(Modality)
    evt_preview_preprocessing = Signal(Modality, Preprocessing)
    evt_preview_transform_preprocessing = Signal(Modality, Preprocessing)
    evt_preprocessing_close = Signal(Modality)
    evt_color = Signal(Modality, object)
    evt_mask = Signal(Modality, bool)
    evt_crop = Signal(Modality, bool)

    _preprocessing_dlg: PreprocessingDialog | None = None
    _mode: bool = False
    item_model: Modality

    def __init__(self, item: QListWidgetItem, parent: QWidget | None = None, color="#808080", valis: bool = False):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.valis = valis
        self.item = item

        self.name_label = hp.make_line_edit(
            self,
            self.item_model.name,
            tooltip="Name of the modality.",
            func=self._on_update_name,
        )
        self.resolution_label = hp.make_line_edit(
            self,
            tooltip="Resolution of the modality.",
            func=self._on_update_resolution,
            validator=QRegularExpressionValidator(QRegularExpression(r"^\d*\.\d+$")),
        )
        self.preprocessing_btn = hp.make_qta_btn(
            self,
            "process",
            tooltip="Click here to set pre-processing parameters.",
            func=self.on_open_preprocessing,
            normal=True,
        )
        self.preview_btn = hp.make_qta_btn(
            self,
            "preview",
            tooltip="Click here to set immediately preview image parameters.",
            func=self.on_preview,
            normal=True,
        )
        self.color_btn = hp.make_swatch(self, color, tooltip="Click here to change color.")
        self.color_btn.evt_color_changed.connect(self._on_change_color)
        self.attach_image_btn = hp.make_qta_btn(
            self,
            "image",
            tooltip="Click here to attach Image file. (.czi, .tiff)",
            normal=True,
            func=self.on_attach_image,
            properties={"with_count": True},
        )
        self.attach_geojson_btn = hp.make_qta_btn(
            self,
            "shapes",
            tooltip="Click here to attach GeoJSON file (.geojson).",
            normal=True,
            func=self.on_attach_geojson,
            properties={"with_count": True},
        )
        self.attach_points_btn = hp.make_qta_btn(
            self,
            "points",
            tooltip="Click here to attach points file (.csv, .txt).",
            normal=True,
            func=self.on_attach_points,
            properties={"with_count": True},
        )

        self.preprocessing_label = hp.make_scrollable_label(
            self, "<no pre-processing>", alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.preprocessing_label.evt_clicked.connect(self.on_open_preprocessing)

        self.modality_icon = QtModalityLabel(self)
        self.remove_btn = hp.make_qta_btn(
            self,
            "delete",
            tooltip="Remove modality from the list.",
            normal=True,
            func=lambda: self.evt_delete.emit(self.item_model),
        )
        self.mask_btn = hp.make_qta_btn(
            self,
            "mask",
            normal=True,
            tooltip="When mask is applied to the image, this icon will be visible.",
        )
        self.mask_btn.setHidden(self.valis)
        self.crop_btn = hp.make_qta_btn(
            self,
            "crop",
            normal=True,
            tooltip="When cropping is applied to the image, this icon will be visible.",
        )
        self.crop_btn.setHidden(self.valis)

        self.lock_btn = QtLockButton(self)
        self.lock_btn.setToolTip("Prevent spatial transformation to the layer.")
        self.lock_btn.set_normal()
        self.lock_btn.auto_connect()
        self.lock_btn.evt_toggled.connect(self._on_lock_preprocessing)

        self.visible_btn = QtVisibleButton(self)
        self.visible_btn.setToolTip("Show/hide image from the canvas.")
        self.visible_btn.set_normal()
        self.visible_btn.auto_connect()
        self.visible_btn.evt_toggled.connect(self._on_show_image)

        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addLayout(hp.make_v_layout(self.preprocessing_btn, self.preview_btn, self.color_btn, stretch_after=True))
        lay.addLayout(
            hp.make_v_layout(self.attach_image_btn, self.attach_geojson_btn, self.attach_points_btn, stretch_after=True)
        )
        lay.addWidget(self.preprocessing_label, alignment=Qt.AlignmentFlag.AlignTop, stretch=True)

        layout = hp.make_form_layout()
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addRow(hp.make_label(self, "Name"), self.name_label)
        layout.addRow(hp.make_label(self, "Pixel size"), self.resolution_label)
        layout.addRow(hp.make_label(self, "Process"), lay)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(1, 1, 1, 1)
        main_layout.setSpacing(1)
        main_layout.addLayout(
            hp.make_v_layout(
                self.modality_icon,
                self.remove_btn,
                self.visible_btn,
                self.lock_btn,
                self.mask_btn,
                self.crop_btn,
                stretch_after=True,
                widget_alignment=Qt.AlignmentFlag.AlignCenter,
            ),
        )
        main_layout.addLayout(layout, stretch=True)

        self.mode = False
        self._set_from_model()

    def _set_from_model(self, _: ty.Any = None) -> None:
        """Update UI elements."""
        self.name_label.setText(self.item_model.name)
        self.modality_icon.state = "image"
        self.resolution_label.setText(f"{self.item_model.pixel_size:.3f}")
        text, tooltip = self.item_model.preprocessing.as_str()
        self.preprocessing_label.setText(text)
        self.preprocessing_label.setToolTip(tooltip)
        n = self.registration_model.get_attachment_count(self.item_model.name, "image")
        self.attach_image_btn.set_count(n)
        n = self.registration_model.get_attachment_count(self.item_model.name, "geojson")
        self.attach_geojson_btn.set_count(n)
        n = self.registration_model.get_attachment_count(self.item_model.name, "points")
        self.attach_points_btn.set_count(n)
        if not self.valis:
            self.mask_btn.setVisible(self.item_model.preprocessing.is_masked())
            self.crop_btn.setVisible(self.item_model.preprocessing.is_cropped())

    @property
    def registration_model(self) -> IWsiReg:
        """Get registration model."""
        parent = self.parent()
        if hasattr(parent, "registration_model"):
            return parent.registration_model
        return self.parent().parent().registration_model

    def on_attach_image(self) -> None:
        """Attach Image file."""
        path = hp.get_filename(
            self, "Attach image file", file_filter="Image files (*.czi *.tiff)", base_dir=CONFIG.last_dir
        )
        if path:
            CONFIG.last_dir = Path(path).parent
            name = Path(path).stem.replace(".ome", "")
            self.registration_model.auto_add_attachment_images(self.item_model.name, name, path)
            self._set_from_model()

    def on_attach_geojson(self) -> None:
        """Attach GeoJSON file."""
        path = hp.get_filename(
            self, "Attach GeoJSON file", file_filter="GeoJSON files (*.geojson)", base_dir=CONFIG.last_dir
        )
        if path:
            CONFIG.last_dir = Path(path).parent
            name = Path(path).stem
            self.registration_model.add_attachment_geojson(self.item_model.name, name, [path])
            self._set_from_model()

    def on_attach_points(self) -> None:
        """Attach points file."""
        path = hp.get_filename(
            self, "Attach points file", file_filter="Points files (*.csv *.txt)", base_dir=CONFIG.last_dir
        )
        if path:
            CONFIG.last_dir = Path(path).parent
            name = Path(path).stem
            self.registration_model.add_attachment_points(self.item_model.name, name, [path])
            self._set_from_model()

    @property
    def color(self) -> np.ndarray:
        """Get color."""
        return self.color_btn.color

    @property
    def hex_color(self) -> str:
        """Get color."""
        return self.color_btn.hex_color

    @property
    def colormap(self) -> object:
        """Get colormap."""
        from qtextra.utils.colormap import napari_colormap

        color = self.color_btn.hex_color.lower()
        colors = {
            "#ff0000ff": "red",
            "#00ff00ff": "green",
            "#0000ffff": "blue",
            "#ffff00ff": "yellow",
            "#ff00ffff": "magenta",
            "#00ffffff": "cyan",
            "#808080ff": "gray",
        }
        return colors.get(color, napari_colormap(color, name=color))

    def _on_lock_preprocessing(self, _state: bool = False) -> None:
        """Show image."""
        with suppress(RuntimeError, AttributeError):
            self._preprocessing_dlg.lock(self.lock_btn.locked)

    def _on_show_image(self, _state: bool = False) -> None:
        """Show image."""
        self.evt_show.emit(self.item_model, self.visible_btn.visible)

    def _on_change_color(self, _: ty.Any = None) -> None:
        """Change color."""
        self.evt_color.emit(self.item_model, self.colormap)

    def _on_update_name(self) -> None:
        """Update name."""
        self.evt_rename.emit(self, self.name_label.text())

    def _on_update_resolution(self) -> None:
        """Update resolution."""
        resolution = self.resolution_label.text()
        if not resolution:
            return
        self.item_model.pixel_size = float(resolution)
        self.evt_resolution.emit(self.item_model)

    def on_open_preprocessing(self) -> None:
        """Open pre-processing dialog."""
        from ._preprocessing import PreprocessingDialog

        try:
            self._preprocessing_dlg.show()
        except (RuntimeError, AttributeError):
            self._preprocessing_dlg = None

        if self._preprocessing_dlg is None:
            self._preprocessing_dlg = PreprocessingDialog(
                self.item_model, parent=self, locked=self.lock_btn.locked, valis=self.valis
            )
            self._preprocessing_dlg.evt_update.connect(self.on_update_preprocessing)
            self._preprocessing_dlg.evt_preview_preprocessing.connect(self.evt_preview_preprocessing.emit)
            self._preprocessing_dlg.evt_set_preprocessing.connect(self.on_set_preprocessing)
            self._preprocessing_dlg.evt_preview_transform_preprocessing.connect(
                self.evt_preview_transform_preprocessing.emit
            )
            self._preprocessing_dlg.evt_close.connect(self._on_close_preprocessing)
            self._preprocessing_dlg.show_below_mouse()
        self.evt_hide_others.emit(self.item_model)

    def _on_close_preprocessing(self) -> None:
        self._preprocessing_dlg = None
        self.evt_preprocessing_close.emit(self.item_model)
        logger.trace(f"Pre-processing dialog closed for {self.item_model.name}.")

    def on_preview(self) -> None:
        """Preview image"""
        self.evt_preview_preprocessing.emit(self.item_model, self.item_model.preprocessing)
        logger.trace(f"Pre-processing previewed for {self.item_model.name}.")

    def on_update_preprocessing(self, preprocessing: Preprocessing) -> None:
        """Update pre-processing."""
        text, tooltip = preprocessing.as_str(valis=self.valis)
        self.preprocessing_label.setText(text)
        self.preprocessing_label.setToolTip(tooltip)

    def on_set_preprocessing(self, preprocessing: Preprocessing) -> None:
        """Set pre-processing."""
        self.item_model.preprocessing = preprocessing
        self._set_from_model()
        self.evt_set_preprocessing.emit(self.item_model)
        logger.debug(f"Pre-processing set for {self.item_model.name}.")

    def toggle_name(self, disabled: bool) -> None:
        """Toggle name."""
        self.name_label.setReadOnly(disabled)

    def toggle_mask(self) -> None:
        """Toggle name."""
        self.mask_btn.setVisible(self.item_model.preprocessing.is_masked())
        self.on_update_preprocessing(self.item_model.preprocessing)

    def toggle_crop(self) -> None:
        """Toggle name."""
        self.crop_btn.setVisible(self.item_model.preprocessing.is_cropped())
        self.on_update_preprocessing(self.item_model.preprocessing)

    def toggle_preview(self, disabled: bool) -> None:
        """Toggle name."""
        hp.disable_widgets(self.preview_btn, disabled=disabled)

    def toggle_visible(self, disabled: bool) -> None:
        """Toggle visibility icon."""
        with hp.qt_signals_blocked(self.visible_btn):
            self.visible_btn.visible = not disabled


class QtModalityList(QtListWidget):
    """List of notifications."""

    evt_delete = Signal(Modality)
    evt_show = Signal(Modality, bool)
    evt_rename = Signal(object, str)
    evt_name = Signal(str, Modality)
    evt_resolution = Signal(Modality)
    evt_hide_others = Signal(Modality)
    evt_set_preprocessing = Signal(Modality)
    evt_preview_preprocessing = Signal(Modality, Preprocessing)
    evt_preview_transform_preprocessing = Signal(Modality, Preprocessing)
    evt_preprocessing_close = Signal(Modality)
    evt_remove = Signal(Modality)
    evt_color = Signal(Modality, object)
    evt_mask = Signal(Modality, bool)
    evt_crop = Signal(Modality, bool)

    def __init__(self, parent: QWidget, valis: bool = False):
        super().__init__(parent)
        self.setSpacing(1)
        # self.setSelectionsMode(QListWidget.SingleSelection)
        self.setMinimumHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.setUniformItemSizes(True)
        self._parent = parent
        self.valis = valis

    @property
    def registration_model(self) -> IWsiReg | ValisReg:
        """Get registration model."""
        return self._parent.registration_model

    def _make_widget(self, item: QListWidgetItem) -> QtModalityItem:
        try:
            colors = [widget.hex_color.lower() for _, _, widget in self.item_model_widget_iter()]
        except AttributeError:
            colors = []
        color = get_next_color(self.count(), other_colors=colors)

        widget = QtModalityItem(item, parent=self, color=color, valis=self.valis)
        widget.evt_delete.connect(self.evt_delete.emit)
        widget.evt_remove.connect(self.remove_item)
        widget.evt_show.connect(self.evt_show.emit)
        widget.evt_rename.connect(self.evt_rename.emit)
        widget.evt_resolution.connect(self.evt_resolution.emit)
        widget.evt_hide_others.connect(self.evt_hide_others.emit)
        widget.evt_preview_preprocessing.connect(self.evt_preview_preprocessing.emit)
        widget.evt_set_preprocessing.connect(self.evt_set_preprocessing.emit)
        widget.evt_preview_transform_preprocessing.connect(self.evt_preview_transform_preprocessing.emit)
        widget.evt_preprocessing_close.connect(self.evt_preprocessing_close.emit)
        widget.evt_color.connect(self.evt_color.emit)
        widget.evt_mask.connect(self.evt_mask.emit)
        widget.evt_crop.connect(self.evt_crop.emit)
        hp.call_later(self, widget._on_show_image, 50)
        return widget

    @Slot(QListWidgetItem)  # type: ignore[misc]
    def remove_item(self, item: QListWidgetItem, force: bool = False):
        """Remove item from the list."""
        item_model: Modality = item.item_model
        if force or hp.confirm(
            self, f"Are you sure you want to remove <b>{item_model.name}</b> from the list?", "Please confirm."
        ):
            super().remove_item(item, force)

    def _check_existing(self, item_model: Modality) -> bool:
        """Check whether model already exists."""
        for item_model_ in self.model_iter():  # noqa: SIM110; type: ignore[var-annotated]
            if item_model_ == item_model:
                return True
        return False

    def populate(self) -> None:
        """Create list of items."""
        registration_model = self.registration_model
        for model in registration_model.modalities.values():
            if registration_model.is_attachment(model.name):
                continue
            self.append_item(model)
        logger.debug("Populated modality list.")

    def depopulate(self) -> None:
        """Remove list of items."""
        registration_model = self.registration_model
        for item in self.item_iter(reverse=True):
            if item.item_model not in registration_model.modalities.values():
                self.remove_item(item, force=True)

    def toggle_name(self, disabled: bool) -> None:
        """Toggle name."""
        for _, _, widget in self.item_model_widget_iter():
            widget.toggle_name(disabled)

    def toggle_mask(self, modality: Modality | None = None) -> None:
        """Toggle mask icon."""
        for _, _, widget in self.item_model_widget_iter():
            widget.toggle_mask()

    def toggle_crop(self, modality: Modality | None = None) -> None:
        """Toggle mask icon."""
        for _, _, widget in self.item_model_widget_iter():
            widget.toggle_crop()

    def toggle_preview(self, disabled: bool) -> None:
        """Toggle toggle button."""
        for _, _, widget in self.item_model_widget_iter():
            widget.toggle_preview(disabled)

    def toggle_visible(self, names: list[str]) -> None:
        """Toggle visibility icon of items."""
        for _, model, widget in self.item_model_widget_iter():
            widget.toggle_visible(model.name not in names)


def get_next_color(n: int, other_colors: list[str] | None = None) -> str:
    """Get next color based on the number of items."""
    if other_colors is None:
        other_colors = []
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#f00ff", "#00ffff"]
    if n < len(colors):
        color = colors[n]
        if color in other_colors:
            n += 1
            return get_next_color(n, other_colors=other_colors)
        return color
    return "#808080"
