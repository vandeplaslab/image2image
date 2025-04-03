"""Item list."""

from __future__ import annotations

import typing as ty
from contextlib import suppress
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_reg.models import Modality, Preprocessing

# from koyo.color import get_next_color
from koyo.typing import PathLike
from loguru import logger
from napari.layers import Image
from qtextra.widgets.qt_button_icon import QtLockButton, QtVisibleButton
from qtpy.QtCore import QEvent, QRegularExpression, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QRegularExpressionValidator
from qtpy.QtWidgets import QDialog, QFrame, QGridLayout, QScrollArea, QSizePolicy, QWidget
from superqt import QLabeledDoubleSlider

from image2image.config import SingleAppConfig, get_elastix_config, get_valis_config
from image2image.qt._wsi._widgets import QtModalityLabel
from image2image.utils.utilities import ensure_list

if ty.TYPE_CHECKING:
    from image2image_reg.workflows import ElastixReg, ValisReg

    from image2image.qt._wsi._preprocessing import PreprocessingDialog
    from image2image.qt.dialog_elastix import ImageElastixWindow
    from image2image.qt.dialog_valis import ImageValisWindow


logger = logger.bind(src="QtModalityList")

PALETTE: dict[str, list[str]] | None = None


class QtModalityItem(QFrame):
    """Widget used to nicely display information about RGBImageChannel.

    This widget permits display of label information as well as modification of color.
    """

    evt_delete = Signal(Modality, bool)
    evt_show = Signal(Modality, bool)
    evt_rename = Signal(object, str)
    evt_resolution = Signal(Modality)
    evt_set_preprocessing = Signal(Modality)
    evt_hide_others = Signal(Modality)
    evt_preview_preprocessing = Signal(Modality, Preprocessing)
    evt_preview_transform_preprocessing = Signal(Modality, Preprocessing)
    evt_preprocessing_close = Signal(Modality)
    evt_color = Signal(Modality, object)
    _evt_color_changed = Signal(str, str)
    evt_mask = Signal(Modality, bool)
    evt_crop = Signal(Modality, bool)

    _preprocessing_dlg: PreprocessingDialog | None = None
    _mode: bool = False
    modality: Modality
    previewing: bool = False

    def __init__(self, modality: Modality, parent: QtModalityList, color="#808080", valis: bool = False):
        self.key = modality.name
        self.modality = modality
        super().__init__(parent)
        self.setMouseTracking(True)
        self._parent = parent
        self.valis = valis

        self.name_label = hp.make_line_edit(
            self,
            self.modality.name,
            tooltip="Name of the modality.",
            func=self._on_update_name,
        )
        self.resolution_label = hp.make_line_edit(
            self,
            tooltip="Resolution of the modality.",
            func=self._on_update_resolution,
            validator=QRegularExpressionValidator(QRegularExpression(r"^[0-9]+(\.[0-9]{1,5})?$")),
        )
        self.output_resolution_label = hp.make_line_edit(
            self,
            tooltip="Output resolution of the modality.<br>If you see an orange border, it means the output resolution"
            " is smaller than the input resolution.",
            func=self._on_update_output_resolution,
            func_clear=self._on_update_output_resolution_clear,
            validator=QRegularExpressionValidator(QRegularExpression(r"^[0-9]+(\.[0-9]{1,5})?$")),
        )
        self.preprocessing_btn = hp.make_qta_btn(
            self,
            "process",
            tooltip="Click here to set pre-processing parameters.",
            func=self.on_open_preprocessing,
            normal=True,
        )
        self.color_btn = hp.make_swatch(self, color, tooltip="Click here to change color.")
        self.color_btn.evt_color_changed.connect(self._on_change_color)
        self.attach_image_btn = hp.make_qta_btn(
            self,
            "image",
            tooltip="Click here to attach Image file. (.czi, .tiff)<br>Right-click to remove images.",
            normal=True,
            func=self.on_attach_image,
            func_menu=lambda _: self.on_edit_attachment(which="image"),
            properties={"with_count": True},
        )
        self.attach_geojson_btn = hp.make_qta_btn(
            self,
            "shapes",
            tooltip="Click here to attach GeoJSON file (.geojson).<br>Right-click to remove shapes.",
            normal=True,
            func=self.on_attach_geojson,
            func_menu=lambda _: self.on_edit_attachment(which="shapes"),
            properties={"with_count": True},
        )
        self.attach_points_btn = hp.make_qta_btn(
            self,
            "points",
            tooltip="Click here to attach points file (.csv, .txt).<br>Right-click to remove points.",
            normal=True,
            func=self.on_attach_points,
            func_menu=lambda _: self.on_edit_attachment(which="points"),
            properties={"with_count": True},
        )

        self.preprocessing_label = hp.make_scrollable_label(
            self,
            "<no pre-processing>",
            alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        self.preprocessing_label.evt_clicked.connect(self.on_open_preprocessing)
        hp.set_expanding_sizer_policy(self.preprocessing_label, vert=True)

        self.modality_icon = QtModalityLabel(self)
        self.open_dir_btn = hp.make_qta_btn(
            self, "folder", tooltip="Open directory containing the image.", normal=True, func=self.on_open_directory
        )
        self.remove_btn = hp.make_qta_btn(
            self,
            "delete",
            tooltip="Remove modality from the list.",
            normal=True,
            func=self.on_remove,
        )
        self.mask_btn = hp.make_qta_label(
            self,
            "mask",
            normal=True,
            tooltip="When mask is applied to the image, this icon will be visible.",
        )
        self.mask_btn.setHidden(self.valis)
        self.crop_btn = hp.make_qta_label(
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

        self.visible_btn = QtVisibleButton(self, state=True, auto_connect=True)
        self.visible_btn.setToolTip("Show/hide image from the canvas.")
        self.visible_btn.set_normal()
        self.visible_btn.evt_toggled.connect(self._on_show_image)

        self.opacity_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal, parent=self)
        self.opacity_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.opacity_slider.setMinimum(0.0)
        self.opacity_slider.setMaximum(1.0)
        self.opacity_slider.setSingleStep(0.01)
        self.opacity_slider.setValue(1.0)
        self.opacity_slider.valueChanged.connect(self.on_change_opacity)

        # self.contrast_slider = _QDoubleRangeSlider(Qt.Orientation.Horizontal, self)
        # # decimals = range_to_decimals(
        # #     self.layer.contrast_limits_range, self.layer.dtype
        # # )
        # # self.contrastLimitsSlider.setRange(*self.layer.contrast_limits_range)
        # # self.contrastLimitsSlider.setSingleStep(10**-decimals)
        # # self.contrastLimitsSlider.setValue(self.layer.contrast_limits)
        # self.contrast_slider.valueChanged.connect(self.on_change_contrast_limits)
        # self.contrast_slider.rangeChanged.connect(self.on_change_contrast_limits_range)
        # self.contrast_slider.setToolTip("Right click for detailed slider popup.")

        grid = QGridLayout(self)
        # widget, row, column, rowspan, colspan
        grid.setSpacing(0)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(3, True)
        grid.setColumnStretch(5, True)
        # column 1
        grid.addWidget(self.modality_icon, 0, 0)
        grid.addWidget(self.open_dir_btn, 1, 0)
        grid.addWidget(self.visible_btn, 2, 0)
        grid.addWidget(self.mask_btn, 3, 0)
        grid.addWidget(self.crop_btn, 4, 0)
        grid.addWidget(self.remove_btn, 5, 0)
        # column 2
        grid.addWidget(self.attach_image_btn, 3, 1)
        grid.addWidget(self.attach_geojson_btn, 4, 1)
        grid.addWidget(self.attach_points_btn, 5, 1)
        # column 3
        grid.addWidget(self.color_btn, 3, 2)
        grid.addWidget(self.lock_btn, 4, 2)
        grid.addWidget(self.preprocessing_btn, 5, 2)
        # column 2-3
        grid.addWidget(hp.make_label(self, "Name"), 0, 1, 1, 2)
        grid.addWidget(hp.make_label(self, "Pixel size"), 1, 1, 1, 2)
        grid.addWidget(hp.make_label(self, "Output size"), 1, 4, 1, 1)
        grid.addWidget(hp.make_label(self, "Opacity"), 2, 1, 1, 2)
        # column 4
        grid.addWidget(self.name_label, 0, 3, 1, 3)
        grid.addWidget(self.resolution_label, 1, 3)
        grid.addWidget(self.output_resolution_label, 1, 5)
        grid.addWidget(self.opacity_slider, 2, 3, 1, 3)
        grid.addWidget(self.preprocessing_label, 3, 3, 4, 3)

        self.mode = False
        self._set_from_model()
        self._old_hex_color = self.hex_color

        self.name_label.installEventFilter(self)
        self.resolution_label.installEventFilter(self)

    def eventFilter(self, recv, event):
        """Event filter."""
        if event.type() == QEvent.Type.FocusOut:
            modality = self.modality
            if not self.resolution_label.hasFocus() and self.resolution_label.text() == "":
                self.resolution_label.setText(f"{modality.pixel_size:.5f}")
            if not self.name_label.hasFocus() and self.name_label.text() == "":
                self.name_label.setText(modality.name)
        return super().eventFilter(recv, event)

    @property
    def layer(self) -> Image | None:
        """Get image layer."""
        layers = self._parent.view.layers
        if self.modality.name in layers:
            layer = layers[self.modality.name]
            return layer
        return None

    def on_change_opacity(self, value: float) -> None:
        """Update opacity."""
        if not self.layer:
            return
        layer = self.layer
        layer.opacity = value

    def _set_from_model(self, _: ty.Any = None) -> None:
        """Update UI elements."""
        self.name_label.setText(self.modality.name)
        self.modality_icon.state = "image"
        self.resolution_label.setText(f"{self.modality.pixel_size:.5f}")
        text, tooltip = self.modality.preprocessing.as_str(valis=self.valis)
        self.preprocessing_label.setText(text)
        self.preprocessing_label.setToolTip(tooltip)
        n = self.registration_model.get_attachment_count(self.modality.name, "image")
        self.attach_image_btn.set_count(n)
        n = self.registration_model.get_attachment_count(self.modality.name, "geojson")
        self.attach_geojson_btn.set_count(n)
        n = self.registration_model.get_attachment_count(self.modality.name, "points")
        self.attach_points_btn.set_count(n)
        if not self.valis:
            self.mask_btn.setVisible(self.modality.preprocessing.is_masked())
            self.crop_btn.setVisible(self.modality.preprocessing.is_cropped())
            self._set_output_resolution(self.modality.output_pixel_size)
        self.setToolTip(f"<b>Modality</b>: {self.modality.name}<br><b>Path</b>: {self.modality.path}")

    def on_open_directory(self) -> None:
        """Open directory where the image is located."""
        from koyo.path import open_directory_alt

        open_directory_alt(self.modality.path)

    def on_remove(self) -> None:
        """Remove image/modality from the list."""
        self.evt_delete.emit(self.modality, False)

    def on_force_remove(self) -> None:
        """Remove image/modality from the list."""
        self.evt_delete.emit(self.modality, True)

    @property
    def CONFIG(self) -> SingleAppConfig:
        """Return instance of configuration."""
        return get_valis_config() if self.valis else get_elastix_config()

    @property
    def registration_model(self) -> ElastixReg:
        """Get registration model."""
        parent = self._parent
        if hasattr(parent, "registration_model"):
            return parent.registration_model
        return self._parent.parent().registration_model

    def on_attach_image(self) -> None:
        """Attach Image file."""
        paths = hp.get_filename(
            self,
            "Attach image file",
            file_filter="Image files (*.czi *.tif *.tiff)",
            base_dir=self.CONFIG.last_dir,
            multiple=True,
        )
        if paths:
            self.CONFIG.update(last_dir=Path(paths[0]).parent)
            for path in paths:
                name = Path(path).stem.replace(".ome", "")
                self.registration_model.auto_add_attachment_images(self.modality.name, name, path)
            self._set_from_model()

    def _remove_modality(self, options: list[str], kind: str) -> list[str]:
        """Select options from a list to remove."""
        to_remove = hp.choose_from_list(self, options, text=f"Please choose <b>{kind}</b> attachment from the list.")
        return to_remove

    def on_edit_attachment(self, which: str = "all") -> None:
        """Remove Image file."""
        from image2image.qt._wsi._attachment import AttachmentEditDialog

        if not self.registration_model.has_attachments(self.modality.name):
            hp.toast(hp.get_main_window(), "Error", "No attachments found for this modality.", icon="warning")
            return

        dlg = AttachmentEditDialog(hp.get_main_window(self), self.modality, self.registration_model, which)
        dlg.show_in_center_of_screen(show=False)
        if dlg.exec_() == QDialog.DialogCode.Accepted:  # type: ignore[attr-defined]
            pass
        self._set_from_model()

    def _get_attachment_metadata(self) -> tuple[str | None, float | None]:
        """Return attachment metadata."""
        from image2image.qt._wsi._attachment import AttachWidget

        dlg = AttachWidget(hp.get_main_window(self), pixel_sizes=(1.0, self.modality.pixel_size))
        dlg.show_in_center_of_screen(show=False)
        if dlg.exec_() == QDialog.DialogCode.Accepted:  # type: ignore[attr-defined]
            return None, dlg.source_pixel_size
        return None, None

    def auto_add_attachments(self, filelist: list[str]) -> None:
        """Add any attachment"""
        shapes, points = [], []
        for file in filelist:
            file = Path(file)
            if file.suffix in [".geojson"]:
                shapes.append(file)
            elif file.suffix in [".csv", ".txt", ".tsv", ".parquet"]:
                points.append(file)
        if shapes or points:
            name, pixel_size = self._get_attachment_metadata()
            if name is None and pixel_size is None:
                hp.toast(hp.get_main_window(), "Error", "No attachment name or pixel size provided.", icon="warning")
                return
            if shapes:
                self._attach_shapes(shapes, name, pixel_size)
            if points:
                self._attach_points(points, name, pixel_size)
            self._set_from_model()

    def _attach_shapes(self, filelist: list[str], name: str | None, pixel_size: float | None) -> None:
        if name:
            self.registration_model.add_attachment_geojson(self.modality.name, name, filelist, pixel_size=pixel_size)
        else:
            for path in filelist:
                name = Path(path).stem
                self.registration_model.add_attachment_geojson(self.modality.name, name, [path], pixel_size=pixel_size)
        logger.debug(f"Attached {len(filelist)} GeoJSON files to {self.modality.name}.")

    def on_attach_geojson(self) -> None:
        """Attach GeoJSON file."""
        shapes = hp.get_filename(
            self,
            "Attach GeoJSON file",
            file_filter="GeoJSON files (*.geojson)",
            base_dir=self.CONFIG.last_dir,
            multiple=True,
        )
        if shapes:
            self.CONFIG.update(last_dir=Path(shapes[0]).parent)
            name, pixel_size = self._get_attachment_metadata()
            if name is None and pixel_size is None:
                hp.toast(self, "Error", "No attachment name or pixel size provided.", icon="warning")
                return
            self._attach_shapes(shapes, name, pixel_size)
            self._set_from_model()

    def _attach_points(self, filelist: list[PathLike], name: str | None, pixel_size: float | None) -> None:
        if name:
            self.registration_model.add_attachment_points(self.modality.name, name, filelist, pixel_size=pixel_size)
        else:
            for path in filelist:
                name = Path(path).stem
                self.registration_model.add_attachment_points(self.modality.name, name, [path], pixel_size=pixel_size)
        logger.debug(f"Attached {len(filelist)} point files to {self.modality.name}.")

    def on_attach_points(self) -> None:
        """Attach points file."""
        paths = hp.get_filename(
            self,
            "Attach points file",
            file_filter="Points files (*.csv *.txt *.tsv *.parquet)",
            base_dir=self.CONFIG.last_dir,
            multiple=True,
        )
        if paths:
            self.CONFIG.update(last_dir=Path(paths[0]).parent)
            name, pixel_size = self._get_attachment_metadata()
            if name is None and pixel_size is None:
                hp.toast(self, "Error", "No attachment name or pixel size provided.", icon="warning")
                return
            self._attach_points(paths, name, pixel_size)
            self._set_from_model()

    @property
    def color(self) -> np.ndarray:
        """Get color."""
        return self.color_btn.color

    @property
    def hex_color(self) -> str:
        """Get color."""
        color = self.color_btn.hex_color
        if len(color) > 7:  # remove the alpha channel
            color = color[:-2]
        return color.lower()

    @property
    def colormap(self) -> object:
        """Get colormap."""
        from napari.utils.colormaps import ensure_colormap

        color = self.hex_color
        colors = {
            "#ff0000": "red",
            "#00ff00": "green",
            "#0000ff": "blue",
            "#ffff00": "yellow",
            "#ff00ff": "magenta",
            "#00ffff": "cyan",
            "#808080": "gray",
        }
        return colors.get(color, ensure_colormap(color))

    def _on_lock_preprocessing(self, _state: bool = False) -> None:
        """Show image."""
        with suppress(RuntimeError, AttributeError):
            self._preprocessing_dlg.lock(self.lock_btn.locked)

    def _on_show_image(self, _state: bool = False) -> None:
        """Show image."""
        self.evt_show.emit(self.modality, self.visible_btn.state)

    def _on_change_color(self, _: ty.Any = None) -> None:
        """Change color."""
        self.evt_color.emit(self.modality, self.colormap)
        self._evt_color_changed.emit(self._old_hex_color, self.hex_color)
        self._old_hex_color = self.hex_color

    def _on_update_name(self) -> None:
        """Update name."""
        name = self.name_label.text()
        if not name or self.modality.name == name:
            return
        self.evt_rename.emit(self, name)

    def _on_update_resolution(self) -> None:
        """Update resolution."""
        resolution = self.resolution_label.text()
        if not resolution or float(resolution) == self.modality.pixel_size:
            return
        self.modality.pixel_size = float(resolution)
        self.evt_resolution.emit(self.modality)

    def _on_update_output_resolution(self) -> None:
        """ "Update output resolution."""
        resolution = self.output_resolution_label.text()
        self.modality.output_pixel_size = float(resolution) if resolution else None
        self._set_output_resolution(self.modality.output_pixel_size)

    def _on_update_output_resolution_clear(self) -> None:
        """ "Update output resolution."""
        self.modality.output_pixel_size = None
        self._set_output_resolution(self.modality.output_pixel_size)

    def _set_output_resolution(self, resolution: float | tuple[float, float] | None = None) -> None:
        """Set output resolution, and potentially warn if the resolution is a bit odd."""
        resolution = resolution[0] if isinstance(resolution, tuple) else resolution
        with hp.qt_signals_blocked(self.output_resolution_label):
            self.output_resolution_label.setText(f"{resolution:.5f}" if resolution is not None else "")
        hp.set_object_name(
            self.output_resolution_label,
            object_name="warning" if resolution is not None and resolution < self.modality.pixel_size else "",
        )

    def on_open_preprocessing(self) -> None:
        """Open pre-processing dialog."""
        from ._preprocessing import PreprocessingDialog

        self.previewing = True
        try:
            self._preprocessing_dlg.show_below_widget(self)
        except (RuntimeError, AttributeError):
            self._preprocessing_dlg = None

        if self._preprocessing_dlg is None:
            self._preprocessing_dlg = PreprocessingDialog(
                self.modality,
                self.hex_color,
                parent=self,
                locked=self.lock_btn.locked,
                valis=self.valis,
            )
            self._preprocessing_dlg.evt_update.connect(self._on_update_preprocessing)
            self._preprocessing_dlg.evt_preview_preprocessing.connect(self.evt_preview_preprocessing.emit)
            self._preprocessing_dlg.evt_set_preprocessing.connect(self.on_set_preprocessing)
            self._preprocessing_dlg.evt_preview_transform_preprocessing.connect(
                self.evt_preview_transform_preprocessing.emit
            )
            self._preprocessing_dlg.evt_close.connect(self._on_close_preprocessing)
            self._preprocessing_dlg.show_below_widget(self)
        self.evt_hide_others.emit(self.modality)

    def _on_close_preprocessing(self) -> None:
        self._preprocessing_dlg = None
        self.previewing = False
        self.evt_preprocessing_close.emit(self.modality)
        logger.trace(f"Pre-processing dialog closed for {self.modality.name}.")

    def on_preview(self) -> None:
        """Preview image"""
        self.evt_preview_preprocessing.emit(self.modality, self.modality.preprocessing)
        logger.trace(f"Pre-processing previewed for {self.modality.name}.")

    def on_update_preprocessing(self) -> None:
        """Update pre-processing"""
        self._on_update_preprocessing(self.modality.preprocessing)

    def _on_update_preprocessing(self, preprocessing: Preprocessing) -> None:
        """Update pre-processing."""
        text, tooltip = preprocessing.as_str(valis=self.valis)
        self.preprocessing_label.setText(text)
        self.preprocessing_label.setToolTip(tooltip)

    def on_set_preprocessing(self, preprocessing: Preprocessing) -> None:
        """Set pre-processing."""
        self.modality.preprocessing = preprocessing
        self._set_from_model()
        self._on_close_preprocessing()
        logger.debug(f"Pre-processing set for {self.modality.name}.")

    def toggle_name(self, disabled: bool) -> None:
        """Toggle name."""
        self.name_label.setReadOnly(disabled)
        tooltip = "Name of the modality."
        if disabled:
            tooltip += " (disabled because registration paths have already been defined)."
        self.name_label.setToolTip(tooltip)

    def toggle_mask(self) -> None:
        """Toggle name."""
        self.mask_btn.setVisible(self.modality.preprocessing.is_masked())
        self._on_update_preprocessing(self.modality.preprocessing)

    #
    def toggle_crop(self) -> None:
        """Toggle name."""
        self.crop_btn.setVisible(self.modality.preprocessing.is_cropped())
        self._on_update_preprocessing(self.modality.preprocessing)

    def toggle_visible(self, state: bool) -> None:
        """Toggle visibility icon."""
        with hp.qt_signals_blocked(self.visible_btn):
            self.visible_btn.state = not state


class QtModalityList(QScrollArea):
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

    def __init__(self, parent: ImageElastixWindow | ImageValisWindow, valis: bool = False):
        super().__init__(parent)
        self.view = parent.view
        self._parent = parent
        self.valis = valis
        self.widgets: dict[Path, QtModalityItem] = {}
        self._dataset_filters: list[str] = []

        # setup UI
        scroll_widget = QWidget()
        self.setWidget(scroll_widget)
        self._layout = hp.make_v_layout(parent=scroll_widget, spacing=2, margin=1, stretch_after=True)

        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # type: ignore[attr-defined]
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # type: ignore[attr-defined]
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # type: ignore[attr-defined]

    def model_widget_iter(self) -> ty.Generator[tuple[Modality, QtModalityItem]]:
        """Iterate over widgets."""
        for widget in self.widget_iter():
            yield widget.modality, widget

    def widget_iter(self) -> ty.Iterable[QtModalityItem]:
        """Iterate over widgets."""
        yield from self.widgets.values()

    def model_iter(self) -> ty.Iterable[QtModalityItem]:
        """Iterate over widgets."""
        for widget in self.widget_iter():
            yield widget.modality

    def key_iter(self) -> ty.Generator[str, None, None]:
        """Iterate over models."""
        yield from self.widgets.keys()

    def get_widget_for_modality(self, key: str | Path | Modality) -> QtModalityItem | None:
        """Get widget for specified item model."""
        if isinstance(key, Modality):
            key = key.path
        if key:
            return self.widgets.get(Path(key))
        return None

    @property
    def registration_model(self) -> ElastixReg | ValisReg:
        """Get registration model."""
        return self._parent.registration_model

    @property
    def used_colors(self) -> list[str]:
        """Refresh colors used by widgets."""
        used = []
        for widget in self.widget_iter():
            used.append(widget.hex_color)
        return used

    def update_preprocessing_info(self) -> None:
        """Update pre-processing info."""
        for widget in self.widget_iter():
            if widget:
                widget.on_update_preprocessing()

    def on_make_modality_item(self, modality: Modality) -> QtModalityItem:
        """Make dataset item."""
        index = self._layout.count() - 1
        color = get_next_color(index)
        while color in self.used_colors:
            color = get_next_color(index)
            index += 1

        widget = QtModalityItem(modality, parent=self, color=color, valis=self.valis)
        widget.evt_delete.connect(self.on_remove)
        # widget.evt_remove.connect(self.remove_item)
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
        self.widgets[Path(modality.path)] = widget
        self._layout.insertWidget(self._layout.count() - 1, widget)
        self.validate()
        hp.call_later(self, widget._on_show_image, 50)
        return widget

    def on_filter_by_dataset_name(self, filters: str | list[str]) -> None:
        """Filter by dataset name."""
        if filters == "":
            filters = []
        self._dataset_filters = ensure_list(filters)

        check_dataset = any(self._dataset_filters)

        for widget in self.widget_iter():
            modality = widget.modality
            visible = (
                True
                if not check_dataset
                else any(filter_ in modality.name.lower() for filter_ in self._dataset_filters)
            )
            widget.setVisible(visible)

    def validate(self) -> None:
        """Validate visibilities."""
        self.on_filter_by_dataset_name(self._dataset_filters)

    def on_show_all(self) -> None:
        """Show all modalities."""
        for _modality, widget in self.model_widget_iter():
            widget.visible_btn.set_state(True, trigger=True)

    def on_hide_all(self) -> None:
        """Hide all modalities."""
        for _modality, widget in self.model_widget_iter():
            widget.visible_btn.set_state(False, trigger=True)

    def on_change_colors(self, kind: str = "normal") -> None:
        """Update colors."""
        global PALETTE

        for index, widget in enumerate(self.widget_iter()):
            color = get_next_color(index, kind)
            widget.color_btn.set_color(color)

    def on_sort_modalities(self, kind: str = "name", reverse: bool = True) -> None:
        """Sort modalities by name."""
        from natsort import index_natsorted, order_by_index

        # get values by which to sort
        widgets = list(self.widget_iter())
        if kind == "name":
            sorted_by = [widget.modality.name for widget in widgets]
        else:
            sorted_by = [widget.modality.path for widget in widgets]
        # obtain order of indices
        indices = index_natsorted(sorted_by, reverse=reverse)
        with suppress(RuntimeError):
            for widget in widgets:
                self._layout.removeWidget(widget)
            for widget in order_by_index(widgets, indices):
                self._layout.insertWidget(0, widget)

    def on_remove(self, modality: Modality, force: bool = False) -> None:
        """Remove image/modality from the list."""
        if (
            force
            or not modality
            or hp.confirm(
                self,
                f"Are you sure you want to remove <b>{modality.name}</b> from the list?",
                "Please confirm.",
            )
        ):
            self.evt_delete.emit(modality)

    def populate(self) -> None:
        """Create list of items."""
        registration_model = self.registration_model
        for name in registration_model.get_image_modalities(with_attachment=False):
            modality = registration_model.modalities[name]
            if not self._check_existing(modality):
                self.on_make_modality_item(modality)
        # remove non-existing widgets
        keys = list(self.key_iter())
        for key in keys:
            modality = self.registration_model.get_modality(path=key)
            if not modality:
                self._remove_by_modality(key)
        logger.debug("Populated modality list.")

    def _check_existing(self, modality: Modality) -> bool:  # type: ignore[override]
        """Check whether model already exists."""
        return any(modality_ == modality for modality_ in self.model_iter())

    def remove_by_modality(self, modality: Modality) -> None:
        """Remove model."""
        self._remove_by_modality(modality)
        self.evt_delete.emit(modality)

    def _remove_by_modality(self, modality: str | Path | Modality) -> None:
        widget = self.get_widget_for_modality(modality)
        if widget:
            self._layout.removeWidget(widget)
        if widget:
            widget.deleteLater()
        path = Path(modality.path if isinstance(modality, Modality) else modality)
        self.widgets.pop(path, None)
        del widget

    def toggle_name(self, disabled: bool) -> None:
        """Toggle name."""
        for widget in self.widget_iter():
            widget.toggle_name(disabled)

    def toggle_mask(self, modality: Modality | None = None) -> None:
        """Toggle mask icon."""
        for widget in self.widget_iter():
            widget.toggle_mask()

    def toggle_crop(self, modality: Modality | None = None) -> None:
        """Toggle mask icon."""
        for widget in self.widget_iter():
            widget.toggle_crop()

    def toggle_visible(self, names: list[str]) -> None:
        """Toggle visibility icon of items."""
        for model, widget in self.model_widget_iter():
            widget.visible_btn.set_state(model.name not in names, trigger=False)

    def get_color(self, modality: Modality) -> str:
        """Get color."""
        widget = self.get_widget_for_modality(modality)
        return widget.hex_color


def get_next_color(n: int, kind: str = "normal") -> str:
    """Get next color."""
    import glasbey
    from koyo.color import get_random_hex_color

    global PALETTE

    if kind in ["protanomaly", "deuteranomaly", "tritanomaly", "normal", "bright"]:
        if PALETTE is None:
            PALETTE = {}

        if kind not in PALETTE:
            if kind in ["protanomaly", "deuteranomaly", "tritanomaly"]:
                PALETTE[kind] = glasbey.create_palette(palette_size=256, colorblind_safe=True, cvd_type=kind)
            elif kind == "bright":
                PALETTE[kind] = glasbey.create_palette(palette_size=256, lightness_bounds=(40, 90))
            else:
                PALETTE[kind] = glasbey.create_palette(palette_size=256)
        return PALETTE[kind][n]
    return get_random_hex_color()
