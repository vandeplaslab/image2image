"""Item list."""

from __future__ import annotations

import typing as ty

import qtextra.helpers as hp
from image2image_reg.models import Modality, Preprocessing
from loguru import logger
from qtextra.widgets.qt_image_button import QtVisibleButton
from qtextra.widgets.qt_list_widget import QtListItem, QtListWidget
from qtpy.QtCore import Qt, Signal, Slot  # type: ignore[attr-defined]
from qtpy.QtWidgets import QHBoxLayout, QListWidgetItem, QSizePolicy, QWidget

if ty.TYPE_CHECKING:
    from image2image_reg.workflows.iwsireg import IWsiReg


logger = logger.bind(src="QtModalityList")


class QtModalityItem(QtListItem):
    """Widget used to nicely display information about RGBImageChannel.

    This widget permits display of label information as well as modification of color.
    """

    evt_show = Signal(Modality, bool)
    evt_name = Signal(Modality)
    evt_resolution = Signal(Modality)
    evt_preview = Signal(Modality)
    evt_preprocessing = Signal(Modality)
    evt_preprocessing_preview = Signal(Modality, Preprocessing)

    _mode: bool = False
    item_model: Modality

    def __init__(self, item: QListWidgetItem, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.item = item

        self.name_label = hp.make_line_edit(
            self,
            self.item_model.name,
            tooltip="Name of the modality.",
            func=self.on_update_name,
        )
        self.resolution_label = hp.make_double_spin_box(
            self,
            value=0.0,
            tooltip="Resolution of the modality.",
            minimum=0.0001,
            maximum=10000,
            n_decimals=3,
            single_step=1,
            func=self.on_update_resolution,
        )
        self.preprocessing_btn = hp.make_qta_btn(
            self,
            "process",
            tooltip="Click here to set pre-processing parameters.",
            func=self.on_open_preprocessing,
            normal=True,
        )
        self.preprocessing_label = hp.make_scrollable_label(
            self, "<no pre-processing>", alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self.mask_icon = hp.make_qta_label(
            self,
            "mask",
            normal=True,
            retain_size=True,
            tooltip="When mask is applied to the image, this icon will be visible.",
        )
        self.crop_icon = hp.make_qta_label(
            self,
            "crop",
            normal=True,
            retain_size=True,
            tooltip="When cropping is applied to the image, this icon will be visible.",
        )

        self.visible_btn = QtVisibleButton(self)
        self.visible_btn.setToolTip("Show/hide image from the canvas.")
        self.visible_btn.set_normal()
        self.visible_btn.auto_connect()
        self.visible_btn.evt_toggled.connect(self.on_show_image)

        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.preprocessing_btn, alignment=Qt.AlignmentFlag.AlignTop)
        lay.addWidget(self.preprocessing_label, alignment=Qt.AlignmentFlag.AlignTop, stretch=True)

        layout = hp.make_form_layout()
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addRow(hp.make_label(self, "Name"), self.name_label)
        layout.addRow(hp.make_label(self, "Pixel size"), self.resolution_label)
        layout.addRow(hp.make_label(self, "Pre-processing"), lay)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(1, 1, 1, 1)
        main_layout.setSpacing(1)
        main_layout.addLayout(layout, stretch=True)
        main_layout.addLayout(
            hp.make_v_layout(self.visible_btn, self.mask_icon, self.crop_icon, stretch_after=True),
        )

        self.mode = False
        self._set_from_model()

    def _set_from_model(self, _: ty.Any = None) -> None:
        """Update UI elements."""
        self.name_label.setText(self.item_model.name)
        self.resolution_label.setValue(self.item_model.pixel_size)
        self.preprocessing_label.setText(self.item_model.preprocessing.as_str())
        self.mask_icon.setVisible(self.item_model.is_masked())  # TODO: change to pre-processing
        self.crop_icon.setVisible(self.item_model.preprocessing.is_cropped())

    def on_show_image(self, _state: bool = False) -> None:
        """Show image."""
        self.evt_show.emit(self.item_model, self.visible_btn.visible)

    def on_update_name(self) -> None:
        """Update name."""
        name = self.name_label.text()
        if name:
            self.item_model.name = name
            self.evt_name.emit(self.item_model)
        else:
            hp.set_object_name(self.name_label, "error")

    def on_update_resolution(self) -> None:
        """Update resolution."""
        self.item_model.pixel_size = self.resolution_label.value()
        self.evt_resolution.emit(self.item_model)

    def on_open_preprocessing(self) -> None:
        """Open pre-processing dialog."""
        from ._preprocessing import PreprocessingDialog

        dlg = PreprocessingDialog(self.item_model, parent=self)
        dlg.evt_update.connect(self.on_update_preprocessing)
        dlg.evt_preview.connect(self.evt_preview.emit)
        dlg.evt_preprocessing.connect(self.on_set_preprocessing)
        dlg.evt_preprocessing_preview.connect(self.evt_preprocessing_preview.emit)
        dlg.show()

    def on_update_preprocessing(self, preprocessing: Preprocessing) -> None:
        """Update pre-processing."""
        self.preprocessing_label.setText(preprocessing.as_str())

    def on_set_preprocessing(self, preprocessing: Preprocessing) -> None:
        """Set pre-processing."""
        self.item_model.preprocessing = preprocessing
        self._set_from_model()
        self.evt_preprocessing.emit(self.item_model)
        logger.debug(f"Pre-processing set for {self.item_model.name}.")

    def toggle_name(self, state: bool) -> None:
        """Toggle name."""
        self.name_label.setReadOnly(state)

    def toggle_mask(self) -> None:
        """Toggle name."""
        self.mask_icon.setVisible(self.item_model.is_masked())  # TODO: change to pre-processing

    def toggle_crop(self) -> None:
        """Toggle name."""
        self.crop_icon.setVisible(self.item_model.preprocessing.is_cropped())


class QtModalityList(QtListWidget):
    """List of notifications."""

    evt_show = Signal(Modality, bool)
    evt_name = Signal(Modality)
    evt_resolution = Signal(Modality)
    evt_preview = Signal(Modality)
    evt_preprocessing = Signal(Modality)
    evt_preprocessing_preview = Signal(Modality, Preprocessing)
    evt_remove = Signal(Modality)

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setSpacing(1)
        # self.setSelectionsMode(QListWidget.SingleSelection)
        self.setMinimumHeight(12)
        self.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.MinimumExpanding)
        self.setUniformItemSizes(True)
        self._parent = parent

    def _make_widget(self, item: QListWidgetItem) -> QtModalityItem:
        widget = QtModalityItem(item, parent=self)
        widget.evt_remove.connect(self.remove_item)
        widget.evt_show.connect(self.evt_show.emit)
        widget.evt_name.connect(self.evt_name.emit)
        widget.evt_resolution.connect(self.evt_resolution.emit)
        widget.evt_preview.connect(self.evt_preview.emit)
        widget.evt_preprocessing.connect(self.evt_preprocessing.emit)
        widget.evt_preprocessing_preview.connect(self.evt_preprocessing_preview.emit)
        widget.on_show_image()
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
        registration_model: IWsiReg = self._parent.registration_model
        for model in registration_model.modalities.values():
            self.append_item(model)
        logger.debug("Populated modality list.")

    def depopulate(self) -> None:
        """Remove list of items."""
        registration_model: IWsiReg = self._parent.registration_model
        for item in self.item_iter(reverse=True):
            if item.item_model not in registration_model.modalities.values():
                self.remove_item(item, force=True)

    def toggle_name(self, state: bool) -> None:
        """Toggle name."""
        for _, _, widget in self.item_model_widget_iter():
            widget.toggle_name(state)

    def toggle_mask(self, modality: Modality | None = None) -> None:
        """Toggle mask icon."""
        for _, _, widget in self.item_model_widget_iter():
            widget.toggle_mask()

    def toggle_crop(self, modality: Modality | None = None) -> None:
        """Toggle mask icon."""
        for _, _, widget in self.item_model_widget_iter():
            widget.toggle_crop()
