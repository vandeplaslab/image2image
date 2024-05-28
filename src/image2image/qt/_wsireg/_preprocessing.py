"""Pre-processing dialog."""

from __future__ import annotations

import typing as ty
from copy import deepcopy

from koyo.secret import hash_parameters
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFormLayout, QWidget

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality


class PreprocessingDialog(QtFramelessTool):
    """Pre-processing."""

    evt_update = Signal(object)  # used to update the model
    evt_preview = Signal(object, object)  # used to preview the entire preprocessing pipeline
    evt_preprocessing_preview = Signal(object, object)  # used to preview the preprocessing (spatial)
    evt_preprocessing = Signal(object)  # used to set the preprocessing

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("index", "channel_index", "int", 100)
        .add("name", "channel_name", "str", 250)
    )

    def __init__(self, modality: Modality, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        self.modality = modality
        self.preprocessing = deepcopy(modality.preprocessing)
        self.original_hash = hash_parameters(**self.preprocessing.to_dict())
        self.set_from_model()

    def set_from_model(self):
        """Set from model."""
        with self.setting_config():
            self.type_choice.setCurrentIndex({"BF": 0, "FL": 1}[self.preprocessing.image_type])
            self.mip_check.setChecked(self.preprocessing.max_intensity_projection)
            self.contrast_check.setChecked(self.preprocessing.contrast_enhance)
            self.invert_check.setChecked(self.preprocessing.invert_intensity)
            self.uint8_check.setChecked(self.preprocessing.as_uint8)
            self.flip_choices.setCurrentIndex({"H": 1, "V": 2}.get(self.preprocessing.flip, 0))
            self.translate_x.setValue(self.preprocessing.translate_x)
            self.translate_y.setValue(self.preprocessing.translate_y)
            self.rotate_spin.setValue(self.preprocessing.rotate_counter_clockwise)
            self.downsample_spin.setValue(self.preprocessing.downsample)
            self.set_selected_channels()

    def set_selected_channels(self) -> None:
        """Set selected channels."""
        channel_names = self.preprocessing.channel_names
        channel_ids = self.preprocessing.channel_indices
        data = []
        for i, name in enumerate(channel_names):
            data.append([i in channel_ids, i, name])
        self.channel_table.reset_data()
        self.channel_table.add_data(data)

    def get_selected_channels(self) -> tuple[list[str], list[int]]:
        """Get selected channels."""
        checked = self.channel_table.get_all_checked()
        channel_names = self.channel_table.get_col_data(self.TABLE_CONFIG.channel_name)
        return [self.channel_table.get_value(self.TABLE_CONFIG.channel_index, i) for i in checked], channel_names

    def on_update_model(self, _=None) -> None:
        """Update model."""
        if self._is_setting_config:
            return
        self.preprocessing.image_type = {"Brightfield": "BF", "Fluorescence": "FL"}[self.type_choice.currentText()]
        self.preprocessing.max_intensity_projection = self.mip_check.isChecked()
        self.preprocessing.contrast_enhance = self.contrast_check.isChecked()
        self.preprocessing.invert_intensity = self.invert_check.isChecked()
        self.preprocessing.channel_indices, self.preprocessing.channel_names = self.get_selected_channels()
        self.preprocessing.as_uint8 = self.uint8_check.isChecked()
        self.preprocessing.flip = {"None": None, "Horizontal": "h", "Vertical": "v"}[self.flip_choices.currentText()]
        self.preprocessing.translate_x = self.translate_x.value()
        self.preprocessing.translate_y = self.translate_y.value()
        self.preprocessing.rotate_counter_clockwise = self.rotate_spin.value()
        self.preprocessing.downsample = self.downsample_spin.value()
        self.evt_update.emit(self.preprocessing)
        self.evt_preprocessing_preview.emit(self.modality, self.preprocessing)

    def on_preview_preprocessing(self) -> None:
        """Preview preprocessing."""
        if not self.preview_check.isChecked():
            return
        self.evt_preview.emit(self.modality, self.preprocessing)

    def accept(self) -> None:
        """Set model."""
        self.evt_update.emit(self.preprocessing)
        self.evt_preprocessing.emit(self.preprocessing)
        super().accept()

    def close(self) -> bool:
        """Hide dialog rather than delete it."""
        new_hash = hash_parameters(**self.preprocessing.to_dict())
        if new_hash != self.original_hash and not hp.confirm(
            self,
            "You've made changes to the pre-processing settings. Closing will discard them. "
            "<br><b>Are you sure you wish to continue?</b>",
        ):
            return
        self.evt_update.emit(self.modality.preprocessing)
        super().close()

    def on_set_defaults(self, _=None) -> None:
        """Set defaults."""
        from image2image_reg.models import Preprocessing

        button = self.defaults_choice_group.checkedButton()
        kind = button.text()
        if not hp.confirm(
            self, f"Are you sure you want to set to <b>{kind}</b> defaults? This will overwrite other settings."
        ):
            return
        if kind == "Brightfield":
            new_preprocessing = Preprocessing.brightfield()
        elif kind == "Fluorescence":
            new_preprocessing = Preprocessing.fluorescence()
        else:
            new_preprocessing = Preprocessing.basic()
        new_preprocessing.channel_names = self.preprocessing.channel_names
        new_preprocessing.channel_indices = self.preprocessing.channel_indices
        self.preprocessing = new_preprocessing
        self.set_from_model()
        self.evt_update.emit(self.preprocessing)
        self.evt_preprocessing_preview.emit(self.modality, self.preprocessing)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""

        self.defaults_choice_lay, self.defaults_choice_group = hp.make_toggle_group(
            self, "Basic", "Brightfield", "Fluorescence", func=self.on_set_defaults
        )

        # intensity preprocessing
        self.type_choice = hp.make_combobox(
            self,
            ["Brightfield", "Fluorescence"],
            tooltip="Image type - this determines how certain pre-processing steps are conducted.",
            func=self.on_update_model,
        )
        self.mip_check = hp.make_checkbox(self, "", tooltip="Max intensity projection", func=self.on_update_model)
        self.contrast_check = hp.make_checkbox(self, "", tooltip="Contrast enhancement", func=self.on_update_model)
        self.invert_check = hp.make_checkbox(self, "", tooltip="Invert intensity", func=self.on_update_model)
        self.channel_table = QtCheckableTableView(
            self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True, double_click_to_check=True
        )
        self.channel_table.setCornerButtonEnabled(False)
        hp.set_font(self.channel_table)
        self.channel_table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
        self.uint8_check = hp.make_checkbox(self, "", func=self.on_update_model)
        self.channel_table.evt_checked.connect(self.on_update_model)

        # spatial preprocessing
        self.flip_choices = hp.make_combobox(
            self,
            ["None", "Horizontal", "Vertical"],
            tooltip="Horizontal or vertial image flip.",
            func=self.on_update_model,
        )
        self.translate_x = hp.make_int_spin_box(
            self,
            value=0,
            minimum=-10000,
            maximum=10000,
            step_size=50,
            suffix="µm",
            tooltip="Translate X",
            func=self.on_update_model,
        )
        self.translate_y = hp.make_int_spin_box(
            self,
            value=0,
            minimum=-10000,
            maximum=10000,
            step_size=50,
            suffix="µm",
            tooltip="Translate Y",
            func=self.on_update_model,
        )
        self.rotate_spin = hp.make_double_spin_box(
            self,
            value=0,
            minimum=-360,
            maximum=360,
            step_size=45,
            suffix="°",
            tooltip="Rotate (counter-clockwise)",
            func=self.on_update_model,
        )
        self.downsample_spin = hp.make_int_spin_box(
            self, default=1, minimum=1, maximum=10, tooltip="Downsample", func=self.on_update_model
        )

        self.preview_check = hp.make_checkbox(self, "", func=self.on_preview_preprocessing)

        layout = hp.make_form_layout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addRow(self._make_close_handle("Pre-processing")[1])
        layout.addRow("Defaults", self.defaults_choice_lay)
        layout.addRow(hp.make_h_line_with_text("Intensity", self))
        layout.addRow("Image type", self.type_choice)
        layout.addRow("Max. intensity projection", self.mip_check)
        layout.addRow("Contrast enhancement", self.contrast_check)
        layout.addRow("Invert intensity", self.invert_check)
        layout.addRow("UInt8 (reduce data size)", self.uint8_check)
        layout.addRow(self.channel_table)
        layout.addRow(hp.make_h_line_with_text("Spatial", self))
        layout.addRow("Flip", self.flip_choices)
        layout.addRow(
            "Translate (x)",
            hp.make_h_layout(
                self.translate_x,
                hp.make_warning_label(self, "Setting this value is not fully supported yet.", small=True),
                stretch_id=(0,),
                margin=0,
                spacing=0,
            ),
        )
        layout.addRow(
            "Translate (y)",
            hp.make_h_layout(
                self.translate_y,
                hp.make_warning_label(self, "Setting this value is not fully supported yet.", small=True),
                stretch_id=(0,),
                margin=0,
                spacing=0,
            ),
        )
        layout.addRow("Rotate (counter-clockwise)", self.rotate_spin)
        layout.addRow("Downsample", self.downsample_spin)
        layout.addRow(hp.make_h_line())
        layout.addRow("Preview", self.preview_check)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Apply", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.close),
                spacing=2,
            )
        )
        return layout
