"""Pre-processing dialog."""

from __future__ import annotations

import typing as ty
from copy import deepcopy
from functools import partial

from koyo.secret import hash_parameters
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import FilterProxyModel, QtCheckableTableView
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFormLayout, QWidget
from superqt.utils import qdebounced

from image2image.config import STATE

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality


def get_methods_for_modality(modality: Modality) -> list[str]:
    """Select options that are appropriate for the modality."""

    def _maybe_pop(option: str) -> None:
        if option not in options:
            options.pop(options.index(option))

    options = [
        "NoProcessing",
        "I2RegPreprocessor",
        "OD",
        "ColorfulStandardizer",  # only for RGB
        "Luminosity",
        "BgColorDistance",
        "StainFlattener",
        "Gray",
        "ChannelGetter (no preview)",
        "HEDeconvolution (no preview)",  # only for RGB
        "HEPreprocessing (no preview)",  # only for RGB
        "MaxIntensityProjection (no preview)",
    ]
    # if not rgb, remove a couple of options
    to_pop = []
    if len(modality.channel_names) != 3 and "R" not in modality.channel_names:
        to_pop.extend(
            [
                "ColorfulStandardizer",
                "Luminosity",
                "BgColorDistance",
                "StainFlattener",
                "Gray",
                "HEDeconvolution (no preview)",
                "HEPreprocessing (no preview)",
            ]
        )
    if not STATE.allow_valis:
        to_pop.extend(
            [
                "OD",
                "ColorfulStandardizer",
                "Luminosity",
                "BgColorDistance",
                "StainFlattener",
                "Gray",
                "ChannelGetter (no preview)",
                "HEDeconvolution (no preview)",
                "HEPreprocessing (no preview)",
                "MaxIntensityProjection (no preview)",
            ]
        )
    for option in set(to_pop):
        _maybe_pop(option)
    return options


class PreprocessingDialog(QtFramelessTool):
    """Pre-processing."""

    evt_update = Signal(object)  # used to update the model
    evt_preview_preprocessing = Signal(object, object)  # used to preview the entire preprocessing pipeline
    evt_preview_transform_preprocessing = Signal(object, object)  # used to preview the preprocessing (spatial)
    evt_set_preprocessing = Signal(object)  # used to set the preprocessing

    TABLE_CONFIG = (
        TableConfig()
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("index", "channel_index", "int", 100, sizing="contents")
        .add("name", "channel_name", "str", 250, sizing="stretch")
    )

    def __init__(
        self,
        modality: Modality,
        parent: QWidget | None = None,
        locked: bool = False,
        valis: bool = False,
    ):
        self.valis = valis
        self.modality = modality
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

        self.preprocessing = deepcopy(modality.preprocessing)
        self.original_hash = hash_parameters(**self.preprocessing.to_dict())
        self.set_from_model()
        self.lock(locked or valis)
        self.on_toggle_available()
        if parent and hasattr(parent, "previewing"):
            parent.previewing = True

    def lock(self, lock: bool) -> None:
        """Lock/unlock widgets."""
        hp.disable_widgets(
            *self.flip_choices_group.buttons(),
            self.translate_x,
            self.translate_left,
            self.translate_right,
            self.translate_y,
            self.translate_up,
            self.translate_down,
            self.rotate_spin,
            self.rotate_bck,
            self.rotate_fwd,
            self.downsample_spin,
            disabled=lock,
        )

    def on_toggle_available(self) -> None:
        """Toggle available options."""
        if not self.valis:
            return
        hp.disable_widgets(
            *self.type_choice_group.buttons(),
            self.mip_check,
            self.equalize_check,
            self.contrast_check,
            self.invert_check,
            self.uint8_check,
            disabled=self.method.currentText() != "I2RegPreprocessor",
        )
        hp.disable_widgets(
            self.channel_table,
            self.filter_by_channel,
            disabled=self.method.currentText() not in ["I2RegPreprocessor", "MaxIntensityProjection (no preview)"],
        )

    @qdebounced(timeout=300, leading=False)
    def on_preview_preprocessing(self) -> None:
        """Preview preprocessing."""
        if not self.preview_check.isChecked():
            return
        self.evt_preview_preprocessing.emit(self.modality, self.preprocessing)

    @qdebounced(timeout=300, leading=False)
    def on_preview_preprocessing_transform(self) -> None:
        """Preview preprocessing."""
        if not self.preview_check.isChecked():
            return
        self.evt_preview_transform_preprocessing.emit(self.modality, self.preprocessing)

    def set_from_model(self) -> None:
        """Set from model."""
        with self.setting_config():
            self._title_label.setText(f"{self.modality.name}")
            method = {
                None: "NoProcessing",
                "None": "NoProcessing",
                "ChannelGetter": "ChannelGetter (no preview)",
                "HEPreprocessing": "HEPreprocessing (no preview)",
                "MaxIntensityProjection": "MaxIntensityProjection (no preview)",
            }.get(self.preprocessing.method, self.preprocessing.method)
            self.method.setCurrentText(method)
            image_type = {"BF": 0, "FL": 1}[self.preprocessing.image_type]
            self.type_choice_group.button(image_type).setChecked(True)
            # spectral
            self.mip_check.setChecked(self.preprocessing.max_intensity_projection)
            self.contrast_check.setChecked(self.preprocessing.contrast_enhance)
            self.equalize_check.setChecked(self.preprocessing.equalize_histogram)
            self.invert_check.setChecked(self.preprocessing.invert_intensity)
            self.uint8_check.setChecked(self.preprocessing.as_uint8)
            # spatial
            flip = {"None": 0, "H": 1, "V": 1}.get(self.preprocessing.flip, 0)
            self.flip_choices_group.button(flip).setChecked(True)
            self.translate_x.setValue(self.preprocessing.translate_x)
            self.translate_y.setValue(self.preprocessing.translate_y)
            self.rotate_spin.setValue(self.preprocessing.rotate_counter_clockwise)
            self.downsample_spin.setValue(self.preprocessing.downsample)
            self.set_selected_channels()
        self.on_preview_preprocessing()

    def set_selected_channels(self) -> None:
        """Set selected channels."""
        channel_names = self.modality.channel_names
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
        image_type = self.type_choice_group.checkedButton().text()
        method = self.method.currentText()
        if "(no preview)" in method:
            method = method.split(" (no preview)")[0]
            logger.warning(f"Method {method} does not support previewing.")
        self.preprocessing.method = method
        self.preprocessing.image_type = {"Brightfield": "BF", "Fluorescence": "FL"}[image_type]
        self.preprocessing.max_intensity_projection = self.mip_check.isChecked()
        self.preprocessing.contrast_enhance = self.contrast_check.isChecked()
        self.preprocessing.equalize_histogram = self.equalize_check.isChecked()
        self.preprocessing.invert_intensity = self.invert_check.isChecked()
        self.preprocessing.channel_indices, self.preprocessing.channel_names = self.get_selected_channels()
        self.preprocessing.as_uint8 = self.uint8_check.isChecked()
        self.evt_update.emit(self.preprocessing)
        self.on_preview_preprocessing()

    def on_update_transform_model(self, _=None) -> None:
        """Update model."""
        if self._is_setting_config:
            return
        flip = self.flip_choices_group.checkedButton().text()
        self.preprocessing.flip = {"None": None, "Horizontal": "h", "Vertical": "v"}[flip]
        # x = self.translate_x.value() * self.modality.pixel_size
        # y = self.translate_y.value() * self.modality.pixel_size
        # set values in physical units (microns)
        self.preprocessing.translate_x = self.translate_x.value()
        self.preprocessing.translate_y = self.translate_y.value()
        self.preprocessing.rotate_counter_clockwise = self.rotate_spin.value()
        self.preprocessing.downsample = self.downsample_spin.value()
        self.evt_update.emit(self.preprocessing)
        self.on_preview_preprocessing_transform()

    def on_set_defaults(self, _=None) -> None:
        """Set defaults."""
        from image2image_reg.models import Preprocessing

        text = self.defaults_choice_group.checkedButton().text()
        if not hp.confirm(
            self, f"Are you sure you want to set to <b>{text}</b> defaults? This will overwrite other settings."
        ):
            return
        if text == "Brightfield":
            new_preprocessing = Preprocessing.brightfield(valis=self.valis)
        elif text == "Fluorescence":
            new_preprocessing = Preprocessing.fluorescence(valis=self.valis)
        else:
            new_preprocessing = Preprocessing.basic(valis=self.valis)
        new_preprocessing.channel_names = self.preprocessing.channel_names
        new_preprocessing.channel_indices = self.preprocessing.channel_indices
        self.preprocessing = new_preprocessing
        self.set_from_model()
        self.evt_update.emit(self.preprocessing)
        self.evt_preview_transform_preprocessing.emit(self.modality, self.preprocessing)
        self.on_preview_preprocessing()

    def on_accept(self) -> None:
        """Accept."""
        self.evt_update.emit(self.preprocessing)
        self.evt_set_preprocessing.emit(self.preprocessing)
        parent = self.parent()
        if parent and hasattr(parent, "previewing"):
            parent.previewing = False
        self.close()

    def on_close(self) -> None:
        """Close model."""
        new_hash = hash_parameters(**self.preprocessing.to_dict())
        if new_hash != self.original_hash and not hp.confirm(
            self,
            "You've made changes to the pre-processing settings. Closing will discard them. "
            "<br><b>Are you sure you wish to continue?</b>",
        ):
            return False
        self.evt_update.emit(self.modality.preprocessing)
        parent = self.parent()
        if parent and hasattr(parent, "previewing"):
            parent.previewing = False
        self.close()
        return None

    def on_rotate(self, value: int) -> None:
        """Increment rotation by specified value."""
        self.rotate_spin.setValue(self.rotate_spin.value() + value)

    def on_translate_x(self, value: int) -> None:
        """Increment translation(x) by specified value."""
        self.translate_x.setValue(self.translate_x.value() + value)

    def on_translate_y(self, value: int) -> None:
        """Increment translation(y) by specified value."""
        self.translate_y.setValue(self.translate_y.value() + value)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""

        self.defaults_choice_lay, self.defaults_choice_group = hp.make_toggle_group(
            self, "Basic", "Brightfield", "Fluorescence", func=self.on_set_defaults
        )

        # pre-processing method
        self.method = hp.make_combobox(
            self,
            get_methods_for_modality(self.modality),
            tooltip="Pre-processing method (only valid for Valis registration)",
            func=[self.on_toggle_available, self.on_update_model],
        )
        self.method.setHidden(not self.valis)

        # intensity preprocessing
        self.type_choice_lay, self.type_choice_group = hp.make_toggle_group(
            self,
            "Brightfield",
            "Fluorescence",
            tooltip="Image type - this determines how certain pre-processing steps are conducted.",
            func=self.on_update_model,
        )
        self.mip_check = hp.make_checkbox(self, "", tooltip="Max intensity projection", func=self.on_update_model)
        self.equalize_check = hp.make_checkbox(
            self, "", tooltip="Equalize histogram enhancement", func=self.on_update_model
        )
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
        if STATE.allow_filters:
            self.table_proxy = FilterProxyModel(self)
            self.table_proxy.setSourceModel(self.channel_table.model())
            self.channel_table.model().table_proxy = self.table_proxy
            self.channel_table.setModel(self.table_proxy)
            self.filter_by_channel = hp.make_line_edit(
                self,
                placeholder="Type in channel name...",
                func_changed=lambda text, col=self.TABLE_CONFIG.channel_name: self.table_proxy.setFilterByColumn(
                    text, col
                ),
            )

        self.uint8_check = hp.make_checkbox(self, "", func=self.on_update_model)
        self.channel_table.evt_checked.connect(self.on_update_model)

        # spatial preprocessing
        self.flip_choices_lay, self.flip_choices_group = hp.make_toggle_group(
            self,
            "None",
            "Horizontal",
            "Vertical",
            tooltip="Horizontal or vertical image flip.",
            func=self.on_update_transform_model,
        )
        self.translate_x = hp.make_int_spin_box(
            self,
            value=0,
            minimum=-10000,
            maximum=10000,
            step_size=10,
            suffix="µm",
            tooltip="Translate X",
            func=self.on_update_transform_model,
        )
        self.translate_left = hp.make_qta_btn(
            self,
            "arrow_left",
            tooltip="Add 500um from current value (it might not be left!)",
            func=partial(self.on_translate_x, value=500),
            normal=True,
            standout=True,
        )
        self.translate_right = hp.make_qta_btn(
            self,
            "arrow_right",
            tooltip="Subtract 500um to the current value (it might not be right!)",
            func=partial(self.on_translate_x, value=-500),
            normal=True,
            standout=True,
        )
        self.translate_y = hp.make_int_spin_box(
            self,
            value=0,
            minimum=-10000,
            maximum=10000,
            step_size=10,
            suffix="µm",
            tooltip="Translate Y",
            func=self.on_update_transform_model,
            normal=True,
            standout=True,
        )
        self.translate_up = hp.make_qta_btn(
            self,
            "arrow_up",
            tooltip="Add 500um from current value (it might not be up!)",
            func=partial(self.on_translate_y, value=500),
            normal=True,
            standout=True,
        )
        self.translate_down = hp.make_qta_btn(
            self,
            "arrow_down",
            tooltip="Subtract 500um to the current value (it might not be down!)",
            func=partial(self.on_translate_y, value=-500),
            normal=True,
            standout=True,
        )
        self.rotate_bck = hp.make_qta_btn(
            self,
            "rotate_left",
            tooltip="Rotate (counter-clockwise)",
            func=partial(self.on_rotate, value=-90),
            normal=True,
            standout=True,
        )
        self.rotate_fwd = hp.make_qta_btn(
            self,
            "rotate_right",
            tooltip="Rotate (clockwise)",
            func=partial(self.on_rotate, value=90),
            normal=True,
            standout=True,
        )
        self.rotate_spin = hp.make_double_spin_box(
            self,
            value=0,
            minimum=-360,
            maximum=360,
            step_size=1,
            suffix="°",
            tooltip="Rotate (counter-clockwise)",
            func=self.on_update_transform_model,
        )
        self.downsample_spin = hp.make_int_spin_box(
            self, default=1, minimum=1, maximum=10, tooltip="Downsample", func=self.on_update_transform_model
        )

        self.preview_check = hp.make_checkbox(self, "", func=self.on_preview_preprocessing, value=True)

        layout = hp.make_form_layout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addRow(self._make_close_handle("Pre-processing")[1])
        layout.addRow(
            hp.make_label(
                self,
                "Pre-processing will be applied to the image before registration. You want images that look most"
                " alike to each other as it will improve registration.<br><b>Note</b> If you are using spatial"
                " transformations, make sure images are not cropped.",
                wrap=True,
            )
        )
        layout.addRow("Defaults", self.defaults_choice_lay)
        if self.valis:
            layout.addRow(hp.make_h_line_with_text("Valis", self))
        method_label = hp.make_label(self, "Method")
        method_label.setHidden(not self.valis)
        layout.addRow(method_label, self.method)
        layout.addRow(hp.make_h_line_with_text("Intensity", self))
        layout.addRow("Image type", self.type_choice_lay)
        layout.addRow("Max. intensity projection", self.mip_check)
        layout.addRow("Histogram equalization", self.equalize_check)
        layout.addRow("Contrast enhancement", self.contrast_check)
        layout.addRow("Invert intensity", self.invert_check)
        layout.addRow("UInt8 (reduce data size)", self.uint8_check)
        layout.addRow(self.channel_table)
        if STATE.allow_filters:
            layout.addRow(hp.make_h_layout(self.filter_by_channel, stretch_id=(0,), spacing=1))
        layout.addRow(hp.make_h_line_with_text("Spatial", self))
        layout.addRow("Flip", self.flip_choices_lay)
        layout.addRow(
            "Translate (x)",
            hp.make_h_layout(
                hp.make_warning_label(
                    self,
                    "Setting this value is not fully supported yet.<br>Positive values might result in cropped"
                    " images.",
                    small=True,
                ),
                self.translate_x,
                self.translate_left,
                self.translate_right,
                stretch_id=(1,),
                margin=0,
                spacing=0,
            ),
        )
        layout.addRow(
            "Translate (y)",
            hp.make_h_layout(
                hp.make_warning_label(
                    self,
                    "Setting this value is not fully supported yet.<br>Positive values might result in cropped"
                    " images.",
                    small=True,
                ),
                self.translate_y,
                self.translate_up,
                self.translate_down,
                stretch_id=(1,),
                margin=0,
                spacing=0,
            ),
        )
        layout.addRow(
            "Rotate (counter-clockwise)",
            hp.make_h_layout(self.rotate_spin, self.rotate_bck, self.rotate_fwd, spacing=1, margin=0, stretch_id=(0,)),
        )
        layout.addRow("Downsample", self.downsample_spin)
        layout.addRow(hp.make_h_line())
        layout.addRow("Preview", self.preview_check)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Apply", func=self.on_accept),
                hp.make_btn(self, "Cancel", func=self.on_close),
                spacing=2,
            )
        )
        return layout
