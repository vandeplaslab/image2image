"""Whole slide registration."""

from __future__ import annotations

import typing as ty
from functools import partial
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_reg.models import Modality
from image2image_reg.workflows.iwsireg import IWsiReg
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from napari.layers import Image
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtpy.QtWidgets import QDialog, QHBoxLayout, QStatusBar, QVBoxLayout, QWidget
from superqt import ensure_main_thread

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_PROJECT_WSIREG_FORMATS, ALLOWED_WSIREG_FORMATS
from image2image.models.data import DataModel
from image2image.qt._dialogs._select import LoadWidget
from image2image.qt._wsireg._list import QtModalityList
from image2image.qt._wsireg._paths import RegistrationMap
from image2image.qt.dialog_base import Window

if ty.TYPE_CHECKING:
    from image2image_reg.models import Preprocessing

    from image2image.qt._wsireg._mask import CropDialog, MaskDialog

MASK_LAYER_NAME = "Mask"
MASK_FILENAME = "mask.tmp"


def guess_preprocessing(reader) -> Preprocessing:
    """Guess pre-processing."""
    from image2image_reg.models import Preprocessing

    if reader.is_rgb:
        return Preprocessing.brightfield()
    return Preprocessing.fluorescence()


class ImageWsiRegWindow(Window):
    """Image viewer dialog."""

    _console = None
    _output_dir = None
    _registration_model: IWsiReg | None = None
    _mask_dlg: MaskDialog | None = None
    _crop_dlg: MaskDialog | None = None

    def __init__(self, parent: QWidget | None, run_check_version: bool = True):
        super().__init__(
            parent, f"image2wsireg: WSI Registration app (v{__version__})", run_check_version=run_check_version
        )
        # if CONFIG.first_time_viewer:
        #     hp.call_later(self, self.on_show_tutorial, 10_000)
        self._setup_config()

    @property
    def registration_model(self) -> IWsiReg | None:
        """Registration model."""
        if self._registration_model is None:
            name = self.name_label.text() or "project"
            name = IWsiReg.format_project_name(name)
            self._registration_model = IWsiReg(name=name, output_dir=CONFIG.output_dir, init=False)
        return self._registration_model

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False
        logger.trace("Setup config for image2wsireg.")

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closing, self.on_remove_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_project, self._on_load_from_project, state=state)
        # connect(self.view.widget.canvas.events.key_press, self.keyPressEvent, state=state)
        connect(self.view.viewer.events.status, self._status_changed, state=state)

        connect(self.modality_list.evt_hide_others, self.on_hide_modalities, state=state)
        connect(self.modality_list.evt_preview_preprocessing, self.on_preview, state=state)
        connect(self.modality_list.evt_name, self.on_update_modality_name, state=state)
        connect(self.modality_list.evt_resolution, self.on_update_modality, state=state)
        connect(self.modality_list.evt_show, self.on_show_modality, state=state)
        connect(self.modality_list.evt_set_preprocessing, self.on_update_modality, state=state)
        connect(self.modality_list.evt_preview_transform_preprocessing, self.on_preview_live, state=state)
        connect(self.modality_list.evt_preprocessing_close, self.on_preview_close, state=state)
        connect(self.modality_list.evt_color, self.on_update_colormap, state=state)

    def on_update_modality_name(self, old_name: str, modality: Modality) -> None:
        """Update modality."""
        self.registration_model.modalities[old_name].name = modality.name
        self.registration_model.modalities[modality.name] = self.registration_model.modalities.pop(old_name)
        layer = self.view.get_layer(old_name)
        if layer:
            layer.name = modality.name
        self.registration_map.populate_images()
        logger.trace(f"Updated modality name: {old_name} -> {modality.name}")

    def on_update_modality(self, modality: Modality) -> None:
        """Preview image."""
        self.registration_model.modalities[modality.name].pixel_size = modality.pixel_size
        self.registration_model.modalities[modality.name].preprocessing = modality.preprocessing
        logger.trace(f"Updated modality: {modality.name}")

    def on_preview(self, modality: Modality, preprocessing: Preprocessing | None = None) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview

        if preprocessing is None:
            preprocessing = modality.preprocessing

        wrapper = self.data_model.get_wrapper()
        if wrapper:
            layer = self.view.get_layer(modality.name)
            if layer and layer.rgb:
                self.view.remove_layer(modality.name)

            widget = self.modality_list.get_widget_for_item_model(modality)
            colormap = "gray" if widget is None else widget.colormap
            with MeasureTimer() as timer:
                reader = wrapper.get_reader_for_path(modality.path)
                scale = reader.scale_for_pyramid(-1)
                image = preprocess_preview(reader.pyramid[-1], reader.is_rgb, scale[0], preprocessing)
                self.view.add_image(
                    image,
                    name=modality.name,
                    scale=scale,
                    blending="additive",
                    colormap=colormap,
                    metadata={
                        "key": reader.key,
                    },
                )
            logger.trace(f"Processed image for preview in {timer()}")

    def on_preview_live(self, modality: Modality, preprocessing: Preprocessing) -> None:
        """Preview image."""
        from image2image.utils.transform import combined_transform

        wrapper = self.data_model.get_wrapper()
        if wrapper:
            reader = wrapper.get_reader_for_path(modality.path)
            image = reader.get_channel(0, -1)
            shape = reader.get_image_shape_for_shape(image.shape)
            scale = reader.scale_for_pyramid(-1)
            matrix = combined_transform(
                shape,
                scale,
                preprocessing.rotate_counter_clockwise,
                (preprocessing.translate_y, preprocessing.translate_x),
                flip_lr=preprocessing.flip == "h",
                flip_ud=preprocessing.flip == "v",
            )
            # matrix = matrix @ scale_transform(scale)
            layer = self.view.get_layer(modality.name)
            if layer:
                layer.affine = matrix

    def on_preview_close(self, modality: Modality) -> None:
        """Preview window was closed."""
        # self.view.remove_layer(f"{modality.name} (preview)")
        # layer = self.view.get_layer(modality.name)
        # if layer:
        #     layer.visible = True

    def on_update_colormap(self, modality: Modality, color: np.ndarray):
        """Update colormap."""
        layer = self.view.get_layer(modality.name)
        if layer:
            layer.colormap = color

    @ensure_main_thread
    def on_load_image(self, model: DataModel, _channel_list: list[str]) -> None:
        """Load fixed image."""
        if model and model.n_paths:
            hp.toast(
                self, "Loaded data", f"Loaded model with {model.n_paths} paths.", icon="success", position="top_left"
            )
            self.on_populate_list()
        else:
            logger.warning(f"Failed to load data - model={model}")

    def on_show_modalities(self) -> None:
        """Show modality images."""
        self.modality_list.toggle_preview(self.use_preview_check.isChecked())
        for _, modality, widget in self.modality_list.item_model_widget_iter():
            self.on_show_modality(modality, state=widget.visible_btn.visible, overwrite=True)

    def on_hide_modalities(self, modality: Modality) -> None:
        """Hide other modalities."""
        if not self.hide_others_check.isChecked():
            return
        for layer in self.view.get_layers_of_type(Image):
            layer.visible = layer.name in (modality.name, f"{modality.name} (preview)")
        self.modality_list.toggle_visible([layer.name for layer in self.view.get_layers_of_type(Image)])

    def on_show_modality(self, modality: Modality, state: bool = True, overwrite: bool = False) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview

        wrapper = self.data_model.get_wrapper()
        if wrapper:
            reader = wrapper.get_reader_for_path(modality.path)
            if self.use_preview_check.isChecked():
                image = preprocess_preview(reader.pyramid[-1], reader.is_rgb, reader.resolution, modality.preprocessing)
            else:
                image = reader.get_channel(0, -1)
            layer = self.view.get_layer(modality.name)
            widget = self.modality_list.get_widget_for_item_model(modality)
            colormap = "gray" if widget is None else widget.color
            if overwrite and layer:
                self.view.remove_layer(modality.name)
            if not state:
                if layer:
                    layer.visible = state
            else:
                self.view.add_image(
                    image,
                    name=modality.name,
                    scale=reader.scale_for_pyramid(-1),
                    blending="additive",
                    colormap=colormap,
                    # contrast_limits=contrast_limits,
                    # affine=model.affine(image.shape),  # type: ignore[arg-type]
                    metadata={
                        "key": reader.key,
                        # "contrast_limits_range": contrast_limits_range,
                        # "contrast_limits": contrast_limits,
                    },
                    visible=state,
                )

    def on_populate_list(self) -> None:
        """Populate list."""
        # Add image(s) to the registration model
        wrapper = self.data_model.get_wrapper()
        if wrapper:
            for reader in wrapper.reader_iter():
                if not self.registration_model.has_modality(path=reader.path):
                    preprocessing = guess_preprocessing(reader)
                    preprocessing.channel_names = reader.channel_names
                    preprocessing.channel_indices = list(range(len(reader.channel_names)))
                    self.registration_model.add_modality(
                        name=reader.clean_name,
                        path=reader.path,
                        pixel_size=reader.resolution,
                        channel_names=reader.channel_names,
                        preprocessing=preprocessing,
                        raise_on_error=False,
                    )
        # Populate table
        self.modality_list.populate()
        self.registration_map.populate()
        self.modality_list.toggle_preview(self.use_preview_check.isChecked())

    def on_remove_image(self, model: DataModel, channel_names: list[str], keys: list[str]) -> None:
        """Remove image."""
        self.on_depopulate_list(keys)

    def on_depopulate_list(self, keys: list[str]) -> None:
        """De-populate list."""
        wrapper = self.data_model.get_wrapper()
        if wrapper:
            for key in keys:
                reader = wrapper.get_reader_for_key(key)
                if reader and self.registration_model.has_modality(path=reader.path):
                    modality = self.registration_model.remove_modality(path=reader.path)
                    if modality:
                        self.view.remove_layer(modality.name)
                        self.view.remove_layer(f"{modality.name} (preview)")
        # Populate table
        self.modality_list.depopulate()
        self.registration_map.depopulate()

    def on_open_mask_dialog(self) -> None:
        """Open mask dialog."""
        if self._mask_dlg is None:
            from image2image.qt._wsireg._mask import MaskDialog

            self._mask_dlg = MaskDialog(self)
            self._mask_dlg.evt_mask.connect(self.modality_list.toggle_mask)
        size = self._mask_dlg.sizeHint()
        self._mask_dlg.show_above_widget(self.mask_btn, x_offset=-size.width() // 8, y_offset=size.height() // 2)

    def on_open_crop_dialog(self) -> None:
        """Open mask dialog."""
        if self._crop_dlg is None:
            from image2image.qt._wsireg._mask import CropDialog

            self._crop_dlg = CropDialog(self)
            self._crop_dlg.evt_mask.connect(self.modality_list.toggle_crop)
        size = self._crop_dlg.sizeHint()
        self._crop_dlg.show_above_widget(self.crop_btn, x_offset=-size.width() // 8, y_offset=size.height() // 2)

    def on_set_output_dir(self) -> None:
        """Set output directory."""
        self.output_dir = hp.get_directory(self, "Select output directory", CONFIG.output_dir)

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if CONFIG.output_dir is None:
                return Path.cwd()
            return Path(CONFIG.output_dir)
        return Path(self._output_dir)

    @output_dir.setter
    def output_dir(self, directory: PathLike) -> None:
        if directory:
            self._output_dir = directory
            CONFIG.output_dir = directory
            formatted_output_dir = f".{self.output_dir.parent}/{self.output_dir.name}"
            self.output_dir_label.setText(hp.hyper(self.output_dir, value=formatted_output_dir))
            logger.debug(f"Output directory set to {self._output_dir}")

    def on_save_to_i2reg(self) -> None:
        """Save project to i2reg."""
        name = self.name_label.text()
        if not name:
            hp.toast(self, "Error", "Please provide a name for the project.", icon="error", position="top_left")
            return
        output_dir = self.output_dir
        if output_dir is None:
            hp.toast(self, "Error", "Please provide an output directory.", icon="error", position="top_left")
            return
        is_valid, errors = self.registration_model.validate(require_paths=True)
        if not is_valid:
            from image2image.qt._dialogs._errors import ErrorsDialog

            dlg = ErrorsDialog(self, errors)
            dlg.show()
            return

        self.registration_model.merge_images = self.write_merged_check.isChecked()
        self.registration_model.name = name
        self.registration_model.output_dir = output_dir
        if self.registration_model.project_dir.exists() and not hp.confirm(
            self, f"Project <b>{self.registration_model.name}</b> already exists.<br><b>Overwrite?</b>"
        ):
            return

        path = self.registration_model.save()
        if path:
            hp.toast(self, "Saved", f"Saved project to {hp.hyper(path)}.", icon="success", position="top_left")
            logger.info(f"Saved project to {path}")

    def on_load_from_project(self, _evt=None):
        """Load a previous project."""
        path_ = hp.get_filename(
            self, "Load I2Reg project", base_dir=CONFIG.output_dir, file_filter=ALLOWED_PROJECT_WSIREG_FORMATS
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: PathLike) -> None:
        if path_:
            path_ = Path(path_)
            project = IWsiReg.from_path(path_.parent if path_.is_file() else path_)
            if project:
                self._registration_model = project
                self.output_dir = project.output_dir
                self.name_label.setText(project.name)
                self._image_widget.on_close_dataset()
                paths = [modality.path for modality in project.modalities.values()]
                if paths:
                    self._image_widget.on_set_path(paths)

    def on_run(self) -> None:
        """Execute registration."""

    def on_queue(self) -> None:
        """Queue registration."""

    def on_close(self) -> None:
        """Close project."""
        if self.registration_model and hp.confirm(
            self, "Are you sure you want to close the project?", "Close project?"
        ):
            self.name_label.setText("")
            self._image_widget.on_close_dataset(force=True)
            self.registration_model.reset_registration_paths()
            self.view.clear()
            self.modality_list.populate()
            self.registration_map.populate()
            logger.trace("Closed project.")

    def on_validate(self, _: ty.Any = None) -> None:
        """Validate project."""
        name = self.name_label.text()
        hp.set_object_name(self.name_label, object_name="error" if not name else "")

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(
            self, add_toolbars=True, allow_extraction=False, disable_controls=False, disable_new_layers=True
        )

        self._image_widget = LoadWidget(
            self,
            self.view,
            select_channels=False,
            available_formats=ALLOWED_WSIREG_FORMATS,
            project_extension=[".i2wsireg.json", ".i2wsireg.toml", ".config.json", ".wsireg", ".i2reg"],
            allow_geojson=True,
        )

        side_widget = QWidget()
        side_widget.setMinimumWidth(400)
        side_widget.setMaximumWidth(400)

        self.modality_list = QtModalityList(self)
        self.registration_map = RegistrationMap(self)
        self.mask_btn = hp.make_btn(
            self, "Mask...", tooltip="Set mask to focus registration...", func=self.on_open_mask_dialog
        )
        self.crop_btn = hp.make_btn(
            self, "Crop...", tooltip="Crop the image to focus registration...", func=self.on_open_crop_dialog
        )

        self.name_label = hp.make_line_edit(
            side_widget, "Name", tooltip="Name of the project", placeholder="e.g. project.wsireg", func=self.on_validate
        )
        self.output_dir_label = hp.make_label(
            side_widget, "Output directory", tooltip="Output directory", enable_url=True
        )
        self.output_dir_label.setText(hp.hyper(self.output_dir))
        self.output_dir_btn = hp.make_qta_btn(
            side_widget, "folder", tooltip="Change output directory", func=self.on_set_output_dir, normal=True
        )
        self.use_preview_check = hp.make_checkbox(
            self,
            "Use preview image",
            tooltip="Use preview image for viewing instead of the first channel only.",
            checked=False,
            func=self.on_show_modalities,
        )
        self.hide_others_check = hp.make_checkbox(
            self,
            "Hide others when previewing",
            tooltip="When previewing, hide other images to reduce clutter.",
            checked=False,
            func=self.on_show_modalities,
        )

        self.write_non_transformed_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write original, not-transformed images (those without any transformations such as target).",
            checked=True,
        )
        self.write_transformed_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write original, transformed images.",
            checked=True,
        )
        self.write_merged_check = hp.make_checkbox(
            self,
            "",
            tooltip="Merge non- and transformed images into a single image.",
            checked=False,
        )
        self.as_uint8 = hp.make_checkbox(
            self,
            "",
            tooltip="Convert to uint8 to reduce file size with minimal data loss.",
            checked=True,
            value=CONFIG.as_uint8,
        )

        hidden_settings = hp.make_advanced_collapsible(side_widget, "Advanced options", allow_checkbox=False)
        hidden_settings.addRow(hp.make_label(self, "Write non-transformed images"), self.write_non_transformed_check)
        hidden_settings.addRow(hp.make_label(self, "Write transformed images"), self.write_transformed_check)
        hidden_settings.addRow(hp.make_label(self, "Merge transformed images"), self.write_merged_check)
        hidden_settings.addRow(hp.make_label(self, "Reduce data size"), self.as_uint8)

        side_layout = hp.make_form_layout(side_widget)
        hp.style_form_layout(side_layout)
        side_layout.addRow(
            hp.make_btn(
                side_widget, "Import project...", tooltip="Load previous project", func=self.on_load_from_project
            )
        )
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(self.modality_list)
        side_layout.addRow(self.mask_btn)
        side_layout.addRow(hp.make_h_layout(self.mask_btn, self.crop_btn))
        side_layout.addRow(hp.make_h_layout(self.use_preview_check, self.hide_others_check, margin=2, spacing=2))
        side_layout.addRow(hp.make_h_line_with_text("Registration paths"))
        side_layout.addRow(self.registration_map)
        side_layout.addRow(hp.make_h_line_with_text("I2Reg project"))
        side_layout.addRow(hp.make_label(side_widget, "Name"), self.name_label)
        side_layout.addRow(
            hp.make_label(side_widget, "Output directory"),
            hp.make_h_layout(self.output_dir_label, self.output_dir_btn, stretch_id=(0,), spacing=1, margin=1),
        )
        side_layout.addRow(
            hp.make_h_layout(
                hp.make_btn(side_widget, "Save...", tooltip="Export I2Reg project", func=self.on_save_to_i2reg),
                hp.make_btn(side_widget, "Close", tooltip="Close I2Reg project", func=self.on_close),
            )
        )
        side_layout.addRow(hidden_settings)
        side_layout.addRow(
            hp.make_h_layout(
                hp.make_btn(side_widget, "Run", tooltip="Immediately execute registration", func=self.on_run),
                hp.make_btn(side_widget, "Queue", tooltip="Add registration to queue", func=self.on_queue),
            )
        )

        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.addWidget(self.view.widget, stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addWidget(side_widget)

        widget = QWidget()  # noqa
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(layout)

        # extra settings
        # self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def on_activate_scalebar(self) -> None:
        """Activate scalebar."""
        self.view.viewer.scale_bar.visible = not self.view.viewer.scale_bar.visible

    def on_show_scalebar(self) -> None:
        """Show scale bar controls for the viewer."""
        from image2image.qt._dialogs._scalebar import QtScaleBarControls

        dlg = QtScaleBarControls(self.view.viewer, self.view.widget)
        dlg.set_px_size(self.data_model.min_resolution)
        dlg.show_above_widget(self.scalebar_btn)

    def on_show_save_figure(self) -> None:
        """Show scale bar controls for the viewer."""
        from qtextra._napari.common.widgets.screenshot_dialog import QtScreenshotDialog

        dlg = QtScreenshotDialog(self.view, self)
        dlg.show_above_widget(self.clipboard_btn)

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        from qtextra.widgets.qt_image_button import QtThemeButton

        from image2image.qt._dialogs._sentry import send_feedback

        self.statusbar = QStatusBar()
        self.statusbar.setSizeGripEnabled(False)

        self.screenshot_btn = hp.make_qta_btn(
            self,
            "save",
            tooltip="Save snapshot of the canvas to file. Right-click to show dialog with more options.",
            func=self.view.widget.on_save_figure,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.screenshot_btn)

        self.clipboard_btn = hp.make_qta_btn(
            self,
            "screenshot",
            tooltip="Take a snapshot of the canvas and copy it into your clipboard. Right-click to show dialog with"
            " more options.",
            func=self.view.widget.clipboard,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.clipboard_btn)
        self.scalebar_btn = hp.make_qta_btn(
            self,
            "ruler",
            tooltip="Show scalebar.",
            func=self.on_activate_scalebar,
            func_menu=self.on_show_scalebar,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.scalebar_btn)

        self.feedback_btn = hp.make_qta_btn(
            self,
            "feedback",
            tooltip="Send feedback to the developers.",
            func=partial(send_feedback, parent=self),
            small=True,
        )
        self.statusbar.addPermanentWidget(self.feedback_btn)

        self.theme_btn = QtThemeButton(self)
        self.theme_btn.auto_connect()
        with hp.qt_signals_blocked(self.theme_btn):
            self.theme_btn.dark = CONFIG.theme == "dark"
        self.theme_btn.clicked.connect(self.on_toggle_theme)
        self.theme_btn.set_small()
        self.statusbar.addPermanentWidget(self.theme_btn)

        self.tutorial_btn = hp.make_qta_btn(
            self, "help", tooltip="Give me a quick tutorial!", func=self.on_show_tutorial, small=True
        )
        self.statusbar.addPermanentWidget(self.tutorial_btn)
        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "ipython",
                tooltip="Open IPython console",
                small=True,
                func=self.on_show_console,
            )
        )
        self.update_status_btn = hp.make_btn(
            self,
            "Update available - click here to download!",
            tooltip="Show information about available updates.",
            func=self.on_show_update_info,
        )
        self.update_status_btn.setObjectName("update_btn")
        self.update_status_btn.hide()
        self.statusbar.addPermanentWidget(self.update_status_btn)
        self.setStatusBar(self.statusbar)

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not CONFIG.confirm_close_wsireg
            or QtConfirmCloseDialog(self, "confirm_close_wsireg", self.on_save_to_project, CONFIG).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and CONFIG.confirm_close_wsireg
            # and (self.data_model.is_valid() or self.registration.is_valid())
            and QtConfirmCloseDialog(self, "confirm_close_wsireg", self.on_save_to_project, CONFIG).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        CONFIG.save()
        evt.accept()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="wsireg", level=0)
