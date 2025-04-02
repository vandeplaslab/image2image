"""Whole slide registration."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
import qtextra.queue.cli_queue as _q
from image2image_reg.workflows.elastix import ElastixReg
from koyo.secret import hash_parameters
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from qtextra.config import THEMES
from qtextra.queue.popup import QUEUE
from qtextra.queue.task import Task
from qtextra.utils.utilities import connect
from qtpy.QtWidgets import QSizePolicy, QWidget

from image2image import __version__
from image2image.config import get_elastix_config
from image2image.enums import ALLOWED_ELASTIX_FORMATS, ALLOWED_PROJECT_ELASTIX_FORMATS, PYRAMID_TO_LEVEL
from image2image.qt._dialog_wsi import ImageWsiWindow
from image2image.qt._dialogs._select import LoadWidget
from image2image.qt._wsi._list import QtModalityList
from image2image.qt._wsi._paths import RegistrationMap
from image2image.utils.utilities import check_image_size, get_i2reg_path, pad_str
from image2image.utils.valis import guess_preprocessing

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality, Preprocessing


_q.N_PARALLEL = get_elastix_config().n_parallel


def make_registration_task(
    project: ElastixReg,
    write_not_registered: bool = False,
    write_transformed: bool = False,
    write_attached: bool = False,
    write_merged: bool = False,
    remove_merged: bool = False,
    as_uint8: bool = False,
    rename: bool = True,
    clip: str = "remove",
) -> Task:
    """Make registration task."""
    task_id = hash_parameters(
        project_dir=project.project_dir,
        write_transformed=write_transformed,
        write_not_registered=write_not_registered,
        write_merged=write_merged,
        as_uint8=as_uint8,
        rename=rename,
    )

    commands = []

    # pre-processing command
    preprocess_command = [
        get_i2reg_path(),
        "--no_color",
        "--debug",
        "elastix",
        "preprocess",
        "--project_dir",
        pad_str(project.project_dir),
    ]
    commands.append(preprocess_command)

    # registration command
    register_command = [
        get_i2reg_path(),
        "--no_color",
        "--debug",
        "elastix",
        "register",
        "--project_dir",
        pad_str(project.project_dir),
        "--no_write",
    ]
    commands.append(register_command)

    # write command
    if any([write_attached, write_transformed, write_not_registered, write_merged]):
        write_command = [
            get_i2reg_path(),
            "--no_color",
            "--debug",
            "elastix",
            "write",
            "--project_dir",
            pad_str(project.project_dir),
            "--write_not_registered" if write_not_registered else "--no_write_not_registered",
            "--write_registered" if write_transformed else "--no_write_registered",
            "--write_attached" if write_attached else "--no_write_attached",
            "--write_merged" if write_merged else "--no_write_merged",
            "--remove_merged" if remove_merged else "--no_remove_merged",
            "--as_uint8" if as_uint8 else "--no_as_uint8",
            "--rename" if rename else "--no_rename",
            "--clip",
            pad_str(clip),
        ]
        commands.append(write_command)

    return Task(
        task_id=task_id,
        task_name=f"{project.project_dir!s}",
        task_name_repr=hp.hyper(project.project_dir, value=project.project_dir.name),
        task_name_tooltip=str(project.project_dir),
        commands=commands,
    )


class ImageElastixWindow(ImageWsiWindow):
    """Image viewer dialog."""

    APP_NAME = "elastix"

    _registration_model: ElastixReg | None = None

    WINDOW_TITLE = f"image2elastix: WSI Registration app (v{__version__})"
    PROJECT_SUFFIX = ".wsireg"
    RUN_DISABLED: bool = False
    OTHER_PROJECT: str = "Valis"
    IS_VALIS = False

    def __init__(
        self,
        parent: QWidget | None,
        run_check_version: bool = True,
        project_dir: PathLike | None = None,
        **_kwargs: ty.Any,
    ):
        self.CONFIG = get_elastix_config()
        super().__init__(parent, run_check_version=run_check_version, project_dir=project_dir)
        self.WINDOW_CONSOLE_ARGS = (("view", "viewer"), "data_model", ("data_model", "wrapper"), "registration_model")

    @property
    def registration_model(self) -> ElastixReg | None:
        """Registration model."""
        if self._registration_model is None:
            name = self.name_label.text() or "project"
            name = ElastixReg.format_project_name(name)
            self._registration_model = ElastixReg(name=name, output_dir=self.CONFIG.output_dir, init=False)
        return self._registration_model

    @staticmethod
    def make_registration_task(
        project: ElastixReg,
        write_not_registered: bool = False,
        write_attached: bool = False,
        write_transformed: bool = False,
        write_merged: bool = False,
        remove_merged: bool = False,
        as_uint8: bool = False,
        rename: bool = False,
        **_kwargs: ty.Any,
    ) -> Task:
        return make_registration_task(
            project,
            write_not_registered=write_not_registered,
            write_attached=write_attached,
            write_transformed=write_transformed,
            write_merged=write_merged,
            remove_merged=remove_merged,
            as_uint8=as_uint8,
            rename=rename,
        )

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dset_dlg.evt_closing, self.on_remove_image, state=state)
        connect(self._image_widget.dset_dlg.evt_import_project, self._on_load_from_project, state=state)
        connect(self._image_widget.dset_dlg.evt_files, self._on_pre_loading_images, state=state)
        connect(self._image_widget.dset_dlg.evt_rejected_files, self.on_maybe_add_attachment, state=state)
        connect(self._image_widget.dset_dlg.evt_resolution, self.on_update_resolution_from_table, state=state)

        connect(self.view.viewer.events.status, self._status_changed, state=state)
        # connect(self.view.widget.canvas.events.key_press, self.keyPressEvent, state=state)

        connect(self.modality_list.evt_show, self.on_show_modality, state=state)
        connect(self.modality_list.evt_rename, self.on_rename_modality, state=state)
        connect(self.modality_list.evt_resolution, self.on_update_resolution_from_list, state=state)
        connect(self.modality_list.evt_color, self.on_update_colormap, state=state)
        connect(self.modality_list.evt_delete, self.on_remove_modality, state=state)
        connect(self.modality_list.evt_hide_others, self.on_hide_modalities, state=state)
        connect(self.modality_list.evt_preview_preprocessing, self.on_preview, state=state)
        connect(self.modality_list.evt_set_preprocessing, self.on_update_preprocessing_of_modality, state=state)
        connect(self.modality_list.evt_preview_transform_preprocessing, self.on_preview_transform, state=state)
        connect(self.modality_list.evt_preprocessing_close, self.on_preview_close, state=state)
        connect(self.registration_map.evt_message, self.statusbar.showMessage)
        connect(self.registration_map.evt_valid, self.on_validate_registrations)

        connect(QUEUE.evt_errored, self.on_registration_finished, state=state)
        connect(QUEUE.evt_finished, self.on_registration_finished, state=state)
        connect(QUEUE.evt_started, lambda _: self.spinner.show(), state=state)
        connect(QUEUE.evt_empty, self.spinner.hide, state=state)

    def on_update_modality_name(self, old_name: str, modality: Modality) -> None:
        """Update modality."""
        self.registration_model.rename_modality(old_name, modality.name)
        layer = self.view.get_layer(old_name)
        if layer:
            layer.name = modality.name
        self.registration_map.populate_images()
        logger.trace(f"Updated modality name: {old_name} -> {modality.name}")

    def on_preview(self, modality: Modality, preprocessing: Preprocessing | None = None) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview

        if preprocessing is None:
            preprocessing = modality.preprocessing
        preprocessing_hash = self._get_preprocessing_hash(modality, preprocessing)
        pyramid = PYRAMID_TO_LEVEL[self.pyramid_level.currentText()]

        wrapper = self.data_model.get_wrapper()
        if wrapper:
            layer = self.view.get_layer(modality.name)
            if layer and layer.rgb:
                self.view.remove_layer(modality.name)

            widget = self.modality_list.get_widget_for_modality(modality)
            colormap = "gray" if widget is None else widget.colormap
            with MeasureTimer() as timer:
                reader = wrapper.get_reader_for_path(modality.path)
                scale = reader.scale_for_pyramid(pyramid)
                image = preprocess_preview(reader.pyramid[pyramid], reader.is_rgb, scale[0], preprocessing)
                self.view.add_image(
                    image,
                    name=modality.name,
                    scale=scale,
                    blending="additive",
                    colormap=colormap,
                    metadata={"key": reader.key, "preview_hash": preprocessing_hash},
                )
            logger.trace(
                f"Processed image {modality.name} for preview in {timer()} at {pyramid} pyramid level ({image.shape})"
            )

    def on_preview_transform(self, modality: Modality, preprocessing: Preprocessing) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview

        if preprocessing is None:
            preprocessing = modality.preprocessing
        preprocessing_hash = self._get_preprocessing_hash(modality, preprocessing)
        pyramid = PYRAMID_TO_LEVEL[self.pyramid_level.currentText()]

        wrapper = self.data_model.get_wrapper()
        if wrapper:
            layer = self.view.get_layer(modality.name)
            if layer and layer.rgb:
                self.view.remove_layer(modality.name)

            widget = self.modality_list.get_widget_for_modality(modality)
            colormap = "gray" if widget is None else widget.colormap
            with MeasureTimer() as timer:
                reader = wrapper.get_reader_for_path(modality.path)
                scale = reader.scale_for_pyramid(pyramid)
                image = preprocess_preview(reader.pyramid[pyramid], reader.is_rgb, scale[0], preprocessing)
                self.view.add_image(
                    image,
                    name=modality.name,
                    scale=scale,
                    blending="additive",
                    colormap=colormap,
                    mmetadata={"key": reader.key, "preview_hash": preprocessing_hash},
                )
            logger.trace(
                f"Processed image {modality.name} for preview transform in {timer()} with {pyramid} pyramid level"
            )

    def on_show_modality(self, modality: Modality, state: bool = True, overwrite: bool = False) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview

        preprocessing_hash = self._get_preprocessing_hash(modality)
        pyramid = PYRAMID_TO_LEVEL[self.pyramid_level.currentText()]
        preview = self.use_preview_check.isChecked()
        layer = self.view.get_layer(modality.name)

        # ensure that the controls are shown
        if widget := self.modality_list.get_widget_for_modality(modality):
            widget.visible_btn.set_state(state, trigger=False)

        # no need to re-process if the layer is already there
        if layer and layer.metadata.get("preview_hash") == preprocessing_hash and not overwrite:
            layer.visible = state
            logger.trace(f"Already processed image {modality.name} with {pyramid} pyramid level ({preprocessing_hash})")
            return

        wrapper = self.data_model.get_wrapper()
        if wrapper:
            with MeasureTimer() as timer:
                reader = wrapper.get_reader_for_path(modality.path)
                channel_axis, _ = reader.get_channel_axis_and_n_channels()
                reader.resolution = modality.pixel_size
                scale = reader.scale_for_pyramid(pyramid)
                image = reader.pyramid[pyramid]
                if pyramid == -1 and reader.n_in_pyramid == 1:
                    image, scale = check_image_size(image, scale, pyramid, channel_axis)

                if preview:
                    image = preprocess_preview(image, reader.is_rgb, scale[0], modality.preprocessing)
                else:
                    image = reader.get_channel(0, pyramid)

                widget = self.modality_list.get_widget_for_modality(modality)
                kws, overwrite = (
                    ({"rgb": True}, True)
                    if (reader.is_rgb and not preview)
                    else (
                        {"colormap": "gray" if widget is None else widget.colormap},
                        overwrite if not reader.is_rgb else True,
                    )
                )
                if overwrite and layer:
                    self.view.remove_layer(modality.name)
                if not state:
                    if layer:
                        layer.visible = state
                else:
                    self.view.add_image(
                        image,
                        name=modality.name,
                        scale=scale,
                        blending="additive",
                        metadata={"key": reader.key, "preview_hash": preprocessing_hash},
                        visible=state,
                        **kws,
                    )
                logger.trace(f"Processed image {modality.name} in {timer()} with {pyramid} pyramid level")

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
                        reader_kws=reader.reader_kws,
                        raise_on_error=False,
                    )
        # Populate table
        self.modality_list.populate()
        self.registration_map.populate()

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
        self.modality_list.populate()
        self.registration_map.depopulate()

    def on_open_mask_dialog(self) -> None:
        """Open mask dialog."""
        if self._mask_dlg is None:
            from image2image.qt._wsi._mask import MaskDialog

            self._mask_dlg = MaskDialog(self)
            self._mask_dlg.evt_mask.connect(self.modality_list.toggle_mask)
            self._mask_dlg.evt_close.connect(self.on_remove_mask_dialog)
        self._mask_dlg.show_in_center_of_widget(self.mask_btn)

    def on_remove_mask_dialog(self) -> None:
        """Remove mask dialog."""
        if self._mask_dlg:
            self.view.remove_layer(self._mask_dlg.MASK_NAME)
            self._mask_dlg = None
            logger.trace("Removed mask dialog.")

    def on_open_crop_dialog(self) -> None:
        """Open crop dialog."""
        if self._crop_dlg is None:
            from image2image.qt._wsi._mask import CropDialog

            self._crop_dlg = CropDialog(self)
            self._crop_dlg.evt_mask.connect(self.modality_list.toggle_crop)
            self._crop_dlg.evt_close.connect(self.on_remove_crop_dialog)
        self._crop_dlg.show_in_center_of_widget(self)

    def on_remove_crop_dialog(self) -> None:
        """Remove mask dialog."""
        if self._crop_dlg:
            self.view.remove_layer(self._crop_dlg.MASK_NAME)
            self._crop_dlg = None
            logger.trace("Removed mask dialog.")

    def on_open_merge_dialog(self) -> None:
        """Open merge dialog."""
        # modalities = self.registration_model.get_image_modalities(with_attachment=False)
        # if not modalities or len(modalities) < 2:
        #     hp.toast(self, "Error", "Need more images to merge..", icon="error", position="top_left")
        #     return
        #
        # hp.select_from_list(self, "Select images to merge")

    def on_save_to_project(self) -> bool:
        """Save project to i2reg."""
        name = self.name_label.text()
        if not name:
            hp.toast(self, "Error", "Please provide a name for the project.", icon="error", position="top_left")
            return False
        output_dir = self.output_dir
        if output_dir is None:
            hp.toast(self, "Error", "Please provide an output directory.", icon="error", position="top_left")
            return False
        if not self._validate():
            return False
        path = self.save_model()
        if path:
            hp.toast(self, "Saved", f"Saved project to {hp.hyper(path)}.", icon="success", position="top_left")
            logger.info(f"Saved project to {path}")
            return True
        return False

    def on_save_with_errors_to_project(self) -> None:
        """Save project to i2reg."""
        name = self.name_label.text()
        if not name:
            hp.toast(self, "Error", "Please provide a name for the project.", icon="error", position="top_left")
            return
        output_dir = self.output_dir
        if output_dir is None:
            hp.toast(self, "Error", "Please provide an output directory.", icon="error", position="top_left")
            return
        path = self.save_model()
        if path:
            hp.toast(self, "Saved", f"Saved project to {hp.hyper(path)}.", icon="success", position="top_left")
            logger.info(f"Saved project to {path}")

    def on_save_as_other_project(self) -> None:
        """Save project to Valis."""
        name = self.name_label.text()
        if not name:
            hp.toast(self, "Error", "Please provide a name for the project.", icon="error", position="top_left")
            return
        output_dir = self.output_dir
        if output_dir is None:
            hp.toast(self, "Error", "Please provide an output directory.", icon="error", position="top_left")
            return
        if not self._validate():
            return
        self.registration_model.merge_images = self.write_merged_check.isChecked()
        self.registration_model.name = name
        self.registration_model.output_dir = output_dir
        if self.registration_model.project_dir.with_suffix(".valis").exists() and not hp.confirm(
            self, f"Project <b>{self.registration_model.name}</b> already exists.<br><b>Overwrite?</b>"
        ):
            return
        try:
            obj = self.registration_model.to_valis(output_dir)
            if obj:
                hp.toast(
                    self, "Saved", f"Saved project to {hp.hyper(obj.project_dir)}.", icon="success", position="top_left"
                )
                logger.trace(f"Saved i2reg project to {obj.project_dir}")
        except (ValueError, FileNotFoundError):
            hp.toast(self, "Error", "Could not save project to I2Reg format.", icon="error", position="top_left")

    def on_load_from_project(self, _evt=None) -> None:
        """Load a previous project."""
        path_ = hp.get_filename(
            self, "Load I2Reg project", base_dir=self.CONFIG.output_dir, file_filter=ALLOWED_PROJECT_ELASTIX_FORMATS
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: PathLike) -> None:
        if path_:
            path_ = Path(path_)
            try:
                project_path = path_.parent if path_.is_file() else path_
                try:
                    project = ElastixReg.from_path(project_path, quick=True)
                except ValueError:
                    ElastixReg.update_paths(project_path, project_path.parent)
                    project = ElastixReg.from_path(project_path, quick=True)
                if project:
                    self._registration_model = project
                    self.output_dir = project.output_dir
                    self.name_label.setText(project.name)
                    self._image_widget.on_close_dataset()
                    modalities = project.get_image_modalities(with_attachment=False)
                    paths = [project.get_modality(modality).path for modality in modalities]
                    if paths:
                        self._image_widget.on_set_path(paths)
            except ValueError:
                logger.exception(f"Could not load project from {path_}")

    def save_model(self) -> Path | None:
        """Save model in the current state."""
        name = self.name_label.text()
        if not name:
            hp.toast(self, "Error", "Please provide a name for the project.", icon="error", position="top_left")
            return None
        output_dir = self.output_dir
        if output_dir is None:
            hp.toast(self, "Error", "Please provide an output directory.", icon="error", position="top_left")
            return None
        self.registration_model.merge_images = self.write_merged_check.isChecked()
        self.registration_model.name = name
        self.registration_model.output_dir = output_dir
        if self.registration_model.project_dir.exists() and not hp.confirm(
            self, f"Project <b>{self.registration_model.name}</b> already exists.<br><b>Overwrite?</b>"
        ):
            return None
        path = self.registration_model.save()
        logger.trace(f"Saved project to {self.registration_model.project_dir}")
        return path

    def on_close(self, force: bool = False) -> None:
        """Close project."""
        if self.registration_model and (
            force or hp.confirm(self, "Are you sure you want to close the project?", "Close project?")
        ):
            self.name_label.setText("")
            self._image_widget.on_close_dataset(force=True)
            self._registration_model = None
            # self.registration_model.reset_registration_paths()
            self.view.clear()
            self.modality_list.populate()
            self.registration_map.populate()
            self.output_dir_label.setText("")
            self.output_dir_label.setToolTip("")
            self.output_dir = None
            logger.trace("Closed project.")

    def _setup_ui(self) -> None:
        """Create panel."""
        self.view = self._make_image_view(
            self, add_toolbars=True, allow_extraction=False, disable_controls=False, disable_new_layers=True
        )
        self.view.toolbar.tools_cross_btn.hide()
        self.view.toolbar.tools_colorbar_btn.hide()
        self.view.toolbar.tools_clip_btn.hide()
        self.view.toolbar.tools_save_btn.hide()
        self.view.toolbar.tools_scalebar_btn.hide()
        self.view.widget.canvas.events.key_press.connect(self.keyPressEvent)
        self.view.viewer.scale_bar.unit = "um"

        self._image_widget = LoadWidget(
            self,
            self.view,
            self.CONFIG,
            allow_channels=False,
            allow_save=False,
            available_formats=ALLOWED_ELASTIX_FORMATS,
            project_extension=[".i2wsireg.json", ".i2wsireg.toml", ".config.json", ".wsireg", ".i2reg"],
            allow_geojson=True,
            allow_import_project=True,
            confirm_czi=True,
        )

        side_widget = QWidget()
        side_widget.setMinimumWidth(450)
        side_widget.setMaximumWidth(450)

        self.modality_list = QtModalityList(self)

        self.registration_map = RegistrationMap(self)
        self.mask_btn = hp.make_btn(
            self,
            "Mask...",
            tooltip="Open dialog where you can specify mask for images to focus the registration on specific part of"
            " the tissue...",
            func=self.on_open_mask_dialog,
        )
        self.crop_btn = hp.make_btn(
            self,
            "Crop...",
            tooltip="Open dialog where you can specify mask for images to crop the image to before the registration"
            " takes place...",
            func=self.on_open_crop_dialog,
        )
        # hp.disable_widgets(self.crop_btn, disabled=True)
        self.merge_btn = hp.make_btn(
            self, "Merge...", tooltip="Specify images to merge...", func=self.on_open_merge_dialog
        )
        self.merge_btn.hide()
        self._make_visibility_options()

        side_layout = hp.make_form_layout(parent=side_widget, margin=3)
        side_layout.addRow(self._image_widget)
        # Modalities
        side_layout.addRow(
            hp.make_h_layout(
                self.apply_btn,
                self.show_all_btn,
                self.hide_all_btn,
                self.recolor_btn,
                self.sort_btn,
                self.filter_modalities_by,
                margin=2,
                spacing=2,
                stretch_id=(5,),
            )
        )
        side_layout.addRow(self.modality_list)
        side_layout.addRow(hp.make_h_layout(self.use_preview_check, self.hide_others_check, margin=2, spacing=2))
        side_layout.addRow(hp.make_h_layout(self.mask_btn, self.crop_btn, self.merge_btn, spacing=2))
        # Registration paths
        self.registration_settings = hp.make_advanced_collapsible(
            side_widget,
            "Registration paths",
            allow_checkbox=False,
            allow_icon=False,
            warning_icon=("warning", {"color": THEMES.get_theme_color("warning")}),
        )
        self.registration_settings.addRow(self.registration_map)
        side_layout.addRow(self.registration_settings)
        # Advanced options
        self.hidden_settings = self._make_hidden_widgets(side_widget)
        side_layout.addRow(self.hidden_settings)
        # Project
        self.project_settings = self._make_output_widgets(side_widget)
        side_layout.addRow(self.project_settings)
        # Execution
        self._make_run_widgets(side_widget)
        self.project_settings.layout().addRow(
            hp.make_h_layout(self.save_btn, self.viewer_btn, self.close_btn, self.run_btn, stretch_id=(3,), spacing=2)
        )

        widget = QWidget()  # noqa
        self.setCentralWidget(widget)
        layout = hp.make_h_layout(parent=widget, spacing=0, margin=0)
        layout.addWidget(self.view.widget, stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addWidget(side_widget)

        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def on_validate_registrations(self, is_valid: bool, errors: list[str]) -> None:
        """Update registration paths."""
        self.registration_settings.warning_label.setToolTip("<br>".join(errors))
        self.registration_settings.set_warning_visible(not is_valid)

    def _make_statusbar(self) -> None:
        super()._make_statusbar()
        self.pyramid_level = hp.make_combobox(
            self,
            list(PYRAMID_TO_LEVEL.keys()),
            tooltip="Index of the polygon to show in the fixed image.\nNegative values are used go from smallest to"
            " highest level.\nValue of 0 means that the highest resolution is shown which will be slow to pre-process.",
            object_name="statusbar_combobox",
        )
        self.pyramid_level.currentIndexChanged.connect(self.on_update_pyramid_level)
        self.pyramid_level.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.statusbar.insertPermanentWidget(0, self.pyramid_level)
        self.statusbar.insertPermanentWidget(1, hp.make_v_line())

        self.spinner, _ = hp.make_loading_gif(self, which="infinity", size=(20, 20), retain_size=False, hide=True)
        self.statusbar.insertPermanentWidget(2, self.spinner)

        self.queue_btn = hp.make_qta_btn(self, "queue", tooltip="Open queue popup.", small=True)
        self.statusbar.insertPermanentWidget(3, self.queue_btn)

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_elastix_tutorial

        if show_elastix_tutorial(self):
            self.CONFIG.update(first_time=False)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="elastix", level=0)
