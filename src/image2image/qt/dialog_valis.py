"""Valis registration dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
import qtextra.queue.cli_queue as _q
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_reg.enums import ValisDetectorMethod, ValisMatcherMethod
from image2image_reg.workflows.valis import ValisReg
from koyo.secret import hash_parameters
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from loguru import logger
from qtextra.config import THEMES
from qtextra.queue.queue_widget import QUEUE
from qtextra.queue.task import Task
from qtextra.utils.utilities import connect
from qtpy.QtWidgets import QWidget

from image2image import __version__
from image2image.config import STATE, ValisConfig, get_valis_config
from image2image.enums import ALLOWED_VALIS_FORMATS
from image2image.qt._dialog_wsi import ImageWsiWindow
from image2image.qt._dialogs._select import LoadWidget
from image2image.qt._wsi._list import QtModalityList
from image2image.utils.utilities import get_i2reg_path, pad_str
from image2image.utils.valis import guess_preprocessing

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality, Preprocessing

_q.N_PARALLEL = get_valis_config().n_parallel


def make_registration_task(
    project: ValisReg,
    write_not_registered: bool = False,
    write_transformed: bool = False,
    write_attached: bool = False,
    write_merged: bool = False,
    remove_merged: bool = False,
    as_uint8: bool = False,
    rename: bool = True,
    clip: str = "remove",
    with_i2reg: bool = True,
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
    register_command = [
        get_i2reg_path() if with_i2reg else "i2reg",
        "--no_color",
        "--debug",
        "valis",
        "register",
        "--project_dir",
        pad_str(project.project_dir),
        "--no_write",
    ]
    commands.append(register_command)
    if any([write_attached, write_transformed, write_not_registered, write_merged]):
        write_command = [
            get_i2reg_path(),
            "--no_color",
            "--debug",
            "valis",
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


class ImageValisWindow(ImageWsiWindow):
    """Image viewer dialog."""

    APP_NAME = "valis"

    _registration_model: ValisReg | None = None

    WINDOW_TITLE = f"image2valis: Valis Registration app (v{__version__})"
    PROJECT_SUFFIX = ".valis"
    RUN_DISABLED: bool = not STATE.allow_valis_run
    OTHER_PROJECT: str = "Elastix"
    IS_VALIS = True

    def __init__(
        self, parent: QWidget | None, run_check_version: bool = True, project_dir: PathLike | None = None, **_kwargs
    ):
        self.CONFIG: ValisConfig = get_valis_config()
        super().__init__(parent, run_check_version=run_check_version, project_dir=project_dir)
        self.WINDOW_CONSOLE_ARGS = (("view", "viewer"), "data_model", ("data_model", "wrapper"), "registration_model")
        if self.CONFIG.first_time:
            hp.call_later(self, self.on_show_tutorial, 10_000)

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = True

    @staticmethod
    def make_registration_task(
        project: ValisReg,
        write_not_registered: bool = False,
        write_transformed: bool = False,
        write_attached: bool = False,
        write_merged: bool = False,
        remove_merged: bool = False,
        as_uint8: bool = False,
        rename: bool = False,
        with_i2reg: bool = True,
        **_kwargs: ty.Any,
    ) -> Task:
        return make_registration_task(
            project,
            write_not_registered=write_not_registered,
            write_transformed=write_transformed,
            write_attached=write_attached,
            write_merged=write_merged,
            remove_merged=remove_merged,
            as_uint8=as_uint8,
            rename=rename,
            with_i2reg=with_i2reg,
        )

    @property
    def registration_model(self) -> ValisReg | None:
        """Registration model."""
        if self._registration_model is None:
            name = self.name_label.text() or "project"
            name = ValisReg.format_project_name(name)
            self._registration_model = ValisReg(name=name, output_dir=self.CONFIG.output_dir, init=False)
        return self._registration_model

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

        connect(self.modality_list.evt_delete, self.on_remove_modality, state=state)
        connect(self.modality_list.evt_rename, self.on_rename_modality, state=state)
        connect(self.modality_list.evt_hide_others, self.on_hide_modalities, state=state)
        connect(self.modality_list.evt_preview_preprocessing, self.on_preview, state=state)
        connect(self.modality_list.evt_show, self.on_show_or_hide_modality, state=state)
        connect(self.modality_list.evt_resolution, self.on_update_resolution_from_list, state=state)
        connect(self.modality_list.evt_set_preprocessing, self.on_update_preprocessing_of_modality, state=state)
        connect(self.modality_list.evt_color, self.on_update_colormap, state=state)
        connect(self.modality_list.evt_preprocessing_close, self.on_preview_close, state=state)

        connect(QUEUE.evt_errored, self.on_registration_finished, state=state)
        connect(QUEUE.evt_finished, self.on_registration_finished, state=state)
        connect(QUEUE.evt_started, lambda _: self.spinner.show(), state=state)
        connect(QUEUE.evt_empty, self.spinner.hide, state=state)

    def _setup_ui(self):
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
            available_formats=ALLOWED_VALIS_FORMATS,
            project_extension=["valis.config.json", ".valis.json", ".valis"],
            allow_channels=False,
            allow_save=False,
            allow_geojson=True,
            confirm_czi=True,
            allow_import_project=True,
        )

        side_widget = QWidget()
        side_widget.setMinimumWidth(450)
        side_widget.setMaximumWidth(450)

        self.modality_list = QtModalityList(self, valis=True)
        self._make_visibility_options()

        self.reference_choice = hp.make_combobox(
            self,
            ["None"],
            tooltip="Reference image for registration. If 'None' is selected, reference image will be automatically"
            " selected.",
            func=self.on_set_reference,
        )
        self.feature_choice = hp.make_combobox(
            self,
            ty.get_args(ValisDetectorMethod),
            tooltip="Feature detection method when performing feature matching between adjacent images.",
            value=self.CONFIG.feature_detector,
        )
        self.matcher_choice = hp.make_combobox(
            self,
            ty.get_args(ValisMatcherMethod),
            tooltip="Feature matching method when performing feature matching between adjacent images.",
            value=self.CONFIG.feature_matcher,
        )
        self.reflection_check = hp.make_checkbox(
            self,
            "",
            tooltip="Check for reflection during registration. Disabling this option can speed-up registration but"
            " make sure that images are not mirrored.",
            value=self.CONFIG.check_reflection,
        )
        self.non_rigid_check = hp.make_checkbox(
            self,
            "",
            tooltip="Perform non-rigid registration (using deformable field).",
            value=self.CONFIG.allow_non_rigid,
        )
        self.micro_check = hp.make_checkbox(
            self,
            "",
            tooltip="Perform refinement registration (using deformable field). This performs additional (slower)"
            " registration using higher resolution images.",
            value=self.CONFIG.allow_micro,
        )
        self.micro_fraction = hp.make_double_spin_box(
            self,
            0,
            1,
            0.05,
            n_decimals=3,
            tooltip="Fraction of the image to use for registration refinement.",
            value=self.CONFIG.micro_fraction,
        )

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
        side_layout.addRow(
            hp.make_h_layout(
                self.use_preview_check,
                self.hide_others_check,
                margin=2,
                spacing=2,
            )
        )
        # Registration paths
        self.registration_settings = hp.make_advanced_collapsible(
            side_widget,
            "Registration configuration",
            allow_checkbox=False,
            allow_icon=False,
            warning_icon=("warning", {"color": THEMES.get_theme_color("warning")}),
        )
        self.registration_settings.addRow(hp.make_label(self, "Reference"), self.reference_choice)
        self.registration_settings.addRow(hp.make_label(self, "Feature detector"), self.feature_choice)
        self.registration_settings.addRow(hp.make_label(self, "Feature matcher"), self.matcher_choice)
        self.registration_settings.addRow(hp.make_label(self, "Check for reflection"), self.reflection_check)
        self.registration_settings.addRow(hp.make_label(self, "Non-rigid registration"), self.non_rigid_check)
        self.registration_settings.addRow(hp.make_label(self, "Refine registration"), self.micro_check)
        self.registration_settings.addRow(hp.make_label(self, "Fraction"), self.micro_fraction)
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

    def on_set_reference(self) -> None:
        """Set reference on the model."""
        reference = self.reference_choice.currentText()
        if reference == "None":
            reference = None
        self.registration_model.set_reference(reference)

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
        if self.registration_model.project_dir.with_suffix(".wsireg").exists() and not hp.confirm(
            self, f"Project <b>{self.registration_model.name}</b> already exists.<br><b>Overwrite?</b>"
        ):
            return
        try:
            obj = self.registration_model.to_i2reg(output_dir)
            if obj:
                hp.toast(
                    self, "Saved", f"Saved project to {hp.hyper(obj.project_dir)}.", icon="success", position="top_left"
                )
                logger.trace(f"Saved elastix project to {obj.project_dir}")
        except (ValueError, FileNotFoundError):
            hp.toast(self, "Error", "Could not save project to elastix format.", icon="error", position="top_left")

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

    def _on_load_from_project(self, path_: PathLike) -> None:
        if path_:
            path_ = Path(path_)
            try:
                project = ValisReg.from_path(path_.parent if path_.is_file() else path_)
                if project:
                    self._registration_model = project
                    self.output_dir = project.output_dir
                    self.name_label.setText(project.name)
                    self.feature_choice.setCurrentText(project.feature_detector)
                    self.matcher_choice.setCurrentText(project.feature_matcher.lower())
                    self.reflection_check.setChecked(project.check_for_reflections)
                    self.non_rigid_check.setChecked(project.non_rigid_registration)
                    self.micro_check.setChecked(project.micro_registration)
                    self.micro_fraction.setValue(project.micro_registration_fraction)
                    self._image_widget.on_close_dataset()
                    paths = [modality.path for modality in project.modalities.values()]
                    if paths:
                        self._image_widget.on_set_path(paths)
            except ValueError:
                logger.exception(f"Could not load project from {path_}")

    def _validate(self) -> None:
        """Validate project."""
        is_valid, errors = self.registration_model.validate(require_paths=True)
        if not is_valid:
            from image2image.qt._dialogs._errors import ErrorsDialog

            dlg = ErrorsDialog(self, errors)
            dlg.show()
        return is_valid

    def _make_statusbar(self) -> None:
        super()._make_statusbar()

        self.spinner, _ = hp.make_loading_gif(self, which="infinity", size=(20, 20), retain_size=False, hide=True)
        self.statusbar.insertPermanentWidget(0, self.spinner)

        self.queue_btn = hp.make_qta_btn(self, "queue", tooltip="Open queue popup.", small=True)
        self.statusbar.insertPermanentWidget(1, self.queue_btn)

    def on_populate_list(self) -> None:
        """Populate list."""
        # Add image(s) to the registration model
        wrapper = self.data_model.get_wrapper()
        if wrapper:
            for reader in wrapper.reader_iter():
                if not self.registration_model.has_modality(path=reader.path):
                    preprocessing = guess_preprocessing(reader, valis=True)
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
        self.populate_reference_list()

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
        self.populate_reference_list()

    def populate_reference_list(self) -> None:
        """Populate reference list."""
        modalities = self.registration_model.get_image_modalities(with_attachment=False)
        hp.combobox_setter(self.reference_choice, clear=True, items=["None"] + modalities)

        try:
            reference = self.registration_model.reference
            if reference is None or reference not in modalities:
                reference = "None"
            self.reference_choice.setCurrentText(reference)
        except Exception as e:
            logger.error(f"Error setting reference: {e}")

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

        self.CONFIG.update(
            feature_detector=self.feature_choice.currentText(),
            feature_matcher=self.matcher_choice.currentText(),
            check_reflection=self.reflection_check.isChecked(),
            allow_micro=self.micro_check.isChecked(),
            micro_fraction=self.micro_fraction.value(),
            allow_non_rigid=self.non_rigid_check.isChecked(),
        )
        self.registration_model.check_for_reflections = self.CONFIG.check_reflection
        self.registration_model.micro_registration = self.CONFIG.allow_micro
        self.registration_model.micro_registration_fraction = self.CONFIG.micro_fraction
        self.registration_model.non_rigid_registration = self.CONFIG.allow_non_rigid
        # set detector and matcher method
        self.registration_model.feature_detector = self.CONFIG.feature_detector
        self.registration_model.feature_matcher = self.CONFIG.feature_matcher.upper()

        # set reference
        reference = self.reference_choice.currentText()
        self.registration_model.set_reference(reference if reference != "None" else None)

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
            self._registration_model = None
            # self.registration_model.set_reference(None)
            self.name_label.setText("")
            self._image_widget.on_close_dataset(force=True)
            self.view.clear()
            self.modality_list.populate()
            self.populate_reference_list()
            self.output_dir_label.setText("")
            self.output_dir_label.setToolTip("")
            self.output_dir = None
            logger.trace("Closed project.")

    def on_rename_modality(self, widget, new_name: str) -> None:
        """Rename modality."""
        modality = widget.modality
        if not new_name:
            hp.set_object_name(widget.name_label, object_name="error")
            return
        if modality.name == new_name:
            return
        if new_name in self.registration_model.modalities:
            hp.toast(self, "Error", f"Name <b>{new_name}</b> already exists.", icon="error", position="top_left")
            widget.name_label.setText(modality.name)
            return
        old_name = modality.name
        modality.name = new_name
        self.on_update_modality_name(old_name, modality)

    def on_update_modality_name(self, old_name: str, modality: Modality) -> None:
        """Update modality."""
        self.registration_model.rename_modality(old_name, modality.name)
        layer = self.view.get_layer(old_name)
        if layer:
            layer.name = modality.name
        self.populate_reference_list()
        logger.trace(f"Updated modality name: {old_name} -> {modality.name}")

    def on_show_or_hide_modality(self, modality: Modality, state: bool) -> None:
        """Show or hide modality."""
        layer = self.view.get_layer(modality.name)
        if layer:
            layer.visible = state
        elif not layer and state:
            self.on_show_modality(modality, state=state)
        logger.trace(f"Show/hide modality: {modality.name} -> {state}")

    def on_show_modality(self, modality: Modality, state: bool = True, overwrite: bool = False) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview_valis

        pyramid = -1
        wrapper = self.data_model.get_wrapper()
        if wrapper:
            with MeasureTimer() as timer:
                reader = wrapper.get_reader_for_path(modality.path)
                reader.resolution = modality.pixel_size
                scale = reader.scale_for_pyramid(pyramid)
                layer = self.view.get_layer(modality.name)
                preprocessing_hash = self._get_preprocessing_hash(modality, pyramid=pyramid)
                # no need to re-process if the layer is already there
                if layer and layer.metadata.get("preview_hash") == preprocessing_hash and not overwrite:
                    layer.visible = state
                    return

                if self.use_preview_check.isChecked():
                    image = preprocess_preview_valis(
                        reader.pyramid[pyramid], reader.is_rgb, scale[0], modality.preprocessing
                    )
                else:
                    image = reader.get_channel(0, pyramid)
                widget = self.modality_list.get_widget_for_modality(modality)
                colormap = "gray" if widget is None else widget.colormap
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
                        colormap=colormap,
                        metadata={"key": reader.key, "preview_hash": preprocessing_hash},
                        visible=state,
                    )
                logger.trace(f"Processed image {modality.name} in {timer()} with {pyramid} pyramid level")

    def on_preview(self, modality: Modality, preprocessing: Preprocessing | None = None) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview_valis

        if preprocessing is None:
            preprocessing = modality.preprocessing

        pyramid = -1
        preprocessing_hash = self._get_preprocessing_hash(modality, preprocessing, pyramid=pyramid)

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
                image = preprocess_preview_valis(reader.pyramid[pyramid], reader.is_rgb, scale[0], preprocessing)
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

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_valis_tutorial

        if show_valis_tutorial(self):
            self.CONFIG.update(first_time=False)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="valis", level=0)
