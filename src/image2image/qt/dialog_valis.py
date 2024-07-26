"""Valis registration dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_reg.enums import ValisDetectorMethod, ValisMatcherMethod
from image2image_reg.workflows.valis import ValisReg
from koyo.secret import hash_obj
from koyo.timer import MeasureTimer
from koyo.typing import PathLike
from koyo.utilities import is_installed
from loguru import logger
from qtextra.queue.queue_widget import QUEUE
from qtextra.queue.task import Task
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_VALIS_FORMATS
from image2image.qt._dialog_wsi import ImageWsiWindow
from image2image.qt._dialogs._select import LoadWidget
from image2image.qt._wsireg._list import QtModalityList
from image2image.utils.utilities import get_i2reg_path
from image2image.utils.valis import guess_preprocessing, hash_preprocessing

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality, Preprocessing


HAS_VALIS = is_installed("valis") and is_installed("pyvips")


def make_registration_task(
    project: ValisReg,
    write_not_registered: bool = False,
    write_transformed: bool = False,
    write_merged: bool = False,
    remove_merged: bool = False,
    as_uint8: bool = False,
) -> Task:
    """Make registration task."""
    task_id = hash_obj(project.project_dir)
    commands = [
        get_i2reg_path(),
        "--no_color",
        "--debug",
        "valis",
        "register",
        "--project_dir",
        f"{project.project_dir!s}",
    ]
    if any([write_transformed, write_not_registered, write_merged]):
        commands.append("--write")
    if write_not_registered:
        commands.append("--write_not_registered")
    if write_transformed:
        commands.append("--write_registered")
    if write_merged:
        commands.append("--write_merged")
    if remove_merged:
        commands.append("--remove_merged")
    if as_uint8:
        commands.append("--as_uint8")
    return Task(
        task_id=task_id,
        task_name=f"{project.project_dir!s}",
        task_name_repr=hp.hyper(project.project_dir, value=project.project_dir.name),
        commands=[commands],
    )


class ImageValisWindow(ImageWsiWindow):
    """Image viewer dialog."""

    _registration_model: ValisReg | None = None

    WINDOW_TITLE = f"image2valis: Valis Registration app (v{__version__})"
    WINDOW_CONFIG_ATTR = "confirm_close_valis"
    WINDOW_CONSOLE_ARGS = (("view", "viewer"), "data_model", ("data_model", "wrapper"), "registration_model")

    def __init__(
        self, parent: QWidget | None, run_check_version: bool = True, project_dir: PathLike | None = None, **_kwargs
    ):
        super().__init__(parent, run_check_version=run_check_version, project_dir=project_dir)

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False
        logger.trace("Setup config for image2wsireg.")

    def make_registration_task(
        self,
        project: ValisReg,
        write_not_registered: bool = False,
        write_transformed: bool = False,
        write_merged: bool = False,
        remove_merged: bool = False,
        as_uint8: bool = False,
    ) -> Task:
        return make_registration_task(
            project,
            write_not_registered=write_not_registered,
            write_transformed=write_transformed,
            write_merged=write_merged,
            remove_merged=remove_merged,
            as_uint8=as_uint8,
        )

    @property
    def registration_model(self) -> ValisReg | None:
        """Registration model."""
        if self._registration_model is None:
            name = self.name_label.text() or "project"
            name = ValisReg.format_project_name(name)
            self._registration_model = ValisReg(name=name, output_dir=CONFIG.output_dir, init=False)
        return self._registration_model

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closing, self.on_remove_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_project, self._on_load_from_project, state=state)
        # connect(self.view.widget.canvas.events.key_press, self.keyPressEvent, state=state)
        connect(self.view.viewer.events.status, self._status_changed, state=state)

        connect(self.modality_list.evt_delete, self.on_remove_modality, state=state)
        connect(self.modality_list.evt_rename, self.on_rename_modality, state=state)
        connect(self.modality_list.evt_hide_others, self.on_hide_modalities, state=state)
        connect(self.modality_list.evt_preview_preprocessing, self.on_preview, state=state)
        connect(self.modality_list.evt_resolution, self.on_update_modality, state=state)
        connect(self.modality_list.evt_show, self.on_show_modality, state=state)
        connect(self.modality_list.evt_set_preprocessing, self.on_update_modality, state=state)
        connect(self.modality_list.evt_color, self.on_update_colormap, state=state)
        connect(self.modality_list.evt_preprocessing_close, self.on_preview_close, state=state)

        connect(QUEUE.evt_errored, self.on_registration_finished, state=state)
        connect(QUEUE.evt_finished, self.on_registration_finished, state=state)
        connect(QUEUE.evt_started, lambda _: self.spinner.show(), state=state)
        connect(QUEUE.evt_empty, self.spinner.hide, state=state)

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
            available_formats=ALLOWED_VALIS_FORMATS,
            project_extension=["valis.config.json", ".valis.json", ".valis"],
            allow_geojson=True,
        )

        side_widget = QWidget()
        side_widget.setMinimumWidth(450)
        side_widget.setMaximumWidth(450)

        self.modality_list = QtModalityList(self, valis=True)

        self.use_preview_check = hp.make_checkbox(
            self,
            "Use preview image",
            tooltip="Use preview image for viewing instead of the first channel only.",
            value=False,
            func=self.on_show_modalities,
        )
        self.hide_others_check = hp.make_checkbox(
            self,
            "Hide others when previewing",
            tooltip="When previewing, hide other images to reduce clutter.",
            checked=False,
            func=self.on_hide_not_previewed_modalities,
        )

        self.reference_choice = hp.make_combobox(
            self,
            ["None"],
            tooltip="Reference image for registration. If 'None' is selected, reference image will be automatically"
            " selected.",
        )
        self.feature_choice = hp.make_combobox(
            self,
            ty.get_args(ValisDetectorMethod),
            tooltip="Feature detection method when performing feature matching between adjacent images.",
            value="svgg",
        )
        self.matcher_choice = hp.make_combobox(
            self,
            ty.get_args(ValisMatcherMethod),
            tooltip="Feature matching method when performing feature matching between adjacent images.",
            value="ransac",
        )
        self.reflection_check = hp.make_checkbox(
            self,
            "",
            tooltip="Check for reflection during registration. Disabling this option can speed-up registration but"
            " make sure that images are not mirrored.",
            value=True,
        )
        self.non_rigid_check = hp.make_checkbox(
            self,
            "",
            tooltip="Perform non-rigid registration (using deformable field).",
            value=True,
        )
        self.micro_check = hp.make_checkbox(
            self,
            "",
            tooltip="Perform micro registration (using deformable field). This performs additional (slower)"
            " registration using higher resolution images.",
            value=True,
        )
        self.micro_fraction = hp.make_double_spin_box(
            self, 0, 1, 0.05, n_decimals=3, tooltip="Fraction of the image to use for micro registration.", value=0.125
        )

        self.name_label = hp.make_line_edit(
            side_widget, "Name", tooltip="Name of the project", placeholder="e.g. *.valis.json", func=self.on_validate
        )
        self.output_dir_label = hp.make_label(
            side_widget, hp.hyper(self.output_dir), tooltip="Output directory", enable_url=True
        )
        self.output_dir_btn = hp.make_qta_btn(
            side_widget, "folder", tooltip="Change output directory", func=self.on_set_output_dir, normal=True
        )

        self.write_not_registered_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write original, not-registered images (those without any transformations such as target).",
            value=True,
        )
        self.write_registered_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write original, registered images.",
            value=True,
        )
        self.write_merged_check = hp.make_checkbox(
            self,
            "",
            tooltip="Merge non- and transformed images into a single image.",
            value=False,
        )
        self.as_uint8 = hp.make_checkbox(
            self,
            "",
            tooltip="Convert to uint8 to reduce file size with minimal data loss.",
            checked=True,
            value=CONFIG.as_uint8,
        )
        self.open_when_finished = hp.make_checkbox(
            self,
            "",
            tooltip="Open images in the viewer when registration is finished.",
            value=True,
        )

        side_layout = hp.make_form_layout(side_widget)
        hp.style_form_layout(side_layout)
        side_layout.addRow(
            hp.make_btn(
                side_widget, "Import project...", tooltip="Load previous project", func=self.on_load_from_project
            )
        )
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        # Modalities
        side_layout.addRow(self.modality_list)
        side_layout.addRow(hp.make_h_layout(self.use_preview_check, self.hide_others_check, margin=2, spacing=2))
        # Registration paths
        side_layout.addRow(hp.make_h_line_with_text("Registration configuration"))
        side_layout.addRow(hp.make_label(self, "Reference"), self.reference_choice)
        side_layout.addRow(hp.make_label(self, "Feature detector"), self.feature_choice)
        side_layout.addRow(hp.make_label(self, "Feature matcher"), self.matcher_choice)
        side_layout.addRow(hp.make_label(self, "Check for reflection"), self.reflection_check)
        side_layout.addRow(hp.make_label(self, "Non-rigid registration"), self.non_rigid_check)
        side_layout.addRow(hp.make_label(self, "Micro registration"), self.micro_check)
        side_layout.addRow(hp.make_label(self, "Fraction"), self.micro_fraction)
        # Project
        side_layout.addRow(hp.make_h_line_with_text("Valis project"))
        side_layout.addRow(hp.make_label(side_widget, "Name"), self.name_label)
        side_layout.addRow(hp.make_label(side_widget, "Output directory", alignment=Qt.AlignmentFlag.AlignLeft))
        side_layout.addRow(
            hp.make_h_layout(self.output_dir_label, self.output_dir_btn, stretch_id=(0,), spacing=1, margin=1),
        )
        # Advanced options
        hidden_settings = hp.make_advanced_collapsible(side_widget, "Advanced options", allow_checkbox=False)
        hidden_settings.addRow(hp.make_label(self, "Write registered images"), self.write_registered_check)
        hidden_settings.addRow(hp.make_label(self, "Write unregistered images"), self.write_not_registered_check)
        hidden_settings.addRow(hp.make_label(self, "Merge transformed images"), self.write_merged_check)
        hidden_settings.addRow(hp.make_label(self, "Reduce data size"), self.as_uint8)
        hidden_settings.addRow(hp.make_label(self, "Open when finished"), self.open_when_finished)
        side_layout.addRow(hidden_settings)

        self.save_btn = hp.make_qta_btn(
            self, "save", normal=True, tooltip="Save Valis project to disk.", func=self.on_save_to_valis
        )
        self.run_btn = hp.make_btn(
            side_widget, "Execute...", tooltip="Immediately execute registration", properties={"with_menu": True}
        )
        menu = hp.make_menu(self.run_btn)
        hp.make_menu_item(
            side_widget,
            "Run registration",
            menu=menu,
            func=self.on_run,
            icon="run",
            tooltip="Perform registration. Images will open in the viewer when finished.",
            disabled=not HAS_VALIS,
        )
        hp.make_menu_item(
            side_widget,
            "Run registration (without saving, not recommended)",
            menu=menu,
            func=self.on_run_no_save,
            icon="run",
            tooltip="Perform registration. Images will open in the viewer when finished. Project will not be"
            " saved before adding to the queue.",
            disabled=not HAS_VALIS,
        )
        hp.make_menu_item(
            side_widget,
            "Queue registration",
            menu=menu,
            func=self.on_queue,
            icon="queue",
            tooltip="Registration task will be added to the queue and you can start it manually.",
            disabled=not HAS_VALIS,
        )
        hp.make_menu_item(
            side_widget,
            "Queue registration (without saving, not recommended)",
            menu=menu,
            func=self.on_queue_no_save,
            icon="queue",
            tooltip="Registration task will be added to the queue and you can start it manually. Project will not be"
            " saved before adding to the queue.",
            disabled=not HAS_VALIS,
        )
        menu.addSeparator()
        hp.make_menu_item(
            side_widget,
            "Open project in viewer",
            menu=menu,
            func=self.on_open_in_viewer,
            icon="viewer",
            tooltip="Open the project in the viewer. This only makes sense if registration is complete.",
        )
        menu.addSeparator()
        hp.make_menu_item(
            side_widget,
            "Close project (without saving)",
            menu=menu,
            func=self.on_close,
            icon="delete",
            tooltip="Close the project without saving.",
        )
        self.run_btn.setMenu(menu)
        # Execution buttons
        side_layout.addRow(hp.make_h_layout(self.save_btn, self.run_btn, stretch_id=(1,), spacing=1))

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
        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def on_save_to_valis(self) -> None:
        """Save project to i2reg."""
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
        path = self.save_model()
        if path:
            hp.toast(self, "Saved", f"Saved project to {hp.hyper(path)}.", icon="success", position="top_left")
            logger.info(f"Saved project to {path}")

    def _on_load_from_project(self, path_: PathLike) -> None:
        if path_:
            path_ = Path(path_)
            project = ValisReg.from_path(path_.parent if path_.is_file() else path_)
            if project:
                self._registration_model = project
                self.output_dir = project.output_dir
                self.name_label.setText(project.name)
                self.reflection_check.setChecked(project.check_for_reflections)
                self.non_rigid_check.setChecked(project.non_rigid_registration)
                self.micro_check.setChecked(project.micro_registration)
                self.micro_fraction.setValue(project.micro_registration_fraction)
                self._image_widget.on_close_dataset()
                paths = [modality.path for modality in project.modalities.values()]
                if paths:
                    self._image_widget.on_set_path(paths)

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
                        raise_on_error=False,
                    )
        # Populate table
        self.modality_list.populate()
        self.modality_list.toggle_preview(self.use_preview_check.isChecked())
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
        self.modality_list.depopulate()
        self.populate_reference_list()

    def populate_reference_list(self) -> None:
        """Populate reference list."""
        modalities = list(self.registration_model.modalities.keys())
        self.reference_choice.clear()
        self.reference_choice.addItem("None")
        self.reference_choice.addItems(modalities)

        reference = self.registration_model.reference
        if reference is None or reference not in modalities:
            reference = "None"
        self.reference_choice.setCurrentText(reference)

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
        self.registration_model.check_for_reflections = self.reflection_check.isChecked()
        self.registration_model.micro_registration = self.micro_check.isChecked()
        self.registration_model.micro_registration_fraction = self.micro_fraction.value()
        self.registration_model.non_rigid_registration = self.non_rigid_check.isChecked()
        # set reference
        reference = self.reference_choice.currentText()
        self.registration_model.set_reference(reference if reference != "None" else None)
        # set detector and matcher method
        self.registration_model.feature_detector = self.feature_choice.currentText()
        self.registration_model.feature_matcher = self.matcher_choice.currentText().upper()

        if self.registration_model.project_dir.exists() and not hp.confirm(
            self, f"Project <b>{self.registration_model.name}</b> already exists.<br><b>Overwrite?</b>"
        ):
            return None
        path = self.registration_model.save()
        logger.trace(f"Saved project to {self.registration_model.project_dir}")
        return path

    def on_close(self) -> None:
        """Close project."""
        if self.registration_model and hp.confirm(
            self, "Are you sure you want to close the project?", "Close project?"
        ):
            self.name_label.setText("")
            self._image_widget.on_close_dataset(force=True)
            self.view.clear()
            self.modality_list.populate()
            self.populate_reference_list()
            logger.trace("Closed project.")

    def on_rename_modality(self, widget, new_name: str) -> None:
        """Rename modality."""
        modality = widget.item_model
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

    def on_show_modality(self, modality: Modality, state: bool = True, overwrite: bool = False) -> None:
        """Preview image."""
        from image2image_reg.utils.preprocessing import preprocess_preview_valis

        pyramid = -1
        wrapper = self.data_model.get_wrapper()
        if wrapper:
            with MeasureTimer() as timer:
                reader = wrapper.get_reader_for_path(modality.path)
                scale = reader.scale_for_pyramid(pyramid)
                layer = self.view.get_layer(modality.name)
                preprocessing_hash = (
                    hash_preprocessing(modality.preprocessing, pyramid=pyramid)
                    if self.use_preview_check.isChecked()
                    else f"pyramid={pyramid}"
                )
                # no need to re-process if the layer is already there
                if layer and layer.metadata.get("preview_hash") == preprocessing_hash:
                    layer.visible = state
                    return

                if self.use_preview_check.isChecked():
                    image = preprocess_preview_valis(
                        reader.pyramid[pyramid], reader.is_rgb, scale[0], modality.preprocessing
                    )
                else:
                    image = reader.get_channel(0, pyramid)
                widget = self.modality_list.get_widget_for_item_model(modality)
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
        preprocessing_hash = hash_preprocessing(modality.preprocessing, pyramid=pyramid)

        wrapper = self.data_model.get_wrapper()
        if wrapper:
            layer = self.view.get_layer(modality.name)
            if layer and layer.rgb:
                self.view.remove_layer(modality.name)

            widget = self.modality_list.get_widget_for_item_model(modality)
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


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="valis", level=0)
