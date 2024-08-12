"""Whole-slide registration mixin."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
import qtpy.QtWidgets as Qw
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_reg.models import Modality
from image2image_reg.workflows import IWsiReg, ValisReg
from koyo.typing import PathLike
from loguru import logger
from napari.layers import Image
from qtextra.queue.popup import QUEUE, QueuePopup
from qtextra.queue.task import Task
from superqt import ensure_main_thread
from superqt.utils import qdebounced

from image2image.models.data import DataModel
from image2image.qt._dialog_mixins import SingleViewerMixin

if ty.TYPE_CHECKING:
    from qtextra._napari.image import NapariImageView
    from qtextra.widgets.qt_collapsible import QtCheckCollapsible

    from image2image.config import ElastixConfig, ValisConfig
    from image2image.qt._wsi._list import QtModalityList


class ImageWsiWindow(SingleViewerMixin):
    """Image viewer dialog."""

    _registration_model: IWsiReg | ValisReg | None = None

    WINDOW_TITLE: str
    PROJECT_SUFFIX: str
    RUN_DISABLED: bool
    OTHER_PROJECT: str
    CONFIG: ValisConfig | ElastixConfig

    # Widgets
    view: NapariImageView
    name_label: Qw.QLabel
    write_registered_check: Qw.QCheckBox
    write_not_registered_check: Qw.QCheckBox
    write_merged_check: Qw.QCheckBox
    as_uint8: Qw.QCheckBox
    use_preview_check: Qw.QCheckBox
    hide_others_check: Qw.QCheckBox
    modality_list: QtModalityList
    open_when_finished: Qw.QCheckBox
    pyramid_level: Qw.QSpinBox

    def __init__(
        self, parent: Qw.QWidget | None, run_check_version: bool = True, project_dir: PathLike | None = None, **_kwargs
    ):
        super().__init__(parent, self.WINDOW_TITLE, run_check_version=run_check_version)
        self._setup_config()
        self.queue_popup = QueuePopup(self)
        self.queue_btn.clicked.connect(self.queue_popup.show)  # noqa
        if project_dir:
            self._on_load_from_project(project_dir)

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False

    @staticmethod
    def make_registration_task(**kwargs) -> Task:
        """Make registration task."""
        raise NotImplementedError("Must implement method")

    @property
    def registration_model(self) -> IWsiReg | ValisReg:
        """Registration model."""
        raise NotImplementedError("Must implement method")

    def _on_load_from_project(self, path_: PathLike) -> None:
        raise NotImplementedError("Must implement method")

    def save_model(self) -> Path | None:
        """Save model in the current state."""
        raise NotImplementedError("Must implement method")

    def on_close(self, force: bool = False) -> None:
        """Close project."""
        raise NotImplementedError("Must implement method")

    def on_show_modality(self, modality: Modality, state: bool = True, overwrite: bool = False) -> None:
        """Preview image."""
        raise NotImplementedError("Must implement method")

    def on_populate_list(self) -> None:
        """Populate list."""
        raise NotImplementedError("Must implement method")

    def on_depopulate_list(self, keys: list[str]) -> None:
        """Populate list."""
        raise NotImplementedError("Must implement method")

    def _validate(self) -> None:
        """Validate project."""
        is_valid, errors = self.registration_model.validate(require_paths=True)
        if not is_valid:
            from image2image.qt._dialogs._errors import ErrorsDialog

            dlg = ErrorsDialog(self, errors)
            dlg.show()
        return is_valid

    def _queue_registration_model(self, add_delayed: bool, save: bool = True, cli: bool = False) -> bool:
        """Queue registration model."""
        if not self.registration_model:
            return False
        if not self._validate():
            return False
        if save and not self.save_model():
            return False
        task = self.make_registration_task(
            project=self.registration_model,
            write_transformed=self.write_registered_check.isChecked(),
            write_attached=self.write_attached_check.isChecked(),
            write_not_registered=self.write_not_registered_check.isChecked(),
            write_merged=self.write_merged_check.isChecked(),
            as_uint8=self.as_uint8.isChecked(),
        )
        if task:
            if cli:
                commands = [" ".join(command) for command in task.command_iter()]
                hp.copy_text_to_clipboard("; ".join(commands))
                logger.trace(f"Copied command to clipboard: {commands}")
                return True
            if QUEUE.is_queued(task.task_id):
                hp.toast(
                    self, "Already queued", "This task is already in the queue.", icon="warning", position="top_left"
                )
                return False
            if QUEUE.is_finished(task.task_id):
                self.queue_popup.queue_list.on_remove_task(task.task_id)
                logger.info(f"Removed finished task {task.task_id}.")
            QUEUE.add_task(task, add_delayed=add_delayed)
        return True

    def on_run(self) -> None:
        """Execute registration."""
        self._queue_registration_model(add_delayed=False)

    def on_run_no_save(self) -> None:
        """Execute registration."""
        self._queue_registration_model(add_delayed=False, save=False)

    def on_queue(self) -> None:
        """Queue registration."""
        self._queue_registration_model(add_delayed=True)

    def on_queue_no_save(self) -> None:
        """Queue registration."""
        self._queue_registration_model(add_delayed=True, save=False)

    def on_copy_to_clipboard(self) -> None:
        """Copy command to clipboard."""
        self._queue_registration_model(add_delayed=False, save=False, cli=True)

    def on_validate_path(self, _: ty.Any = None) -> None:
        """Validate project path."""
        name = self.name_label.text()
        object_name = ""
        if not name:
            object_name = "error"
        if name and self.output_dir and (self.output_dir / name).with_suffix(self.PROJECT_SUFFIX).exists():
            object_name = "warning"
        hp.set_object_name(self.name_label, object_name=object_name)

    def on_remove_modality(self, modality: Modality) -> None:
        """Remove modality from the project."""
        keys = self.data_model.get_key_for_path(modality.path)
        for key in keys:
            self._image_widget.dataset_dlg.on_remove_dataset(key)

    def on_update_modality(self, modality: Modality) -> None:
        """Preview image."""
        self.registration_model.modalities[modality.name].pixel_size = modality.pixel_size
        self.registration_model.modalities[modality.name].preprocessing = modality.preprocessing
        logger.trace(f"Updated modality: {modality.name}")

    def on_update_colormap(self, modality: Modality, color: np.ndarray):
        """Update colormap."""
        layer = self.view.get_layer(modality.name)
        if layer:
            layer.colormap = color

    def _on_pre_loading_images(self, filelist: list[str]) -> None:
        """Before files are loaded, we can check whether all files come from the same directory."""
        # if path is already defined, let's not do anything
        if self.output_dir_label.text():
            return
        common_output_dir = []
        for path in filelist:
            path = Path(path)
            if path.is_file():
                path = path.parent
            if path.name in ["Images"]:
                path = path.parent
            if path.suffix in [".wsireg", ".valis"]:
                path = path.parent
            common_output_dir.append(path)
        if len(set(common_output_dir)) == 1:
            self.output_dir = common_output_dir[0]
            self.output_dir_label.setText(hp.hyper(self.output_dir))
            logger.trace(f"Automatically set output directory to {self.output_dir}")

    def on_maybe_add_attachment(self, filelist: list[str]) -> None:
        """Add attachment modalities to the project."""
        if not filelist:
            return
        # select where to attach the modalities
        if not self.registration_model.modalities:
            hp.toast(self, "Error", "Please load fixed image first.", icon="error", position="top_left")
            return

        options = {key: key for key in self.registration_model.modalities}
        choice = hp.choose(self, options, "Select modality to attach to")
        if not choice:
            return

        for widget in self.modality_list.widget_iter():
            if widget and widget.item_model.name == choice:
                widget.auto_add_attachments(filelist)
                break

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

    @qdebounced(timeout=250)
    def on_update_pyramid_level(self) -> None:
        """Update pyramid level."""
        self._on_update_pyramid_level()

    def _on_update_pyramid_level(self) -> None:
        """Update pyramid level."""
        level = self.pyramid_level.value()
        if level == 0 and not hp.confirm(
            self,
            "Please confirm if you wish to preview the full-resolution image.<br><b>Pre-processing might take"
            " a bit of time.</b>",
            "Please confirm",
        ):
            self.pyramid_level.setValue(-1)
            return
        self.on_show_modalities()
        logger.trace(f"Updated pyramid level to {level}")

    @qdebounced(timeout=250)
    def on_show_modalities(self, _: ty.Any = None) -> None:
        """Show modality images."""
        self._on_show_modalities()

    def _on_show_modalities(self) -> None:
        """Show modality images."""
        self.CONFIG.update(use_preview=self.use_preview_check.isChecked())
        self.modality_list.toggle_preview(self.CONFIG.use_preview)
        for _, modality, widget in self.modality_list.item_model_widget_iter():
            self.on_show_modality(modality, state=widget.visible_btn.visible, overwrite=True)

    def on_hide_not_previewed_modalities(self) -> None:
        """Hide any modality that is not previewed."""
        modalities = []
        for _, modality, widget in self.modality_list.item_model_widget_iter():
            if widget._preprocessing_dlg is not None:
                modalities.append(modality)
        self.on_hide_modalities(modalities)
        logger.trace("Hiding not previewed modalities.")

    def on_hide_modalities(self, modality: Modality | list[Modality], hide: bool | None = None) -> None:
        """Hide other modalities."""
        if hide is None:
            hide = self.hide_others_check.isChecked()
        self.CONFIG.update(hide_others=hide)
        if not hide:
            return
        if not isinstance(modality, list):
            modality = [modality]
        visible_modalities = [mod.name for mod in modality]
        if not visible_modalities:
            return
        for layer in self.view.get_layers_of_type(Image):
            layer.visible = layer.name in visible_modalities
        self.modality_list.toggle_visible([layer.name for layer in self.view.get_layers_of_type(Image)])
        logger.trace(f"Showing {visible_modalities}")

    def on_open_in_viewer(self) -> None:
        """Open registration in viewer."""
        if self.registration_model:
            path = self.registration_model.project_dir / "Images"
            if path.exists():
                self.on_open_viewer("--image_dir", str(path))
                logger.trace("Opening viewer.")

    def on_open_in_viewer_and_close_project(self) -> None:
        """Open registration in viewer and then close it."""
        self.on_open_in_viewer()
        self.on_close(force=True)

    def on_preview_close(self) -> None:
        """Preview was closed."""
        self.on_show_modalities()

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
        """Update modality name."""
        self.registration_model.modalities[modality.name] = self.registration_model.modalities.pop(old_name)
        self.on_show_modality(modality)
        self.on_populate_list()
        logger.trace(f"Updated modality name from {old_name} to {modality.name}")

    def on_registration_finished(self, task: Task, _: ty.Any = None) -> None:
        """Open registration in viewer."""
        if self.open_when_finished.isChecked():
            path = Path(task.task_name) / "Images"
            if path.exists():
                self.on_open_viewer("--image_dir", str(path))
                logger.trace("Registration finished - opening viewer.")
            else:
                hp.toast(self, "Error", f"Failed to open viewer for {path!s}.", icon="error", position="top_left")

    def on_remove_image(self, model: DataModel, channel_names: list[str], keys: list[str]) -> None:
        """Remove image."""
        self.on_depopulate_list(keys)

    def on_save_to_project(self) -> None:
        """Save project."""
        raise NotImplementedError("Must implement method")

    def on_save_as_other_project(self) -> None:
        """Save as other project."""
        raise NotImplementedError("Must implement method")

    def on_save_and_close_project(self) -> None:
        """Save and close project."""
        self.on_save_to_project()
        self.on_close(force=True)

    def _make_output_widgets(self, side_widget: Qw.QWidget) -> None:
        self.name_label = hp.make_line_edit(
            side_widget,
            "Name",
            tooltip="Name of the project",
            placeholder=f"e.g. project{self.PROJECT_SUFFIX}",
            func=self.on_validate_path,
            func_changed=self.on_validate_path,
        )
        self.output_dir_label = hp.make_label(
            side_widget, hp.hyper(self.output_dir), tooltip="Output directory", enable_url=True
        )
        self.output_dir_btn = hp.make_qta_btn(
            side_widget,
            "folder",
            tooltip="Change output directory",
            func=self.on_set_output_dir,
            normal=True,
            standout=True,
        )

    def _make_hidden_widgets(self, side_widget: Qw.QWidget) -> QtCheckCollapsible:
        self.write_not_registered_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write original, not-registered images (those without any transformations such as target).",
            value=self.CONFIG.write_not_registered,
            func=self.on_update_config,
        )
        self.write_registered_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write registered images.",
            value=self.CONFIG.write_registered,
            func=self.on_update_config,
        )
        self.write_attached_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write attached modalities (images, shapes or points).",
            value=self.CONFIG.write_attached,
            func=self.on_update_config,
        )
        self.write_merged_check = hp.make_checkbox(
            self,
            "",
            tooltip="Merge non- and transformed images into a single image.",
            value=self.CONFIG.write_merged,
            func=self.on_update_config,
        )
        self.rename_check = hp.make_checkbox(
            self,
            "",
            tooltip="Rename images during writing. By default names will be written as:"
            " <source_name>_to_<target_name>.ome.tiff, however, they can also be named as <original_name>.ome.tiff",
            value=self.CONFIG.rename,
            func=self.on_update_config,
        )
        self.as_uint8 = hp.make_checkbox(
            self,
            "",
            tooltip="Convert to uint8 to reduce file size with minimal data loss. This will result in change of the"
            " dynamic range of the image to between 0-255.",
            value=self.CONFIG.as_uint8,
            func=self.on_update_config,
        )
        self.open_when_finished = hp.make_checkbox(
            self,
            "",
            tooltip="Open images in the viewer when registration is finished.",
            value=self.CONFIG.open_when_finished,
            func=self.on_update_config,
        )

        hidden_settings = hp.make_advanced_collapsible(side_widget, "Export options", allow_checkbox=False)
        hidden_settings.addRow(hp.make_label(self, "Write registered images"), self.write_registered_check)
        hidden_settings.addRow(hp.make_label(self, "Write unregistered images"), self.write_not_registered_check)
        hidden_settings.addRow(hp.make_label(self, "Write attached modalities"), self.write_attached_check)
        hidden_settings.addRow(hp.make_label(self, "Merge merged images"), self.write_merged_check)
        hidden_settings.addRow(hp.make_label(self, "Rename images"), self.rename_check)
        hidden_settings.addRow(hp.make_label(self, "Reduce data size"), self.as_uint8)
        hidden_settings.addRow(hp.make_label(self, "Open when finished"), self.open_when_finished)
        return hidden_settings

    def on_update_config(self, _: ty.Any = None) -> None:
        """Update values in config."""
        self.CONFIG.write_not_registered = self.write_not_registered_check.isChecked()
        self.CONFIG.write_registered = self.write_registered_check.isChecked()
        self.CONFIG.write_attached = self.write_attached_check.isChecked()
        self.CONFIG.write_merged = self.write_merged_check.isChecked()
        self.CONFIG.rename = self.rename_check.isChecked()
        self.CONFIG.as_uint8 = self.as_uint8.isChecked()
        self.CONFIG.open_when_finished = self.open_when_finished.isChecked()

    def on_clear_project(self) -> None:
        """Clear project."""
        if hp.confirm(self, "Are you sure you wish to clear all project data?<br><b>This action cannot be undone.</b>"):
            self.registration_model.clear(clear_all=True)

    def on_save_menu(self) -> None:
        """Save menu."""
        menu = hp.make_menu(self.save_btn)
        hp.make_menu_item(self, "Save", menu=menu, func=self.on_save_to_project, icon="save")
        hp.make_menu_item(self, "Save and close", menu=menu, func=self.on_save_and_close_project, icon="save")
        hp.show_right_of_mouse(menu)

    def on_view_menu(self) -> None:
        """Save menu."""
        menu = hp.make_menu(self.viewer_btn)
        hp.make_menu_item(self, "Open in viewer", menu=menu, func=self.on_open_in_viewer, icon="viewer")
        hp.make_menu_item(
            self, "Open in viewer and close", menu=menu, func=self.on_open_in_viewer_and_close_project, icon="viewer"
        )
        hp.show_right_of_mouse(menu)

    def on_close_menu(self) -> None:
        """Save menu."""
        menu = hp.make_menu(self.close_btn)
        hp.make_menu_item(self, "Close", menu=menu, func=self.on_close, icon="delete")
        hp.make_menu_item(
            self, "Close (without confirmation)", menu=menu, func=lambda: self.on_close(True), icon="delete"
        )
        hp.show_right_of_mouse(menu)

    def on_run_menu(self) -> None:
        """Menu."""
        menu = hp.make_menu(self.run_btn)
        hp.make_menu_item(
            self,
            "Run registration",
            menu=menu,
            func=self.on_run,
            icon="run",
            tooltip="Perform registration. Images will open in the viewer when finished.",
            disabled=self.RUN_DISABLED,
        )
        hp.make_menu_item(
            self,
            "Run registration (without saving, not recommended)",
            menu=menu,
            func=self.on_run_no_save,
            icon="run",
            tooltip="Perform registration. Images will open in the viewer when finished. Project will not be"
            " saved before adding to the queue.",
            disabled=self.RUN_DISABLED,
        )
        hp.make_menu_item(
            self,
            "Queue registration",
            menu=menu,
            func=self.on_queue,
            icon="queue",
            tooltip="Registration task will be added to the queue and you can start it manually.",
            disabled=self.RUN_DISABLED,
        )
        hp.make_menu_item(
            self,
            "Queue registration (without saving, not recommended)",
            menu=menu,
            func=self.on_queue_no_save,
            icon="queue",
            tooltip="Registration task will be added to the queue and you can start it manually. Project will not be"
            " saved before adding to the queue.",
            disabled=self.RUN_DISABLED,
        )
        hp.make_menu_item(
            self,
            "Copy command to clipboard",
            menu=menu,
            func=self.on_copy_to_clipboard,
            icon="cli",
            tooltip="Copy the registration command to clipboard so it can be executed externally.",
            disabled=self.RUN_DISABLED,
        )
        menu.addSeparator()
        hp.make_menu_item(
            self,
            f"Save as {self.OTHER_PROJECT} project",
            menu=menu,
            func=self.on_save_as_other_project,
            icon="save",
            tooltip=f"Tries to save the project as {self.OTHER_PROJECT} project, without any guarantees.",
        )
        menu.addSeparator()
        hp.make_menu_item(
            self,
            "Clear project (remove all data)",
            menu=menu,
            func=self.on_clear_project,
            icon="delete",
            tooltip="Clears all data from the project, excluding the configuration file.",
        )
        hp.show_above_widget(menu, self.run_btn, y_offset=-100, x_offset=-150)

    def _make_run_widgets(self, side_widget: Qw.QWidget) -> None:
        self.save_btn = hp.make_qta_btn(
            self,
            "save",
            normal=True,
            tooltip="Save Valis project to disk.",
            func=self.on_save_to_project,
            func_menu=self.on_save_menu,
            standout=True,
        )
        self.viewer_btn = hp.make_qta_btn(
            side_widget,
            "viewer",
            func=self.on_open_in_viewer,
            func_menu=self.on_view_menu,
            tooltip="Open the project in the viewer. This only makes sense if registration is complete.",
            standout=True,
        )
        self.close_btn = hp.make_qta_btn(
            side_widget,
            "delete",
            tooltip="Close project (without saving)",
            func=self.on_close,
            func_menu=self.on_close_menu,
            standout=True,
        )
        self.run_btn = hp.make_btn(
            side_widget,
            "Execute...",
            tooltip="Immediately execute registration",
            properties={"with_menu": True},
            func=self.on_run_menu,
        )
