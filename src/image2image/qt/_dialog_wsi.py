"""Whole-slide registration mixin."""

from __future__ import annotations

import os
import typing as ty
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
import qtpy.QtWidgets as Qw
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_reg.models import Modality, Preprocessing
from image2image_reg.workflows import ElastixReg, ValisReg
from koyo.typing import PathLike
from loguru import logger
from qtextra.config import THEMES
from qtextra.queue.popup import QUEUE, QueuePopup
from qtextra.queue.task import Task
from qtpy.QtCore import QRegularExpression, Qt
from qtpy.QtGui import QKeyEvent, QRegularExpressionValidator
from superqt import ensure_main_thread
from superqt.utils import qdebounced

import image2image.constants as C
from image2image.config import ValisConfig
from image2image.enums import LEVEL_TO_PYRAMID, PYRAMID_TO_LEVEL
from image2image.models.data import DataModel
from image2image.qt._dialog_mixins import SingleViewerMixin
from image2image.utils.valis import hash_preprocessing

if ty.TYPE_CHECKING:
    from qtextra.widgets.qt_collapsible import QtCheckCollapsible
    from qtextraplot._napari.image import NapariImageView

    from image2image.config import ElastixConfig
    from image2image.qt._wsi._list import QtModalityItem, QtModalityList
    from image2image.qt._wsi._mask import CropDialog, MaskDialog


class ImageWsiWindow(SingleViewerMixin):
    """Image viewer dialog."""

    _registration_model: ElastixReg | ValisReg | None = None

    WINDOW_TITLE: str
    PROJECT_SUFFIX: str
    RUN_DISABLED: bool
    OTHER_PROJECT: str
    IS_VALIS: bool
    CONFIG: ValisConfig | ElastixConfig

    # Widgets
    view: NapariImageView
    name_label: Qw.QLabel
    write_registered_check: Qw.QCheckBox
    write_not_registered_check: Qw.QCheckBox
    write_merged_check: Qw.QCheckBox
    as_uint8: Qw.QCheckBox
    clip_combo: Qw.QComboBox
    use_preview_check: Qw.QCheckBox
    hide_others_check: Qw.QCheckBox
    modality_list: QtModalityList
    pyramid_level: Qw.QComboBox
    hidden_settings: QtCheckCollapsible
    project_settings: Qw.QGroupBox

    _mask_dlg: MaskDialog | None = None
    _crop_dlg: CropDialog | None = None

    def __init__(
        self,
        parent: Qw.QWidget | None,
        run_check_version: bool = True,
        project_dir: PathLike | None = None,
        **_kwargs: ty.Any,
    ):
        super().__init__(parent, self.WINDOW_TITLE, run_check_version=run_check_version)
        self._setup_config()
        self.queue_popup = QueuePopup(self)
        self.queue_btn.clicked.connect(self.queue_popup.show)  # noqa
        if project_dir:
            self._on_load_from_project(project_dir)
        self.setup_i2reg_path()
        self.on_set_write_warning()

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = True

    @staticmethod
    def make_registration_task(**kwargs: ty.Any) -> Task:
        """Make registration task."""
        raise NotImplementedError("Must implement method")

    @property
    def registration_model(self) -> ElastixReg | ValisReg:
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

    def _validate(self) -> bool:
        """Validate project."""
        is_valid, errors = self.registration_model.validate(require_paths=True)
        if not is_valid:
            from image2image.qt._dialogs._errors import ErrorsDialog

            dlg = ErrorsDialog(self, errors)
            dlg.show()
        return is_valid

    def _queue_registration_model(self, add_delayed: bool, save: bool = True, cli: bool = False) -> bool | Task:
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
            rename=self.rename_check.isChecked(),
            clip=self.clip_combo.currentText(),
            with_i2reg=not self.RUN_DISABLED,
        )
        if task:
            if cli:
                return task
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

    def on_copy_to_clipboard(self, which: str = "all") -> None:
        """Copy command to clipboard."""
        task = self._queue_registration_model(add_delayed=False, save=False, cli=True)
        commands_ = [" ".join(command) for command in task.command_iter()]  # type: ignore[union-attr]
        commands = []
        if which == "preprocess":
            commands = [commands_[0]]
        if which == "registration":
            commands = [commands_[1]]
        elif which == "transformation" and len(commands_) == 3:
            commands = [commands_[2]]
        elif which == "all":
            commands = commands_
        if commands:
            commands_str = "; ".join(commands)
            commands_str = commands_str.replace("--no_color --debug", "--debug")
            hp.copy_text_to_clipboard(commands_str)
            logger.trace(f"Copied command to clipboard: {commands_str}")

    def on_validate_path(self, _: ty.Any = None) -> None:
        """Validate project path."""
        name = self.name_label.text()
        object_name = ""
        try:
            if not name:
                object_name = "error"
            if name and self.output_dir and (self.output_dir / name).with_suffix(self.PROJECT_SUFFIX).exists():
                object_name = "warning"
        except Exception:  # noqa
            object_name = "error"
        hp.set_object_name(self.name_label, object_name=object_name)

    def on_remove_modality(self, modality: Modality) -> None:
        """Remove modality from the project."""
        keys = self.data_model.get_key_for_path(modality.path)
        for key in keys:
            self._image_widget.dset_dlg.on_remove_dataset(key)
        self.modality_list.populate()

    def on_update_resolution_from_table(self, key: str) -> None:
        """Update resolution."""
        reader = self.data_model.get_reader_for_key(key)
        if not reader:
            return
        modality = self.registration_model.get_modality(name_or_path=reader.path)
        if not modality:
            return
        modality.pixel_size = reader.resolution
        item: QtModalityItem = self.modality_list.get_widget_for_modality(modality)
        if item:
            item.resolution_label.setText(f"{reader.resolution:.5f}")
        self.on_update_resolution_of_modality(modality)

    def on_update_resolution_from_list(self, modality: Modality) -> None:
        """Update resolution."""
        keys = self.data_model.get_key_for_path(modality.path)
        for key in keys:
            self._image_widget.dset_dlg.on_set_resolution(key, modality.pixel_size)

    def on_update_preprocessing_of_modality(self, modality: Modality) -> None:
        """Preview image."""
        self.registration_model.modalities[modality.name].preprocessing = modality.preprocessing
        logger.trace(f"Updated preprocessing of modality: {modality.name}")

    def on_update_resolution_of_modality(self, modality: Modality) -> None:
        """Preview image."""
        if self.registration_model.modalities[modality.name].pixel_size != modality.pixel_size:
            self.registration_model.modalities[modality.name].pixel_size = modality.pixel_size
            self._on_show_modalities()
            logger.trace(f"Updated resolution of modality: {modality.name} ({modality.pixel_size})")

    def on_update_colormap(self, modality: Modality, color: np.ndarray):
        """Update colormap."""
        layer = self.view.get_layer(modality.name)
        if layer:
            layer.colormap = color

    def _get_preprocessing_hash(
        self,
        modality: Modality,
        preprocessing: Preprocessing | None = None,
        preview: bool | None = None,
        pyramid: int | None = None,
    ) -> str:
        preview = preview if preview is not None else self.use_preview_check.isChecked()
        pyramid = pyramid or PYRAMID_TO_LEVEL[self.pyramid_level.currentText()]
        if preprocessing is None:
            preprocessing = modality.preprocessing

        return (
            hash_preprocessing(preprocessing, pyramid=pyramid, pixel_size=modality.pixel_size)
            if preview
            else f"pyramid={pyramid}; pixel_size={modality.pixel_size}"
        )

    def _on_pre_loading_images(self, filelist: list[str | dict[str, ty.Any]]) -> None:
        """Before files are loaded, we can check whether all files come from the same directory."""
        # if path is already defined, let's not do anything
        if self.output_dir_label.text():
            return
        common_output_dir = []
        for path_or_dict in filelist:
            path = path_or_dict if isinstance(path_or_dict, str) else path_or_dict["path"]
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
            self.output_dir_label.setToolTip(str(self.output_dir))
            logger.trace(f"Automatically set output directory to {self.output_dir}")

    def on_maybe_add_attachment(self, filelist: list[str]) -> None:
        """Add attachment modalities to the project."""
        if not filelist:
            return
        # select where to attach the modalities
        if not self.registration_model.modalities:
            hp.toast(self, "Error", "Please load fixed image first.", icon="error", position="top_left")
            return

        options = {key: key for key in self.registration_model.get_image_modalities(with_attachment=False)}
        if len(options) == 0:
            hp.toast(self, "Error", "No modalities to attach to.", icon="error", position="top_left")
            return
        choice = options.popitem()[0] if len(options) == 1 else hp.choose(self, options, "Select modality to attach to")

        if not choice:
            return

        for widget in self.modality_list.widget_iter():
            if widget and widget.modality.name == choice:
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

    def on_update_pyramid_level(self) -> None:
        """Update pyramid level."""
        self._on_update_pyramid_level()

    def _on_update_pyramid_level(self) -> None:
        """Update pyramid level."""
        level = PYRAMID_TO_LEVEL[self.pyramid_level.currentText()]
        if level == 0 and not hp.confirm(
            self,
            "Please confirm if you wish to preview the full-resolution image.<br><b>Pre-processing might take"
            " a bit of time.</b>",
            "Please confirm",
        ):
            self.pyramid_level.setCurrentText(LEVEL_TO_PYRAMID[-1])
            return
        self.on_show_modalities()
        logger.trace(f"Updated pyramid level to {level}")

    def on_recolor(self) -> None:
        """Recolor."""
        menu = hp.make_menu(self)
        hp.make_menu_item(
            self,
            "Update colors - use standard palette",
            menu=menu,
            func=lambda _: self.modality_list.on_change_colors("normal"),
        )
        hp.make_menu_item(
            self,
            "Update colors - use bright palette",
            menu=menu,
            func=lambda _: self.modality_list.on_change_colors("bright"),
        )
        hp.make_menu_item(
            self,
            "Update colors - use protanomaly-friendly palette",
            menu=menu,
            func=lambda _: self.modality_list.on_change_colors("protanomaly"),
        )
        hp.make_menu_item(
            self,
            "Update colors - use deuteranomaly-friendly palette",
            menu=menu,
            func=lambda _: self.modality_list.on_change_colors("deuteranomaly"),
        )
        hp.make_menu_item(
            self,
            "Update colors - use tritanomaly-friendly palette",
            menu=menu,
            func=lambda _: self.modality_list.on_change_colors("tritanomaly"),
        )
        menu.addSeparator()
        hp.make_menu_item(
            self,
            "Update colors - use random colors",
            menu=menu,
            func=lambda _: self.modality_list.on_change_colors("random"),
        )
        hp.show_below_widget(menu, self.recolor_btn)

    def on_sort(self) -> None:
        """Sort modalities."""
        menu = hp.make_menu(self)
        hp.make_menu_item(
            self,
            "Sort by name (ascending)",
            menu=menu,
            func=lambda _: self.modality_list.on_sort_modalities("name", reverse=True),
        )
        hp.make_menu_item(
            self,
            "Sort by name (descending)",
            menu=menu,
            func=lambda _: self.modality_list.on_sort_modalities("name", reverse=False),
        )
        hp.make_menu_item(
            self,
            "Sort by path (ascending)",
            menu=menu,
            func=lambda _: self.modality_list.on_sort_modalities("path", reverse=True),
        )
        hp.make_menu_item(
            self,
            "Sort by path (descending)",
            menu=menu,
            func=lambda _: self.modality_list.on_sort_modalities("path", reverse=False),
        )
        hp.show_below_widget(menu, self.sort_btn)

    def on_apply(self) -> None:
        """Apply."""
        menu = hp.make_menu(self)
        hp.make_menu_item(self, "Select channels...", menu=menu, func=self._on_select_channels)
        if self.registration_model.n_registrations == 0:
            hp.make_menu_item(self, "Rename...", menu=menu, func=self._on_rename)
        menu.addSeparator()
        hp.make_menu_from_options(
            self,
            menu,
            [
                "Intensity: Histogram equalization (only)",
                "Intensity: Contrast enhancement (only)",
                "Intensity: Histogram equalization & Contrast enhancement",
                None,
                "Spatial: Downsample x1",
                "Spatial: Downsample x2",
                "Spatial: Downsample x3",
                None,
                "Default: Brightfield",
                "Default: Fluorescence",
                "Default: DAPI",
                "Default: PAS",
                "Default: H&E",
                "Default: postAF(Brightfield)",
                "Default: postAF(EGFP)",
                "Default: DAPI",
                None,
                "Channel(id): 0",
                "Channel(name): DAPI",
                "Channel(name): EGFP",
            ],
            func=self._on_apply_action,
        )
        hp.show_below_widget(menu, self.apply_btn)

    def _on_select_channels(self) -> None:
        """Select channels."""
        channel_names: list[str] = []
        for name in self.registration_model.get_image_modalities(with_attachment=False):
            modality = self.registration_model.modalities[name]
            channel_names.extend(modality.channel_names)
        selected = hp.choose_from_list(self, list(set(channel_names)))

        if not selected:
            return
        for name in self.registration_model.get_image_modalities(with_attachment=False):
            modality = self.registration_model.modalities[name]
            modality.preprocessing.select_channels(channel_names=selected)
        self.modality_list.update_preprocessing_info()
        self.on_show_modalities()

    def _on_rename(self) -> None:
        """Rename channels."""
        from qtextra.dialogs.qt_text_replace import QtTextReplace

        names = list(self.registration_model.get_image_modalities(with_attachment=False))
        dlg = QtTextReplace(self, names)
        if dlg.exec_() == Qw.QDialog.DialogCode.Accepted:
            new_names = dlg.new_texts
            for old_name, new_name in zip(names, new_names):
                if old_name == new_name:
                    continue
                modality = self.registration_model.modalities[old_name]
                widget = self.modality_list.get_widget_for_modality(modality)
                self.on_rename_modality(widget, new_name)
                widget._set_from_model()

    def _on_apply_action(self, option: str) -> None:
        """Apply action."""
        from image2image.qt._wsi._preprocessing import handle_default

        # if not self.registration_model.get_image_modalities(with_attachment=False) or not hp.confirm(
        #     self,
        #     "Please confirm if you wish to apply the selected pre-processing to all modalities.",
        #     "Please confirm",
        # ):
        #     return

        histogram, contrast, factor, default, channel_id, channel_name = None, None, None, None, None, None
        if option.startswith("Intensity:"):
            histogram = "Histogram equalization (only)" in option or "&" in option
            contrast = "Contrast enhancement (only)" in option or "&" in option
        if option.startswith("Spatial"):
            factor = int(option.split("x")[-1])
        if option.startswith("Default"):
            default = option.split(": ")[-1]
        if option.startswith("Channel(id)"):
            channel_id = int(option.split(": ")[-1])
        if option.startswith("Channel(name)"):
            channel_name = option.split(": ")[-1]

        if not any(v is None for v in [histogram, contrast, factor, default]):
            logger.warning("Failed to parse pre-processing option.")
            return

        for name in self.registration_model.get_image_modalities(with_attachment=False):
            modality = self.registration_model.modalities[name]
            if not modality.preprocessing:
                continue
            if default is not None:
                modality.preprocessing.update_from_another(
                    handle_default(default, modality.preprocessing, valis=self.IS_VALIS)
                )
            if histogram is not None:
                modality.preprocessing.equalize_histogram = histogram
            if contrast is not None:
                modality.preprocessing.contrast_enhance = contrast
            if factor is not None:
                modality.preprocessing.downsample = factor
            if channel_id is not None:
                modality.preprocessing.select_channel(channel_id=channel_id)
            if channel_name is not None:
                modality.preprocessing.select_channel(channel_name=channel_name)

            logger.trace(
                f"Updated pre-processing of {name} with default={default}; histogram={histogram}; contrast={contrast};"
                f" factor={factor}; channel_id={channel_id}; channel_name={channel_name}"
            )
        self.modality_list.update_preprocessing_info()
        self.on_show_modalities()

    @qdebounced(timeout=250)
    def on_show_modalities(self, _: ty.Any = None) -> None:
        """Show modality images."""
        self._on_show_modalities()

    def _on_show_modalities(self) -> None:
        """Show modality images."""
        self.CONFIG.update(use_preview=self.use_preview_check.isChecked())
        # self.on_hide_not_previewed_modalities()
        for modality, widget in self.modality_list.model_widget_iter():
            self.on_show_modality(modality, state=widget.visible_btn.state)  # , overwrite=True)

    def on_hide_not_previewed_modalities(self) -> None:
        """Hide any modality that is not previewed."""
        hide = self.hide_others_check.isChecked()
        visible_modalities, maybe_visible_modalities, hidden_modalities = [], [], []
        widget: QtModalityItem
        for modality, widget in self.modality_list.model_widget_iter():
            # if hide is enabled, we should only display the modality currently being pre-processed
            if hide:
                if (
                    widget._preprocessing_dlg is not None
                    or (self._crop_dlg is not None and self._crop_dlg.current_modality == modality)
                    or (self._mask_dlg is not None and self._mask_dlg.current_modality == modality)
                ):
                    visible_modalities.append(modality.name)
                else:
                    if widget.visible_btn.state:
                        maybe_visible_modalities.append(modality.name)
                    hidden_modalities.append(modality.name)
            # otherwise, let's use the user selection
            else:
                if widget.visible_btn.state:
                    visible_modalities.append(modality.name)
                else:
                    hidden_modalities.append(modality.name)

        if not visible_modalities:
            visible_modalities = maybe_visible_modalities
            hidden_modalities = []

        # actually show/hide widgets
        visible_modalities = list(set(visible_modalities))
        hidden_modalities = list(set(hidden_modalities))
        for modality, widget in self.modality_list.model_widget_iter():
            if modality.name in hidden_modalities:
                widget.visible_btn.set_state(False, trigger=True)
            if modality.name in visible_modalities:
                widget.visible_btn.set_state(True, trigger=True)

    def on_hide_modalities(
        self, visible_modalities: Modality | list[Modality] | list[str], hide: bool | None = None
    ) -> None:
        """Hide other modalities."""
        self.on_hide_not_previewed_modalities()

    def on_open_in_viewer(self) -> None:
        """Open registration in viewer."""
        if self.registration_model:
            path = self.registration_model.project_dir / "Images"
            if path.exists():
                self.on_open_viewer("--file_dir", str(path))
                hp.toast(
                    self,
                    "Opening viewer...",
                    f"Opening viewer for {hp.hyper(path, path.name)}.",
                    icon="info",
                    position="top_left",
                )
                logger.trace("Opening viewer.")
            else:
                logger.warning("No image registration model available.")

    def on_open_in_viewer_and_close_project(self) -> None:
        """Open registration in viewer and then close it."""
        self.on_open_in_viewer()
        self.on_close(force=True)

    def on_preview_close(self, modality: Modality) -> None:
        """Preview was closed."""
        preview = self.use_preview_check.isChecked()
        for modality_, widget in self.modality_list.model_widget_iter():
            if modality.name == modality_.name:
                self.on_show_modality(modality_, state=widget.visible_btn.state, overwrite=not preview)

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
        """Update modality name."""
        self.registration_model.modalities[modality.name] = self.registration_model.modalities.pop(old_name)
        self.on_show_modality(modality)
        self.on_populate_list()
        logger.trace(f"Updated modality name from {old_name} to {modality.name}")

    def on_registration_finished(self, task: Task, _: ty.Any = None) -> None:
        """Open registration in viewer."""
        if self.CONFIG.open_when_finished:
            path = Path(task.task_name) / "Images"
            if path.exists() and len(list(path.glob("*.ome.tiff"))) > 0:
                self.on_open_viewer("--file_dir", str(path))
                logger.trace("Registration finished - opening viewer.")
            else:
                hp.toast(self, "Error", f"Failed to open viewer for {path!s}.", icon="error", position="top_left")

    def on_remove_image(self, model: DataModel, channel_names: list[str], keys: list[str]) -> None:
        """Remove image."""
        self.on_depopulate_list(keys)

    def on_save_to_project(self) -> bool:
        """Save project."""
        raise NotImplementedError("Must implement method")

    def on_save_with_errors_to_project(self) -> bool:
        """Save project."""
        raise NotImplementedError("Must implement method")

    def on_save_as_other_project(self) -> None:
        """Save as other project."""
        raise NotImplementedError("Must implement method")

    def on_save_and_close_project(self) -> None:
        """Save and close project."""
        saved = self.on_save_to_project()
        if saved:
            self.on_close(force=True)

    def _make_output_widgets(self, side_widget: Qw.QWidget) -> Qw.QGroupBox:
        # regex = r"^[a-zA-Z0-9_\-.,=]"
        regex = r"^[a-zA-Z0-9_\-.,=]{1,50}$"

        self.name_label = hp.make_line_edit(
            side_widget,
            "",
            tooltip="Name of the project",
            placeholder=f"e.g. project{self.PROJECT_SUFFIX}",
            func=self.on_validate_path,
            func_changed=self.on_validate_path,
            validator=QRegularExpressionValidator(QRegularExpression(regex)),
        )
        self.output_dir_label = hp.make_label(side_widget, "", tooltip="", enable_url=True)
        self.output_dir_btn = hp.make_qta_btn(
            side_widget,
            "folder",
            tooltip="Change output directory",
            func=self.on_set_output_dir,
            normal=True,
            standout=True,
        )

        project_settings = hp.make_group_box(self, f"{self.APP_NAME.capitalize()} project")
        project_layout = hp.make_form_layout(parent=project_settings, margin=(6, 6, 6, 6))
        project_layout.addRow(hp.make_label(side_widget, "Name"), self.name_label)
        project_layout.addRow(
            hp.make_h_layout(
                hp.make_label(side_widget, "Output directory", alignment=Qt.AlignmentFlag.AlignLeft),
                self.output_dir_btn,
                stretch_id=(0,),
                spacing=1,
                margin=1,
            ),
        )
        project_layout.addRow(self.output_dir_label)
        return project_settings

    def _make_visibility_options(self) -> None:
        self.apply_btn = hp.make_qta_btn(
            self,
            "magic",
            tooltip="Apply pre-processing to all modalities",
            func=self.on_apply,
            normal=True,
            standout=True,
        )
        self.show_all_btn = hp.make_qta_btn(
            self,
            "visible_on",
            tooltip="Show all modalities",
            func=self.modality_list.on_show_all,
            normal=True,
            standout=True,
        )
        self.hide_all_btn = hp.make_qta_btn(
            self,
            "visible_off",
            tooltip="Hide all modalities",
            func=self.modality_list.on_hide_all,
            normal=True,
            standout=True,
        )
        self.recolor_btn = hp.make_qta_btn(
            self,
            "color_palette",
            tooltip="Change colors...",
            func=self.on_recolor,
            normal=True,
            standout=True,
        )
        self.sort_btn = hp.make_qta_btn(
            self,
            "sort",
            tooltip="Sort list by...",
            func=self.on_sort,
            normal=True,
            standout=True,
        )
        self.filter_modalities_by = hp.make_line_edit(
            self,
            placeholder="Type in modality name...",
            func_changed=self.modality_list.on_filter_by_dataset_name,
        )

        self.use_preview_check = hp.make_checkbox(
            self,
            "Use preview image",
            tooltip="Use preview image for viewing instead of the first channel only.",
            value=self.CONFIG.use_preview,
            func=self.on_show_modalities,
        )
        self.hide_others_check = hp.make_checkbox(
            self,
            "Hide others",
            tooltip="When previewing, hide other images to reduce clutter.",
            checked=self.CONFIG.hide_others,
            func=self.on_hide_not_previewed_modalities,
        )

    def _make_hidden_widgets(self, side_widget: Qw.QWidget) -> QtCheckCollapsible:
        self.write_check = hp.make_checkbox(
            self,
            "",
            tooltip="Quickly toggle between writing or not writing of images.",
            value=False,
            func=self.on_toggle_write,
        )
        self.write_registered_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write registered images.",
            value=self.CONFIG.write_registered,
            func=self.on_update_config,
        )
        self.write_not_registered_check = hp.make_checkbox(
            self,
            "",
            tooltip="Write original, not-registered images (those without any transformations such as target).",
            value=self.CONFIG.write_not_registered,
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
        hp.disable_widgets(self.write_merged_check, disabled=True)

        self.rename_check = hp.make_checkbox(
            self,
            "",
            tooltip="Rename images during writing. By default names will be written as:"
            " <source_name>_to_<target_name>.ome.tiff, however, they can also be named as <original_name>.ome.tiff",
            value=self.CONFIG.rename,
            func=self.on_update_config,
        )
        self.clip_combo = hp.make_combobox(
            self,
            ["ignore", "clip", "remove", "part-remove"],
            value=self.CONFIG.clip,
            tooltip="What to do about points/shapes outside of the image when using non-linear transformation.<br>"
            "<b>ignore</b> will do nothing to points outside of the image.<br>"
            "<b>clip</b> will clip points to the image size.<br>"
            "<b>remove</b> will remove points outside of the image.<br>"
            "<b>part-remove</b> will remove points outside of the image, but keep the part that is inside.",
        )

        self.as_uint8 = hp.make_checkbox(
            self,
            "",
            tooltip=C.UINT8_TIP,
            value=self.CONFIG.as_uint8,
            func=self.on_update_config,
        )

        hidden_settings = hp.make_advanced_collapsible(
            side_widget,
            "Export options",
            allow_checkbox=False,
            allow_icon=False,
            warning_icon=("warning", {"color": THEMES.get_theme_color("warning")}),
        )
        hidden_settings.addRow(hp.make_label(self, "Write/don't write"), self.write_check)
        hidden_settings.addRow(hp.make_label(self, "Write registered images"), self.write_registered_check)
        hidden_settings.addRow(hp.make_label(self, "Write unregistered images"), self.write_not_registered_check)
        hidden_settings.addRow(hp.make_label(self, "Write attached modalities"), self.write_attached_check)
        hidden_settings.addRow(hp.make_label(self, "Write merged images"), self.write_merged_check)
        hidden_settings.addRow(hp.make_label(self, "Rename images"), self.rename_check)
        hidden_settings.addRow(hp.make_label(self, "Clip"), self.clip_combo)
        hidden_settings.addRow(
            hp.make_label(self, "Reduce data size"),
            hp.make_h_layout(
                self.as_uint8,
                hp.make_warning_label(
                    self,
                    C.UINT8_WARNING,
                    normal=True,
                    icon_name=("warning", {"color": THEMES.get_theme_color("warning")}),
                ),
                spacing=2,
                stretch_id=(0,),
            ),
        )
        return hidden_settings

    def on_update_config(self, _: ty.Any = None) -> None:
        """Update values in config."""
        self.CONFIG.write_not_registered = self.write_not_registered_check.isChecked()
        self.CONFIG.write_registered = self.write_registered_check.isChecked()
        self.CONFIG.write_attached = self.write_attached_check.isChecked()
        self.CONFIG.write_merged = self.write_merged_check.isChecked()
        self.CONFIG.rename = self.rename_check.isChecked()
        self.CONFIG.as_uint8 = self.as_uint8.isChecked()
        self.CONFIG.clip = self.clip_combo.currentText()
        self.on_set_write_warning()

    def on_toggle_write(self) -> None:
        """Toggle between write/no-write of images."""
        write = self.write_check.isChecked()
        self.write_registered_check.setChecked(write)
        self.write_not_registered_check.setChecked(write)
        self.write_attached_check.setChecked(write)
        self.write_merged_check.setChecked(write)
        self.on_set_write_warning()

    def on_set_write_warning(self) -> None:
        """Enable warning."""
        tooltip = []
        if not any(
            [
                self.CONFIG.write_not_registered,
                self.CONFIG.write_registered,
                self.CONFIG.write_attached,
                self.CONFIG.write_merged,
            ]
        ):
            tooltip.append("- Current settings will not export any images as all <b>write</b> options are disabled.")
        if not self.CONFIG.write_attached and self.registration_model and self.registration_model.has_attachments():
            tooltip.append("- There are attachments in the project but they won't be exported.")
        if self.CONFIG.as_uint8:
            tooltip.append(
                "- Images will be converted to uint8 to reduce file size. This can lead to data loss and should be used"
                " with caution."
            )
        self.hidden_settings.warning_label.setToolTip("<br>".join(tooltip))
        self.hidden_settings.set_warning_visible(len(tooltip) > 0)

    def on_clear_project(self) -> None:
        """Clear project."""
        if hp.confirm(self, "Are you sure you wish to clear all project data?<br><b>This action cannot be undone.</b>"):
            self.registration_model.clear(clear_all=True)

    def on_save_menu(self) -> None:
        """Save menu."""
        menu = hp.make_menu(self.save_btn)
        hp.make_menu_item(self, "Save", menu=menu, func=self.on_save_to_project, icon="save")
        hp.make_menu_item(
            self,
            "Save (ignore errors, not recommended)",
            menu=menu,
            func=self.on_save_with_errors_to_project,
            icon="save",
        )
        hp.make_menu_item(self, "Save and close", menu=menu, func=self.on_save_and_close_project, icon="save")
        hp.show_above_widget(menu, self.save_btn)

    def on_view_menu(self) -> None:
        """Save menu."""
        menu = hp.make_menu(self.viewer_btn)
        hp.make_menu_item(self, "Open in viewer", menu=menu, func=self.on_open_in_viewer, icon="viewer")
        hp.make_menu_item(
            self, "Open in viewer and close", menu=menu, func=self.on_open_in_viewer_and_close_project, icon="viewer"
        )
        menu.addSeparator()
        hp.make_menu_item(
            self,
            "Open when finished",
            menu=menu,
            checkable=True,
            checked=self.CONFIG.open_when_finished,
            func=lambda _: self.CONFIG.update(open_when_finished=not self.CONFIG.open_when_finished),
        )
        hp.show_above_widget(menu, self.viewer_btn)

    def on_close_menu(self) -> None:
        """Save menu."""
        menu = hp.make_menu(self.close_btn)
        hp.make_menu_item(self, "Close", menu=menu, func=self.on_close, icon="delete")
        hp.make_menu_item(
            self, "Close (without confirmation)", menu=menu, func=lambda: self.on_close(True), icon="delete"
        )
        hp.show_above_widget(menu, self.close_btn)

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

        menu.addSeparator()
        hp.make_menu_item(
            self,
            "Copy preprocessing, registration + transformation command to clipboard",
            menu=menu,
            func=lambda: self.on_copy_to_clipboard("all"),
            icon="cli",
            tooltip="Copy the registration command to clipboard so it can be executed externally.",
        )
        hp.make_menu_item(
            self,
            "Copy preprocessing command to clipboard",
            menu=menu,
            func=lambda: self.on_copy_to_clipboard("preprocess"),
            icon="cli",
            tooltip="Copy the registration command to clipboard so it can be executed externally.",
        )
        hp.make_menu_item(
            self,
            "Copy registration command to clipboard",
            menu=menu,
            func=lambda: self.on_copy_to_clipboard("registration"),
            icon="cli",
            tooltip="Copy the registration command to clipboard so it can be executed externally.",
        )
        hp.make_menu_item(
            self,
            "Copy transformation command to clipboard",
            menu=menu,
            func=lambda: self.on_copy_to_clipboard("transformation"),
            icon="cli",
            tooltip="Copy the registration command to clipboard so it can be executed externally.",
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
        hp.show_above_mouse(menu)

    def _make_run_widgets(self, side_widget: Qw.QWidget) -> None:
        self.save_btn = hp.make_qta_btn(
            self,
            "save",
            normal=True,
            tooltip=f"Save {self.APP_NAME.capitalize()} project to disk.",
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
            normal=True,
        )
        self.close_btn = hp.make_qta_btn(
            side_widget,
            "delete",
            tooltip="Close project (without saving)",
            func=self.on_close,
            func_menu=self.on_close_menu,
            standout=True,
            normal=True,
        )
        self.run_btn = hp.make_btn(
            side_widget,
            "Execute...",
            tooltip="Immediately execute registration",
            properties={"with_menu": True},
            func=self.on_run_menu,
        )

    def _make_config_menu(self) -> Qw.QMenu:
        menu = super()._make_config_menu()
        menu.addSeparator()
        hp.make_menu_item(self, "Set i2reg path", menu=menu, icon="env", func=self.on_set_i2reg_path)
        return menu

    def on_set_i2reg_path(self) -> None:
        """Set i2reg path."""
        env_path = Path(self.CONFIG.env_i2reg) if self.CONFIG.env_i2reg else Path.cwd()
        base_dir = env_path.parent if env_path.is_file() else env_path
        env_path = hp.get_filename(self, "Select i2reg executable", base_dir)
        if env_path and Path(env_path).exists():
            self.RUN_DISABLED = False
            os.environ["IMAGE2IMAGE_I2REG_PATH"] = str(env_path)
            self.CONFIG.update(env_i2reg=str(env_path))

    def setup_i2reg_path(self) -> None:
        """Set i2reg path."""
        if not os.environ.get("IMAGE2IMAGE_I2REG_PATH", None) and self.CONFIG.env_i2reg:
            env_path = Path(self.CONFIG.env_i2reg)
            if env_path.exists():
                self.RUN_DISABLED = False
                os.environ["IMAGE2IMAGE_I2REG_PATH"] = str(env_path)
                logger.trace(f"Set i2reg path to {env_path}.")

    @qdebounced(timeout=50, leading=True)
    def keyPressEvent(self, evt: QKeyEvent) -> None:
        """Key press event."""
        if hasattr(evt, "native"):
            evt = evt.native
        try:
            key = evt.key()
            ignore = self._handle_key_press(key)
            if ignore:
                evt.ignore()
            if not evt.isAccepted():
                return None
            return super().keyPressEvent(evt)
        except RuntimeError:
            return None

    @qdebounced(timeout=100, leading=True)
    def on_handle_key_press(self, key: int) -> bool:
        """Handle key-press event"""
        return self._handle_key_press(key)

    def _handle_key_press(self, key: int) -> bool:
        ignore = False
        if key == Qt.Key.Key_G:
            self.on_toggle_grid()
        elif key == Qt.Key.Key_H:
            self.hide_others_check.setChecked(not self.hide_others_check.isChecked())
        elif key == Qt.Key.Key_P:
            self.use_preview_check.setChecked(not self.use_preview_check.isChecked())
        return ignore
