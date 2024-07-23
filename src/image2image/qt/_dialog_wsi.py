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
    from image2image.qt._wsireg._list import QtModalityList


class ImageWsiWindow(SingleViewerMixin):
    """Image viewer dialog."""

    _registration_model: IWsiReg | ValisReg | None = None

    WINDOW_TITLE: str

    # Widgets
    name_label: Qw.QLabel
    write_registered_check: Qw.QCheckBox
    write_not_registered_check: Qw.QCheckBox
    write_merged_check: Qw.QCheckBox
    as_uint8: Qw.QCheckBox
    use_preview_check: Qw.QCheckBox
    hide_others_check: Qw.QCheckBox
    modality_list: QtModalityList
    open_when_finished: Qw.QCheckBox

    def __init__(
        self, parent: Qw.QWidget | None, run_check_version: bool = True, project_dir: PathLike | None = None, **_kwargs
    ):
        super().__init__(parent, self.WINDOW_TITLE, run_check_version=run_check_version)
        # if CONFIG.first_time_wsireg:
        #     hp.call_later(self, self.on_show_tutorial, 10_000)
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
        logger.trace("Setup config for image2wsireg.")

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

    def on_close(self) -> None:
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

    def _queue_registration_model(self, add_delayed: bool, save: bool = True) -> bool:
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
            write_not_registered=self.write_not_registered_check.isChecked(),
            write_merged=self.write_merged_check.isChecked(),
            as_uint8=self.as_uint8.isChecked(),
        )
        if task:
            if QUEUE.is_queued(task.task_id):
                hp.toast(
                    self, "Already queued", "This task is already in the queue.", icon="warning", position="top_left"
                )
                return False
            QUEUE.add_task(task, add_delayed=add_delayed)
        return True

    def on_run(self) -> None:
        """Execute registration."""
        if self._queue_registration_model(add_delayed=False):
            self.queue_popup.show()

    def on_run_no_save(self) -> None:
        """Execute registration."""
        if self._queue_registration_model(add_delayed=False, save=False):
            self.queue_popup.show()

    def on_queue(self) -> None:
        """Queue registration."""
        if self._queue_registration_model(add_delayed=True):
            self.queue_popup.show()

    def on_queue_no_save(self) -> None:
        """Queue registration."""
        if self._queue_registration_model(add_delayed=True, save=False):
            self.queue_popup.show()

    def on_validate(self, _: ty.Any = None) -> None:
        """Validate project."""
        name = self.name_label.text()
        hp.set_object_name(self.name_label, object_name="error" if not name else "")

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
            "Please confirm if you wish to preview the full-resolution image. Pre-processing might take a"
            " bit of time.",
            "Please confirm",
        ):
            return
        self.on_show_modalities()
        logger.trace(f"Updated pyramid level to {level}")

    @qdebounced(timeout=250)
    def on_show_modalities(self, _: ty.Any = None) -> None:
        """Show modality images."""
        self._on_show_modalities()

    def _on_show_modalities(self) -> None:
        self.modality_list.toggle_preview(self.use_preview_check.isChecked())
        for _, modality, widget in self.modality_list.item_model_widget_iter():
            self.on_show_modality(modality, state=widget.visible_btn.visible, overwrite=True)

    def on_hide_not_previewed_modalities(self) -> None:
        """Hide any modality that is not previewed."""
        modalities = []
        for _, modality, widget in self.modality_list.item_model_widget_iter():
            if widget._preprocessing_dlg is not None:
                modalities.append(modality)
        self.on_hide_modalities(modalities)

    def on_hide_modalities(self, modality: Modality | list[Modality], hide: bool | None = None) -> None:
        """Hide other modalities."""
        hide = self.hide_others_check.isChecked() if hide is not None else hide
        if not hide:
            return
        if not isinstance(modality, list):
            modality = [modality]
        for layer in self.view.get_layers_of_type(Image):
            layer.visible = layer.name in [mod.name for mod in modality]
        self.modality_list.toggle_visible([layer.name for layer in self.view.get_layers_of_type(Image)])

    def on_open_in_viewer(self) -> None:
        """Open registration in viewer."""
        if self.registration_model:
            path = self.registration_model.project_dir / "Images"
            if path.exists():
                self.on_open_viewer("--image_dir", str(path))
                logger.trace("Opening viewer.")

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
