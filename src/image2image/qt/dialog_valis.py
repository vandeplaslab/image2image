"""Valis registration dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_reg.workflows.valis import ValisReg
from koyo.typing import PathLike
from loguru import logger
from qtextra.queue.popup import QueuePopup
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_WSIREG_FORMATS
from image2image.qt._dialog_mixins import SingleViewerMixin
from image2image.qt._dialogs._select import LoadWidget
from image2image.qt._wsireg._list import QtModalityList


class ImageValisWindow(SingleViewerMixin):
    """Image viewer dialog."""

    _console = None
    _output_dir = None
    _registration_model: ValisReg | None = None

    WINDOW_HAS_QUEUE = True

    def __init__(
        self, parent: QWidget | None, run_check_version: bool = True, project_dir: PathLike | None = None, **_kwargs
    ):
        super().__init__(
            parent, f"image2valis: Valis Registration app (v{__version__})", run_check_version=run_check_version
        )
        # if CONFIG.first_time_wsireg:
        #     hp.call_later(self, self.on_show_tutorial, 10_000)
        self._setup_config()
        self.queue_popup = QueuePopup(self)
        self.queue_btn.clicked.connect(self.queue_popup.show)  # noqa
        # if project_dir:
        #     self._on_load_from_project(project_dir)

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.only_last_pyramid = True
        READER_CONFIG.init_pyramid = False
        READER_CONFIG.split_czi = False
        logger.trace("Setup config for image2wsireg.")

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
        # connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        # connect(self._image_widget.dataset_dlg.evt_closing, self.on_remove_image, state=state)
        # connect(self._image_widget.dataset_dlg.evt_project, self._on_load_from_project, state=state)
        # # connect(self.view.widget.canvas.events.key_press, self.keyPressEvent, state=state)
        # connect(self.view.viewer.events.status, self._status_changed, state=state)
        #
        # connect(self.modality_list.evt_delete, self.on_remove_modality, state=state)
        # connect(self.modality_list.evt_rename, self.on_rename_modality, state=state)
        # connect(self.modality_list.evt_hide_others, self.on_hide_modalities, state=state)
        # connect(self.modality_list.evt_preview_preprocessing, self.on_preview, state=state)
        # connect(self.modality_list.evt_resolution, self.on_update_modality, state=state)
        # connect(self.modality_list.evt_show, self.on_show_modality, state=state)
        # connect(self.modality_list.evt_set_preprocessing, self.on_update_modality, state=state)
        # connect(self.modality_list.evt_preview_transform_preprocessing, self.on_preview_transform, state=state)
        # connect(self.modality_list.evt_preprocessing_close, self.on_preview_close, state=state)
        # connect(self.modality_list.evt_color, self.on_update_colormap, state=state)
        # connect(self.registration_map.evt_message, self.statusbar.showMessage)
        #
        # connect(QUEUE.evt_errored, self.on_registration_finished, state=state)
        # connect(QUEUE.evt_finished, self.on_registration_finished, state=state)

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
            project_extension=[".valis.json"],
            allow_geojson=True,
        )

        side_widget = QWidget()
        side_widget.setMinimumWidth(400)
        side_widget.setMaximumWidth(400)

        self.modality_list = QtModalityList(self)

        self.use_preview_check = hp.make_checkbox(
            self,
            "Use preview image",
            tooltip="Use preview image for viewing instead of the first channel only.",
            value=False,
            # func=self.on_show_modalities,
        )
        self.hide_others_check = hp.make_checkbox(
            self,
            "Hide others when previewing",
            tooltip="When previewing, hide other images to reduce clutter.",
            checked=False,
            # func=self.on_hide_not_previewed_modalities,
        )

        self.reference_choice = hp.make_combobox(
            self,
            ["None"],
            tooltip="Reference image for registration. If 'None' is selected, reference image will be automatically"
            " selected.",
        )
        self.feature_choice = hp.make_combobox(
            self, [], tooltip="Feature detection method when performing feature matching between adjacent images."
        )
        self.matcher_choice = hp.make_combobox(
            self, [], tooltip="Feature matching method when performing feature matching between adjacent images."
        )
        self.reflection_check = hp.make_checkbox(
            self,
            "",
            tooltip="Check for reflection during registration. Disabling this option can speed-up registration but"
            " make sure that images are not mirrored.",
            value=True,
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
        # Project
        side_layout.addRow(hp.make_h_line_with_text("Valis project"))
        side_layout.addRow(hp.make_label(side_widget, "Name"), self.name_label)
        side_layout.addRow(hp.make_label(side_widget, "Output directory", alignment=Qt.AlignmentFlag.AlignLeft))
        side_layout.addRow(
            hp.make_h_layout(self.output_dir_label, self.output_dir_btn, stretch_id=(0,), spacing=1, margin=1),
        )
        # Advanced options
        hidden_settings = hp.make_advanced_collapsible(side_widget, "Advanced options", allow_checkbox=False)
        hidden_settings.addRow(hp.make_label(self, "Reduce data size"), self.as_uint8)
        hidden_settings.addRow(hp.make_label(self, "Open when finished"), self.open_when_finished)
        side_layout.addRow(hidden_settings)

        self.save_btn = hp.make_qta_btn(
            self, "save", normal=True, tooltip="Save i2reg project to disk.", func=self.on_save_to_i2reg
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
        )
        hp.make_menu_item(
            side_widget,
            "Run registration (without saving, not recommended)",
            menu=menu,
            func=self.on_run_no_save,
            icon="run",
            tooltip="Perform registration. Images will open in the viewer when finished. Project will not be"
            " saved before adding to the queue.",
        )
        hp.make_menu_item(
            side_widget,
            "Queue registration",
            menu=menu,
            func=self.on_queue,
            icon="queue",
            tooltip="Registration task will be added to the queue and you can start it manually.",
        )
        hp.make_menu_item(
            side_widget,
            "Queue registration (without saving, not recommended)",
            menu=menu,
            func=self.on_queue_no_save,
            icon="queue",
            tooltip="Registration task will be added to the queue and you can start it manually. Project will not be"
            " saved before adding to the queue.",
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

    def _on_load_from_project(self, path_: PathLike) -> None:
        if path_:
            path_ = Path(path_)
            project = ValisReg.from_path(path_.parent if path_.is_file() else path_)
            if project:
                self._registration_model = project
                self.output_dir = project.output_dir
                self.name_label.setText(project.name)
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

    def _queue_registration_model(self, add_delayed: bool, save: bool = True) -> bool:
        """Queue registration model."""
        # if not self.registration_model:
        #     return False
        # if not self._validate():
        #     return False
        # if save and not self.save_model():
        #     return False
        # task = make_registration_task(
        #     self.registration_model,
        #     write_transformed=self.write_registered_check.isChecked(),
        #     write_not_registered=self.write_not_registered_check.isChecked(),
        #     write_merged=self.write_merged_check.isChecked(),
        #     as_uint8=self.as_uint8.isChecked(),
        # )
        # if task:
        #     if QUEUE.is_queued(task.task_id):
        #         hp.toast(
        #             self, "Already queued", "This task is already in the queue.", icon="warning", position="top_left"
        #         )
        #         return False
        #     QUEUE.add_task(task, add_delayed=add_delayed)
        # return True

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

    def _make_statusbar(self) -> None:
        super()._make_statusbar()

        self.queue_btn = hp.make_qta_btn(self, "queue", tooltip="Open queue popup.", small=True)
        self.statusbar.insertPermanentWidget(0, self.queue_btn)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="valis", level=0)
