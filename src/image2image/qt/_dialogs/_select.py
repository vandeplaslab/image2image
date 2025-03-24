"""Widget for loading data."""

from __future__ import annotations

import typing as ty
from functools import partial
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.typing import PathLike
from loguru import logger
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_label_icon import QtActiveIcon
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout, QWidget

from image2image.config import SingleAppConfig
from image2image.enums import VIEW_TYPE_TRANSLATIONS
from image2image.models.data import DataModel
from image2image.models.transform import TransformData, TransformModel
from image2image.qt._dialogs._dataset import DatasetDialog

if ty.TYPE_CHECKING:
    from qtextraplot._napari.image.wrapper import NapariImageView

    from image2image.qt._dialog_base import Window


logger = logger.bind(src="LoadDialog")


class LoadWidget(QWidget):
    """Widget for loading data."""

    IS_FIXED: bool = True
    INFO_TEXT = ""
    CHANNEL_FIXED: bool | None = None

    def __init__(
        self,
        parent: Window | None,
        view: NapariImageView | None,
        config: SingleAppConfig,
        n_max: int = 0,
        available_formats: str | None = None,
        allow_geojson: bool = False,
        allow_channels: bool = True,
        allow_iterate: bool = False,
        allow_transform: bool = False,
        allow_save: bool = True,
        project_extension: list[str] | None = None,
        show_split_czi: bool = True,
        confirm_czi: bool = False,
        allow_import_project: bool = False,
        allow_export_project: bool = False,
    ):
        """Init."""
        self.allow_geojson = allow_geojson
        self.allow_import_project = allow_import_project
        self.allow_export_project = allow_export_project
        self.CONFIG = config
        super().__init__(parent)
        self.view = view
        self.n_max = n_max
        self.model: DataModel = DataModel(is_fixed=self.IS_FIXED)

        # Dataset dialog
        self.transform_model = TransformModel()
        self.transform_model.add_transform(
            "Identity matrix", TransformData.from_array(np.eye(3, dtype=np.float64)), with_inverse=False, silent=True
        )
        self.dset_dlg = DatasetDialog(
            self,
            self.model,
            self.view,
            self.transform_model,
            self.CONFIG,
            self.IS_FIXED,
            self.n_max,
            allow_geojson=self.allow_geojson,
            allow_iterate=allow_iterate,
            allow_transform=allow_transform,
            allow_channels=allow_channels,
            allow_save=allow_save,
            available_formats=available_formats,
            project_extension=project_extension,
            show_split_czi=show_split_czi,
            confirm_czi=confirm_czi,
        )
        self.dset_dlg.evt_loading.connect(lambda: self.active_icon.set_active(True))
        self.dset_dlg.evt_loaded.connect(lambda _: self.active_icon.set_active(False))
        if parent is not None and hasattr(parent, "evt_dropped"):
            connect(parent.evt_dropped, self.dset_dlg.on_drop)

        self._setup_ui()

    def on_open_dataset_dialog(self) -> None:
        """Select transformation data."""
        self.dset_dlg.show()

    def _setup_ui(self) -> QFormLayout:
        """Setup UI."""
        self.info_text = hp.make_label(
            self,
            self.INFO_TEXT,
            bold=True,
            wrap=True,
            alignment=Qt.AlignmentFlag.AlignCenter,
            visible=self.INFO_TEXT != "",
        )

        self.active_icon = QtActiveIcon()
        self.import_btn = hp.make_qta_btn(
            self, "import", func=self.on_import_project, tooltip="Open project file.", standout=True, normal=True
        )
        if not self.allow_import_project:
            self.import_btn.setVisible(False)
        self.export_btn = hp.make_qta_btn(
            self, "export", func=self.on_export_project, tooltip="Save project file.", standout=True, normal=True
        )
        if not self.allow_export_project:
            self.export_btn.setVisible(False)
        self.add_btn = hp.make_qta_btn(
            self, "add", func=self.on_select_dataset, tooltip="Add image(s) to the viewer.", standout=True, normal=True
        )
        self.more_btn = hp.make_btn(
            self,
            "Select channels...",
            func=self.on_open_dataset_dialog,
            tooltip="Open dialog to add/remove images or adjust pixel size.",
        )

        layout = hp.make_form_layout(parent=self, margin=0)
        layout.addRow(self.info_text)
        layout.addRow(
            hp.make_h_layout(
                self.import_btn,
                self.add_btn,
                hp.make_qta_btn(
                    self,
                    "delete",
                    func=self.on_close_dataset,
                    func_menu=self.on_clear_menu,
                    tooltip="Remove image(s) from the viewer.<br>Right-click to open menu.",
                    standout=True,
                    normal=True,
                ),
                self.export_btn,
                self.active_icon,
                self.more_btn,
                stretch_id=3,
                spacing=2,
            )
        )
        return layout

    def on_clear_menu(self) -> None:
        """Clear data."""
        menu = hp.make_menu(self.add_btn)
        hp.make_menu_item(self, "Close", menu=menu, func=self.on_close_dataset, icon="save")
        hp.make_menu_item(
            self,
            "Close all (without confirmation)",
            menu=menu,
            func=partial(self.on_close_dataset, force=True),
            icon="delete",
        )
        hp.show_below_widget(menu, self.add_btn, x_offset=50)

    def on_set_path(
        self,
        paths: PathLike | ty.Sequence[PathLike],
        transform_data: dict[str, TransformData] | None = None,
        resolution: dict[str, float] | None = None,
        reader_kws: dict[str, dict] | None = None,
    ) -> None:
        """Set the path and immediately load it."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.dset_dlg._on_load_dataset(paths, transform_data, resolution, reader_kws)

    def on_import_project(self) -> None:
        """Open project."""
        self.dset_dlg.on_import_project()

    def on_export_project(self) -> None:
        """Save project."""
        self.dset_dlg.on_export_project()

    def on_select_dataset(self, _evt: ty.Any = None) -> None:
        """Load data."""
        self.dset_dlg.on_select_dataset()

    def on_close_dataset(self, _evt: ty.Any = None, force: bool = False) -> None:
        """Load data."""
        self.dset_dlg.on_close_dataset(force=force)

    def on_clear_data(self, _evt: ty.Any = None) -> None:
        """Clear data."""
        if hp.confirm(self, "Are you sure you want to clear all data?"):
            while self.model.n_paths > 0:
                self.dset_dlg.on_close_dataset(force=True)

    def channel_list(self) -> list[str]:
        """List of visible channels."""
        return self.dset_dlg.channel_list()


class FixedWidget(LoadWidget):
    """Widget for loading fixed data."""

    # class attrs
    IS_FIXED = True
    INFO_TEXT = "Select 'fixed' data..."

    def __init__(
        self,
        parent: Window | None,
        view: NapariImageView,
        config: SingleAppConfig,
        n_max: int = 0,
        allow_geojson: bool = False,
        allow_iterate: bool = False,
        allow_transform: bool = False,
        allow_channels: bool = True,
        project_extension: list[str] | None = None,
        allow_import_project: bool = False,
        allow_export_project: bool = False,
    ):
        super().__init__(
            parent,
            view,
            config,
            n_max,
            allow_geojson=allow_geojson,
            allow_channels=allow_channels,
            allow_transform=allow_transform,
            allow_iterate=allow_iterate,
            project_extension=project_extension,
            allow_import_project=allow_import_project,
            allow_export_project=allow_export_project,
        )

        if parent is not None and hasattr(parent, "evt_fixed_dropped"):
            connect(parent.evt_fixed_dropped, self.dset_dlg.on_drop)


class MovingWidget(LoadWidget):
    """Widget for loading moving data."""

    # class attrs
    IS_FIXED = False
    CHANNEL_FIXED = False
    INFO_TEXT = "Select 'moving' data..."

    # events
    evt_dataset_select = Signal(str)
    evt_show_transformed = Signal(str)
    evt_view_type = Signal(object)

    def __init__(
        self,
        parent: Window | None,
        view: NapariImageView,
        config: SingleAppConfig,
        n_max: int = 0,
        allow_geojson: bool = False,
        allow_channels: bool = True,
        project_extension: list[str] | None = None,
        allow_iterate: bool = False,
        allow_transform: bool = False,
        allow_import_project: bool = False,
        allow_export_project: bool = False,
    ):
        super().__init__(
            parent,
            view,
            config,
            n_max,
            allow_geojson=allow_geojson,
            allow_channels=allow_channels,
            project_extension=project_extension,
            allow_transform=allow_transform,
            allow_iterate=allow_iterate,
            allow_import_project=allow_import_project,
            allow_export_project=allow_export_project,
        )

        # extra events
        connect(self.dset_dlg.evt_loaded, self._on_update_choice)
        connect(self.dset_dlg.evt_closed, self._on_clear_choice)

        if parent is not None and hasattr(parent, "evt_moving_dropped"):
            self.dset_dlg._on_loaded_dataset = self.dset_dlg._on_loaded_dataset_with_preselection
            connect(parent.evt_moving_dropped, self.dset_dlg.on_drop)

    def _on_update_choice(self, _model: object | None = None, _channel_list: list[str] | None = None) -> None:
        """Update list of available options."""
        datasets = self.model.dataset_names("image")
        hp.combobox_setter(self.dataset_choice, clear=True, items=datasets)
        self._on_update_transformed_choice()

    def _on_update_transformed_choice(self) -> None:
        """Update list of available options."""
        current = self.dataset_choice.currentText()
        hp.combobox_setter(
            self.transformed_choice, clear=True, items=["None", *self.model.get_channel_names_for_keys([current])]
        )
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _on_clear_choice(self, _model: object) -> None:
        """Clear list of available options."""
        self._on_update_choice()
        self._on_update_transformed_choice()
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _setup_ui(self) -> QFormLayout:
        """Setup UI."""
        layout = super()._setup_ui()

        self.view_type_choice = hp.make_combobox(
            self,
            data=VIEW_TYPE_TRANSLATIONS,
            value=str(READER_CONFIG.view_type),
            func=self._on_update_view_type,
            tooltip="Select what kind of image should be displayed.<br><b>Overlay</b> will use the 'true' image and can"
            " be overlaid with other images.<br><b>Random</b> will display single image with random intensity.",
        )
        layout.addRow(hp.make_label(self, "View type"), self.view_type_choice)
        self.dataset_choice = hp.make_combobox(
            self,
            tooltip="Select which dataset should be used in registration process.",
            func=self._on_toggle_dataset,
        )
        layout.addRow(hp.make_label(self, "Dataset"), self.dataset_choice)
        self.transformed_choice = hp.make_combobox(
            self,
            tooltip="Select which image channels should be displayed in the moving modality (and in the fixed"
            " modality).",
            func=self._on_toggle_transformed,
        )
        layout.addRow(hp.make_label(self, "Channel"), self.transformed_choice)
        return layout

    def _on_update_view_type(self, value: str) -> None:
        """Update view type."""
        READER_CONFIG.view_type = value.lower()
        self.evt_view_type.emit(value.lower())  # noqa

    def toggle_transformed(self) -> None:
        """Toggle visibility of transformed image."""
        hp.increment_combobox(self.transformed_choice, 1)

    def _on_toggle_dataset(self, value: str) -> None:
        """Toggle visibility of dataset."""
        self._on_update_transformed_choice()
        self.evt_dataset_select.emit(value)

    def _on_toggle_transformed(self, value: str) -> None:
        """Toggle visibility of transformed."""
        READER_CONFIG.show_transformed = value != "None"
        self.evt_show_transformed.emit(value)  # noqa
