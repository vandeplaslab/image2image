"""Widget for loading data."""
import typing as ty
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from koyo.typing import PathLike
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout, QWidget

from image2image._dialogs import (
    OverlayChannelsDialog,
    SelectImagesDialog,
    SelectTransformDialog,
)
from image2image.config import CONFIG
from image2image.enums import VIEW_TYPE_TRANSLATIONS
from image2image.models.data import DataModel
from image2image.models.transform import TransformModel
from image2image.utilities import style_form_layout


class LoadMixin(QWidget):
    """Load data mixin."""

    IS_FIXED: bool
    INFO_TEXT = "Select data..."

    def __init__(self, parent: ty.Optional[QWidget], view, n_max: int = 0):
        """Init."""
        super().__init__(parent=parent)
        self._setup_ui()
        self.view = view
        self.n_max = n_max
        self.model: DataModel = DataModel(is_fixed=self.IS_FIXED)
        self.dataset_dlg = SelectImagesDialog(self, self.model, self.IS_FIXED, self.n_max)

    def _setup_ui(self) -> QFormLayout:
        """Setup UI."""
        raise NotImplementedError("Must implement method")

    def on_set_path(
        self,
        paths: ty.Union[PathLike, ty.Sequence[PathLike]],
        affine: ty.Optional[ty.Dict[str, np.ndarray]] = None,
        resolution: ty.Optional[ty.Dict[str, float]] = None,
    ) -> None:
        """Set the path and immediately load it."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self.dataset_dlg._on_load_dataset(paths, affine, resolution)

    def on_select_dataset(self, _evt=None) -> None:
        """Load data."""
        self.dataset_dlg.on_select_dataset()

    def _on_add_dataset(self) -> None:
        """Select channels from the list."""
        self.dataset_dlg.show()


class LoadWidget(LoadMixin):
    """Widget for loading data."""

    evt_toggle_channel = Signal(str, bool)
    evt_toggle_all_channels = Signal(bool)

    IS_FIXED: bool = True

    def __init__(self, parent: ty.Optional[QWidget], view, n_max: int = 0):
        """Init."""
        super().__init__(parent, view, n_max)
        self.channel_dlg = OverlayChannelsDialog(self, self.model, self.view) if self.view else None
        if self.channel_dlg is None:
            self.select_btn.hide()

    def _setup_ui(self) -> QFormLayout:
        """Setup UI."""
        layout = hp.make_form_layout()
        style_form_layout(layout)
        self.info_text = hp.make_label(
            self,
            self.INFO_TEXT,
            bold=True,
            wrap=True,
            alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
        )
        layout.addRow(self.info_text)  # noqa
        layout.addRow(hp.make_btn(self, "Add/remove dataset...", func=self._on_add_dataset))

        self.select_btn = hp.make_btn(self, "Select channels...", func=self._on_select_channels)
        layout.addRow(self.select_btn)
        self.setLayout(layout)
        return layout

    def _on_select_channels(self) -> None:
        """Select channels from the list."""
        self.channel_dlg.show()


class MovingWidget(LoadWidget):
    """Widget for loading moving data."""

    # class attrs
    IS_FIXED = False
    INFO_TEXT = "Select 'moving' data..."

    # events
    evt_show_transformed = Signal(str)
    evt_view_type = Signal(object)

    def __init__(self, parent: ty.Optional[QWidget], view):
        super().__init__(parent, view)

        # extra events
        connect(self.dataset_dlg.evt_loaded, self._on_update_choice)
        connect(self.dataset_dlg.evt_closed, self._on_clear_choice)

    def _on_update_choice(self, _model, _channel_list) -> None:
        """Update list of available options."""
        hp.combobox_setter(self.transformed_choice, clear=True, items=["None", *self.model.channel_names()])
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _on_clear_choice(self, _model) -> None:
        """Clear list of available options."""
        hp.combobox_setter(self.transformed_choice, clear=True, items=["None"])
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _setup_ui(self) -> QFormLayout:
        """Setup UI."""
        layout = super()._setup_ui()

        self.view_type_choice = hp.make_combobox(
            self,
            data=VIEW_TYPE_TRANSLATIONS,
            value=str(CONFIG.view_type),
            func=self._on_update_view_type,
            tooltip="Select what kind of image should be displayed.<br><b>Overlay</b> will use the 'true' image and can"
            " be overlaid with other images.<br><b>Random</b> will display single image with random intensity.",
        )
        layout.addRow(hp.make_label(self, "View type"), self.view_type_choice)

        self.transformed_choice = hp.make_combobox(
            self,
            tooltip="Select which image should be displayed on the fixed modality.",
            func=self._on_toggle_transformed,
        )
        layout.addRow(
            hp.make_label(self, "Overlay"),
            self.transformed_choice,
        )
        return layout

    def _on_update_view_type(self, value: str):
        """Update view type."""
        CONFIG.view_type = value  # type: ignore
        self.evt_view_type.emit(value)  # noqa

    def _on_toggle_transformed(self, value: str) -> None:
        """Toggle visibility of transformed."""
        CONFIG.show_transformed = value != "None"
        self.evt_show_transformed.emit(value)  # noqa


class FixedWidget(LoadWidget):
    """Widget for loading fixed data."""

    # class attrs
    IS_FIXED = True
    INFO_TEXT = "Select 'fixed' data..."


class LoadWithTransformWidget(LoadMixin):
    """Widget for loading data."""

    IS_FIXED = True

    def __init__(self, parent: ty.Optional[QWidget], view):
        """Init."""
        super().__init__(parent, view)
        self.transform_model = TransformModel()
        self.transform_model.add_transform("Identity matrix", np.eye(3, dtype=np.float64))
        self.transform_dlg = SelectTransformDialog(self, self.model, self.transform_model, self.view)

    def _setup_ui(self) -> QFormLayout:
        """Setup UI."""
        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(
            hp.make_label(
                self,
                self.INFO_TEXT,
                bold=True,
                wrap=True,
                alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
            )
        )
        layout.addRow(hp.make_btn(self, "Add/remove dataset...", func=self._on_add_dataset))
        layout.addRow(hp.make_btn(self, "Select transformation...", func=self._on_select_transform))
        return layout

    def _on_select_transform(self) -> None:
        """Select transformation data."""
        self.transform_dlg.show()
