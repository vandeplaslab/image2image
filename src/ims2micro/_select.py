"""Widget for loading IMS data."""
import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from koyo.typing import PathLike
from loguru import logger
from qtextra.utils.utilities import connect
from qtpy.QtCore import QRegExp, Qt, Signal
from qtpy.QtGui import QRegExpValidator
from qtpy.QtWidgets import QWidget
from superqt.utils import thread_worker

from ims2micro._dialogs import OverlayTableDialog, SelectChannelsTableDialog
from ims2micro.config import CONFIG
from ims2micro.enums import ALLOWED_FORMATS, VIEW_TYPE_TRANSLATIONS
from ims2micro.models import DataModel
from ims2micro.utilities import style_form_layout


class LoadWidget(QWidget):
    """Widget for loading IMS data."""

    evt_loading = Signal()
    evt_loaded = Signal(object, object)
    evt_closed = Signal(object)
    evt_toggle_channel = Signal(str, bool)

    IS_MICROSCOPY: bool
    INFO_TEXT = "Select 'fixed' data..."
    FILE_TITLE = "Select 'fixed' data..."
    FILE_FORMATS = ALLOWED_FORMATS

    def __init__(self, parent, view):
        """Init."""
        super().__init__(parent=parent)
        self._setup_ui()
        self.view = view
        self.model = DataModel()
        self.table_dlg = OverlayTableDialog(self, self.model, self.view)

    def _setup_ui(self):
        """Setup UI."""
        self.load_btn = hp.make_btn(self, "Add image...", func=self.on_select_dataset)
        self.close_btn = hp.make_btn(self, "Remove image...", func=self._on_close_dataset)

        self.resolution_edit = hp.make_line_edit(self, placeholder="Enter spatial resolution...", text="1.0")
        self.resolution_edit.setValidator(QRegExpValidator(QRegExp(r"^[0-9]+(\.[0-9]+)?$")))  # noqa
        self.resolution_edit.textChanged.connect(self._on_set_resolution)  # noqa
        self.channel_choice = hp.make_btn(self, "Select channels...", func=self._on_select_channels)

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(hp.make_label(self, self.INFO_TEXT, bold=True, wrap=True, alignment=Qt.AlignCenter))
        layout.addRow(hp.make_h_layout(self.load_btn, self.close_btn))
        layout.addRow(self.channel_choice)
        layout.addRow(hp.make_label(self, "Pixel size (μm)"), self.resolution_edit)
        self.setLayout(layout)
        return layout

    def _on_close_dataset(self, force: bool = False) -> bool:
        """Close dataset."""
        from ims2micro._dialogs import CloseDatasetDialog

        if self.model.n_paths:
            paths = None
            if not force:  # only ask user if not forced
                dlg = CloseDatasetDialog(self, self.model)
                if dlg.exec_():
                    paths = dlg.paths
            else:
                paths = self.model.paths
            self.model.remove_paths(paths)
            self.evt_closed.emit(self.model)
            if not self.model.n_paths:
                self.resolution_edit.setText("1.0")
            return True
        return False

    def on_set_path(self, paths: ty.Union[PathLike, ty.List[PathLike]]):
        """Set the path and immediately load it."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self._on_load_dataset(paths)

    def on_select_dataset(self, _evt=None):
        """Load data."""
        path = hp.get_filename(
            self,
            title=self.FILE_TITLE,
            base_dir=CONFIG.microscopy_dir if self.IS_MICROSCOPY else CONFIG.imaging_dir,
            file_filter=self.FILE_FORMATS,
        )
        # path = self.FILENAME
        if path:
            if self.IS_MICROSCOPY:
                CONFIG.microscopy_dir = str(Path(path).parent)
            else:
                CONFIG.imaging_dir = str(Path(path).parent)
            self._on_load_dataset([path])

    def _on_load_dataset(self, paths: ty.List[PathLike]):
        """Load data."""
        self.evt_loading.emit()
        model = self.model
        model.add_paths(paths)
        func = thread_worker(
            model.load,
            start_thread=True,
            connect={"returned": self._on_loaded_dataset, "errored": lambda: self.evt_loaded.emit(None, None)},
        )
        func()
        logger.info(f"Started loading dataset - '{model.paths}'")

    def _on_loaded_dataset(self, model: "DataModel"):
        """Finished loading data."""
        # setup resolution
        self.resolution_edit.setText(str(model.resolution))
        # select what should be loaded
        dlg = SelectChannelsTableDialog(self, model)
        channel_list = []
        if dlg.exec_():
            channel_list = dlg.channels
        logger.info(f"Selected channels: {channel_list}")
        if not channel_list:
            model.remove_paths(model.just_added)
            model, channel_list = None, None
            logger.warning("No channels selected - dataset not loaded")

        # load data into an image
        self.evt_loaded.emit(model, channel_list)

    def _on_set_resolution(self, _evt=None):
        """Specify resolution."""
        self.model.resolution = float(self.resolution_edit.text())

    def _on_select_channels(self):
        """Select channels from the list."""
        self.table_dlg.show()


class MovingWidget(LoadWidget):
    """Widget for loading IMS data."""

    # class attrs
    IS_MICROSCOPY = False
    INFO_TEXT = "Select 'moving' data..."
    FILE_TITLE = "Select 'moving' data..."

    # events
    evt_show_transformed = Signal(str)
    evt_view_type = Signal(object)

    def __init__(self, parent, view):
        super().__init__(parent, view)

        # extra events
        connect(self.evt_loaded, self._on_update_choice)
        connect(self.evt_closed, self._on_clear_choice)

    def _on_update_choice(self, _model, _channel_list):
        """Update list of available options."""
        hp.combobox_setter(self.transformed_choice, clear=True, items=["None", *self.model.channel_names()])
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _on_clear_choice(self, _model):
        """Clear list of available options."""
        hp.combobox_setter(self.transformed_choice, clear=True, items=["None"])
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _setup_ui(self):
        """Setup UI."""
        layout = super()._setup_ui()

        self.view_type_choice = hp.make_combobox(
            self,
            data=VIEW_TYPE_TRANSLATIONS,
            value=str(CONFIG.view_type),
            func=self._on_update_view_type,
        )
        layout.addRow(hp.make_label(self, "View type"), self.view_type_choice)

        self.transformed_choice = hp.make_combobox(
            self,
            tooltip="Select which image should be displayed on the microscopy modality.",
            func=self._on_toggle_transformed,
        )
        layout.addRow(
            hp.make_label(self, "Overlay"),
            self.transformed_choice,
        )
        return layout

    def _on_update_view_type(self, value: str):
        """Update view type."""
        CONFIG.view_type = value
        self.evt_view_type.emit(value)

    def _on_toggle_transformed(self, value: str):
        """Toggle visibility of transformed."""
        CONFIG.show_transformed = value != "None"
        self.evt_show_transformed.emit(value)


class FixedWidget(LoadWidget):
    """Widget for loading Microscopy data."""

    # class attrs
    IS_MICROSCOPY = True
