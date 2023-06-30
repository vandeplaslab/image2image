"""Widget for loading data."""
from functools import partial

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
from ims2micro.utilities import log_exception, style_form_layout


class LoadWidget(QWidget):
    """Widget for loading data."""

    evt_loading = Signal()
    evt_loaded = Signal(object, object)
    evt_closed = Signal(object)
    evt_toggle_channel = Signal(str, bool)
    evt_toggle_all_channels = Signal(bool)

    IS_FIXED: bool
    INFO_TEXT = "Select 'fixed' data..."
    FILE_TITLE = "Select 'fixed' data..."
    FILE_FORMATS = ALLOWED_FORMATS

    def __init__(self, parent, view):
        """Init."""
        super().__init__(parent=parent)
        self._setup_ui()
        self.view = view
        self.model = DataModel(is_fixed=self.IS_FIXED)
        self.table_dlg = OverlayTableDialog(self, self.model, self.view)

    def _setup_ui(self):
        """Setup UI."""
        self.load_btn = hp.make_btn(self, "Add image...", func=self.on_select_dataset)
        self.close_btn = hp.make_btn(self, "Remove image...", func=self._on_close_dataset)

        self.resolution_edit = hp.make_line_edit(self, placeholder="Enter spatial resolution...", text="1.0")
        self.resolution_edit.setValidator(QRegExpValidator(QRegExp(r"^[0-9]+(\.[0-9]+)?$")))  # noqa
        self.resolution_edit.textChanged.connect(self._on_set_resolution)  # noqa
        self.channel_choice = hp.make_btn(self, "Select channels...", func=self._on_select_channels)
        self.extract_btn = hp.make_qta_btn(self, "add", func=self._on_extract_channels, normal=True)

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(hp.make_label(self, self.INFO_TEXT, bold=True, wrap=True, alignment=Qt.AlignCenter))
        layout.addRow(hp.make_h_layout(self.load_btn, self.close_btn))
        layout.addRow(hp.make_h_layout(self.channel_choice, self.extract_btn, stretch_id=0))
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
            if paths:
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
            base_dir=CONFIG.fixed_dir if self.IS_FIXED else CONFIG.moving_dir,
            file_filter=self.FILE_FORMATS,
        )
        # path = self.FILENAME
        if path:
            if self.IS_FIXED:
                CONFIG.fixed_dir = str(Path(path).parent)
            else:
                CONFIG.moving_dir = str(Path(path).parent)
            self._on_load_dataset([path])

    def _on_load_dataset(self, paths: ty.List[PathLike]):
        """Load data."""
        self.evt_loading.emit()
        model = self.model
        model.add_paths(paths)
        func = thread_worker(
            model.load,
            start_thread=True,
            connect={"returned": self._on_loaded_dataset, "errored": self._on_failed_dataset},
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

    def _on_failed_dataset(self, exception: Exception):
        """Failed to load dataset."""
        logger.error("Error occurred while loading dataset.")
        log_exception(exception)
        self.evt_loaded.emit(None, None)

    def _on_set_resolution(self, _evt=None):
        """Specify resolution."""
        self.model.resolution = float(self.resolution_edit.text())

    def _on_select_channels(self):
        """Select channels from the list."""
        self.table_dlg.show()

    def _on_extract_channels(self):
        """Extract channels from the list."""
        from ims2micro._dialogs import ExtractChannelsDialog

        if not self.model.get_extractable_paths():
            logger.warning("No paths to extract data from.")
            return

        dlg = ExtractChannelsDialog(self, self.model)
        if dlg.exec_():
            path = dlg.path_to_extract
            mzs = dlg.mzs
            ppm = dlg.ppm
            if path and mzs:
                reader = self.model.get_reader(path)

                func = thread_worker(  # noqa
                    partial(reader.extract, mzs=mzs, ppm=ppm),
                    start_thread=True,
                    connect={"returned": self._on_update_dataset, "errored": self._on_failed_update_dataset},
                )
                func()

    def _on_update_dataset(self, result: ty.Tuple[Path, ty.List[str]]):
        """Finished loading data."""
        path, channel_list = result
        # load data into an image
        self.evt_loaded.emit(self.model, channel_list)

    @staticmethod
    def _on_failed_update_dataset(exception: Exception):
        """Failed to load dataset."""
        logger.error("Error occurred while extracting images.", exception)
        log_exception(exception)


class MovingWidget(LoadWidget):
    """Widget for loading moving data."""

    # class attrs
    IS_FIXED = False
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
        CONFIG.view_type = value
        self.evt_view_type.emit(value)

    def _on_toggle_transformed(self, value: str):
        """Toggle visibility of transformed."""
        CONFIG.show_transformed = value != "None"
        self.evt_show_transformed.emit(value)


class FixedWidget(LoadWidget):
    """Widget for loading fixed data."""

    # class attrs
    IS_FIXED = True
