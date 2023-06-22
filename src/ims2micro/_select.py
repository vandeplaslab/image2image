"""Widget for loading IMS data."""
import typing as ty
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from koyo.typing import PathLike
from loguru import logger
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QCheckBox, QWidget
from superqt.utils import thread_worker

from ims2micro.config import CONFIG
from ims2micro.enums import ALLOWED_IMAGING_FORMATS, ALLOWED_MICROSCOPY_FORMATS, VIEW_TYPE_TRANSLATIONS
from ims2micro.models import ImagingModel, MicroscopyModel

if ty.TYPE_CHECKING:
    from ims2micro.models import DataModel


class LoadWidget(QWidget):
    """Widget for loading IMS data."""

    evt_loading = Signal()
    evt_loaded = Signal(object)
    evt_closed = Signal(object)

    IS_MICROSCOPY: bool
    INFO_TEXT: str = ""
    FILE_TITLE = "Select data file"
    FILE_FORMATS: str

    def __init__(self, parent=None):
        """Init."""
        super().__init__(parent=parent)
        self._setup_ui()
        self.model: ty.Optional[DataModel] = MicroscopyModel() if self.IS_MICROSCOPY else ImagingModel()

    def _setup_ui(self):
        """Setup UI."""
        self.info_label = hp.make_label(self, "")
        self.load_btn = hp.make_qta_btn(self, "open", func=self._on_select_dataset, small=True)
        self.close_btn = hp.make_qta_btn(self, "close", func=self._on_close_dataset, small=True)

        self.resolution_edit = hp.make_line_edit(self, placeholder="Enter spatial resolution...")
        self.resolution_edit.textChanged.connect(self._on_set_resolution)
        self.channel_choice = hp.make_btn(self, "Select channels...", func=self._on_select_channels)

        layout = hp.make_form_layout()
        layout.addRow(hp.make_label(self, self.INFO_TEXT, bold=True, wrap=True, alignment=Qt.AlignCenter))
        layout.addRow(hp.make_h_layout(self.info_label, self.load_btn, self.close_btn, stretch_id=0))
        layout.addRow(hp.make_label(self, "Resolution (μm)"), self.resolution_edit)
        layout.addRow(self.channel_choice)
        self.setLayout(layout)
        return layout

    def _on_close_dataset(self):
        """Close dataset."""
        self.evt_closed.emit(self.model)
        self.info_label.setText("")
        self.resolution_edit.setText("1.0")

    def on_set_path(self, paths: ty.Union[PathLike, ty.List[PathLike]]):
        """Set the path and immediately load it."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self._on_load_dataset(paths)

    def _on_select_dataset(self, evt=None):
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
            connect={"returned": self._on_loaded_dataset, "errored": lambda: self.evt_loaded.emit(None)},
        )
        func()
        logger.info(f"Started loading dataset - '{model.paths}'")

    def _on_loaded_dataset(self, model: "DataModel"):
        """Finished loading data."""
        self.resolution_edit.setText(str(model.resolution))
        self.info_label.setText(f"Loaded '{model.n_paths}' paths")
        self.evt_loaded.emit(model)

    def _on_set_resolution(self, _evt=None):
        """Specify resolution."""
        self.model.resolution = float(self.resolution_edit.text())

    def _on_select_channels(self):
        """Select channels from list."""
        print("Select channels...")


class IMSWidget(LoadWidget):
    """Widget for loading IMS data."""

    # class attrs
    IS_MICROSCOPY = False
    INFO_TEXT = "Select IMS dataset - supported formats: .imzML, .tdf/.tsf (Bruker), .data, .npy"
    FILE_TITLE = "Select IMS dataset..."
    FILE_FORMATS = ALLOWED_IMAGING_FORMATS

    # events
    evt_show_transformed = Signal(bool)
    evt_view_type = Signal(object)

    def _setup_ui(self):
        """Setup UI."""
        layout = super()._setup_ui()

        self.view_type_choice = hp.make_combobox(
            self, data=VIEW_TYPE_TRANSLATIONS, value="Random", func=self._on_update_view_type
        )
        layout.addRow(hp.make_label(self, "View type"), self.view_type_choice)
        layout.addRow(
            hp.make_label(self, "Show transformed"),
            hp.make_checkbox(self, func=self._on_toggle_transformed, value=True),
        )
        return layout

    def _on_update_view_type(self, value: str):
        """Update view type."""
        CONFIG.view_type = value
        self.evt_view_type.emit(value)

    def _on_toggle_transformed(self, state: bool):
        """Toggle visibility of transformed."""
        CONFIG.show_transformed = state
        self.evt_show_transformed.emit(state)


class MicroscopyWidget(LoadWidget):
    """Widget for loading Microscopy data."""

    # class attrs
    IS_MICROSCOPY = True
    INFO_TEXT = "Select microscopy data - supported formats: .tiff, .czi, .jpg, .png"
    FILE_TITLE = "Select microscopy dataset..."
    FILE_FORMATS = ALLOWED_MICROSCOPY_FORMATS

    # events
    evt_toggle_channel = Signal(str, bool)

    def __init__(self, parent=None):
        """Init."""
        super().__init__(parent=parent)

    def _on_loaded_dataset(self, model: "DataModel"):
        """Finished loading data."""
        super()._on_loaded_dataset(model)

    def _on_set_channels(self, model: "DataModel"):
        """Specify channel names."""
        wrapper = model.get_reader()
        wrapper.channel_names()
        # self.__on_set_channels(channel_names)

    def _on_toggle_channel(self, state: bool, name: str):
        """Toggle channel."""
        self.evt_toggle_channel.emit(name, state)
