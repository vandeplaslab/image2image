"""Widget for loading IMS data."""
from functools import partial
from pathlib import Path

from qtpy.QtWidgets import QWidget, QCheckBox
from qtpy.QtCore import Signal
import qtextra.helpers as hp
import typing as ty
from superqt.utils import thread_worker
from ims2micro.utilities import sanitize_path

if ty.TYPE_CHECKING:
    from ims2micro.models import DataModel


class LoadWidget(QWidget):
    """Widget for loading IMS data."""

    evt_loading = Signal()
    evt_loaded = Signal(object)
    evt_closed = Signal()

    INFO_TEXT: str = ""
    FILE_TITLE = "Select data file"
    FILE_FORMATS: str

    def __init__(self, parent=None):
        """Init."""
        super().__init__(parent=parent)
        self._setup_ui()
        self.model: ty.Optional[DataModel] = None

    def _setup_ui(self):
        """Setup UI."""
        self.text_edit = hp.make_line_edit(self, placeholder="Enter path to data file...")
        self.text_edit.editingFinished.connect(self._on_load_set_data)

        self.load_btn = hp.make_qta_btn(self, "open", func=self._on_load_data)
        self.close_btn = hp.make_qta_btn(self, "close", func=self._on_close_data)

        self.resolution_edit = hp.make_line_edit(self, placeholder="Enter spatial resolution...")
        self.resolution_edit.textChanged.connect(self._on_set_resolution)

        layout = hp.make_v_layout(
            hp.make_label(self, self.INFO_TEXT),
            hp.make_h_layout(
                self.text_edit,
                self.load_btn,
                self.close_btn,
                stretch_id=0,
            ),
            hp.make_h_layout(hp.make_label(self, "Spatial resolution (um)"), self.resolution_edit, stretch_id=1),
        )
        self.setLayout(layout)
        return layout

    def _on_close_data(self):
        """Close dataset."""
        self.evt_closed.emit()

    def _on_load_set_data(self, evt=None):
        """Load data."""
        path = Path(self.text_edit.text())
        assert path.exists(), f"File does not exist: {path}"

        self._on_read_data()

    def _on_load_data(self, evt=None):
        """Load data."""
        path = hp.get_filename(self, title=self.FILE_TITLE, base_dir="", file_filter=self.FILE_FORMATS)
        # path = str(sanitize_path(path))
        if path:
            with hp.qt_signals_blocked(self.text_edit):
                self.text_edit.setText(str(path))
            self._on_read_data()

    def _on_read_data(self, evt=None):
        """Load data."""
        raise NotImplementedError("Must implement method")

    def _on_set_resolution(self, evt=None):
        """Specify resolution."""
        if self.model is None:
            return
        self.model.resolution = float(self.resolution_edit.text())


class IMSWidget(LoadWidget):
    """Widget for loading IMS data."""

    INFO_TEXT = "Select IMS dataset - supported formats: .imzML, .d (Bruker), .data (ionglow)."
    FILE_TITLE = "Select IMS dataset..."
    FILE_FORMATS = "Bruker QTOF (*.tsf); Bruker IMS-QTOF (*.tdf); imzML (*.imzML); ionglow (*.metadata.h5);"

    def _on_load_data(self, evt=None):
        """Load data."""
        # path = hp.get_filename(self, title=self.FILE_TITLE, base_dir="", file_filter=self.FILE_FORMATS)
        path = r"D:\ims2micro_test\test.d\analysis.tsf"
        if path:
            with hp.qt_signals_blocked(self.text_edit):
                self.text_edit.setText(str(path))
            self._on_read_data()

    def _on_read_data(self, evt=None):
        """Load data."""
        from ims2micro.models import ImagingModel

        def _execute():
            model = self.model
            func = thread_worker(model.load, start_thread=True, connect={"returned": self.evt_loaded.emit})
            func()  # noqa

        path = Path(self.text_edit.text())
        if self.model is None or self.model.path != path:
            self.model = None
            self.model = ImagingModel(path=path)
        _execute()


class MicroscopyWidget(LoadWidget):
    """Widget for loading Microscopy data."""

    INFO_TEXT = "Select microscopy data - supported formats: .tiff, .ome.tiff, .czi"
    FILE_TITLE = "Select microscopy dataset..."
    FILE_FORMATS = "CZI (*.czi); OME-TIFF (*.ome.tiff); TIFF (*.tiff); JPEG (*.jpg); PNG (*.png);"

    evt_toggle_channel = Signal(str, bool)

    def _on_read_data(self, evt=None):
        """Load data."""
        from ims2micro.models import MicroscopyModel

        def _execute():
            model = self.model
            func = thread_worker(
                model.load, start_thread=True, connect={"returned": [self.evt_loaded.emit, self._on_set_channels]}
            )
            func()  # noqa

        path = Path(self.text_edit.text())
        if self.model is None or self.model.path != path:
            self.model = None
            self.model = MicroscopyModel(path=path)
        _execute()

    def _on_load_data(self, evt=None):
        """Load data."""
        # path = hp.get_filename(self, title=self.FILE_TITLE, base_dir="", file_filter=self.FILE_FORMATS)
        path = r"D:\ims2micro_test\example.ome.tiff"
        path = str(sanitize_path(path))
        if path:
            with hp.qt_signals_blocked(self.text_edit):
                self.text_edit.setText(path)
            self._on_read_data()

    def _on_set_channels(self, model):
        """Specify channel names."""
        wrapper = model.get_reader()
        channel_names = wrapper.channel_names()
        hp.clear_layout(self.checkbox_layout)
        for channel in channel_names:
            cb = QCheckBox(self)
            cb.setText(channel)
            cb.setChecked(True)
            self.checkbox_layout.addWidget(cb)
            cb.stateChanged.connect(partial(self._on_toggle_channel, name=channel))

    def _on_toggle_channel(self, state: bool, name: str):
        """Toggle channel."""
        self.evt_toggle_channel.emit(name, state)
        print("toggling")

    def _setup_ui(self):
        """Setup UI."""
        layout = super()._setup_ui()

        self.channels_widget = QWidget()
        self.checkbox_layout = hp.make_h_layout()

        channel_layout = hp.make_v_layout()
        channel_layout.addWidget(hp.make_label(self.channels_widget, "Channels (check to show/hide):"))
        channel_layout.addLayout(self.checkbox_layout)
        self.channels_widget.setLayout(channel_layout)
        layout.addWidget(self.channels_widget)
        return layout
