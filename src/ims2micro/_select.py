"""Widget for loading IMS data."""
import typing as ty
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QCheckBox, QWidget
from superqt.utils import thread_worker
from loguru import logger

from ims2micro.config import CONFIG
from ims2micro.enums import ALLOWED_MICROSCOPY_FORMATS, ALLOWED_IMAGING_FORMATS, VIEW_TYPE_TRANSLATIONS

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
        self.model: ty.Optional[DataModel] = None

    def _setup_ui(self):
        """Setup UI."""
        self.text_edit = hp.make_line_edit(self, placeholder="Enter path to data file...")
        self.text_edit.editingFinished.connect(self.on_load_dataset)

        self.load_btn = hp.make_qta_btn(self, "open", func=self._on_select_dataset, small=True)
        self.close_btn = hp.make_qta_btn(self, "close", func=self._on_close_dataset, small=True)

        self.resolution_edit = hp.make_line_edit(self, placeholder="Enter spatial resolution...")
        self.resolution_edit.textChanged.connect(self._on_set_resolution)

        layout = hp.make_form_layout()
        layout.addRow(hp.make_label(self, self.INFO_TEXT, bold=True, wrap=True, alignment=Qt.AlignCenter))
        layout.addRow(
            hp.make_h_layout(self.text_edit, self.load_btn, self.close_btn, stretch_id=0),
        )
        layout.addRow(
            hp.make_label(self, "Resolution (μm)"),
            self.resolution_edit,
        )
        self.setLayout(layout)
        return layout

    def _on_close_dataset(self):
        """Close dataset."""
        self.evt_closed.emit(self.model)
        self.text_edit.setText("")
        self.resolution_edit.setText("1.0")
        self.model = None

    def on_set_path(self, path: str):
        """Set the path and immediately load it."""
        self.text_edit.setText(str(path))
        self.on_load_dataset()

    def on_load_dataset(self, evt=None):
        """Load data."""
        path = Path(self.text_edit.text())
        assert path.exists(), f"File does not exist: {path}"
        self._on_load_dataset()

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
            with hp.qt_signals_blocked(self.text_edit):
                self.text_edit.setText(str(path))
            self._on_load_dataset()

    def _on_load_dataset(self, evt=None):
        """Load data."""
        raise NotImplementedError("Must implement method")

    def _on_loaded_dataset(self, model: "DataModel"):
        """Finished loading data."""
        self.resolution_edit.setText(str(model.resolution))
        self.evt_loaded.emit(model)

    def _on_set_resolution(self, evt=None):
        """Specify resolution."""
        if self.model is None:
            return
        self.model.resolution = float(self.resolution_edit.text())


class IMSWidget(LoadWidget):
    """Widget for loading IMS data."""

    # class attrs
    IS_MICROSCOPY = False
    INFO_TEXT = "Select IMS dataset - supported formats: .imzML, .tdf/.tsf (Bruker), .data"
    FILE_TITLE = "Select IMS dataset..."
    FILE_FORMATS = ALLOWED_IMAGING_FORMATS
    FILENAME = r"D:\ims2micro_test\test.d\analysis.tsf"  # noqa
    FILENAME = (  # noqa
        r"D:\2023_02_17_Olof\VAN0052-RK-3\IMS\230119_isbergo_VAN0052_RK_3_2_05um_area3_neg_IMS.d\analysis.tsf"
    )
    FILENAME = (  # noqa
        r"D:\2023_02_17_Olof\VAN0052-RK-3\IMS\230119_isbergo_VAN0052_RK_3_2_05um_area2_neg_IMS.d\analysis.tsf"
    )

    # events
    evt_show_transformed = Signal(bool)
    evt_view_type = Signal(object)

    def _on_load_dataset(self, evt=None):
        """Load data."""
        from ims2micro.models import ImagingModel

        def _execute():
            self.evt_loading.emit()
            model = self.model
            func = thread_worker(model.load, start_thread=True, connect={"returned": self._on_loaded_dataset})
            func()
            logger.info(f"Started loading imaging data - '{model.path}'")

        path = Path(self.text_edit.text())
        if self.model is None or self.model.path != path:
            if self.model is not None:
                self._on_close_dataset()
            self.model = ImagingModel(path=path)
        _execute()

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
    FILENAME = r"D:\ims2micro_test\example.ome.tiff"
    FILENAME = r"D:\2023_02_17_Olof\VAN0052-RK-3\coregistration\Set1\230119_isbergo_VAN0052_RK_3_2_05um_area3_neg_coreg-postIMS-registered.ome.tiff"  # noqa
    FILENAME = r"D:\2023_02_17_Olof\VAN0052-RK-3\coregistration\Set1\230211_isbergo_VAN0052_RK_3_2_05um_area2_neg_coreg-postIMS-registered.ome.tiff"  # noqa

    # events
    evt_toggle_channel = Signal(str, bool)

    def __init__(self, parent=None):
        """Init."""
        super().__init__(parent=parent)
        self.evt_closed.connect(lambda model: self.__on_set_channels())

    def _on_load_dataset(self, evt=None):
        """Load data."""
        from ims2micro.models import MicroscopyModel

        def _execute():
            self.evt_loading.emit()
            model = self.model
            func = thread_worker(
                model.load,
                start_thread=True,
                connect={"returned": self._on_loaded_dataset},
            )
            func()
            logger.info(f"Started loading microscopy data - '{model.path}'")

        path = Path(self.text_edit.text())
        if self.model is None or self.model.path != path:
            if self.model is not None:
                self._on_close_dataset()
            self.model = MicroscopyModel(path=path)
        _execute()

    def _on_loaded_dataset(self, model: "DataModel"):
        """Finished loading data."""
        super()._on_loaded_dataset(model)
        self._on_set_channels(model)

    def _on_set_channels(self, model):
        """Specify channel names."""
        wrapper = model.get_reader()
        channel_names = wrapper.channel_names()
        self.__on_set_channels(channel_names)

    def __on_set_channels(self, channel_names: ty.Optional[ty.List[str]] = None):
        hp.clear_layout(self.checkbox_layout)
        if channel_names is None:
            channel_names = []
        for channel in channel_names:
            cb = QCheckBox(self)
            cb.setText(channel)
            cb.setChecked(True)
            cb.stateChanged.connect(partial(self._on_toggle_channel, name=channel))
            self.checkbox_layout.addWidget(cb)

    def _on_toggle_channel(self, state: bool, name: str):
        """Toggle channel."""
        self.evt_toggle_channel.emit(name, state)

    def _setup_ui(self):
        """Setup UI."""
        layout = super()._setup_ui()

        self.channels_widget = QWidget()
        self.checkbox_layout = hp.make_h_layout()

        channel_layout = hp.make_v_layout()
        channel_layout.addWidget(
            hp.make_label(
                self.channels_widget,
                "Channels (check to show/hide)",
                alignment=Qt.AlignCenter,
            )
        )
        channel_layout.addLayout(self.checkbox_layout)
        self.channels_widget.setLayout(channel_layout)
        layout.addRow(self.channels_widget)
        return layout
