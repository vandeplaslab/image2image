"""Widget for loading IMS data."""
from qtpy.QtWidgets import QWidget
import qtextra.helpers as hp


class LoadWidget(QWidget):
    """Widget for loading IMS data."""

    INFO_TEXT = ""

    def __init__(self, parent=None):
        """Init."""
        super().__init__(parent=parent)
        self._setup_ui()

    def _setup_ui(self):
        """Setup UI."""
        label = hp.make_label(self, self.INFO_TEXT)
        self.text_edit = hp.make_line_edit(self, placeholder="Enter path to data file...")
        self.text_edit.textChanged.connect(self._on_load_data)
        self.load_btn = hp.make_qta_btn(self, "open", func=self._on_load_data)

        self.resolution_edit = hp.make_line_edit(self, placeholder="Enter spatial resolution...")
        self.resolution_edit.textChanged.connect(self._on_set_resolution)

        layout = hp.make_v_layout(
            label,
            hp.make_h_layout(
                self.text_edit,
                self.load_btn,
                stretch_id=0,
            ),
            hp.make_h_layout(hp.make_label(self, "Spatial resolution (um)"), self.resolution_edit, stretch_id=1),
            stretch_id=0,
        )
        self.setLayout(layout)

    def _on_load_data(self, evt=None):
        """Load data."""
        raise NotImplementedError("Must implement method")

    def _on_set_resolution(self, evt=None):
        """Specify resolution."""


class IMSWidget(LoadWidget):
    """Widget for loading IMS data."""

    INFO_TEXT = "Select IMS dataset - supported formats: .imzML, .d (Bruker), .data (ionglow)."

    def _on_load_data(self, evt=None):
        """Load data."""


class MicroscopyWidget(LoadWidget):
    """Widget for loading Microscopy data."""

    INFO_TEXT = "Select microscopy data - supported formats: .tiff, .ome.tiff, .czi"

    def _on_load_data(self, evt=None):
        """Load data."""
