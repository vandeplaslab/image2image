"""Launcher application for image2image."""

from __future__ import annotations

import typing as ty

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.system import IS_MAC_ARM, IS_PYINSTALLER
from koyo.utilities import is_installed
from qtextra.config import THEMES
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_logger import QtLoggerDialog
from qtextra.widgets.qt_tile import QtTileWidget, Tile
from qtpy.QtWidgets import QVBoxLayout, QWidget

from image2image import __version__
from image2image.config import APP_CONFIG
from image2image.utils._appdirs import USER_LOG_DIR

# to add apps: volume viewer, sync viewer,
REGISTER_TEXT = "Co-register your microscopy and imaging mass spectrometry data."
VIEWER_TEXT = "Overlay your microscopy and imaging mass spectrometry data."
CROP_TEXT = "Crop your microscopy data to reduce it's size (handy for Image Fusion)."
CONVERT_TEXT = "Convert multi-scene CZI images or other formats to OME-TIFF."
MERGE_TEXT = "Merge multiple OME-TIFF images into a single file."
FUSION_TEXT = "Export your data for Image Fusion in MATLAB compatible format."
WSIREG_TEXT = "Register whole slide microscopy images<br>(<b>i2reg</b>)."
VALIS_TEXT = "Register whole slide microscopy images<br>(<b>Valis</b>)."
HAS_CONVERT = not IS_PYINSTALLER and not IS_MAC_ARM
CONVERT_WARNING = ""
if HAS_CONVERT:
    CONVERT_WARNING = "<i>Not available on Apple Silicon due to a bug I can't find...</i>"
HAS_VALIS = is_installed("valis") and is_installed("pyvips")
VALIS_WARNING = ""
if not HAS_VALIS:
    VALIS_WARNING = "<br><br><i>Valis or pyvips is not installed.</i>"


def _make_tile(
    parent: QWidget,
    title: str,
    description: str,
    icon: str,
    func: ty.Callable,
    warning: str = "",
    icon_kws: dict | None = None,
) -> QtTileWidget:
    """Make tile."""
    return QtTileWidget(
        parent, Tile(title=title, description=description, icon=icon, func=func, warning=warning, icon_kws=icon_kws)
    )


class Launcher(QtDialog):
    """General launcher application."""

    def __init__(self, parent=None):
        super().__init__(parent, title=f"image2image Launcher (v{__version__})")
        self.console = None
        self.logger = QtLoggerDialog(self, USER_LOG_DIR)
        self.setFixedSize(self.sizeHint())

    def make_panel(self) -> QVBoxLayout:
        """Make panel."""
        from image2image.qt.dialog_base import Window

        layout = hp.make_form_layout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addRow(hp.make_h_line_with_text("Viewers", self, position="left", bold=True))
        # viewer apps
        tile = _make_tile(self, "Viewer<br>App", VIEWER_TEXT, "viewer", Window.on_open_viewer)
        layout.addRow(hp.make_h_layout(tile, stretch_after=True, spacing=2, margin=2))

        # register apps
        layout.addRow(hp.make_h_line_with_text("Registration", self, position="left", bold=True))
        tile_reg = _make_tile(self, "Registration<br>App", REGISTER_TEXT, "register", Window.on_open_register)
        tile_wsireg = _make_tile(self, "WsiReg<br>App", WSIREG_TEXT, "wsireg", Window.on_open_wsireg)
        tile_valis = _make_tile(
            self,
            "Valis<br>App",
            VALIS_TEXT,
            "valis",
            Window.on_open_valis,
            # if HAS_VALIS
            # else lambda: hp.warn_pretty(self, "Valis or pyvips is not installed on this machine."),
            icon_kws=None if HAS_VALIS else {"color": THEMES.get_hex_color("warning")},
            warning=VALIS_WARNING,
        )
        layout.addRow(hp.make_h_layout(tile_reg, tile_wsireg, tile_valis, stretch_after=True, spacing=2, margin=2))

        # utility apps
        layout.addRow(hp.make_h_line_with_text("Utilities", self, position="left", bold=True))
        tile_crop = _make_tile(self, "Image Crop<br>App", CROP_TEXT, "crop", Window.on_open_crop)
        tile_convert = _make_tile(
            self,
            "Image to OME-TIFF<br>App",
            CONVERT_TEXT,
            "convert",
            Window.on_open_convert if HAS_CONVERT else lambda: hp.warn_pretty(self, "Not available on Apple Silicon."),
            icon_kws=None if HAS_CONVERT else {"color": THEMES.get_hex_color("warning")},
            warning=CONVERT_WARNING,
        )
        tile_merge = _make_tile(self, "Merge OME-TIFFs<br>App", MERGE_TEXT, "merge", Window.on_open_merge)
        tile_fusion = _make_tile(self, "Fusion Preparation<br>App", FUSION_TEXT, "fusion", Window.on_open_fusion)
        layout.addRow(
            hp.make_h_layout(tile_crop, tile_convert, tile_merge, tile_fusion, stretch_after=True, spacing=2, margin=2)
        )

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout, stretch=1)
        main_layout.addWidget(hp.make_h_line())
        main_layout.addWidget(hp.make_btn(self, "Show logger...", func=self.on_show_logger))
        main_layout.addWidget(hp.make_btn(self, "Show IPython console...", func=self.on_show_console))
        main_layout.addStretch(1)
        return main_layout

    def on_show_logger(self) -> None:
        """View console."""
        self.logger.show()

    def on_show_console(self) -> None:
        """View console."""
        if self.console is None:
            from qtextra.dialogs.qt_console import QtConsoleDialog

            self.console = QtConsoleDialog(self)
            self.console.push_variables({"window": self, "APP_CONFIG": APP_CONFIG, "READER_CONFIG": READER_CONFIG})
        self.console.show()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(tool="launcher", level=0, dev=True)
