"""Launcher application for image2image."""

from __future__ import annotations

import typing as ty

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.system import IS_MAC_ARM, IS_PYINSTALLER
from qtextra.config import THEMES
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_logger import QtLoggerDialog
from qtextra.widgets.qt_tile import QtTileWidget, Tile
from qtpy.QtWidgets import QGridLayout, QVBoxLayout, QWidget

from image2image import __version__
from image2image.config import CONFIG
from image2image.utils._appdirs import USER_LOG_DIR

# to add apps: volume viewer, sync viewer,
REGISTER_TEXT = "Co-register your microscopy and imaging mass spectrometry data."
VIEWER_TEXT = "Overlay your microscopy and imaging mass spectrometry data."
CROP_TEXT = "Crop your microscopy data to reduce it's size (handy for Image Fusion)."
CONVERT_TEXT = "Convert multi-scene CZI images or other formats to OME-TIFF."
MERGE_TEXT = "Merge multiple OME-TIFF images into a single file."
FUSION_TEXT = "Export your data for Image Fusion in MATLAB compatible format."
WSIREG_TEXT = "Register whole slide microscopy images."
CONVERT_UNAVAILABLE = IS_PYINSTALLER and IS_MAC_ARM
if CONVERT_UNAVAILABLE:
    CONVERT_TEXT += "<br><br><i>Not available on Apple Silicon due to a bug I can't find...</i>"


def _make_tile(parent: QWidget, title: str, description: str, icon: str, func: ty.Callable, **icon_kws) -> QtTileWidget:
    """Make tile."""
    return QtTileWidget(parent, Tile(title=title, description=description, icon=icon, func=func, icon_kws=icon_kws))


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

        tile_layout = QGridLayout()
        tile_layout.setSpacing(2)
        tile_layout.setColumnStretch(0, 1)
        tile_layout.setColumnStretch(4, 1)
        # First row
        # register app
        register = _make_tile(self, "Registration<br>App", REGISTER_TEXT, "register", Window.on_open_register)
        tile_layout.addWidget(register, 0, 1)
        # viewer app
        viewer = _make_tile(self, "Viewer<br>App", VIEWER_TEXT, "viewer", Window.on_open_viewer)
        tile_layout.addWidget(viewer, 0, 2)
        crop = _make_tile(self, "WsiReg<br>App", WSIREG_TEXT, "wsireg", Window.on_open_wsireg)
        tile_layout.addWidget(crop, 0, 3)
        # Second row
        # crop app
        crop = _make_tile(
            self, "Image Crop<br>App", CROP_TEXT, "crop", Window.on_open_crop, icon_kws={"color": "#ff0000"}
        )
        tile_layout.addWidget(crop, 1, 1)
        # convert app
        convert = _make_tile(
            self,
            "Image to OME-TIFF<br>App",
            CONVERT_TEXT,
            "change",
            Window.on_open_convert
            if not CONVERT_UNAVAILABLE
            else lambda: hp.warn_pretty(self, "Not available on Apple Silicon."),
            icon_kws=dict(color=THEMES.get_hex_color("warning")) if CONVERT_UNAVAILABLE else None,
        )
        tile_layout.addWidget(convert, 1, 2)
        # merge app
        merge = _make_tile(self, "Merge OME-TIFFs<br>App", MERGE_TEXT, "merge", Window.on_open_merge)
        tile_layout.addWidget(merge, 1, 3)
        # Third row
        # export app
        export = _make_tile(self, "Fusion Preparation<br>App", FUSION_TEXT, "fusion", Window.on_open_fusion)
        tile_layout.addWidget(export, 2, 2)

        main_layout = QVBoxLayout()
        main_layout.addLayout(tile_layout, stretch=1)
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
            self.console.push_variables({"window": self, "CONFIG": CONFIG, "READER_CONFIG": READER_CONFIG})
        self.console.show()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(tool="launcher", level=0)
