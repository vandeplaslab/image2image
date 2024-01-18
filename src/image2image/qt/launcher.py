"""Launcher application for image2image."""
from __future__ import annotations

import qtextra.helpers as hp
from loguru import logger
from qtextra.config import THEMES
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_logger import QtLoggerDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout

from image2image.utils._appdirs import USER_LOG_DIR

REGISTER_TEXT = "<b>Registration App</b><br>Co-register your microscopy and imaging mass spectrometry data."
VIEWER_TEXT = "<b>Viewer App</b><br>Overlay your microscopy and imaging mass spectrometry data."
CROP_TEXT = "<b>Crop App</b><br>Crop your microscopy data to reduce it's size (handy for Image Fusion)."
CONVERT_TEXT = "<b>CZI to OME-TIFF App</b><br>Convert your multi-scene CZI image to OME-TIFF."
EXPORT_TEXT = "<b>Fusion Preparation App</b><br>Export your data for Image Fusion in MATLAB compatible format."
# to add apps: volume viewer, sync viewer,


class Launcher(QtDialog):
    """General launcher application."""

    def __init__(self, parent=None):
        super().__init__(parent, title="image2image Launcher")
        self.console = None
        self.logger = QtLoggerDialog(self, USER_LOG_DIR)

    def make_panel(self) -> QVBoxLayout:
        """Make panel."""
        layout = QVBoxLayout()
        # register app
        btn = hp.make_qta_btn(self, "register", tooltip="Open registration application.", func=self.on_register)
        btn.set_xxlarge()
        layout.addLayout(
            hp.make_h_layout(
                btn,
                hp.make_label(
                    self,
                    REGISTER_TEXT,
                    alignment=Qt.AlignHCenter,  # type: ignore[attr-defined]
                    wrap=True,
                    enable_url=True,
                ),
                stretch_id=1,
                alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
            ),
        )
        # viewer app
        btn = hp.make_qta_btn(self, "viewer", tooltip="Open viewer application.", func=self.on_viewer)
        btn.set_xxlarge()
        layout.addLayout(
            hp.make_h_layout(
                btn,
                hp.make_label(self, VIEWER_TEXT, alignment=Qt.AlignHCenter, wrap=True),  # type: ignore[attr-defined]
                stretch_id=0,
                alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
            ),
        )
        # crop app
        btn = hp.make_qta_btn(self, "crop", tooltip="Open crop application.", func=self.on_crop)
        btn.set_xxlarge()
        layout.addLayout(
            hp.make_h_layout(
                btn,
                hp.make_label(self, CROP_TEXT, alignment=Qt.AlignHCenter, wrap=True),  # type: ignore[attr-defined]
                stretch_id=0,
                alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
            ),
        )
        # convert app
        btn = hp.make_qta_btn(self, "change", tooltip="Open czi2tiff application.", func=self.on_convert)
        btn.set_xxlarge()
        layout.addLayout(
            hp.make_h_layout(
                btn,
                hp.make_label(self, CONVERT_TEXT, alignment=Qt.AlignHCenter, wrap=True),  # type: ignore[attr-defined]
                stretch_id=0,
                alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
            ),
        )
        # export app
        btn = hp.make_qta_btn(self, "export", tooltip="Open export application.", func=self.on_export)
        btn.set_xxlarge()
        layout.addLayout(
            hp.make_h_layout(
                btn,
                hp.make_label(self, EXPORT_TEXT, alignment=Qt.AlignHCenter, wrap=True),  # type: ignore[attr-defined]
                stretch_id=0,
                alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
            ),
        )
        layout.addWidget(hp.make_h_line())
        layout.addWidget(hp.make_btn(self, "Show logger...", func=self.on_show_logger))
        layout.addWidget(hp.make_btn(self, "Show IPython console...", func=self.on_show_console))
        layout.addStretch(1)
        return layout

    def on_show_logger(self) -> None:
        """View console."""
        self.logger.show()

    def on_show_console(self) -> None:
        """View console."""
        if self.console is None:
            from qtextra.dialogs.qt_console import QtConsoleDialog

            self.console = QtConsoleDialog(self)
        self.console.show()

    @staticmethod
    def on_convert():
        """Open registration application."""
        from image2image.qt.dialog_convert import ImageConvertWindow

        logger.debug("Opening czi2tiff application.")
        dlg = ImageConvertWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(500, 500)
        dlg.show()

    @staticmethod
    def on_export():
        """Open registration application."""
        from image2image.qt.dialog_fusion import ImageFusionWindow

        logger.debug("Opening export application.")
        dlg = ImageFusionWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(500, 500)
        dlg.show()

    @staticmethod
    def on_register():
        """Open registration application."""
        from image2image.qt.dialog_register import ImageRegistrationWindow

        logger.debug("Opening registration application.")
        dlg = ImageRegistrationWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(1200, 700)
        dlg.show()

    @staticmethod
    def on_viewer():
        """Open registration application."""
        from image2image.qt.dialog_viewer import ImageViewerWindow

        logger.debug("Opening viewer application.")
        dlg = ImageViewerWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(1200, 700)
        dlg.show()

    @staticmethod
    def on_crop():
        """Open registration application."""
        from image2image.qt.dialog_crop import ImageCropWindow

        logger.debug("Opening crop application.")
        dlg = ImageCropWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(1200, 700)
        dlg.show()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(
        # dev=True,
        tool="launcher",
        level=0,
    )
