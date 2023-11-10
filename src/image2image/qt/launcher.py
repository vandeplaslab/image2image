"""Launcher application for image2image."""
from __future__ import annotations

import qtextra.helpers as hp
from qtextra.config import THEMES
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout

REGISTER_TEXT = "<b>Registration App</b><br>Co-register your microscopy and imaging mass spectrometry data."
VIEWER_TEXT = "<b>Viewer App</b><br>Overlay your microscopy and imaging mass spectrometry data."
EXPORT_TEXT = "<b>Export App</b><br>Export your data for Image Fusion in MATLAB compatible format."
CROP_TEXT = "<b>Crop App</b><br>Crop your microscopy data to reduce it's size (handy for Image Fusion)."
SYNC_TEXT = "<b>Sync App</b><br>(coming)"


class Launcher(QtDialog):
    """General launcher application."""

    def __init__(self, parent=None):
        super().__init__(parent, title="Image2Image Launcher")

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
        # # sync app
        # btn = hp.make_qta_btn(self, "sync", tooltip="Open sync application (coming).", func=self.on_viewer)
        # hp.disable_widgets(btn, disabled=True)
        # btn.set_xxlarge()
        # layout.addLayout(
        #     hp.make_h_layout(
        #         btn,
        #         hp.make_label(self, SYNC_TEXT, alignment=Qt.AlignHCenter, wrap=True),  # type: ignore[attr-defined]
        #         stretch_id=0,
        #         alignment=Qt.AlignCenter,  # type: ignore[attr-defined]
        #     ),
        # )
        layout.addStretch(1)
        return layout

    @staticmethod
    def on_export():
        """Open registration application."""
        from image2image.qt.dialog_export import ImageExportWindow

        dlg = ImageExportWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(500, 500)
        dlg.show()

    @staticmethod
    def on_register():
        """Open registration application."""
        from image2image.qt.dialog_register import ImageRegistrationWindow

        dlg = ImageRegistrationWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(1200, 700)
        dlg.show()

    @staticmethod
    def on_viewer():
        """Open registration application."""
        from image2image.qt.dialog_viewer import ImageViewerWindow

        dlg = ImageViewerWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(1200, 700)
        dlg.show()

    @staticmethod
    def on_crop():
        """Open registration application."""
        from image2image.qt.dialog_crop import ImageCropWindow

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
