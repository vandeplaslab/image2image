"""Launcher application for image2image."""
import qtextra.helpers as hp
from qtextra.config import THEMES
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout


class Launcher(QtDialog):
    """General launcher application."""

    def __init__(self, parent):
        super().__init__(parent, title="App Launcher")

    def make_panel(self) -> QHBoxLayout:
        """Make panel."""
        btn = hp.make_qta_btn(self, "register", tooltip="Open registration application.", func=self.on_register)
        btn.set_xxlarge()

        reg_layout = QVBoxLayout()
        reg_layout.addWidget(btn, alignment=Qt.AlignHCenter)
        reg_layout.addWidget(hp.make_label(self, "<b>Registration App</b>", alignment=Qt.AlignHCenter))

        btn = hp.make_qta_btn(self, "viewer", tooltip="Open viewer application.", func=self.on_viewer)
        btn.set_xxlarge()
        viewer_layout = QVBoxLayout()
        viewer_layout.addWidget(btn, alignment=Qt.AlignHCenter)
        viewer_layout.addWidget(hp.make_label(self, "<b>Viewer App</b>", alignment=Qt.AlignHCenter))

        layout = QHBoxLayout()
        layout.addLayout(reg_layout)
        layout.addLayout(viewer_layout)
        return layout

    def on_register(self):
        """Open registration application."""
        from image2image.dialog_register import ImageRegistrationWindow

        dlg = ImageRegistrationWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(1200, 500)
        dlg.show()

    def on_viewer(self):
        """Open registration application."""
        from image2image.dialog_viewer import ImageViewerWindow

        dlg = ImageViewerWindow(None)
        THEMES.set_theme_stylesheet(dlg)
        dlg.setMinimumSize(1200, 500)
        dlg.show()


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(
        # dev=True,
        tool="launcher",
        level=0,
    )
