"""Launcher application for image2image."""

from __future__ import annotations

import typing as ty
from functools import partial

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from qtextra.config import THEMES
from qtextra.dialogs.qt_logger import QtLoggerDialog
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_tile import QtTileWidget, Tile
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QVBoxLayout, QWidget

import image2image.constants as C
from image2image import __version__
from image2image.config import STATE, get_app_config
from image2image.utils._appdirs import USER_LOG_DIR


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
        self.logger_dlg = QtLoggerDialog(self, USER_LOG_DIR)
        self.setFixedSize(self.sizeHint())
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QVBoxLayout:
        """Make panel."""
        from image2image.qt._dialog_base import Window

        layout = hp.make_form_layout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addRow(hp.make_h_line_with_text("Viewers", self, position="left", bold=True))
        # viewer apps
        layout.addRow(
            hp.make_h_layout(
                _make_tile(
                    self,
                    "Viewer<br>App",
                    C.VIEWER_TEXT,
                    "viewer",
                    partial(self.on_open_app, Window.on_open_viewer),
                ),
                stretch_after=True,
                spacing=2,
                margin=2,
            )
        )

        # register apps
        layout.addRow(hp.make_h_line_with_text("Registration", self, position="left", bold=True))
        layout.addRow(
            hp.make_h_layout(
                _make_tile(
                    self,
                    "Registration<br>App",
                    C.REGISTER_TEXT,
                    "register",
                    partial(self.on_open_app, Window.on_open_register),
                ),
                _make_tile(
                    self,
                    "Elastix<br>App",
                    C.ELASTIX_TEXT,
                    "elastix",
                    partial(self.on_open_app, Window.on_open_elastix),
                ),
                # _make_tile(
                #     self,
                #     "Valis<br>App",
                #     VALIS_TEXT,
                #     "valis",
                #     partial(self.on_open_app, Window.on_open_valis),
                #     icon_kws=None if STATE.allow_valis else {"color": THEMES.get_hex_color("warning")},
                #     warning=VALIS_WARNING,
                # ),
                stretch_after=True,
                spacing=2,
                margin=2,
            ),
        )

        # utility apps
        layout.addRow(hp.make_h_line_with_text("Utilities", self, position="left", bold=True))
        layout.addRow(
            hp.make_h_layout(
                _make_tile(
                    self,
                    "Image Crop<br>App",
                    C.CROP_TEXT,
                    "crop",
                    partial(self.on_open_app, Window.on_open_crop),
                ),
                _make_tile(
                    self,
                    "Image to OME-TIFF<br>App",
                    C.CONVERT_TEXT,
                    "convert",
                    partial(self.on_open_app, Window.on_open_convert),
                    icon_kws=None if STATE.allow_convert else {"color": THEMES.get_hex_color("warning")},
                    warning=C.CONVERT_WARNING,
                ),
                _make_tile(
                    self,
                    "Merge OME-TIFFs<br>App",
                    C.MERGE_TEXT,
                    "merge",
                    partial(self.on_open_app, Window.on_open_merge),
                ),
                _make_tile(
                    self,
                    "Fusion Preparation<br>App",
                    C.FUSION_TEXT,
                    "fusion",
                    partial(self.on_open_app, Window.on_open_fusion),
                ),
                stretch_after=True,
                spacing=2,
                margin=2,
            )
        )

        self.progress_label = hp.make_label(self, "")
        self.spinner, _ = hp.make_loading_gif(self, which="infinity", size=(40, 40), retain_size=False, hide=True)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout, stretch=1)
        main_layout.addWidget(hp.make_h_line())
        main_layout.addWidget(hp.make_btn(self, "Show logger...", func=self.on_show_logger))
        main_layout.addWidget(hp.make_btn(self, "Show IPython console...", func=self.on_show_console))
        main_layout.addLayout(hp.make_h_layout(self.spinner, self.progress_label, stretch_id=(1,), spacing=2))
        main_layout.addStretch(1)
        return main_layout

    def on_open_app(self, func: ty.Callable) -> None:
        """Open app."""
        self.spinner.show()
        self.progress_label.setText("Opening application - this should only take a moment...")
        func()
        hp.call_later(self, self._open_finished, 5000)

    def _open_finished(self) -> None:
        """Open app."""
        self.spinner.hide()
        self.progress_label.setText("")

    def on_show_logger(self) -> None:
        """View console."""
        self.logger_dlg.show()

    def on_show_console(self) -> None:
        """View console."""
        if self.console is None:
            from qtextra.dialogs.qt_console import QtConsoleDialog

            self.console = QtConsoleDialog(self)
            self.console.push_variables(
                {"window": self, "APP_CONFIG": get_app_config(), "READER_CONFIG": READER_CONFIG}
            )
        self.console.show()

    @staticmethod
    def on_toggle_theme() -> None:
        """Toggle theme."""
        THEMES.theme = "dark" if THEMES.theme == "light" else "light"
        get_app_config().theme = THEMES.theme

    def keyPressEvent(self, evt):
        """Key press event."""
        if evt.key() == Qt.Key.Key_Escape:
            evt.ignore()
        else:
            super().keyPressEvent(evt)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(tool="launcher", level=0, dev=True)
