"""Take screenshot dialog."""
import typing as ty
from functools import partial

import qtextra.helpers as hp
from qtpy.QtWidgets import QLayout

from image2image.config import CONFIG

if ty.TYPE_CHECKING:
    from qtextra._napari.image.wrapper import NapariImageView

# from qtextra.widgets.qt_color_button import QtColorSwatch
from qtextra.widgets.qt_dialog import QtFramelessPopup


class QtScreenshotDialog(QtFramelessPopup):
    """Popup to control screenshot/clipboard."""

    def __init__(self, wrapper: "NapariImageView", parent=None):
        self.wrapper = wrapper
        super().__init__(parent=parent)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QLayout:
        """Make layout."""
        size = self.wrapper.widget.canvas.size
        self.size_x = hp.make_int_spin_box(self, 50, 10000, 50, default=size[0], tooltip="Width of the screenshot.")
        self.size_y = hp.make_int_spin_box(self, 50, 10000, 50, default=size[1], tooltip="Height of the screenshot.")

        self.scale = hp.make_double_spin_box(
            self, 0.1, 100, 0.5, n_decimals=2, default=1, tooltip="Scale of the screenshot."
        )
        self.canvas_only = hp.make_checkbox(
            self,
            "",
            "Only screenshot the canvas",
            value=True,
        )
        self.clipboard_btn = hp.make_btn(
            self,
            "Copy to clipboard",
            tooltip="Copy screenshot to clipboard",
            func=self.on_copy_to_clipboard,
        )
        self.save_btn = hp.make_btn(
            self,
            "Save to file",
            tooltip="Save screenshot to file",
            func=self.on_save_figure,
        )

        layout = hp.make_form_layout()
        hp.style_form_layout(layout)
        layout.addRow("Width", self.size_x)
        layout.addRow("Height", self.size_y)
        layout.addRow("Scale", self.scale)
        layout.addRow("Canvas only", self.canvas_only)
        layout.addRow(self.clipboard_btn)
        layout.addRow(self.save_btn)
        return layout

    def on_save_figure(self):
        """Save figure."""
        from napari._qt.dialogs.screenshot_dialog import HOME_DIRECTORY, ScreenshotDialog

        save_func = partial(
            self.wrapper.widget.screenshot,
            size=(self.size_y.value(), self.size_x.value()),
            scale=self.scale.value(),
            canvas_only=self.canvas_only.isChecked(),
        )

        dialog = ScreenshotDialog(save_func, self, str(CONFIG.output_dir) or HOME_DIRECTORY, history=[])
        if dialog.exec_():
            pass

    def on_copy_to_clipboard(self):
        """Copy canvas to clipboard."""
        self.wrapper.widget.clipboard(
            size=(self.size_y.value(), self.size_x.value()),
            scale=self.scale.value(),
            canvas_only=self.canvas_only.isChecked(),
        )
