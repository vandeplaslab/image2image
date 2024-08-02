""" "Attachment widgets."""

from __future__ import annotations

from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtWidgets import QFormLayout, QWidget


class AttachWidget(QtDialog):
    """Dialog window to attach widgets to a parent widget."""

    attachment_name: str = ""
    source_pixel_size: float = 1.0

    def __init__(self, parent: QWidget, pixel_sizes: tuple[float, float], title: str = "Attach modality..."):
        # pixel_sizes are specified as (default (unknown), modality)
        self._pixel_sizes = pixel_sizes
        super().__init__(parent, title=title)

    def on_update(self) -> None:
        """Update values."""
        self.attachment_name = self.name.text()
        self.source_pixel_size = self._pixel_sizes[self.defaults_choice_group.checkedId()]

    def accept(self) -> None:
        """Accept."""
        self.on_update()
        return super().accept()

    def reject(self) -> None:
        """Reject."""
        self.attachment_name = None
        self.source_pixel_size = None
        return super().reject()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.name = hp.make_line_edit(
            self, "", placeholder="Name of the 'attachment' modality", func_changed=self.on_update
        )
        default, modality = self._pixel_sizes
        options = [f"{default:.3f} (coordinates are in µm)", f"{modality:.3f} (coordinates are in px)"]
        self.defaults_choice_lay, self.defaults_choice_group = hp.make_toggle_group(self, *options, func=self.on_update)

        layout = hp.make_form_layout(self)
        layout.addRow(
            hp.make_label(
                self,
                "You can specify multiple attachment modalities. They can be associated by <b>name</b> which will treat"
                " as as separate files but metadata such as <b>pixel size</b> will be shared.<br>It's essential that"
                " you specify the <b>pixel size</b> of the attachment modality because the coordinates must be"
                " transformed correctly before applying registration.<br><br>"
                "Value of <b>1.0</b> means that coordinates are in µm (micrometers).<br>"
                "Other values indicate that coordinates are in px (pixels).<br>",
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(hp.make_h_line(self))
        layout.addRow("Name (optional)", self.name)
        layout.addRow("Pixel size", self.defaults_choice_lay)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout
