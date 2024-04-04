"""Save image(s) to disk dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtWidgets import QFormLayout

from image2image.config import CONFIG

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


class ExportImageDialog(QtDialog):
    """Dialog that lets you select what should be imported."""

    def __init__(self, parent, model: DataModel, key: str):
        self.model = model
        self.key = key
        super().__init__(parent)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        reader = self.model.get_reader_for_key(self.key)
        info = (
            f"<b>RGB</b>: {reader.is_rgb}<br>"
            f"<b>Number of channels</b>: {reader.n_channels}<br>"
            f"<b>Image shape</b>: {reader.image_shape}<br>"
            f"<b>Resolution</b>: {reader.resolution}<br>"
            f"<b>Data type</b>: {reader.dtype}<br>"
        )
        self.info_label = hp.make_label(
            self,
            info,
            tooltip="Export image(s) to OME-TIFF format.",
        )
        self.tile_size = hp.make_combobox(
            self,
            ["256", "512", "1024", "2048", "4096"],
            tooltip="Specify size of the tile. Default is 512",
            default="512",
            value=f"{CONFIG.tile_size}",
        )
        self.as_uint8 = hp.make_checkbox(
            self,
            "Reduce data size (uint8 - dynamic range 0-255)",
            tooltip="Convert to uint8 to reduce file size with minimal data loss.",
            checked=True,
            value=CONFIG.as_uint8,
        )

        layout = hp.make_form_layout()
        hp.style_form_layout(layout)
        layout.addRow(self.info_label)
        layout.addRow("Tile size", self.tile_size)
        layout.addRow(self.as_uint8)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def accept(self):
        """Accept."""
        CONFIG.tile_size = int(self.tile_size.currentText())
        CONFIG.as_uint8 = self.as_uint8.isChecked()
        reader = self.model.get_reader_for_key(self.key)
        base_dir = reader.path.parent
        filename = f"{reader.path.stem}-exported".replace(".ome", "") + ".ome.tiff"
        # export image
        filename = hp.get_save_filename(
            self,
            "Save image filename...",
            base_dir,
            base_filename=filename,
            file_filter="OME-TIFF (*.ome.tiff);;",
        )
        if not filename or Path(filename).exists():
            return
        reader.to_ome_tiff(filename, as_uint8=CONFIG.as_uint8, tile_size=CONFIG.tile_size)
        hp.toast(self, "Image saved", f"Saved image {hp.hyper(Path(filename), self.key)} as OME-TIFF.", icon="info")
        return super().accept()
