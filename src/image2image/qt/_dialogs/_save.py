"""Save image(s) to disk dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

from qtextra import helpers as hp
from qtextra.config import THEMES
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtWidgets import QFormLayout, QWidget

import image2image.constants as C
from image2image.config import SingleAppConfig

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader


class ExportImageDialog(QtDialog):
    """Dialog that lets you select what should be imported."""

    def __init__(self, parent: QWidget, reader: BaseReader, config: SingleAppConfig):
        self.CONFIG = config
        self.reader = reader
        super().__init__(parent)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        reader = self.reader
        info = (
            f"<b>RGB</b>: {reader.is_rgb}<br>"
            f"<b>Number of channels</b>: {reader.n_channels}<br>"
            f"<b>Image shape</b>:({reader.image_shape[0]:,}, {reader.image_shape[1]:,})<br>"
            f"<b>Resolution</b>: {reader.resolution} µm<br>"
            f"<b>Data type</b>: {reader.dtype}<br>"
        )
        self.info_label = hp.make_label(
            self,
            info,
            tooltip="Export image(s) to OME-TIFF format.",
        )
        if not info:
            self.info_label.hide()
        self.tile_size = hp.make_combobox(
            self,
            ["256", "512", "1024", "2048", "4096"],
            tooltip="Specify size of the tile. Default is 512",
            default="512",
            value=f"{self.CONFIG.tile_size}",
        )
        self.as_uint8 = hp.make_checkbox(
            self,
            "",
            tooltip=C.UINT8_TIP,
            checked=True,
            value=self.CONFIG.as_uint8,
        )

        layout = hp.make_form_layout()
        layout.addRow(self.info_label)
        layout.addRow("Tile size", self.tile_size)
        layout.addRow(
            hp.make_label(self, "Reduce data size"),
            hp.make_h_layout(
                self.as_uint8,
                hp.make_warning_label(
                    self,
                    C.UINT8_WARNING,
                    normal=True,
                    icon_name=("warning", {"color": THEMES.get_theme_color("warning")}),
                ),
                spacing=2,
                stretch_id=(0,),
            ),
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def accept(self) -> None:
        """Accept."""
        self.CONFIG.tile_size = int(self.tile_size.currentText())
        self.CONFIG.as_uint8 = self.as_uint8.isChecked()
        reader = self.reader
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
            return None
        filename = reader.to_ome_tiff(filename, as_uint8=self.CONFIG.as_uint8, tile_size=self.CONFIG.tile_size)
        hp.toast(self, "Image saved", f"Saved image {hp.hyper(filename, reader.key)} as OME-TIFF.", icon="info")
        return super().accept()
