""" "Attachment widgets."""

from __future__ import annotations

import typing as ty
from functools import partial
from pathlib import Path

from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtCore import Qt
from qtpy.QtGui import QDoubleValidator
from qtpy.QtWidgets import QFormLayout, QLineEdit, QTableWidgetItem, QWidget

if ty.TYPE_CHECKING:
    from image2image_reg.models import Modality
    from image2image_reg.workflows import ElastixReg, ValisReg


class AttachWidget(QtDialog):
    """Dialog window to attach widgets to a parent widget."""

    attachment_name: str = ""
    source_pixel_size: float = 1.0

    def __init__(self, parent: QWidget | None, pixel_sizes: tuple[float, float], title: str = "Attach modality..."):
        # pixel_sizes are specified as (default (unknown), modality)
        self._pixel_sizes = pixel_sizes
        super().__init__(parent, title=title)

    def on_update(self) -> None:
        """Update values."""
        self.source_pixel_size = self._pixel_sizes[self.defaults_choice_group.checkedId()]

    def accept(self) -> None:
        """Accept."""
        self.on_update()
        return super().accept()

    def reject(self) -> None:
        """Reject."""
        self.source_pixel_size = None
        return super().reject()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""

        default, modality = self._pixel_sizes
        options = [
            f"<b>{default:.5f}</b><br>coordinates are in µm<br><b>don't need to rescale</b>",
            f"<b>{modality:.5f}</b><br>coordinates are in px<br><b>need to rescale</b>",
        ]
        self.defaults_choice_lay, self.defaults_choice_group = hp.make_toggle_group(
            self, *options, func=self.on_update, orientation="vertical"
        )

        layout = hp.make_form_layout(parent=self)
        layout.addRow(
            hp.make_label(
                self,
                "Please select the modality to which to attach the attachment modality<br><br>"
                "It is essential that the <b>pixel size</b> is correctly specified because the coordinates must be"
                " scales correctly before applying registration.<br><br>"
                "Value of <b>1.0</b> means that coordinates are in µm (micrometers).<br>"
                "Other values indicate that coordinates are in px (pixels).<br>",
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(hp.make_h_line(self))
        layout.addRow("Pixel size", self.defaults_choice_lay)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout


class AttachmentEditDialog(QtDialog):
    """Dialog where it's possible to edit the attachment."""

    TABLE_CONFIG = (
        TableConfig()  # type: ignore
        .add("name", "name", "str", 0, sizing="stretch")
        .add("pixel size (um)", "resolution", "str", 0, sizing="contents")
        .add("type", "type", "str", 0, sizing="contents")
        .add("", "remove", "button", 0, sizing="contents")
    )

    def __init__(
        self, parent: QWidget | None, modality: Modality, registration_model: ElastixReg | ValisReg, which: str = "all"
    ):
        self.modality = modality
        self.which = which
        self.registration_model = registration_model
        self.name_mapping: dict[tuple[str, str], str] = {}
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.on_populate_table()
        self.setMinimumWidth(600)

    def on_populate_table(self) -> None:
        """Load data."""

        def _insert_row(name: str, path: Path, pixel_size: float, attachment_type: str) -> None:
            # get model information
            index = self.table.rowCount()
            self.table.insertRow(index)

            # add name item
            name_item = QLineEdit(name)
            name_item.setToolTip(str(path))
            name_item.setObjectName("table_cell")
            name_item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_item.editingFinished.connect(
                partial(self.on_update_attachment, name=name, attachment_type=attachment_type, name_edit=name_item)
            )
            self.table.setCellWidget(index, self.TABLE_CONFIG.name, name_item)

            # add resolution item
            res_item = QLineEdit(f"{pixel_size:.5f}")
            res_item.setObjectName("table_cell")
            res_item.setAlignment(Qt.AlignmentFlag.AlignCenter)
            res_item.setValidator(QDoubleValidator(0, 1000, 4))
            res_item.editingFinished.connect(
                partial(self.on_update_attachment, name=name, attachment_type=attachment_type, resolution_edit=res_item)
            )
            self.table.setCellWidget(index, self.TABLE_CONFIG.resolution, res_item)

            # add type item
            type_item = QTableWidgetItem(attachment_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            type_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(index, self.TABLE_CONFIG.type, type_item)

            # remove button
            self.table.setCellWidget(
                index,
                self.TABLE_CONFIG.remove,
                hp.make_qta_btn(
                    self,
                    "delete",
                    func=partial(self.on_remove_attachment, name=name, attachment_type=attachment_type),
                    tooltip="Remove image from project. You will <b>not</b> be asked to confirm removal..",
                ),
            )
            self.name_mapping[(name, attachment_type)] = name

        # clear table
        while self.table.rowCount() > 0:
            self.table.removeRow(0)

        # add images
        if self.which in ["all", "image"]:
            attached_images = self.registration_model.get_attachment_list(self.modality.name, "image")
            for image in attached_images:
                modality = self.registration_model.get_modality(image)
                _insert_row(modality.name, modality.path, modality.pixel_size, "image")
        # add shapes
        if self.which in ["all", "shapes"]:
            attached_shapes = self.registration_model.get_attachment_list(self.modality.name, "geojson")
            for name in attached_shapes:
                shapes = self.registration_model.attachment_shapes[name]
                _insert_row(name, shapes["files"], shapes["pixel_size"], "geojson")

        # add shapes
        if self.which in ["all", "points"]:
            attached_points = self.registration_model.get_attachment_list(self.modality.name, "points")
            for name in attached_points:
                points = self.registration_model.attachment_points[name]
                _insert_row(name, points["files"], points["pixel_size"], "points")

    def on_update_attachment(
        self,
        name: str,
        attachment_type: str,
        resolution_edit: QLineEdit | None = None,
        name_edit: QLineEdit | None = None,
    ) -> None:
        """Update attachment."""
        new_name = name_edit.text() if name_edit else name
        new_resolution = float(resolution_edit.text()) if resolution_edit else None
        # check mapping
        alt_name = self.name_mapping.get((name, attachment_type))
        if alt_name and alt_name != name:
            name = alt_name

        if attachment_type == "image":
            if new_name != name:
                self.registration_model.rename_modality(name, new_name)
            if new_resolution is not None:
                self.registration_model.modalities[new_name].pixel_size = new_resolution
        elif attachment_type == "geojson":
            if new_name != name:
                self.registration_model.attachment_shapes[new_name] = self.registration_model.attachment_shapes.pop(
                    name
                )
            if new_resolution is not None:
                self.registration_model.attachment_shapes[new_name]["pixel_size"] = new_resolution
        elif attachment_type == "points":
            if new_name != name:
                self.registration_model.attachment_points[new_name] = self.registration_model.attachment_points.pop(
                    name
                )
            if new_resolution is not None:
                self.registration_model.attachment_points[new_name]["pixel_size"] = new_resolution
        self.name_mapping[(name, attachment_type)] = new_name

    def on_remove_attachment(self, name: str, attachment_type: str) -> None:
        """Remove attachment from the registration model."""
        if attachment_type == "image":
            self.registration_model.remove_attachment_image(name)
        elif attachment_type == "geojson":
            self.registration_model.remove_attachment_geojson(name)
        elif attachment_type == "points":
            self.registration_model.remove_attachment_points(name)
        self.on_populate_table()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        # _, header_layout = self._make_hide_handle(title="Modalities")

        self.table = hp.make_table(self, self.TABLE_CONFIG)

        layout = hp.make_form_layout(parent=self)
        layout.addRow(
            hp.make_label(
                self,
                "Please edit the attached modalities.<br><br>"
                "You can edit the <b>name</b> and <b>pixel size</b> of the attached modalities.<br>",
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self.table)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout
