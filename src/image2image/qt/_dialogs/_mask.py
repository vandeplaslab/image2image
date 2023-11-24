"""Export mask."""
from __future__ import annotations

import typing as ty
from pathlib import Path

from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtWidgets import QFormLayout

from image2image.config import CONFIG
from image2image.enums import DEFAULT_TRANSFORM_NAME

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel
    from image2image.qt.dialog_viewer import ImageViewerWindow
    from image2image_reader.readers.geojson_reader import GeoJSONReader

logger = logger.bind(src="MaskDialog")


class MasksDialog(QtFramelessTool):
    """Dialog to display fiducial marker information."""

    HIDE_WHEN_CLOSE = False

    TABLE_GEO_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True)
        .add("name", "name", "str", 100)
        .add("path", "path", "str", 0, hidden=True)
        .add("key", "key", "str", 0, hidden=True)
    )

    TABLE_IMAGE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True)
        .add("name", "name", "str", 100)
        .add("output shape", "shape", "str", 100)
        .add("path", "path", "str", 0, hidden=True)
        .add("key", "key", "str", 0, hidden=True)
    )

    def __init__(self, parent: ImageViewerWindow):
        super().__init__(parent)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.on_load()

    def on_load(self) -> None:
        """Load data."""
        parent: ImageViewerWindow = self.parent()  # type: ignore[assignment]
        data_model = parent.data_model
        wrapper = data_model.wrapper
        images, masks = [], []
        # get images
        if wrapper:
            for reader in wrapper.reader_iter():
                if reader.reader_type == "image":
                    shape = reader.image_shape

                    images.append([False, reader.name, f"{shape[0]} * {shape[1]}", reader.path, reader.key])
                if reader.reader_type == "shapes":
                    if not reader.is_identity_transform():
                        logger.warning(
                            f"For the time being, only masks with '{DEFAULT_TRANSFORM_NAME}' transform are compatible."
                        )
                        continue
                    masks.append([True, reader.name, reader.path, reader.key])
        logger.debug(f"Discovered {len(images)} images and {len(masks)} masks.")
        # update table
        self.table_geo.reset_data()
        self.table_geo.add_data(masks)
        self.table_image.reset_data()
        self.table_image.add_data(images)

    def _validate(self) -> tuple[bool, tuple | None]:
        parent: ImageViewerWindow = self.parent()  # type: ignore[assignment]
        data_model = parent.data_model
        wrapper = data_model.wrapper
        if not wrapper:
            hp.toast(
                self, "Could not export masks.", "Could not export masks - there is not data loaded.", icon="error"
            )
            return False, None
        # get the image(s) with identity transform
        mask_shape = None
        for _, reader in wrapper.data.items():
            if reader.reader_type == "image" and reader.is_identity_transform():
                mask_shape = reader.image_shape
                break
        if mask_shape is None:
            hp.toast(
                self,
                "Could not export masks.",
                "Could not export masks - no image with identity transform.",
                icon="error",
            )
            return False, None
        masks: list[str] = [
            self.table_geo.get_value(self.TABLE_GEO_CONFIG.key, index) for index in self.table_geo.get_all_checked()
        ]
        images: list[str] = [
            self.table_image.get_value(self.TABLE_IMAGE_CONFIG.key, index)
            for index in self.table_image.get_all_checked()
        ]
        if not masks:
            hp.toast(self, "Could not export masks.", "Could not export masks - no masks selected.", icon="error")
            return False, None
        if not images:
            hp.toast(self, "Could not export masks.", "Could not export masks - no images selected.", icon="error")
            return False, None
        return True, (data_model, masks, images, mask_shape)

    def on_export(self) -> None:
        """Export masks."""
        valid, data = self._validate()
        if not valid or data is None:
            return
        # export masks
        data_model, masks, images, mask_shape = data
        output_dir_ = hp.get_directory(self, "Select output directory", base_dir=CONFIG.output_dir)
        if output_dir_:
            output_dir = Path(output_dir_)
            self._on_export(mask_shape, masks, images, data_model, output_dir, preview=False)

    def on_preview(self) -> None:
        """Export masks."""
        valid, data = self._validate()
        if not valid or data is None:
            return
        # export masks
        data_model, masks, images, mask_shape = data
        logger.debug(f"Exporting {len(masks)} masks for {len(images)} images with {mask_shape} shape.")
        self._on_export(mask_shape, masks, images, data_model, preview=True)

    def _on_export(
        self,
        mask_shape: tuple[int, int],
        masks: list[str],
        images: list[str],
        data_model: DataModel,
        output_dir: Path | None = None,
        preview: bool = False,
    ) -> None:
        """Export masks."""
        from image2image.utils.mask import write_masks

        if not preview and not output_dir:
            raise ValueError("Must provide output directory if exporting.")

        parent: ImageViewerWindow = self.parent()  # type: ignore[assignment]
        for mask_key in masks:
            mask_reader: GeoJSONReader = data_model.get_reader_for_key(mask_key)  # type: ignore[assignment]
            if not mask_reader:
                raise ValueError(f"Could not find mask reader for '{mask_key}'")
            mask = mask_reader.to_mask(mask_shape)
            mask_indexed = mask_reader.to_mask(mask_shape, with_index=True)
            display_name, shapes = mask_reader.to_shapes()
            for image_key in images:
                image_reader = data_model.get_reader_for_key(image_key)
                if not image_reader:
                    raise ValueError(f"Could not find image reader for '{image_key}'")
                transformed_mask = image_reader.warp(mask)
                transformed_mask_indexed = image_reader.warp(mask_indexed)

                # at um level
                transform = image_reader.transform
                # preview
                if preview:
                    parent.view.add_image(
                        transformed_mask,
                        name=f"{mask_reader.name} | {image_reader.name}",
                        colormap="red",
                        affine=transform,
                        scale=image_reader.scale,
                        contrast_limits=(0, 1),
                        keep_auto_contrast=False,
                    )
                # export
                elif output_dir:
                    name = mask_reader.path.stem

                    output_path = output_dir / f"{name}-{image_reader.path.stem}.h5"
                    logger.debug(f"Exporting mask to '{output_path}'")
                    write_masks(
                        output_path,
                        display_name,
                        transformed_mask,
                        shapes,
                        display_name,
                        metadata={"polygon_index": transformed_mask_indexed},
                    )
                    logger.debug(f"Exported mask to '{output_path}'")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle("Export masks")

        self.table_geo = QtCheckableTableView(
            self, config=self.TABLE_GEO_CONFIG, enable_all_check=False, sortable=False
        )
        self.table_geo.setCornerButtonEnabled(False)
        hp.set_font(self.table_geo)
        self.table_geo.setup_model(
            self.TABLE_GEO_CONFIG.header,
            self.TABLE_GEO_CONFIG.no_sort_columns,
            self.TABLE_GEO_CONFIG.hidden_columns,
        )

        self.table_image = QtCheckableTableView(
            self, config=self.TABLE_IMAGE_CONFIG, enable_all_check=False, sortable=False
        )
        self.table_image.setCornerButtonEnabled(False)
        hp.set_font(self.table_image)
        self.table_image.setup_model(
            self.TABLE_IMAGE_CONFIG.header,
            self.TABLE_IMAGE_CONFIG.no_sort_columns,
            self.TABLE_IMAGE_CONFIG.hidden_columns,
        )

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(hp.make_h_line_with_text("Masks to export."))
        layout.addRow(self.table_geo)
        layout.addRow(hp.make_h_line_with_text("Images to export masks for."))
        layout.addRow(self.table_image)
        layout.addRow(hp.make_h_line_with_text("Export"))
        layout.addRow(hp.make_btn(self, "Preview", func=self.on_preview))
        layout.addRow(hp.make_btn(self, "Export", func=self.on_export))
        return layout
