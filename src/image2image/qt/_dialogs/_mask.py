"""Export mask."""

from __future__ import annotations

import typing as ty
from functools import partial
from pathlib import Path

import numpy as np
from image2image_io.enums import DEFAULT_TRANSFORM_NAME
from koyo.system import IS_MAC
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout
from superqt.utils import GeneratorWorker, create_worker, ensure_main_thread

from image2image.config import CONFIG
from image2image.utils.utilities import log_exception_or_error

if ty.TYPE_CHECKING:
    from image2image_io.readers._base_reader import BaseReader
    from image2image_io.readers.shapes_reader import ShapesReader

    from image2image.models.data import DataModel
    from image2image.qt.dialog_viewer import ImageViewerWindow

logger = logger.bind(src="MaskDialog")


class MasksDialog(QtFramelessTool):
    """Dialog to display fiducial marker information."""

    HIDE_WHEN_CLOSE = False

    TABLE_GEO_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("name", "name", "str", 100)
        .add("display name", "display_name", "str", 100)
        .add("path", "path", "str", 0, hidden=True)
        .add("key", "key", "str", 0, hidden=True)
    )

    TABLE_IMAGE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("name", "name", "str", 100)
        .add("output shape", "shape", "str", 100, sizing="fixed")
        .add("path", "path", "str", 0, hidden=True)
        .add("key", "key", "str", 0, hidden=True)
    )

    worker_preview: GeneratorWorker | None = None
    worker_export: GeneratorWorker | None = None

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
                        hp.toast(
                            self,
                            "Incompatible mask transform",
                            f"For the time being, only masks with '{DEFAULT_TRANSFORM_NAME}' transform are compatible."
                            f" If this is an issue, please get in touch to discuss further.",
                        )
                        continue
                    masks.append([True, reader.name, reader.display_name, reader.path, reader.key])
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
        # get the image(s) with identity transform - this is the image to which the GeoJSON mask is to be transformed
        # from and therefore is the 'original' image shape from which to warp from.
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
        # retrieve the masks and images
        masks: list[str] = [
            self.table_geo.get_value(self.TABLE_GEO_CONFIG.key, index) for index in self.table_geo.get_all_checked()
        ]
        display_names: list[str] = [
            self.table_geo.get_value(self.TABLE_GEO_CONFIG.display_name, index)
            for index in self.table_geo.get_all_checked()
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
        return True, (data_model, masks, display_names, images, mask_shape)

    @staticmethod
    def _on_transform_mask(
        mask_shape: tuple[int, int],
        masks: list[str],
        display_names: list[str],
        images: list[str],
        data_model: DataModel,
    ) -> ty.Generator[tuple[ShapesReader, BaseReader, np.ndarray, np.ndarray, str, dict, int, int], None, None]:
        """Export masks."""
        n = int(len(masks) * len(images))
        for index, mask_key in enumerate(masks, start=1):
            mask_reader: ShapesReader = data_model.get_reader_for_key(mask_key)  # type: ignore[assignment]
            if not mask_reader:
                raise ValueError(f"Could not find mask reader for '{mask_key}'")
            mask = mask_reader.to_mask(mask_shape)
            mask_indexed = mask_reader.to_mask(mask_shape, with_index=True)
            _, shapes = mask_reader.to_shapes()
            display_name = display_names[index - 1]
            for image_key in images:
                image_reader = data_model.get_reader_for_key(image_key)
                if not image_reader:
                    raise ValueError(f"Could not find image reader for '{image_key}'")
                transformed_mask = image_reader.warp(mask)
                transformed_mask_indexed = image_reader.warp(mask_indexed)
                yield (
                    mask_reader,
                    image_reader,
                    transformed_mask,
                    transformed_mask_indexed,
                    display_name,
                    shapes,
                    index,
                    n,
                )

    def _on_export_run(
        self,
        fmt: str,
        mask_shape: tuple[int, int],
        masks: list[str],
        display_names: list[str],
        images: list[str],
        data_model: DataModel,
        output_dir: Path,
    ) -> ty.Generator[tuple[int, int], None, None]:
        from image2image_io.utils.mask import write_masks_as_geojson, write_masks_as_hdf5, write_masks_as_image

        if output_dir is None:
            raise ValueError("Output directory is None.")

        for (
            mask_reader,
            image_reader,
            transformed_mask,
            transformed_mask_indexed,
            display_name,
            shapes,
            current,
            total,
        ) in self._on_transform_mask(mask_shape, masks, display_names, images, data_model):
            name = display_name or mask_reader.path.stem
            output_dir = Path(output_dir)
            extension = {"hdf5": "h5", "binary": "png", "geojson": "geojson"}[fmt]
            output_path = output_dir / f"{name}_ds={image_reader.path.stem}.{extension}"
            logger.debug(f"Exporting mask to '{output_path}'")
            if fmt == "hdf5":
                write_masks_as_hdf5(
                    output_path,
                    display_name,
                    transformed_mask,
                    shapes,
                    display_name,
                    metadata={"polygon_index": transformed_mask_indexed},
                )
            elif fmt == "binary":
                write_masks_as_image(output_path, transformed_mask)
            elif fmt == "geojson":
                write_masks_as_geojson(output_path, shapes, display_name)
            else:
                raise ValueError(f"Unsupported format '{fmt}'")
            logger.debug(f"Exported mask to '{output_path}'")
            yield current, total

    def on_cancel(self, which: str) -> None:
        """Cancel cropping."""
        if which == "preview" and self.worker_preview:
            self.worker_preview.quit()
            logger.trace("Requested aborting of the preview process.")
        elif which == "export" and self.worker_export:
            self.worker_export.quit()
            logger.trace("Requested aborting of the export process.")

    def on_export(self, fmt: str) -> None:
        """Export masks."""
        valid, data = self._validate()
        if not valid or data is None:
            return
        # export masks
        data_model, masks, display_names, images, mask_shape = data
        output_dir_ = hp.get_directory(self, "Select output directory", base_dir=CONFIG.output_dir)
        if output_dir_:
            output_dir = Path(output_dir_)
            if IS_MAC:
                hp.disable_widgets(self.export_btn.active_btn, disabled=True)
                self.export_btn.active = True
                try:
                    for res in self._on_export_run(
                        fmt, mask_shape, masks, display_names, images, data_model, output_dir
                    ):
                        self._on_exported(res)
                except Exception as exc:
                    self._on_error(exc)
                finally:
                    self._on_finished("export")
            else:
                self.worker_export = create_worker(
                    self._on_export_run,
                    fmt=fmt,
                    mask_shape=mask_shape,
                    masks=masks,
                    display_names=display_names,
                    images=images,
                    data_model=data_model,
                    output_dir=output_dir,
                    _connect={
                        "yielded": self._on_exported,
                        "errored": self._on_error,
                        "finished": partial(self._on_finished, which="export"),
                        "aborted": partial(self._on_aborted, which="export"),
                    },
                    _worker_class=GeneratorWorker,
                )
                hp.disable_widgets(self.export_btn.active_btn, disabled=True)
                self.export_btn.active = True

    def on_preview(self) -> None:
        """Export masks."""
        valid, data = self._validate()
        if not valid or data is None:
            return
        # export masks
        data_model, masks, display_names, images, mask_shape = data
        logger.debug(f"Exporting {len(masks)} masks for {len(images)} images with {mask_shape} shape.")
        if IS_MAC:
            hp.disable_widgets(self.preview_btn.active_btn, disabled=True)
            self.preview_btn.active = True
            try:
                for res in self._on_transform_mask(mask_shape, masks, display_names, images, data_model):
                    self._on_preview(res)
            except Exception as exc:
                self._on_error(exc)
            finally:
                self._on_finished("preview")
        else:
            self.worker_preview = create_worker(
                self._on_transform_mask,
                mask_shape=mask_shape,
                masks=masks,
                display_names=display_names,
                images=images,
                data_model=data_model,
                _connect={
                    "yielded": self._on_preview,
                    "errored": self._on_error,
                    "finished": partial(self._on_finished, which="preview"),
                    "aborted": partial(self._on_aborted, which="preview"),
                },
                _worker_class=GeneratorWorker,
            )
            hp.disable_widgets(self.preview_btn.active_btn, disabled=True)
            self.preview_btn.active = True

    @ensure_main_thread()
    def _on_preview(self, res) -> None:
        """Preview."""
        mask_reader, image_reader, transformed_mask, _, display_name, _, current, total = res
        parent: ImageViewerWindow = self.parent()  # type: ignore[assignment]
        transform = image_reader.transform
        parent.view.add_image(
            transformed_mask,
            name=f"{display_name} - {mask_reader.name} | {image_reader.name}",
            colormap="red",
            affine=transform,
            scale=image_reader.scale,
            contrast_limits=(0, 1),
            keep_auto_contrast=False,
        )
        self.preview_btn.setRange(0, total)
        self.preview_btn.setValue(current)

    def _on_exported(self, args: tuple[int, int]) -> None:
        current, total = args
        self.export_btn.setRange(0, total)
        self.export_btn.setValue(current)

    @ensure_main_thread()
    def _on_aborted(self, which: str) -> None:
        """Update CSV."""
        if which == "preview":
            self.worker_preview = None
        else:
            self.worker_export = None

    @ensure_main_thread()
    def _on_finished(self, which: str) -> None:
        """Failed exporting of the CSV."""
        if which == "preview":
            self.worker_preview = None
            btn = self.preview_btn
        else:
            self.worker_export = None
            btn = self.export_btn
        hp.disable_widgets(btn.active_btn, disabled=False)
        btn.active = False

    def _on_error(self, exc: Exception) -> None:
        hp.toast(self, "Failed", f"Failed export or preview: {exc}", icon="error")
        log_exception_or_error(exc)

    def on_select(self, row: int) -> None:
        """Select channels."""
        display_name = self.table_geo.get_value(self.TABLE_GEO_CONFIG.display_name, row)

        new_display_name = hp.get_text(self, "Enter new display name", "Display name", display_name)
        if not new_display_name or display_name == new_display_name:
            return
        self.table_geo.set_value(self.TABLE_GEO_CONFIG.display_name, row, new_display_name)

    def on_export_menu(self):
        """Export masks."""
        menu = hp.make_menu(self, "Tools")
        hp.make_menu_item(self, "Save as HDF5", menu=menu, func=partial(self.on_export, "hdf5"), icon="hdf5")
        hp.make_menu_item(
            self, "Save as binary image", menu=menu, func=partial(self.on_export, "binary"), icon="binary"
        )
        hp.make_menu_item(self, "Save as GeoJSON", menu=menu, func=partial(self.on_export, "geojson"), icon="json")
        hp.show_left_of_mouse(menu)

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
        self.table_geo.doubleClicked.connect(lambda index: self.on_select(index.row()))

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

        self.preview_btn = hp.make_active_progress_btn(
            self,
            "Preview",
            tooltip="Preview mask in the viewer.",
            func=self.on_preview,
            cancel_func=partial(self.on_cancel, which="preview"),
        )

        self.export_btn = hp.make_active_progress_btn(
            self,
            "Export masks...",
            tooltip="Export mask as a AutoIMS compatible HDF5, GeoJSON or binary file.",
            func=self.on_export_menu,
            cancel_func=partial(self.on_cancel, which="export"),
        )

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(hp.make_h_line_with_text("Masks to export."))
        layout.addRow(self.table_geo)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> You can double-click on the 'display name' column value to change how the mask will be"
                " named.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(hp.make_h_line_with_text("Images to export masks for."))
        layout.addRow(self.table_image)
        layout.addRow(hp.make_h_line_with_text("Export"))
        layout.addRow(self.preview_btn)
        layout.addRow(self.export_btn)
        # layout.addRow(hp.make_h_layout(self.preview_btn, self.preview_cancel_btn, spacing=2))
        # layout.addRow(hp.make_h_layout(self.export_btn, self.export_cancel_btn, spacing=2))
        return layout
