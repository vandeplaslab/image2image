"""Viewer dialog."""

from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from functools import partial
from math import ceil, floor
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.utilities import pluralize
from loguru import logger
from napari.layers import Shapes
from napari.layers.shapes._shapes_constants import Box
from qtextra.config import THEMES
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import GeneratorWorker, create_worker, qdebounced

import image2image.constants as C
from image2image import __version__
from image2image.config import get_crop_config
from image2image.enums import ALLOWED_PROJECT_CROP_FORMATS
from image2image.qt._dialog_mixins import SingleViewerMixin
from image2image.qt._dialogs._select import LoadWidget
from image2image.utils.utilities import ensure_extension, init_shapes_layer, log_exception_or_error, write_project

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader
    from qtextraplot._napari.image.wrapper import NapariImageView

    from image2image.models.data import DataModel


def parse_crop_info(data: tuple[int, int, int, int] | np.ndarray | None = None) -> str:
    """Parse crop data."""
    if data is None:
        return "No data"
    if isinstance(data, np.ndarray):
        left, right, top, bottom = data[:, 1].min(), data[:, 1].max(), data[:, 0].min(), data[:, 0].max()
        return (
            f"Shape: Polygon"
            f"\nNumber of points: {len(data)}"
            f"\nHorizontal: {left:.0f}-{right:.0f} ({right - left:.0f})"
            f"\nVertical: {top:.0f}-{bottom:.0f} ({bottom - top:.0f})"
        )
    left, right, top, bottom = data
    return (
        f"Shape: Rectangle"
        f"\nHorizontal: {left:.0f}-{right:.0f} ({right - left:.0f})"
        f"\nVertical: {top:.0f}-{bottom:.0f} ({bottom - top:.0f})"
    )


class ImageCropWindow(SingleViewerMixin):
    """Image viewer dialog."""

    APP_NAME = "crop"

    view: NapariImageView

    worker_crop: GeneratorWorker | None = None
    worker_preview_crop: GeneratorWorker | None = None
    worker_mask: GeneratorWorker | None = None
    worker_preview_mask: GeneratorWorker | None = None

    _editing = False

    def __init__(self, parent: QWidget | None = None, run_check_version: bool = True, **kwargs: ty.Any):
        self.CONFIG = get_crop_config()
        super().__init__(
            parent,
            f"image2image: Crop images app (v{__version__})",
            run_check_version=run_check_version,
        )
        if self.CONFIG.first_time:
            hp.call_later(self, self.on_show_tutorial, 10_000)
        self.on_set_write_warning()

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.view_type = "overlay"  # type: ignore[assignment]
        READER_CONFIG.init_pyramid = True
        READER_CONFIG.auto_pyramid = True
        READER_CONFIG.split_czi = True
        READER_CONFIG.split_roi = True
        READER_CONFIG.split_rgb = False
        READER_CONFIG.only_last_pyramid = False

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dset_dlg.evt_import_project, self._on_load_from_project, state=state)
        connect(self._image_widget.dset_dlg.evt_export_project, self.on_save_to_project, state=state)
        connect(self._image_widget.dset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dset_dlg.evt_closed, self.on_close_image, state=state)
        connect(self._image_widget.dset_dlg.evt_channel, self.on_toggle_channel, state=state)
        connect(self._image_widget.dset_dlg.evt_channel_all, self.on_toggle_all_channels, state=state)
        connect(self.view.viewer.events.status, self._status_changed, state=state)
        connect(self._image_widget.dset_dlg.evt_resolution, self.on_update_transform, state=state)
        connect(self._image_widget.dset_dlg.evt_transform, self.on_update_transform, state=state)

    @ensure_main_thread
    def on_load_image(self, model: DataModel, channel_list: list[str]) -> None:
        """Load fixed image."""
        if model and model.n_paths:
            self._on_load_image(model, channel_list)
            hp.toast(
                self, "Loaded data", f"Loaded model with {model.n_paths} paths.", icon="success", position="top_left"
            )
        else:
            logger.warning(f"Failed to load data - model={model}")
        self._move_layer(self.view, self.crop_layer)

    def on_close_image(self, model: DataModel) -> None:
        """Close fixed image."""
        self._close_model(model, self.view, "view", exclude_names=["Mask"])

    def on_load_from_project(self) -> None:
        """Load previous data."""
        path_ = hp.get_filename(
            self, "Load i2c project", base_dir=self.CONFIG.output_dir, file_filter=ALLOWED_PROJECT_CROP_FORMATS
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: str) -> None:
        if path_:
            from image2image.models.data import load_crop_setup_from_file
            from image2image.models.utilities import _remove_missing_from_dict

            path = Path(path_)
            self.CONFIG.update(output_dir=str(path.parent))

            # load data from config file
            try:
                paths, paths_missing, transform_data, resolution, crop = load_crop_setup_from_file(path)
            except ValueError as e:
                hp.warn_pretty(self, f"Failed to load config from {path}\n{e}", "Failed to load config")
                logger.exception(e)
                return

            # locate paths that are missing
            if paths_missing:
                from image2image.qt._dialogs import LocateFilesDialog

                locate_dlg = LocateFilesDialog(self, self.CONFIG, paths_missing)
                if locate_dlg.exec_():  # type: ignore[attr-defined]
                    paths = locate_dlg.fix_missing_paths(paths_missing, paths)

            # clean-up affine matrices
            transform_data = _remove_missing_from_dict(transform_data, paths)
            resolution = _remove_missing_from_dict(resolution, paths)

            # add paths
            if paths:
                self._image_widget.on_set_path(paths, transform_data, resolution)

            # add crop
            data = []
            for crop_ in crop:
                left, right, top, bottom = crop_["left"], crop_["right"], crop_["top"], crop_["bottom"]
                rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
                with self._editing_crop():
                    data.append((rect, "rectangle"))
                    self.crop_layer.data = [(rect, "rectangle")]
            if data:
                self.crop_layer.add(data)

    def on_save_to_project(self) -> None:
        """Save data to config file."""
        if not self._validate():
            return

        filename = "crop.i2c.json"
        path_ = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=self.CONFIG.output_dir,
            file_filter=ALLOWED_PROJECT_CROP_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2c")
            self.CONFIG.update(output_dir=str(path.parent))
            regions = self.get_crop_areas()
            config = get_project_data(self.data_model, regions)
            write_project(path, config)
            hp.toast(
                self,
                "Exported project",
                f"Saved project to<br><b>{hp.hyper(path)}</b>",
                icon="success",
                position="top_left",
            )

    def _validate(self, index: int = 0) -> bool:
        if not self.data_model.is_valid():
            hp.toast(self, "Invalid data", "Image data is invalid.", icon="error")
            return False
        data = self._get_crop_area_for_index(index)
        if isinstance(data, tuple):
            left, right, top, bottom = data
            if left == right:
                hp.toast(
                    self, "Invalid crop area", f"Left and right values are the same for area={index}", icon="error"
                )
                return False
            if top == bottom:
                hp.toast(
                    self, "Invalid crop area", f"Top and bottom values are the same for area={index}.", icon="error"
                )
                return False
        return True

    def on_preview_mask(self):
        """Preview image cropping."""

    def on_open_mask(self):
        """Mask images."""

    def on_preview_crop(self):
        """Preview image cropping."""
        regions = self.get_crop_areas()
        if not regions:
            return

        if regions:
            self.worker_preview_crop = create_worker(
                preview_crop_regions,
                data_model=self.data_model,
                regions=regions,
                _start_thread=True,
                _connect={
                    "aborted": partial(self._on_aborted_crop, which="preview"),
                    "finished": partial(self._on_finished_crop, which="preview"),
                    "yielded": self._on_preview_crop_yield,
                },
            )
            hp.disable_widgets(self.preview_crop_btn.active_btn, disabled=True)
            self.preview_crop_btn.active = True

    @ensure_main_thread()
    def _on_aborted_crop(self, which: str) -> None:
        """Update CSV."""
        if which == "preview":
            self.worker_preview_crop = None
        else:
            self.worker_crop = None

    @ensure_main_thread()
    def _on_preview_crop_yield(self, args: tuple[str, str, np.ndarray, float, int, int]) -> None:
        self.__on_preview_crop_yield(args)

    def __on_preview_crop_yield(self, args: tuple[str, str, np.ndarray, float, int, int]) -> None:
        name, channel_name, array, resolution, current, total = args
        self.preview_crop_btn.setRange(0, total)
        self.preview_crop_btn.setValue(current)
        if array.size == 0:
            return
        self.view.viewer.add_image(array, name=f"{name}-{channel_name}", scale=(resolution, resolution))
        self._move_layer(self.view, self.crop_layer)

    @ensure_main_thread()
    def _on_finished_crop(self, which: str) -> None:
        """Failed exporting of the CSV."""
        if which == "preview":
            self.worker_preview_crop = None
            btn = self.preview_crop_btn
        else:
            self.worker_crop = None
            btn = self.crop_btn
        hp.disable_widgets(btn.active_btn, disabled=False)
        btn.active = False

    def on_cancel(self, which: str) -> None:
        """Cancel cropping."""
        if which == "preview" and self.worker_preview_crop:
            self.worker_preview_crop.quit()
            logger.trace("Requested aborting of the preview process.")
        elif which == "crop" and self.worker_crop:
            self.worker_crop.quit()
            logger.trace("Requested aborting of the crop process.")

    def on_open_crop(self) -> None:
        """Save data."""
        regions = self.get_crop_areas()
        if not regions:
            hp.toast(self, "No regions", "No regions to crop.", icon="error")
            return

        output_dir_ = hp.get_directory(self, "Select output directory", self.CONFIG.output_dir)
        if not output_dir_:
            hp.toast(self, "No output directory", "No output directory selected.", icon="error")
            return

        self.CONFIG.update(output_dir=output_dir_)
        if regions:
            self.worker_crop = create_worker(
                export_crop_regions,
                data_model=self.data_model,
                output_dir=Path(output_dir_),
                regions=regions,
                tile_size=self.CONFIG.tile_size,
                as_uint8=self.CONFIG.as_uint8,
                _start_thread=True,
                _connect={
                    "aborted": partial(self._on_aborted_crop, which="crop"),
                    "finished": partial(self._on_finished_crop, which="crop"),
                    "yielded": self._on_export_crop_yield,
                    "errored": self._on_export_crop_error,
                },
            )
            hp.disable_widgets(self.crop_btn.active_btn, disabled=True)
            self.crop_btn.active = True

    @ensure_main_thread()
    def _on_export_crop_yield(self, args: tuple[Path, int, int]) -> None:
        filename, current, total = args
        self.crop_btn.setRange(0, total)
        self.crop_btn.setValue(current)
        logger.info(f"Exported {filename}")
        self.statusbar.showMessage(f"Exported {filename} {current}/{total}")

    def _on_export_crop_error(self, exc: Exception) -> None:
        hp.toast(self, "Export failed", f"Failed to export: {exc}", icon="error")
        log_exception_or_error(exc)

    def get_crop_areas(self) -> list[tuple[int, int, int, int] | np.ndarray]:
        """Get all crop areas."""
        regions = []
        for index in range(len(self.crop_layer.data)):
            if not self._validate(index):
                return []
            # get crop area
            data = self._get_crop_area_for_index(index)
            if isinstance(data, np.ndarray):
                regions.append(data)
            else:
                left, right, top, bottom = data
                regions.append((left, right, top, bottom))
        return regions

    def on_edit_crop(self):
        """Edit crop area."""
        self.crop_layer.mode = "add_polygon"
        self.view.select_one_layer(self.crop_layer)

    def on_reset_crop(self) -> None:
        """Reset crop area."""
        n = len(self.crop_layer.data)
        if n > 0 and not hp.confirm(
            self,
            f"Are you sure you want to reset the crop areas? There {pluralize('is', n)} currently <b>{n}</b> "
            f"highlighted {pluralize('area', n)}.",
            "Reset crop area",
        ):
            return
        self.crop_layer.data = []
        self.crop_info.setText(parse_crop_info())
        self.crop_layer.mode = "add_polygon"
        self._move_layer(self.view, self.crop_layer)
        logger.info("Reset crop areas.")

    def on_update_ui_from_index(self) -> None:
        """Update crop rect based on the current index."""
        if not self.crop_layer.data:
            return
        current_index = self.index_choice.value()
        self._on_update_crop_from_canvas(current_index)
        self.crop_layer.mode = "select"
        self.crop_layer.selected_data = (current_index,) if current_index != -1 else ()
        if self.crop_layer.selected_data:
            self.crop_layer._set_highlight()
        logger.trace(f"Updated rectangle (from index {current_index}).")

    def on_update_crop_from_canvas(self, _evt: ty.Any = None) -> None:
        """Update crop values."""
        if self._editing:
            return
        n = len(self.crop_layer.data)
        with hp.qt_signals_blocked(self.index_choice):
            self.index_choice.setMaximum(n - 1)
            self.index_choice.setValue(n - 1)
        self._on_update_crop_from_canvas(self.index_choice.value())

    def _get_crop_area_for_index(self, index: int = 0) -> tuple[int, int, int, int] | np.ndarray:
        """Return crop area."""
        n = len(self.crop_layer.data)
        if index > n or index < 0:
            return 0, 0, 0, 0
        shape = self.crop_layer.shape_type[index]
        if shape == "rectangle":
            rect = self.crop_layer.interaction_box(index)
            rect = rect[Box.LINE_HANDLE]
            xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
            xmin = max(0, xmin)
            ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
            ymin = max(0, ymin)
            return floor(xmin), ceil(xmax), floor(ymin), ceil(ymax)
        return self.crop_layer.data[index]

    def _on_update_crop_from_canvas(self, index: int = 0) -> None:
        n = len(self.crop_layer.data)
        if index > n or index < 0:
            self.crop_info.setText("No data")
            return
        data = self._get_crop_area_for_index(index)
        self.crop_info.setText(parse_crop_info(data))
        logger.trace(f"Updated region {index} (from canvas).")

    @property
    def crop_layer(self) -> Shapes:
        """Crop layer."""
        if "Mask" not in self.view.layers:
            layer = self.view.viewer.add_shapes(
                None,
                edge_width=5,
                name="Mask",
                face_color="green",
                edge_color="white",
                opacity=0.5,
            )
            visual = self.view.widget.canvas.layer_to_visual[layer]
            init_shapes_layer(layer, visual)
            connect(self.crop_layer.events.set_data, self.on_update_crop_from_canvas, state=True)
        return self.view.layers["Mask"]

    def _setup_ui(self):
        """Create panel."""
        settings_widget = QWidget(self)
        settings_widget.setMinimumWidth(400)
        settings_widget.setMaximumWidth(400)

        self.view = self._make_image_view(
            self, add_toolbars=False, allow_extraction=False, disable_controls=True, disable_new_layers=True
        )
        self.view.widget.canvas.events.key_press.connect(self.keyPressEvent)
        self.view.viewer.scale_bar.unit = "um"

        self._image_widget = LoadWidget(
            self,
            self.view,
            self.CONFIG,
            project_extension=[".i2c.json", ".i2c.toml"],
            allow_import_project=True,
            allow_export_project=True,
        )

        self.index_choice = hp.make_int_spin_box(
            self,
            -1,
            0,
            tooltip="Index of the drawn shape. Value of -1 means that the last shape is used.",
            func=self.on_update_ui_from_index,
        )

        self.crop_info = hp.make_label(self, "", alignment=Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.crop_info.setMinimumHeight(100)

        crop_layout = hp.make_form_layout()
        crop_layout.setSpacing(2)
        crop_layout.addRow(hp.make_label(self, "Crop area"), self.index_choice)
        crop_layout.addRow(hp.make_label(self, "Information"), self.crop_info)

        self.init_btn = hp.make_btn(
            self, "Initialize crop area", tooltip="Edit crop area (interactively)", func=self.on_edit_crop
        )
        self.reset_btn = hp.make_btn(
            self, "Reset crop area", tooltip="Reset crop area to center of the image.", func=self.on_reset_crop
        )

        self.preview_crop_btn = hp.make_active_progress_btn(
            self,
            "Preview (crop)",
            tooltip="Preview how cropped image would look like.",
            func=self.on_preview_crop,
            cancel_func=partial(self.on_cancel, which="preview"),
        )
        self.crop_btn = hp.make_active_progress_btn(
            self,
            "Export to OME-TIFF (crop)...",
            tooltip="Crop images and save as OME-TIFF files.",
            func=self.on_open_crop,
            cancel_func=partial(self.on_cancel, which="crop"),
        )

        self.preview_mask_btn = hp.make_active_progress_btn(
            self,
            "Preview (crop)",
            tooltip="Preview how masked image would look like.",
            func=self.on_preview_mask,
            cancel_func=partial(self.on_cancel, which="preview"),
        )
        self.preview_mask_btn.hide()

        self.mask_btn = hp.make_active_progress_btn(
            self,
            "Export to OME-TIFF (mask)...",
            tooltip="Mask images and save as OME-TIFF files.",
            func=self.on_open_mask,
            cancel_func=partial(self.on_cancel, which="crop"),
        )
        self.mask_btn.hide()

        self.as_uint8 = hp.make_checkbox(
            settings_widget,
            "",
            tooltip=C.UINT8_TIP,
            value=self.CONFIG.as_uint8,
            func=self.on_update_config,
        )
        self.tile_size = hp.make_combobox(
            settings_widget,
            ["256", "512", "1024", "2048", "4096"],
            tooltip="Specify size of the tile. Default is 512",
            default="512",
            value=f"{self.CONFIG.tile_size}",
            func=self.on_update_config,
        )

        self.hidden_settings = hp.make_advanced_collapsible(
            settings_widget,
            "Export transformed image",
            allow_checkbox=False,
            allow_icon=False,
            warning_icon=("warning", {"color": THEMES.get_theme_color("warning")}),
        )
        self.hidden_settings.addRow(hp.make_label(self, "Tile size"), self.tile_size)
        self.hidden_settings.addRow(
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

        side_layout = hp.make_form_layout(parent=settings_widget)
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line_with_text("Image crop position"))
        side_layout.addRow(crop_layout)
        side_layout.addRow(hp.make_h_layout(self.init_btn, self.reset_btn, spacing=2))
        side_layout.addRow(hp.make_h_line_with_text("Export"))
        side_layout.addRow(self.hidden_settings)
        side_layout.addRow(self.preview_crop_btn)
        side_layout.addRow(self.crop_btn)
        side_layout.addRow(self.preview_mask_btn)
        side_layout.addRow(self.mask_btn)
        side_layout.addRow(hp.make_h_line_with_text("Layer controls"))
        side_layout.addRow(self.view.widget.controls)
        side_layout.addRow(self.view.widget.layerButtons)
        side_layout.addRow(self.view.widget.layers)
        side_layout.addRow(self.view.widget.viewerButtons)

        widget = QWidget()  # noqa
        self.setCentralWidget(widget)
        layout = QHBoxLayout()
        layout.addWidget(self.view.widget, stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addWidget(settings_widget)

        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(layout, stretch=True)

        # extra settings
        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def on_update_config(self, _: ty.Any = None) -> None:
        """Update values in config."""
        self.CONFIG.as_uint8 = self.as_uint8.isChecked()
        self.CONFIG.tile_size = int(self.tile_size.currentText())
        self.on_set_write_warning()

    def on_set_write_warning(self) -> None:
        """Set warning."""
        tooltip = []
        if self.CONFIG.as_uint8:
            tooltip.append(
                "- Images will be converted to uint8 to reduce file size. This can lead to data loss and should be used"
                " with caution."
            )
        self.hidden_settings.warning_label.setToolTip("<br>".join(tooltip))
        self.hidden_settings.set_warning_visible(len(tooltip) > 0)

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update({"viewer": self.view.viewer, "data_model": self.data_model})
        return variables

    @contextmanager
    def _editing_crop(self):
        self._editing = True
        yield
        self._editing = False

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_crop_tutorial

        show_crop_tutorial(self)
        self.CONFIG.update(first_time=False)

    @qdebounced(timeout=50, leading=True)
    def keyPressEvent(self, evt: QKeyEvent) -> None:
        """Key press event."""
        if hasattr(evt, "native"):
            evt = evt.native
        try:
            key = evt.key()
            ignore = self._handle_key_press(key)
            if ignore:
                evt.ignore()
            if not evt.isAccepted():
                return None
            return super().keyPressEvent(evt)
        except RuntimeError:
            return None

    @qdebounced(timeout=100, leading=True)
    def on_handle_key_press(self, key: int) -> bool:
        """Handle key-press event"""
        return self._handle_key_press(key)

    def _handle_key_press(self, key: int) -> bool:
        ignore = False
        return ignore


def get_project_data(data_model: DataModel, regions: list[tuple[int, int, int, int] | np.ndarray]) -> dict:
    """Write project."""
    schema_version = "1.2"
    data_ = data_model.to_dict()
    regions = [
        {"yx": region.tolist()}
        if isinstance(region, np.ndarray)
        else {"left": region[0], "right": region[1], "top": region[2], "bottom": region[3]}
        for region in regions
    ]
    data = {
        "schema_version": schema_version,
        "tool": "crop",
        "version": __version__,
        "crop": regions,
        "images": data_["images"],
    }
    return data


def _crop_regions_iter(
    data_model: DataModel, polygon_or_bbox: list[tuple[int, int, int, int] | np.ndarray], skip_empty: bool = True
) -> ty.Generator[tuple[Path, ty.Any, int, str, np.ndarray, tuple[int, int, int, int]], None, None]:
    """Crop regions."""
    if isinstance(polygon_or_bbox, tuple):
        left, right, top, bottom = polygon_or_bbox
        for path, reader, channel_index, channel_name, cropped_channel, (
            left,
            right,
            top,
            bottom,
        ) in data_model.crop_bbox_iter(left, right, top, bottom):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (left, right, top, bottom),
            )
    else:
        assert isinstance(polygon_or_bbox, np.ndarray), f"Invalid type: {type(polygon_or_bbox)}"
        for path, reader, channel_index, channel_name, cropped_channel, (
            left,
            right,
            top,
            bottom,
        ) in data_model.crop_polygon_iter(polygon_or_bbox):
            yield (
                path,
                reader,
                channel_index,
                channel_name,
                cropped_channel,
                (left, right, top, bottom),
            )


def _get_new_image_shape(reader: BaseReader, left: int, right: int, top: int, bottom: int) -> tuple[int, ...]:
    """Get new image shape."""
    x_size = right - left
    y_size = bottom - top
    channel_axis, n_channels = reader.get_channel_axis_and_n_channels()
    if reader.is_rgb:
        return y_size, x_size, n_channels
    if channel_axis is None:
        return y_size, x_size
    if channel_axis == 0:
        return n_channels, y_size, x_size
    if channel_axis == 1:
        return y_size, n_channels, x_size
    return y_size, x_size, n_channels


def export_crop_regions(
    data_model: DataModel,
    output_dir: Path,
    regions: list[tuple[int, int, int, int] | np.ndarray],
    tile_size: int = 512,
    as_uint8: bool = True,
) -> ty.Generator[tuple[Path, int, int], None, None]:
    """Crop images."""
    from image2image_io.writers import OmeTiffWrapper

    n = len(regions)
    for current, polygon_or_bbox in enumerate(regions, start=1):
        for path, reader in data_model.wrapper.path_reader_iter():
            output_path, dtype, shape, rgb = None, None, None, []
            wrapper = OmeTiffWrapper()
            for channel, (left, right, top, bottom) in reader.crop_region_iter(polygon_or_bbox):
                output_path = output_dir / f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
                dtype = reader.dtype
                shape = _get_new_image_shape(reader, left, right, top, bottom)
                break

            if dtype is None or shape is None or output_path is None:
                logger.warning(f"Skipping {path} as it has no data.")
                continue
            if output_path.with_suffix(".ome.tiff").exists():
                logger.warning(f"Skipping {output_path} as it already exists.")
                yield output_path, current, n
            else:
                with wrapper.write(
                    channel_names=reader.channel_names,
                    resolution=reader.resolution,
                    dtype=dtype,
                    shape=shape,
                    name=output_path.name,
                    output_dir=output_dir,
                    tile_size=tile_size,
                    as_uint8=as_uint8,
                ):
                    channel_index = 0
                    for channel, _ in reader.crop_region_iter(polygon_or_bbox):
                        if channel is None:
                            continue
                        if reader.is_rgb:
                            rgb.append(channel)
                        else:
                            wrapper.add_channel(channel_index, reader.channel_names[channel_index], channel)
                            channel_index += 1
                    if rgb:
                        wrapper.add_channel([0, 1, 2], ["R", "G", "B"], np.dstack(rgb))
                yield wrapper.path, current, n


def preview_crop_regions(
    data_model: DataModel, regions: list[tuple[int, int, int, int] | np.ndarray]
) -> ty.Generator[tuple[str, str, np.ndarray, float, int, int], None, None]:
    """Preview images."""
    n = len(regions)
    for current, polygon_or_bbox in enumerate(regions, start=1):
        for path, reader, _channel_index, channel_name, cropped, (left, right, top, bottom) in _crop_regions_iter(
            data_model, polygon_or_bbox
        ):
            name = f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
            yield name, channel_name, cropped, reader.resolution, current, n


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="crop", level=0)
