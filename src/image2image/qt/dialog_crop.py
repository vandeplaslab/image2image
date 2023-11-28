"""Viewer dialog."""
from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from functools import partial
from math import ceil, floor
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_reader.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from koyo.utilities import pluralize
from loguru import logger
from napari.layers import Image, Shapes
from napari.layers.shapes._shapes_constants import Box
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QMenuBar, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import GeneratorWorker, create_worker

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_CROP_FORMATS
from image2image.qt._select import LoadWidget
from image2image.qt.dialog_base import Window
from image2image.utils.utilities import ensure_extension, init_shapes_layer, log_exception_or_error, write_project

if ty.TYPE_CHECKING:
    from image2image.models.data import DataModel


class ImageCropWindow(Window):
    """Image viewer dialog."""

    worker_crop: GeneratorWorker | None = None
    worker_preview: GeneratorWorker | None = None

    image_layer: list[Image] | None = None
    shape_layer: list[Shapes] | None = None
    _console = None
    _editing = False

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, f"image2crop: Crop and export microscopy data app (v{__version__})")

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closed, self.on_close_image, state=state)
        connect(self._image_widget.evt_toggle_channel, self.on_toggle_channel, state=state)
        connect(self._image_widget.evt_toggle_all_channels, self.on_toggle_all_channels, state=state)

    def on_toggle_channel(self, name: str, state: bool) -> None:
        """Toggle channel."""
        self._toggle_channel(self.data_model, self.view, name, state, "view")

    def on_toggle_all_channels(self, state: bool, channel_names: list[str] | None = None) -> None:
        """Toggle channel."""
        self._toggle_all_channels(self.data_model, self.view, state, "view", channel_names=channel_names)

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

    def _on_load_image(self, model: DataModel, channel_list: list[str] | None = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading fixed data with {model.n_paths} paths...")
            self.plot_image_layers(channel_list)
            self.view.viewer.reset_view()
        logger.info(f"Loaded data in {timer()}")

    def plot_image_layers(self, channel_list: list[str] | None = None) -> None:
        """Plot image layers."""
        self.image_layer, self.shape_layer = self._plot_image_layers(
            self.data_model, self.view, channel_list, "view", True
        )

    def on_close_image(self, model: DataModel) -> None:
        """Close fixed image."""
        self._close_model(model, self.view, "view")

    def on_load_from_project(self) -> None:
        """Load previous data."""
        path = hp.get_filename(self, "Load i2c project", base_dir=CONFIG.output_dir, file_filter=ALLOWED_CROP_FORMATS)
        if path:
            from image2image.models.data import load_crop_setup_from_file
            from image2image.models.utilities import _remove_missing_from_dict

            path_ = Path(path)
            CONFIG.output_dir = str(path_.parent)

            # load data from config file
            try:
                paths, paths_missing, transform_data, resolution, crop = load_crop_setup_from_file(path_)
            except ValueError as e:
                hp.warn(self, f"Failed to load transformation from {path_}\n{e}", "Failed to load transformation")
                return

            # locate paths that are missing
            if paths_missing:
                from image2image.qt._dialogs import LocateFilesDialog

                locate_dlg = LocateFilesDialog(self, paths_missing)
                if locate_dlg.exec_():  # noqa
                    paths = locate_dlg.fix_missing_paths(paths_missing, paths)

            # clean-up affine matrices
            transform_data = _remove_missing_from_dict(transform_data, paths)
            resolution = _remove_missing_from_dict(resolution, paths)
            # add paths
            if paths:
                self._image_widget.on_set_path(paths, transform_data, resolution)
            # add crop
            with hp.qt_signals_blocked(self.left_edit, self.right_edit, self.top_edit, self.bottom_edit):
                crop_ = crop[0]
                self.left_edit.setText(str(crop_["left"]))
                self.right_edit.setText(str(crop_["right"]))
                self.top_edit.setText(str(crop_["top"]))
                self.bottom_edit.setText(str(crop_["bottom"]))
        self.on_update_rect_from_ui()

    def on_save_to_project(self) -> None:
        """Save data to config file."""
        if not self._validate():
            return

        filename = "crop.i2c.json"
        path_ = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_CROP_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2c")
            CONFIG.output_dir = str(path.parent)
            left, right, top, bottom = self._get_crop_area()
            config = get_project_data(self.data_model, left, right, top, bottom)
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
        left, right, top, bottom = self._get_crop_area_for_index(index)
        if left == right:
            hp.toast(self, "Invalid crop area", f"Left and right values are the same for area={index}", icon="error")
            return False
        if top == bottom:
            hp.toast(self, "Invalid crop area", f"Top and bottom values are the same for area={index}.", icon="error")
            return False
        if bottom - top < 128:
            hp.toast(
                self,
                "Invalid crop area",
                f"The specified top and bottom areas are too small for area={index}. "
                f"They should be larger than 128 pixels.",
                icon="error",
            )
            return False
        if right - left < 128:
            hp.toast(
                self,
                "Invalid crop area",
                f"The specified left and right areas are too small for area={index}. "
                f"They should be larger than 128 pixels.",
                icon="error",
            )
            return False
        return True

    def on_preview(self):
        """Preview image cropping."""
        regions = []
        for index in range(len(self.crop_layer.data)):
            if not self._validate(index):
                return

            # get crop area
            left, right, top, bottom = self._get_crop_area_for_index(index)
            regions.append((left, right, top, bottom))

        if regions:
            self.worker_preview = create_worker(
                preview_regions,
                data_model=self.data_model,
                regions=regions,
                _start_thread=True,
                _connect={
                    "aborted": partial(self._on_aborted, which="preview"),
                    "finished": partial(self._on_finished, which="preview"),
                    "yielded": self._on_preview_yield,
                },
            )
            hp.disable_widgets(self.preview_btn.active_btn, disabled=True)
            self.preview_btn.active = True

    @ensure_main_thread()
    def _on_aborted(self, which: str) -> None:
        """Update CSV."""
        if which == "preview":
            self.worker_preview = None
        else:
            self.worker_crop = None

    @ensure_main_thread()
    def _on_preview_yield(self, args: tuple[str, np.ndarray, int, int]) -> None:
        self.__on_preview_yield(args)

    def __on_preview_yield(self, args: tuple[str, np.ndarray, int, int]) -> None:
        name, array, current, total = args
        self.preview_btn.setRange(0, total)
        self.preview_btn.setValue(current)
        self.view.viewer.add_image(array, name=name)
        self._move_layer(self.view, self.crop_layer)

    @ensure_main_thread()
    def _on_finished(self, which: str) -> None:
        """Failed exporting of the CSV."""
        if which == "preview":
            self.worker_preview = None
            btn = self.preview_btn
        else:
            self.worker_crop = None
            btn = self.crop_btn
        hp.disable_widgets(btn.active_btn, disabled=False)
        btn.active = False

    def on_cancel(self, which: str) -> None:
        """Cancel cropping."""
        if which == "preview" and self.worker_preview:
            self.worker_preview.quit()
            logger.trace("Requested aborting of the preview process.")
        elif which == "crop" and self.worker_crop:
            self.worker_crop.quit()
            logger.trace("Requested aborting of the crop process.")

    def on_crop(self) -> None:
        """Save data."""
        output_dir_ = hp.get_directory(self, "Select output directory", CONFIG.output_dir)
        if not output_dir_:
            return

        regions = []
        for index in range(len(self.crop_layer.data)):
            if not self._validate(index):
                return

            # get crop area
            left, right, top, bottom = self._get_crop_area_for_index(index)
            regions.append((left, right, top, bottom))

        if regions:
            self.worker_crop = create_worker(
                crop_regions,
                data_model=self.data_model,
                output_dir=Path(output_dir_),
                regions=regions,
                _start_thread=True,
                _connect={
                    "aborted": partial(self._on_aborted, which="crop"),
                    "finished": partial(self._on_finished, which="crop"),
                    "yielded": self._on_export_yield,
                    "errored": self._on_export_error,
                },
            )
            hp.disable_widgets(self.crop_btn.active_btn, disabled=True)
            self.crop_btn.active = True

    @ensure_main_thread()
    def _on_export_yield(self, args: tuple[Path, int, int]) -> None:
        filename, current, total = args
        self.crop_btn.setRange(0, total)
        self.crop_btn.setValue(current)
        logger.info(f"Exported {filename}")
        self.statusbar.showMessage(f"Exported {filename} {current}/{total}")

    def _on_export_error(self, exc: Exception) -> None:
        hp.toast(self, "Export failed", f"Failed to export: {exc}", icon="error")
        log_exception_or_error(exc)

    def _get_crop_area(self) -> tuple[int, int, int, int]:
        left = self.left_edit.text()
        left = int(left or 0)  # type: ignore
        top = self.top_edit.text()
        top = int(top or 0)  # type: ignore
        right = self.right_edit.text()
        right = int(right or 0)  # type: ignore
        bottom = self.bottom_edit.text()
        bottom = int(bottom or 0)  # type: ignore
        return left, right, top, bottom  # type: ignore

    def _get_default_crop_area(self) -> tuple[int, int, int, int]:
        (_, y, x) = self.view.viewer.camera.center
        top, bottom = y - 256, y + 256
        left, right = x - 256, x + 256
        return max(0, left), max(0, right), max(0, top), max(0, bottom)

    def on_edit_crop(self):
        """Edit crop area."""
        self.crop_layer.mode = "select"
        if len(self.crop_layer.data) == 0:
            left, right, top, bottom = self._get_default_crop_area()
            rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
            self.crop_layer.data = [(rect, "rectangle")]
        self.crop_layer.selected_data = [0]
        self.crop_layer.mode = "select"

    def on_reset_crop(self):
        """Reset crop area."""
        n = len(self.crop_layer.data)
        if not hp.confirm(
            self,
            f"Are you sure you want to reset the crop areas? There {pluralize('is', n)} currently <b>{n}</b> "
            f"highlighted {pluralize('area', n)}.",
            "Reset crop area",
        ):
            return
        self.crop_layer.data = []
        with hp.qt_signals_blocked(self.left_edit, self.right_edit, self.top_edit):
            left, right, top, bottom = self._get_default_crop_area()
            self.left_edit.setText(f"{left:.0f}")
            self.right_edit.setText(f"{right:.0f}")
            self.top_edit.setText(f"{top:.0f}")
            self.bottom_edit.setText(f"{bottom:.0f}")
        self.on_update_rect_from_ui()
        self.crop_layer.mode = "select"
        self.crop_layer.selected_data = (0,)
        self._move_layer(self.view, self.crop_layer)

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

    def on_update_rect_from_ui(self, _: int | None = None) -> None:
        """Update crop rect."""
        if self.crop_layer.data:
            self.crop_layer.data = []

        left, right, top, bottom = self._get_crop_area()
        rect = np.asarray([[top, left], [top, right], [bottom, right], [bottom, left]])
        with self._editing_crop():
            self.crop_layer.data = [(rect, "rectangle")]
        self._move_layer(self.view, self.crop_layer)
        logger.trace("Updated rectangle (from edit).")

    def on_update_crop_from_canvas(self, _evt: ty.Any = None) -> None:
        """Update crop values."""
        if self._editing:
            return
        n = len(self.crop_layer.data)
        with hp.qt_signals_blocked(self.index_choice):
            self.index_choice.setMaximum(n - 1)
            self.index_choice.setValue(n - 1)
        self._on_update_crop_from_canvas(self.index_choice.value())

    def _get_crop_area_for_index(self, index: int = 0) -> tuple[int, int, int, int]:
        """Return crop area."""
        n = len(self.crop_layer.data)
        if index > n or index < 0:
            return 0, 0, 0, 0
        rect = self.crop_layer.interaction_box(index)
        rect = rect[Box.LINE_HANDLE]
        xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
        xmin = max(0, xmin)
        ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
        ymin = max(0, ymin)
        return floor(xmin), ceil(xmax), floor(ymin), ceil(ymax)

    def _on_update_crop_from_canvas(self, index: int = 0) -> None:
        n = len(self.crop_layer.data)
        if index > n or index < 0:
            with hp.qt_signals_blocked(self.left_edit, self.right_edit, self.top_edit):
                self.left_edit.setText("")
                self.right_edit.setText("")
                self.top_edit.setText("")
                self.bottom_edit.setText("")
            return
        left, right, top, bottom = self._get_crop_area_for_index(index)
        with hp.qt_signals_blocked(self.left_edit, self.right_edit, self.top_edit):
            self.left_edit.setText(str(left))
            self.right_edit.setText(str(right))
            self.top_edit.setText(str(top))
            self.bottom_edit.setText(str(bottom))
        logger.trace("Updated rectangle (from canvas).")

    @property
    def crop_layer(self) -> Shapes:
        """Crop layer."""
        if "Crop rectangle" not in self.view.layers:
            layer = self.view.viewer.add_shapes(
                None,
                edge_width=5,
                name="Crop rectangle",
                face_color="green",
                edge_color="white",
                opacity=0.5,
            )
            visual = self.view.widget.layer_to_visual[layer]
            init_shapes_layer(layer, visual)
            connect(self.crop_layer.events.set_data, self.on_update_crop_from_canvas, state=True)
        return self.view.layers["Crop rectangle"]

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(
            self, add_toolbars=False, allow_extraction=False, disable_controls=True, disable_new_layers=True
        )
        self._image_widget = LoadWidget(self, self.view)

        self.index_choice = hp.make_int_spin_box(
            self,
            -1,
            0,
            tooltip="Index of the drawn shape. Value of -1 means that the last shape is used.",
            func=self.on_update_ui_from_index,
        )

        self.left_edit = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")
        self.top_edit = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")
        self.right_edit = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")
        self.bottom_edit = hp.make_label(self, alignment=Qt.AlignmentFlag.AlignCenter, object_name="crop_label")

        crop_layout = QFormLayout()  # noqa
        crop_layout.addRow(hp.make_label(self, "Crop area"), self.index_choice)
        crop_layout.addRow(
            hp.make_label(self, "Horizontal"),
            hp.make_h_layout(self.left_edit, hp.make_label(self, "-"), self.right_edit),
        )
        crop_layout.addRow(
            hp.make_label(self, "Vertical"),
            hp.make_h_layout(self.top_edit, hp.make_label(self, "-"), self.bottom_edit),
        )

        self.init_btn = hp.make_btn(
            self, "Initialize crop area", tooltip="Edit crop area (interactively)", func=self.on_edit_crop
        )
        self.reset_btn = hp.make_btn(
            self, "Reset crop area", tooltip="Reset crop area to center of the image.", func=self.on_reset_crop
        )

        self.preview_btn = hp.make_active_progress_btn(
            self,
            "Preview",
            tooltip="Preview crop area.",
            func=self.on_preview,
            cancel_func=partial(self.on_cancel, which="preview"),
        )

        self.crop_btn = hp.make_active_progress_btn(
            self,
            "Export to OME-TIFF...",
            tooltip="Export cropped image to OME-TIFF file.",
            func=self.on_crop,
            cancel_func=partial(self.on_cancel, which="crop"),
        )

        side_layout = hp.make_form_layout()
        side_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        hp.style_form_layout(side_layout)
        side_layout.addRow(
            hp.make_btn(self, "Import project...", tooltip="Load previous project", func=self.on_load_from_project)
        )
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line_with_text("Image crop position"))
        side_layout.addRow(crop_layout)
        side_layout.addRow(hp.make_h_layout(self.init_btn, self.reset_btn))
        side_layout.addRow(hp.make_h_line_with_text("Export"))
        side_layout.addRow(self.preview_btn)
        side_layout.addRow(self.crop_btn)
        side_layout.addRow(
            hp.make_btn(
                self,
                "Export project...",
                tooltip="Export configuration to a project file. Information such as image path and crop"
                " information are saved. (This does not save the cropped image)",
                func=self.on_save_to_project,
            )
        )
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
        layout.addLayout(side_layout)
        main_layout = QVBoxLayout(widget)
        main_layout.addLayout(layout)

        # extra settings
        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    def _make_menu(self) -> None:
        """Make menu items."""
        # File menu
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(
            self,
            "Add image (.tiff, .png, .jpg, .imzML, .tdf, .tsf, + others)...",
            "Ctrl+I",
            menu=menu_file,
            func=self._image_widget.on_select_dataset,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(self, "Show scale bar controls...", "Ctrl+S", menu=menu_tools, func=self.on_show_scalebar)
        menu_tools.addSeparator()
        hp.make_menu_item(self, "Show IPython console...", "Ctrl+T", menu=menu_tools, func=self.on_show_console)

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def on_show_scalebar(self):
        """Show scale bar controls for the viewer."""
        from qtextra._napari.common.component_controls.qt_scalebar_controls import QtScaleBarControls

        dlg = QtScaleBarControls(self.view.viewer, self.view.widget)
        dlg.show_below_widget(self._image_widget)

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update({"viewer": self.view.viewer, "data_model": self.data_model})
        return variables

    # noinspection PyAttributeOutsideInit
    @contextmanager
    def _editing_crop(self):
        self._editing = True
        yield
        self._editing = False

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not CONFIG.confirm_close_crop
            or QtConfirmCloseDialog(
                self,
                "confirm_close_crop",
                self.on_save_to_project,
                CONFIG,
            ).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and CONFIG.confirm_close_crop
            and self.data_model.is_valid()
            and QtConfirmCloseDialog(
                self,
                "confirm_close_crop",
                self.on_save_to_project,
                CONFIG,
            ).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        CONFIG.save()
        READER_CONFIG.save()
        evt.accept()


def get_project_data(data_model: DataModel, left: int, right: int, top: int, bottom: int) -> dict:
    """Write project."""
    schema_version = "1.0"
    data_ = data_model.to_dict()
    data = {
        "schema_version": schema_version,
        "tool": "crop",
        "version": __version__,
        "crop": [{"left": left, "right": right, "top": top, "bottom": bottom}],
        "images": data_["images"],
    }
    return data


def crop_regions(
    data_model: DataModel, output_dir: Path, regions: list[tuple[int, int, int, int]]
) -> ty.Generator[tuple[Path, int, int], None, None]:
    """Crop images."""
    from image2image_reader._writer import write_ome_tiff

    n = len(regions)
    for current, (left, right, top, bottom) in enumerate(regions, start=1):
        for path, reader, cropped in data_model.crop(left, right, top, bottom):
            logger.info(f"Exporting {path} with shape {cropped.shape}...")
            output_path = output_dir / f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
            if output_path.with_suffix(".ome.tiff").exists():
                logger.warning(f"Skipping {output_path} as it already exists.")
                yield output_path, current, n
                continue
            filename = write_ome_tiff(output_path, cropped, reader)
            yield Path(filename), current, n


def preview_regions(
    data_model: DataModel, regions: list[tuple[int, int, int, int]]
) -> ty.Generator[tuple[str, np.ndarray, int, int], None, None]:
    """Preview images."""
    n = len(regions)
    for current, (left, right, top, bottom) in enumerate(regions, start=1):
        for path, _reader, cropped in data_model.crop(left, right, top, bottom):
            name = f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
            yield name, cropped, current, n


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="crop", level=0)
