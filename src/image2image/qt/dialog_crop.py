"""Viewer dialog."""
from __future__ import annotations

import typing as ty
from contextlib import contextmanager
from math import ceil, floor
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_reader.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Shapes
from napari.layers.shapes._shapes_constants import Box
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator
from qtpy.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QMenuBar, QVBoxLayout, QWidget
from superqt import ensure_main_thread
from superqt.utils import create_worker

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

    def on_toggle_all_channels(self, state: bool) -> None:
        """Toggle channel."""
        self._toggle_all_channels(self.data_model, self.view, state, "view")

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
        # self.on_indicator("fixed", False)

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

    def _validate(self) -> bool:
        if not self.data_model.is_valid():
            hp.toast(self, "Invalid data", "Data is invalid.", icon="error")
            return False
        left, right, top, bottom = self._get_crop_area()
        if left == right:
            hp.toast(self, "Invalid crop area", "Left and right values are the same.", icon="error")
            return False
        if top == bottom:
            hp.toast(self, "Invalid crop area", "Top and bottom values are the same.", icon="error")
            return False
        if bottom - top < 128:
            hp.toast(
                self,
                "Invalid crop area",
                "The specified top and bottom areas are too small. They should be larger than 128 pixels.",
                icon="error",
            )
            return False
        if right - left < 128:
            hp.toast(
                self,
                "Invalid crop area",
                "The specified left and right areas are too small. They should be larger than 128 pixels.",
                icon="error",
            )
            return False
        return True

    def on_preview(self):
        """Preview image cropping."""
        if not self._validate():
            return

        left, right, top, bottom = self._get_crop_area()
        create_worker(
            self._on_preview,
            data_model=self.data_model,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
            _start_thread=True,
            _connect={
                "yielded": self._on_preview_yield,
            },
        )

    @staticmethod
    def _on_preview(
        data_model: DataModel, left: int, right: int, top: int, bottom: int
    ) -> ty.Generator[tuple[str, np.ndarray], None, None]:
        for path, _reader, cropped in data_model.crop(left, right, top, bottom):
            name = f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
            yield name, cropped

    def _on_preview_yield(self, args: tuple[str, np.ndarray]) -> None:
        name, array = args
        self.view.viewer.add_image(array, name=name)
        self._move_layer(self.view, self.crop_layer)

    def on_crop(self) -> None:
        """Save data."""
        if not self._validate():
            return

        left, right, top, bottom = self._get_crop_area()
        output_dir_ = hp.get_directory(self, "Select output directory", CONFIG.output_dir)
        if output_dir_:
            create_worker(
                self._on_export,
                data_model=self.data_model,
                output_dir=Path(output_dir_),
                left=left,
                right=right,
                top=top,
                bottom=bottom,
                _start_thread=True,
                _connect={
                    "started": lambda: hp.toast(self, "Export started", "Export started.", icon="info"),
                    "yielded": self._on_export_yield,
                    "errored": self._on_export_error,
                },
            )

    @staticmethod
    def _on_export(
        data_model: DataModel, output_dir: Path, left: int, right: int, top: int, bottom: int
    ) -> ty.Generator[Path, None, None]:
        from image2image_reader._writer import write_ome_tiff

        for path, reader, cropped in data_model.crop(left, right, top, bottom):
            logger.info(f"Exporting {path} with shape {cropped.shape}...")
            output_path = output_dir / f"{path.stem}_x={left}-{right}_y={top}-{bottom}".replace(".ome", "")
            filename = write_ome_tiff(output_path, cropped, reader)
            yield Path(filename)

    def _on_export_yield(self, filename: Path) -> None:
        logger.info(f"Exported {filename}")
        self.statusbar.showMessage(f"Exported {filename}")

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

    def on_update_rect_from_ui(self, _: ty.Optional[int] = None) -> None:
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
        if n == 0:
            return
        if n > 1:
            hp.toast(
                self,
                "Multiple rectangles detected!",
                "There are more than one crop rectangles. Only the first one will be used. Please remove "
                " all the others as they won't be used.",
                icon="error",
            )
        rect = self.crop_layer.interaction_box(0)
        rect = rect[Box.LINE_HANDLE]
        xmin, xmax = np.min(rect[:, 1]), np.max(rect[:, 1])
        xmin = max(0, xmin)
        ymin, ymax = np.min(rect[:, 0]), np.max(rect[:, 0])
        ymin = max(0, ymin)
        with hp.qt_signals_blocked(self.left_edit, self.right_edit, self.top_edit):
            self.left_edit.setText(str(floor(xmin)))
            self.right_edit.setText(str(ceil(xmax)))
            self.top_edit.setText(str(floor(ymin)))
            self.bottom_edit.setText(str(ceil(ymax)))
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
            connect(self.crop_layer.events.data, self.on_update_crop_from_canvas, state=True)
        return self.view.layers["Crop rectangle"]

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(self, add_toolbars=False, allow_extraction=False, disable_controls=True)
        self._image_widget = LoadWidget(self, self.view)

        # self.index_choice = hp.make_int_spin_box(
        #     self,
        #     -1,
        #     0,
        #     tooltip="Index of the drawn shape. Value of -1 means that the last shape is used.",
        #     func=self.on_update_rect_from_ui,
        # )

        self.left_edit = hp.make_line_edit(
            self, placeholder="Left", validator=QIntValidator(0, 75_000), func=self.on_update_rect_from_ui
        )
        self.left_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.top_edit = hp.make_line_edit(
            self, placeholder="Top", validator=QIntValidator(0, 75_000), func=self.on_update_rect_from_ui
        )
        self.top_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.right_edit = hp.make_line_edit(
            self, placeholder="Right", validator=QIntValidator(0, 75_000), func=self.on_update_rect_from_ui
        )
        self.right_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]
        self.bottom_edit = hp.make_line_edit(
            self, placeholder="Bottom", validator=QIntValidator(0, 75_000), func=self.on_update_rect_from_ui
        )
        self.bottom_edit.setAlignment(Qt.AlignCenter)  # type: ignore[attr-defined]

        crop_layout = QFormLayout()  # noqa
        # crop_layout.addRow(hp.make_label(self, "Shape index"), self.index_choice)
        crop_layout.addRow(
            hp.make_label(self, "Horizontal"),
            hp.make_h_layout(self.left_edit, hp.make_label(self, "-"), self.right_edit),
        )
        crop_layout.addRow(
            hp.make_label(self, "Vertical"),
            hp.make_h_layout(self.top_edit, hp.make_label(self, "-"), self.bottom_edit),
        )

        self.edit_btn = hp.make_btn(
            self, "Initialize crop area", tooltip="Edit crop area (interactively)", func=self.on_edit_crop
        )
        self.reset_btn = hp.make_btn(
            self, "Reset crop area", tooltip="Reset crop area to center of the image.", func=self.on_reset_crop
        )
        self.sync_btn = hp.make_btn(
            self,
            "Synchronize",
            tooltip="Synchronize the extents of the first shape and the values displayed above. Sometimes (rarely),"
            " these are out of sync.",
            func=self.on_update_crop_from_canvas,
        )

        side_layout = hp.make_form_layout()
        hp.style_form_layout(side_layout)
        side_layout.addRow(
            hp.make_btn(self, "Import project...", tooltip="Load previous project", func=self.on_load_from_project)
        )
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line_with_text("Image crop position"))
        side_layout.addRow(crop_layout)
        side_layout.addRow(hp.make_h_layout(self.edit_btn, self.sync_btn, self.reset_btn))
        side_layout.addRow(hp.make_h_line_with_text("Export"))
        side_layout.addRow(hp.make_btn(self, "Preview", tooltip="Preview crop area.", func=self.on_preview))
        side_layout.addRow(
            hp.make_btn(
                self, "Export to OME-TIFF...", tooltip="Export cropped image to OME-TIFF file.", func=self.on_crop
            )
        )
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

        # Help menu
        menu_help = self._make_help_menu()

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(menu_help.menuAction())
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


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="crop", level=0)
