"""Viewer dialog."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.typing import PathLike
from loguru import logger
from napari.layers import Image, Points, Shapes
from napari.utils.events import Event
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QDialog, QHBoxLayout, QVBoxLayout, QWidget
from superqt.utils import qdebounced

from image2image import __version__
from image2image.config import VIEWER_CONFIG
from image2image.enums import ALLOWED_PROJECT_VIEWER_FORMATS
from image2image.qt._dialog_mixins import SingleViewerMixin
from image2image.qt._dialogs._select import LoadWithTransformWidget
from image2image.utils.utilities import ensure_extension

if ty.TYPE_CHECKING:
    from image2image_io.readers import ShapesReader
    from qtextra._napari.image.wrapper import NapariImageView
    from image2image.models.transform import TransformModel

MASK_LAYER_NAME = "Mask"
MASK_FILENAME = "mask.tmp"


class ImageViewerWindow(SingleViewerMixin):
    """Image viewer dialog."""

    APP_NAME = "viewer"

    image_layer: list[Image] | None = None
    shape_layer: list[Shapes] | None = None
    points_layer: list[Points] | None = None
    view: NapariImageView

    def __init__(
        self,
        parent: QWidget | None,
        run_check_version: bool = True,
        image_path: PathLike | list[PathLike] | None = None,
        image_dir: PathLike | None = None,
        **_kwargs: ty.Any,
    ):
        self.CONFIG = VIEWER_CONFIG
        super().__init__(parent, f"image2image: Viewer app (v{__version__})", run_check_version=run_check_version)
        if self.CONFIG.first_time:
            hp.call_later(self, self.on_show_tutorial, 10_000)
        self._setup_config()
        self.on_open_images(image_dir, image_path)

    def on_open_images(
        self, image_dir: PathLike | None = None, image_path: PathLike | list[PathLike] | None = None
    ) -> None:
        """Open images."""
        if image_path:
            if not isinstance(image_path, list):
                image_path = [image_path]
            self._image_widget.on_set_path(image_path)
        if image_dir:
            image_path = list(Path(image_dir).rglob("*"))
            self._image_widget.on_set_path(image_path)

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.view_type = "overlay"
        READER_CONFIG.split_roi = True
        READER_CONFIG.split_rgb = False
        READER_CONFIG.only_last_pyramid = False
        READER_CONFIG.init_pyramid = True

    @property
    def transform_model(self) -> TransformModel:
        """Return transform model."""
        return self._image_widget.transform_model

    def setup_events(self, state: bool = True) -> None:
        """Setup events."""
        # wrapper
        connect(self._image_widget.dataset_dlg.evt_loaded, self.on_load_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_closing, self.on_closing_image, state=state)
        connect(self._image_widget.dataset_dlg.evt_import_project, self._on_load_from_project, state=state)
        connect(self._image_widget.dataset_dlg.evt_export_project, self.on_save_to_project, state=state)
        connect(self._image_widget.dataset_dlg.evt_resolution, self.on_update_transform, state=state)
        connect(self._image_widget.dataset_dlg.evt_resolution, self.on_update_mask_reader, state=state)
        connect(self._image_widget.transform_dlg.evt_transform, self.on_update_transform, state=state)
        connect(self._image_widget.evt_toggle_channel, self.on_toggle_channel, state=state)
        connect(self._image_widget.evt_toggle_all_channels, self.on_toggle_all_channels, state=state)
        # viewer
        connect(self.view.viewer.events.status, self._status_changed, state=state)
        # temporary images
        connect(self._image_widget.evt_update_temp, self.on_plot_temporary, state=state)
        connect(self._image_widget.evt_remove_temp, self.on_remove_temporary, state=state)
        connect(self._image_widget.evt_add_channel, self.on_add_temporary_to_viewer, state=state)

    def on_load_from_project(self, _evt=None):
        """Load a previous project."""
        path_ = hp.get_filename(
            self, "Load i2v project", base_dir=self.CONFIG.output_dir, file_filter=ALLOWED_PROJECT_VIEWER_FORMATS
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: str) -> None:
        if path_:
            from image2image.models.data import load_viewer_setup_from_file
            from image2image.models.utilities import _remove_missing_from_dict

            path = Path(path_)
            if any(v in path.name for v in [".i2r.json", ".i2r.toml"]):
                self._image_widget.transform_dlg._on_add_transform(path)
                return

            self.CONFIG.update(output_dir=str(path.parent))

            # load data from config file
            try:
                paths, paths_missing, transform_data, resolution, reader_kws = load_viewer_setup_from_file(path)
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
            reader_kws = _remove_missing_from_dict(reader_kws, paths)
            # add paths
            if paths:
                self._image_widget.on_set_path(paths, transform_data, resolution, reader_kws)

            # add affine matrices to the transform model
            for name, matrix in transform_data.items():
                self.transform_model.add_transform(name, matrix)

    def on_save_to_project(self) -> None:
        """Export project."""
        model = self.data_model
        if model.n_paths == 0:
            logger.warning("Cannot save project - there are no images loaded.")
            hp.toast(
                self,
                "Cannot save project",
                "Cannot save project - there are no images loaded.",
                icon="warning",
                position="top_left",
            )
            return
        # get filename which is based on the moving dataset
        filename = model.get_filename() + ".i2v.json"
        path_ = hp.get_save_filename(
            self,
            "Save i2v project",
            base_dir=self.CONFIG.output_dir,
            file_filter=ALLOWED_PROJECT_VIEWER_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2v")
            self.CONFIG.update(output_dir=str(path.parent))
            model.to_file(path)
            hp.long_toast(
                self,
                "Exported i2v project",
                f"Saved project to<br><b>{hp.hyper(path)}</b>",
                icon="success",
                position="top_left",
            )

    def on_create_mask(self) -> None:
        """Add shapes mask to the viewer."""
        from image2image_io.readers import ShapesReader
        from qtextra.widgets.qt_pick_option import QtScrollablePickOption

        wrapper = self.data_model.get_wrapper()
        options = get_resolution_options(wrapper)
        if not options:
            hp.toast(
                self,
                "No images loaded",
                "No images loaded. Please load an image first.",
                icon="warning",
                position="top_left",
            )
            return

        dlg = QtScrollablePickOption(
            self,
            "Please select the resolution (pixel size) at which the mask should be created at. This is <b>important</b>"
            " as this will determine the transformation matrix used to export the mask. If you are viewing multiple "
            " images, you should use the resolution of the image you are currently viewing.",
            options,
            orientation="vertical",
        )
        which = None
        if dlg.exec_() == QDialog.DialogCode.Accepted:  # type: ignore[attr-defined]
            which = dlg.option
        if not which:
            hp.toast(
                self,
                "No resolution selected",
                "No resolution selected. Please try again.",
                icon="warning",
                position="top_left",
            )
            return

        scale = (which, which)
        affine = np.eye(3)
        if wrapper:
            if MASK_FILENAME not in wrapper.data:
                reader = ShapesReader.create(MASK_FILENAME, channel_names=[MASK_LAYER_NAME])
                self.data_model.add_paths([MASK_FILENAME])
                wrapper.add(reader)
                self._image_widget.dataset_dlg._on_loaded_dataset(self.data_model, select=False)
            else:
                reader = wrapper.get_reader_for_key(MASK_FILENAME)
                affine = wrapper.get_affine(reader, reader.resolution)
                scale = reader.scale

        if f"{MASK_LAYER_NAME} | mask.tmp" not in self.view.viewer.layers:
            layer = self.view.viewer.add_shapes(
                name=f"{MASK_LAYER_NAME} | mask.tmp",
                face_color="green",
                edge_color="red",
                opacity=0.5,
                edge_width=2,
                scale=scale,
                affine=affine,
            )
            connect(layer.events.set_data, self.on_update_mask_reader, state=True)
        hp.toast(
            self,
            "Change resolution",
            "You can always change the pixel size by going to <b>More options...</b>.",
            icon="info",
            position="top_left",
            duration=10_000,
        )

    def on_update_mask_reader(self, _event: Event | None = None) -> None:
        """Update reader based on layer data."""
        from image2image_io.readers.shapes_reader import napari_to_shapes_data

        wrapper = self.data_model.wrapper
        layer: Shapes = self.view.get_layer(f"{MASK_LAYER_NAME} | mask.tmp")
        if wrapper and layer:
            try:
                reader: ShapesReader = wrapper.get_reader_for_key(MASK_FILENAME)
                if reader:
                    reader._channel_names = [MASK_LAYER_NAME]
                    reader.shape_data = napari_to_shapes_data(layer.name, layer.data, layer.shape_type)
            except KeyError:
                return

    def on_save_masks(self) -> None:
        """Export masks."""
        # Ask user which layer(s) to export (select layer(s) from list) - only shapes
        # Specify output dimensions (select layer(s) from list) - only images
        from image2image.qt._viewer._mask import MasksDialog

        dlg = MasksDialog(self)
        dlg.show()

    def _setup_ui(self):
        """Create panel."""
        self.view = self._make_image_view(
            self, add_toolbars=False, allow_extraction=False, disable_controls=True, disable_new_layers=True
        )
        self.view.viewer.scale_bar.unit = "um"
        self.view.widget.canvas.events.key_press.connect(self.keyPressEvent)

        side_widget = QWidget()
        side_widget.setMinimumWidth(400)
        side_widget.setMaximumWidth(400)

        self._image_widget = LoadWithTransformWidget(
            self,
            self.view,
            self.CONFIG,
            allow_geojson=True,
            project_extension=[".i2v.json", ".i2v.toml", ".i2r.json", ".i2r.toml"],
            allow_iterate=True,
            allow_import_project=True,
            allow_export_project=True,
        )

        self.create_mask_btn = hp.make_btn(
            side_widget,
            "Create mask",
            tooltip="Create mask using shapes. The mask can be subsequently exported as a HDF5 file.",
            func=self.on_create_mask,
        )
        self.export_mask_btn = hp.make_btn(
            side_widget,
            "Export GeoJSON/shape masks...",
            tooltip="Export masks/regions in a AutoIMS compatible format",
            func=self.on_save_masks,
        )

        side_layout = hp.make_form_layout(side_widget)
        hp.style_form_layout(side_layout)
        side_layout.addRow(self._image_widget)
        side_layout.addRow(hp.make_h_line_with_text("Masks"))
        side_layout.addRow(self.create_mask_btn)
        side_layout.addRow(self.export_mask_btn)
        side_layout.addRow(hp.make_h_line_with_text("Layer controls"))
        side_layout.addRow(self.view.widget.controls)
        side_layout.addRow(self.view.widget.layerButtons)
        side_layout.addRow(self.view.widget.layers)
        side_layout.addRow(self.view.widget.viewerButtons)

        layout = QHBoxLayout()
        layout.addWidget(self.view.widget, stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addWidget(side_widget)

        widget = QWidget()
        self.setCentralWidget(widget)

        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(layout, stretch=True)

        # extra settings
        self._make_menu()
        self._make_icon()
        self._make_statusbar()

    # def _make_focus_layout(self) -> QFormLayout:
    #     self.lock_btn = hp.make_lock_btn(
    #         self,
    #         func=self.on_lock,
    #         normal=True,
    #         tooltip="Lock the area of interest. Press <b>L</b> on your keyboard to lock.",
    #     )
    #     # self.set_current_focus_btn = hp.make_btn(self, "Set current range", func=self.on_set_focus)
    #     self.x_center = hp.make_double_spin_box(
    #         self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the x-axis."
    #     )
    #     self.y_center = hp.make_double_spin_box(
    #         self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the y-axis."
    #     )
    #     self.zoom = hp.make_double_spin_box(self, -1e5, 1e5, step_size=0.5, n_decimals=4, tooltip="Zoom factor.")
    #     self.use_focus_btn = hp.make_btn(
    #         self,
    #         "Zoom-in",
    #         func=self.on_apply_focus,
    #         tooltip="Zoom-in to an area of interest. Press <b>Z</b> on your keyboard to zoom-in.",
    #     )
    #
    #     layout = hp.make_form_layout()
    #     hp.style_form_layout(layout)
    #     layout.addRow(hp.make_label(self, "Center (x)"), self.x_center)
    #     layout.addRow(hp.make_label(self, "Center (y)"), self.y_center)
    #     layout.addRow(hp.make_label(self, "Zoom"), self.zoom)
    #     layout.addRow(hp.make_h_layout(self.lock_btn, self.use_focus_btn, stretch_id=1))
    #     return layout

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update(
            {
                "transforms_model": self.transform_model,
                "viewer": self.view.viewer,
                "data_model": self.data_model,
                "wrapper": self.data_model.wrapper,
            }
        )
        return variables

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_viewer_tutorial

        show_viewer_tutorial(self)
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
        if key == Qt.Key.Key_4:
            self.on_toggle_zoom()
            ignore = True
        return ignore


def get_resolution_options(wrapper) -> dict[str, str]:
    """Get resolution options."""
    resolutions: dict[float, list[str]] = {}
    for reader in wrapper.reader_iter():
        if reader.reader_type != "image":
            continue
        if reader.resolution not in resolutions:
            resolutions[reader.resolution] = []
        resolutions[reader.resolution].append(reader.name)
    options = {}
    for resolution, names in resolutions.items():
        datasets = ", ".join(names)
        if len(datasets) > 120:
            datasets = f"{datasets[:120]}..."
        options[resolution] = f"{resolution:.3f}µm\nLike: {datasets}"
    return options


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="viewer", level=0)
