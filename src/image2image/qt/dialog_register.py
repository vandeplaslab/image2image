"""Registration dialog."""
from __future__ import annotations

import typing as ty
from contextlib import contextmanager, suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from image2image_io.enums import ViewType
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image
from napari.layers.points.points import Mode, Points
from napari.layers.utils._link_layers import link_layers
from napari.utils.events import Event
from qtextra._napari.image.wrapper import NapariImageView
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtextra.widgets.qt_image_button import QtImagePushButton
from qtextra.widgets.qt_mini_toolbar import QtMiniToolbar
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QMenuBar, QSizePolicy, QVBoxLayout, QWidget
from superqt.utils import ensure_main_thread, qdebounced

from image2image import __version__
from image2image.config import CONFIG
from image2image.enums import ALLOWED_EXPORT_REGISTER_FORMATS, ALLOWED_IMPORT_REGISTER_FORMATS
from image2image.models.data import DataModel
from image2image.models.transformation import Transformation
from image2image.qt._dialogs._select import FixedWidget, MovingWidget
from image2image.qt.dialog_base import Window
from image2image.utils.utilities import (
    _get_text_data,
    _get_text_format,
    ensure_extension,
    get_colormap,
    init_points_layer,
)

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform

    from image2image.qt._dialogs import FiducialsDialog


def has_any_points(*layers: Points) -> bool:
    """Return True if any of the layers has any points."""
    return any(layer.data.shape[0] > 0 for layer in layers if layer is not None)


def get_error_state(n_fixed: int, transform_model: Transformation) -> tuple[str, str]:
    """Retrieve error state based on the transformation model."""
    error_label = "<need more points>"
    error_style = "reg_error"
    if n_fixed > 4:
        error = transform_model.error()
        if transform_model.moving_model:
            # success error is half of the spatial resolution
            success_error = transform_model.moving_model.min_resolution * 0.5
            # acceptable error is 2/3 of the spatial resolution
            acceptable_error = transform_model.moving_model.min_resolution * 0.67
            error_label = f"{error:.2f}"
            error_style = "reg_error"
            if error < 0.01:
                error_label += " (unlikely)"
                error_style = "reg_error"
            elif error < success_error:
                error_label += " (very good)"
                error_style = "reg_success"
            elif error < acceptable_error:
                error_label += " (decent)"
                error_style = "reg_warning"
    return error_label, error_style


def get_random_image(array: list[np.ndarray]) -> list[np.ndarray]:
    """Retrieve random image."""
    array_ = array[0]
    nan_mask = np.isnan(array_)
    fill_value = np.nan
    if not np.any(nan_mask):
        nan_mask = array_ == 0
    array_ = np.random.randint(128, 255, array_.shape)
    if np.any(nan_mask):
        array_ = array_.astype(np.float32) / 255
        array_[nan_mask] = fill_value

    return [array_]


class ImageRegistrationWindow(Window):
    """Image registration dialog."""

    view_fixed: NapariImageView
    view_moving: NapariImageView
    fixed_image_layer: list[Image] | None = None
    moving_image_layer: list[Image] | None = None
    _fiducials_dlg, _console = None, None
    _zooming = False
    _current_index = -1

    # events
    evt_predicted = Signal()
    evt_fixed_dropped = Signal("QEvent")
    evt_moving_dropped = Signal("QEvent")

    def __init__(self, parent: QWidget | None):
        super().__init__(parent, f"image2register: Simple image registration tool (v{__version__})", delay_events=True)
        self.transform_model = Transformation(
            fixed_model=self.fixed_model,
            moving_model=self.moving_model,
            fixed_points=self.fixed_points_layer.data,
            moving_points=self.moving_points_layer.data,
        )
        if CONFIG.first_time_register:
            hp.call_later(self, self.on_show_tutorial, 10_000)

    @contextmanager
    def zooming(self) -> ty.Generator[None, None, None]:
        """Context manager to set editing."""
        self._zooming = True
        yield
        self._zooming = False

    @property
    def transform(self) -> ProjectiveTransform | None:
        """Retrieve transform."""
        transform = self.transform_model
        if transform.is_valid():
            return transform.transform
        return None

    @property
    def transformed_moving_image_layer(self) -> Image | None:
        """Return transformed, moving image layer."""
        if "Transformed" in self.view_fixed.layers:
            return self.view_fixed.layers["Transformed"]
        return None

    @property
    def fixed_points_layer(self) -> Points:
        """Fixed points layer."""
        if "Fixed (points)" not in self.view_fixed.layers:
            layer = self.view_fixed.viewer.add_points(  # noqa
                None,
                size=self.fixed_point_size.value(),
                name="Fixed (points)",
                face_color="green",
                edge_color="white",
                symbol="ring",
            )
            visual = self.view_fixed.widget.layer_to_visual[layer]
            init_points_layer(layer, visual, False)
            connect(layer.events.data, self.on_run, state=True)
            connect(layer.events.data, self.fiducials_dlg.on_load, state=True)
            connect(layer.events.add_point, partial(self.on_predict, "fixed"), state=True)
        return self.view_fixed.layers["Fixed (points)"]

    @property
    def moving_points_layer(self) -> Points:
        """Fixed points layer."""
        if "Moving (points)" not in self.view_moving.layers:
            layer = self.view_moving.viewer.add_points(  # noqa
                None,
                size=self.moving_point_size.value(),
                name="Moving (points)",
                face_color="green",
                edge_color="white",
                symbol="ring",
            )
            visual = self.view_moving.widget.layer_to_visual[layer]
            init_points_layer(layer, visual, True)
            connect(layer.events.data, self.on_run, state=True)
            connect(layer.events.data, self.fiducials_dlg.on_load, state=True)
            connect(layer.events.add_point, partial(self.on_predict, "moving"), state=True)
        return self.view_moving.layers["Moving (points)"]

    def setup_events(self, state: bool = True) -> None:
        """Additional setup."""
        # fixed widget
        connect(self._fixed_widget.dataset_dlg.evt_project, self._on_load_from_project, state=state)
        connect(self._fixed_widget.dataset_dlg.evt_loading, partial(self.on_indicator, which="fixed"), state=state)
        connect(self._fixed_widget.dataset_dlg.evt_loaded, self.on_load_fixed, state=state)
        connect(self._fixed_widget.dataset_dlg.evt_closed, self.on_close_fixed, state=state)
        connect(self._fixed_widget.evt_toggle_channel, partial(self.on_toggle_channel, which="fixed"), state=state)
        connect(
            self._fixed_widget.evt_toggle_all_channels, partial(self.on_toggle_all_channels, which="fixed"), state=state
        )
        connect(self._fixed_widget.evt_swap, self.on_swap, state=state)

        # moving widget
        connect(self._moving_widget.dataset_dlg.evt_project, self._on_load_from_project, state=state)
        connect(self._moving_widget.dataset_dlg.evt_loading, partial(self.on_indicator, which="moving"), state=state)
        connect(self._moving_widget.dataset_dlg.evt_loaded, self.on_load_moving, state=state)
        connect(self._moving_widget.dataset_dlg.evt_closed, self.on_close_moving, state=state)
        connect(self._moving_widget.evt_toggle_channel, partial(self.on_toggle_channel, which="moving"), state=state)
        connect(self._moving_widget.evt_show_transformed, self.on_toggle_transformed_moving, state=state)
        connect(
            self._moving_widget.evt_toggle_all_channels,
            partial(self.on_toggle_all_channels, which="moving"),
            state=state,
        )
        connect(self._moving_widget.evt_swap, self.on_swap, state=state)
        connect(self._moving_widget.evt_view_type, self.on_change_view_type, state=state)
        # views
        connect(
            self.view_fixed.viewer.camera.events,
            # partial(self.on_sync_views, which="fixed"),
            self.on_sync_views_fixed,
            state=True,
        )
        connect(
            self.view_moving.viewer.camera.events,
            # partial(self.on_sync_views, which="moving"),
            self.on_sync_views_moving,
            state=True,
        )
        logger.trace("Connected events...")

    def on_indicator(self, which: str, state: bool = True) -> None:
        """Set indicator."""
        indicator = self.moving_indicator if which == "moving" else self.fixed_indicator
        indicator.setVisible(state)

    def on_close_fixed(self, model: DataModel) -> None:
        """Close fixed image."""
        self._close_model(model, self.view_fixed, "fixed view")

    @property
    def fixed_model(self) -> DataModel:
        """Return transform model."""
        return self._fixed_widget.model

    @property
    def moving_model(self) -> DataModel:
        """Return transform model."""
        return self._moving_widget.model

    def _on_add_channel(self, which: str, name: str) -> None:
        """Add the missing channel if it's available in the reader/wrapper."""
        if which == "fixed":
            self._plot_fixed_layers([name])
        else:
            self._plot_moving_layers([name])

    @ensure_main_thread
    def on_load_fixed(self, model: DataModel, channel_list: list[str]) -> None:
        """Load fixed image."""
        if model and model.n_paths:
            self._on_load_fixed(model, channel_list)
            hp.toast(
                self,
                "Loaded fixed data",
                f"Loaded fixed model with {model.n_paths} paths.",
                icon="success",
                position="top_left",
            )
        else:
            logger.warning(f"Failed to load fixed data - model={model}")
        self.on_indicator("fixed", False)

    def _on_load_fixed(self, model: DataModel, channel_list: list[str] | None = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading fixed data with {model.n_paths} paths...")
            self._plot_fixed_layers(channel_list)
            self.view_fixed.viewer.reset_view()
        logger.info(f"Loaded fixed data in {timer()}")

    def _plot_fixed_layers(self, channel_list: list[str] | None = None) -> None:
        self.fixed_image_layer, _ = self._plot_image_layers(
            self.fixed_model, self.view_fixed, channel_list, "fixed view"
        )
        if isinstance(self.fixed_image_layer, list) and len(self.fixed_image_layer) > 1:
            link_layers(self.fixed_image_layer, attributes=("opacity",))

    def on_toggle_channel(self, name: str, state: bool, which: str) -> None:
        """Toggle channel."""
        view = self.view_fixed if which == "fixed" else self.view_moving
        if name not in view.layers:
            logger.warning(f"Layer '{name}' not found in the {which} view.")
            self._on_add_channel(which, name)
            return
        view.layers[name].visible = state

    def on_toggle_all_channels(self, state: bool, channel_names: str | None, which: str) -> None:
        """Toggle channel."""
        view = self.view_fixed if which == "fixed" else self.view_moving
        for layer in view.layers:
            if isinstance(layer, Image):
                layer.visible = state

    def on_close_moving(self, model: DataModel) -> None:
        """Close moving image."""
        self._close_model(model, self.view_moving, "moving view")
        if self.transformed_moving_image_layer:
            self.view_fixed.layers.remove(self.transformed_moving_image_layer)

    @ensure_main_thread
    def on_load_moving(self, model: DataModel, channel_list: list[str]) -> None:
        """Open modality."""
        if model and model.n_paths:
            self._on_load_moving(model, channel_list)
            hp.toast(
                self,
                "Loaded moving data",
                f"Loaded moving model with {model.n_paths} paths.",
                icon="success",
                position="top_left",
            )
        else:
            logger.warning(f"Failed to load moving data - model={model}")
        self.on_indicator("moving", False)

    def _on_load_moving(self, model: DataModel, channel_list: list[str] | None = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading moving data with {model.n_paths} paths...")
            self._plot_moving_layers(channel_list)
            self.on_apply(update_data=True)
            self.view_moving.viewer.reset_view()
        logger.info(f"Loaded moving data in {timer()}")

    def _plot_moving_layers(self, channel_list: list[str] | None = None) -> None:
        READER_CONFIG.view_type = ViewType(READER_CONFIG.view_type)
        is_overlay = READER_CONFIG.view_type == ViewType.OVERLAY
        wrapper = self.moving_model.wrapper
        if not wrapper:
            return
        if channel_list is None:
            channel_list = wrapper.channel_names()

        moving_image_layer = []
        for index, (name, array, _) in enumerate(wrapper.channel_image_for_channel_names_iter(channel_list)):
            initial_affine = (
                self.transform_model.moving_initial_affine
                if self.transform_model.moving_initial_affine is not None
                else np.eye(3)
            )
            if not is_overlay:
                array = get_random_image(array)
            logger.trace(f"Adding '{name}' to moving view...")
            with MeasureTimer() as timer:
                colormap = get_colormap(index, self.view_moving.layers) if is_overlay else "turbo"
                is_visible = True if (is_overlay and index == 0) else (not is_overlay)
                if name in self.view_moving.layers:
                    is_visible = self.view_moving.layers[name].visible
                    del self.view_moving.layers[name]
                moving_image_layer.append(
                    self.view_moving.viewer.add_image(
                        array,
                        name=name,
                        blending="additive",
                        colormap=colormap,
                        visible=is_visible and name in channel_list,
                        affine=initial_affine,
                    )
                )
            logger.trace(f"Added '{name}' to fixed view in {timer()}.")
        # hide away other layers if user selected 'random' view
        if READER_CONFIG.view_type == ViewType.RANDOM:
            for index, layer in enumerate(moving_image_layer):
                if index > 0:
                    layer.visible = False
        self.moving_image_layer = moving_image_layer

    def on_change_view_type(self, _view_type: str) -> None:
        """Change view type."""
        if self.moving_model.n_paths:
            channel_list = self._moving_widget.channel_dlg.channel_list()
            self._plot_moving_layers(channel_list)
            self.on_apply(update_data=True)

    def on_swap(self, key: str, source: str) -> None:
        """Swap fixed and moving images."""
        # # swap image from 'fixed' to 'moving' or vice versa
        # if source == "fixed":
        #     wrapper = self.fixed_model.wrapper
        #     if not wrapper:
        #         return
        #     wrapper = wrapper.pop

    def on_toggle_transformed_moving(self, value: str) -> None:
        """Toggle visibility of transformed moving image."""
        self.on_apply(update_data=True, name=value)

    def _select_layer(self, which: str) -> None:
        """Select layer."""
        view, layer = (
            (self.view_fixed, self.fixed_points_layer)
            if which == "fixed"
            else (self.view_moving, self.moving_points_layer)
        )
        self._move_layer(view, layer)

    def _get_mode_button(self, which: str, mode: Mode) -> QtImagePushButton | None:
        if which == "fixed":
            widgets = {
                Mode.ADD: self.fixed_add_btn,
                Mode.SELECT: self.fixed_move_btn,
                Mode.PAN_ZOOM: self.fixed_zoom_btn,
            }
        else:
            widgets = {
                Mode.ADD: self.moving_add_btn,
                Mode.SELECT: self.moving_move_btn,
                Mode.PAN_ZOOM: self.moving_zoom_btn,
            }
        return widgets.get(mode, None)

    def on_mode(self, which: str, evt: ty.Any = None, mode: Mode | None = None) -> None:
        """Update mode."""
        widget = self._get_mode_button(which, mode or evt.mode)
        if widget:
            widget.setChecked(True)

    def on_panzoom(self, which: str, _evt: ty.Any = None) -> None:
        """Switch to `panzoom` tool."""
        self._select_layer(which)
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "pan_zoom"

    def on_add(self, which: str, _evt: ty.Any = None) -> None:
        """Add point to the image."""
        self._select_layer(which)
        # extract button and layer based on the appropriate mode
        widget = self.fixed_add_btn if which == "fixed" else self.moving_add_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # make sure the 'add' mode is active
        layer.mode = "add" if widget.isChecked() else "pan_zoom"

    def on_move(self, which: str, _evt: ty.Any = None) -> None:
        """Move points."""
        self._select_layer(which)
        widget = self.fixed_move_btn if which == "fixed" else self.moving_move_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "select" if widget.isChecked() else "pan_zoom"
        layer.selected_data = []

    def on_activate_initial(self):
        """Activate initial button."""
        hp.disable_widgets(self.initial_btn, disabled=has_any_points(self.moving_points_layer))

    def on_remove(self, which: str, _evt: ty.Any = None) -> None:
        """Remove point to the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # there is no data to remove
        if layer.data.shape[0] == 0:
            return

        data = layer.data
        layer.data = np.delete(data, -1, 0)
        self.fiducials_dlg.on_load()
        self.on_run()

    def on_remove_selected(self, which: str, _evt: ty.Any = None) -> None:
        """Remove selected points from the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # there is no data to remove
        if layer.data.shape[0] == 0:
            return
        layer.remove_selected()
        self.fiducials_dlg.on_load()
        self.on_run()

    def on_clear(self, which: str, force: bool = True) -> None:
        """Remove point to the image."""
        if force or hp.confirm(self, "Are you sure you want to remove all data points from the points layer?"):
            layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
            layer.data = np.zeros((0, 2))
            self.evt_predicted.emit()  # noqa
            self.on_clear_transformation()
            self.fiducials_dlg.on_load()
            self.on_run()

    def on_clear_transformation(self) -> None:
        """Clear transformation and remove image."""
        if self.transform_model.is_valid():
            self.transform_model.clear(clear_model=False, clear_initial=False)
        if self.transformed_moving_image_layer:
            with suppress(ValueError):
                self.view_fixed.layers.remove(self.transformed_moving_image_layer)

    @ensure_main_thread
    def on_run(self, _evt: ty.Any = None) -> None:
        """Compute transformation."""
        self._on_run()

    def _on_run(self) -> None:
        if not self.fixed_points_layer or not self.moving_points_layer:
            return

        # execute transform calculation
        n_fixed = len(self.fixed_points_layer.data)
        n_moving = len(self.moving_points_layer.data)
        self.on_activate_initial()
        if 3 <= n_fixed == n_moving >= 3:
            method = "affine"
            self.transform_model.transformation_type = method
            self.transform_model.time_created = datetime.now()
            self.transform_model.fixed_points = self.fixed_points_layer.data
            self.transform_model.moving_points = self.moving_points_layer.data
            self.transform_model.transform = self.transform_model.compute()
            error_label, error_style = get_error_state(n_fixed, self.transform_model)
            self.transform_error.setText(error_label)
            hp.update_widget_style(self.transform_error, error_style)
            self.transform_info.setText(self.transform_model.about("\n", error=False, n=False, split_by_dim=True))
            logger.info(self.transform_model.about("; "))
            self.on_apply()
        else:
            if n_fixed <= 3 or n_moving <= 3:
                logger.warning("There must be at least three points before we can compute the transformation.")
                self.transform_error.setText("<need more points>")
                hp.update_widget_style(self.transform_error, "reg_error")
                self.transform_info.setText("Need at least three points to compute the transformation.")
                self.transform_model.clear(clear_model=False, clear_initial=False)
            elif n_fixed != n_moving:
                logger.warning("The number of `fixed` and `moving` points must be equal.")
                self.transform_error.setText("<need more points>")
                self.transform_info.setText("The number of `fixed` and `moving` points must be equal.")

    def on_save_to_project(self, _evt: ty.Any = None) -> None:
        """Export transformation."""
        transform = self.transform_model
        if not transform.is_valid():
            logger.warning("Cannot save transformation - no transformation has been computed.")
            hp.warn(self, "Cannot save transformation - no transformation has been computed.")
            return
        is_recommended, reason = transform.is_recommended()
        if not is_recommended:
            hp.warn(self, f"Saving transformations in this state is not recommended.<br><br>Reason<br>{reason}")

        # get filename which is based on the moving dataset
        filename = self.moving_model.get_filename() + "_transform.i2r.json"
        path_ = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_EXPORT_REGISTER_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2r")
            CONFIG.output_dir = str(path.parent)
            transform.to_file(path)
            hp.toast(
                self,
                "Exported transformation",
                f"Saved project to<br><b>{hp.hyper(path)}</b>",
                icon="success",
                position="top_left",
            )

    def on_load_from_project(self, _evt: ty.Any = None) -> None:
        """Import transformation."""
        path_ = hp.get_filename(
            self, "Load transformation", base_dir=CONFIG.output_dir, file_filter=ALLOWED_IMPORT_REGISTER_FORMATS
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: str) -> None:
        if path_:
            from image2image.models.transformation import load_transform_from_file
            from image2image.qt._dialogs import ImportSelectDialog, LocateFilesDialog

            # load transformation
            path = Path(path_)
            CONFIG.output_dir = str(path.parent)

            # get info on which settings should be imported
            dlg = ImportSelectDialog(self)
            if dlg.exec_():  # type: ignore[attr-defined]
                config = dlg.config
                logger.trace(f"Loaded configuration from {path}\n{config}")

                # reset all widgets
                if config["fixed_image"]:
                    self._fixed_widget.dataset_dlg.on_close_dataset(force=True)
                if config["moving_image"]:
                    self._moving_widget.dataset_dlg.on_close_dataset(force=True)

                # load data from config file
                try:
                    (
                        transformation_type,
                        fixed_paths,
                        fixed_paths_missing,
                        fixed_points,
                        moving_paths,
                        moving_paths_missing,
                        moving_points,
                        _fixed_resolution,
                        _moving_resolution,
                    ) = load_transform_from_file(path, **config)
                except (ValueError, KeyError) as e:
                    hp.warn(self, f"Failed to load transformation from {path}\n{e}", "Failed to load transformation")
                    return

                # locate paths that are missing
                if fixed_paths_missing or moving_paths_missing:
                    locate_dlg = LocateFilesDialog(
                        self,
                        fixed_paths_missing,  # type: ignore[arg-type]
                        moving_paths_missing,
                    )
                    if locate_dlg.exec_():  # type: ignore[attr-defined]
                        if fixed_paths_missing:
                            fixed_paths = locate_dlg.fix_missing_paths(  # type: ignore[assignment]
                                fixed_paths_missing,
                                fixed_paths,  # type: ignore[arg-type]
                            )
                        if moving_paths_missing:
                            moving_paths = locate_dlg.fix_missing_paths(  # type: ignore[assignment]
                                moving_paths_missing,
                                moving_paths,  # type: ignore[arg-type]
                            )
                # reset initial transform
                self.transform_model.clear(clear_data=False, clear_model=False, clear_initial=True)
                # set new paths
                if fixed_paths:
                    self._fixed_widget.on_set_path(fixed_paths)
                if moving_paths:
                    self._moving_widget.on_set_path(moving_paths)
                # update points
                if moving_points is not None:
                    self._update_layer_points(self.moving_points_layer, moving_points, block=False)
                if fixed_points is not None:
                    self._update_layer_points(self.fixed_points_layer, fixed_points, block=False)
                if moving_points is not None and fixed_points is not None:
                    self.fiducials_dlg.on_load()  # update table
                    self.on_run()
                # force update of the text
                self.on_update_text(block=False)

    @property
    def fiducials_dlg(self) -> FiducialsDialog:
        """Return fiducials dialog."""
        if self._fiducials_dlg is None:
            from image2image.qt._dialogs import FiducialsDialog

            self._fiducials_dlg = FiducialsDialog(self)
            self._fiducials_dlg.evt_update.connect(self.on_run)
        return self._fiducials_dlg

    def on_show_fiducials(self):
        """View fiducials table."""
        self.fiducials_dlg.show()

    def on_show_initial(self):
        """Show initial transform dialog."""
        from image2image.qt._dialogs._preprocess import PreprocessMovingDialog

        # add image
        #     self.on_add_transformed_moving()
        # open dialog
        initial = PreprocessMovingDialog(self)
        initial.show_above_widget(self.initial_btn, x_offset=20)

    def on_show_shortcuts(self) -> None:
        """View shortcuts table."""
        from image2image.qt._dialogs._shortcuts import RegisterShortcutsDialog

        dlg = RegisterShortcutsDialog(self)
        dlg.show()

    def _get_console_variables(self) -> dict:
        variables = super()._get_console_variables()
        variables.update(
            {
                "transform_model": self.transform_model,
                "fixed_viewer": self.view_fixed.viewer,
                "fixed_model": self.fixed_model,
                "moving_viewer": self.view_moving.viewer,
                "moving_model": self.moving_model,
            }
        )
        return variables

    @ensure_main_thread
    def on_apply(self, update_data: bool = False, name: str | None = None) -> None:
        """Apply transformation."""
        self._on_apply(update_data, name)

    def _on_apply(self, update_data: bool = False, name: str | None = None) -> None:
        if self.moving_image_layer is None or (
            self.transform is None
        ):  # or self.transform_model.moving_initial_affine is None
            # ):
            logger.warning("Cannot apply transformation - no transformation has been computed.")
            return

        if name is None or name == "None":
            moving_image_layer = self.moving_image_layer[0]
        else:
            index = [layer.name for layer in self.moving_image_layer].index(name)
            moving_image_layer = self.moving_image_layer[index]

        # retrieve affine matrix which might be composite of initial + transform or just initial
        affine = self.transform.params  # if self.transform is not None else self.transform_model.moving_initial_affine

        # add image and apply transformation
        if self.transformed_moving_image_layer:
            self.transformed_moving_image_layer.affine = affine
            if update_data:
                self.transformed_moving_image_layer.data = moving_image_layer.data
                self.transformed_moving_image_layer.colormap = moving_image_layer.colormap
                self.transformed_moving_image_layer.reset_contrast_limits()
        else:
            self.view_fixed.viewer.add_image(
                moving_image_layer.data,
                name="Transformed",
                blending="translucent",
                affine=affine,
                colormap=moving_image_layer.colormap,
                opacity=CONFIG.opacity_moving / 100,
            )
        self.transformed_moving_image_layer.visible = READER_CONFIG.show_transformed
        self._move_layer(self.view_fixed, self.transformed_moving_image_layer, -1, False)
        self._select_layer("fixed")

    def on_add_transformed_moving(self, name: str | None = None):
        """Add transformed moving image."""
        if self.transformed_moving_image_layer is None and self.moving_image_layer:
            if name is None or name == "None":
                moving_image_layer = self.moving_image_layer[0]
            else:
                index = [layer.name for layer in self.moving_image_layer].index(name)
                moving_image_layer = self.moving_image_layer[index]
            self.view_fixed.viewer.add_image(
                moving_image_layer.data,
                name="Transformed",
                blending="translucent",
                affine=np.eye(3),
                colormap=moving_image_layer.colormap,
                opacity=CONFIG.opacity_moving / 100,
            )

    @ensure_main_thread
    def on_predict(self, which: str, _evt: ty.Any = None) -> None:
        """Predict transformation from either image."""
        self._on_predict(which)

    def _on_predict(self, which: str) -> None:
        if self.transform is None:
            logger.warning("Cannot predict - no transformation has been computed.")
            return

        if self.fixed_points_layer.data.size == self.moving_points_layer.data.size:
            return
        self.on_update_text()
        if which == "fixed":
            # predict point position in the moving image -> inverse transform
            predict_for_layer = self.moving_points_layer
            transformed_last_point = self.transform.inverse(self.fixed_points_layer.data[-1])
            transformed_last_point = self.transform_model.apply_moving_initial_transform(
                transformed_last_point, inverse=False
            )
            logger.trace("Predicted moving points based on fixed points...")
        else:
            # predict point position in the fixed image -> transform
            predict_for_layer = self.fixed_points_layer
            transformed_last_point = self.transform_model.apply_moving_initial_transform(
                self.moving_points_layer.data[-1], inverse=True
            )
            transformed_last_point = self.transform(transformed_last_point)
            logger.trace("Predicted fixed points based on moving points...")

        transformed_data = predict_for_layer.data
        transformed_data = np.append(transformed_data, transformed_last_point, axis=0)
        # don't predict positions if the number of points is lower than the number already present in the image
        if predict_for_layer.data.shape[0] > len(transformed_data):
            return

        self._update_layer_points(predict_for_layer, transformed_data)
        self.evt_predicted.emit()  # noqa
        self._on_run()

    @staticmethod
    def _update_layer_points(layer: Points, data: np.ndarray, block: bool = True) -> None:
        """Update points layer."""
        if block:
            with layer.events.data.blocker(), suppress(IndexError):
                layer.data = data
                layer.properties = _get_text_data(data)
        else:
            with suppress(IndexError):
                layer.data = data
                layer.properties = _get_text_data(data)

    def on_update_layer(self, which: str, _value: ty.Any = None) -> None:
        """Update points layer."""
        CONFIG.size_fixed = self.fixed_point_size.value()
        CONFIG.size_moving = self.moving_point_size.value()
        CONFIG.opacity_fixed = self.fixed_opacity.value()
        CONFIG.opacity_moving = self.moving_opacity.value()

        # update point size
        if self.fixed_points_layer and which == "fixed":
            self.fixed_points_layer.size = CONFIG.size_fixed
            with suppress(IndexError):
                self.fixed_points_layer.current_size = CONFIG.size_fixed
        if self.moving_points_layer and which == "moving":
            self.moving_points_layer.size = CONFIG.size_moving
            with suppress(IndexError):
                self.moving_points_layer.current_size = CONFIG.size_moving
        if self.fixed_image_layer and which == "fixed":
            self.fixed_image_layer[0].opacity = CONFIG.opacity_fixed / 100
        if self.transformed_moving_image_layer and which == "moving":
            self.transformed_moving_image_layer.opacity = CONFIG.opacity_moving / 100

    def on_update_text(self, _: ty.Any = None, block: bool = False) -> None:
        """Update text data in each layer."""
        CONFIG.label_color = self.text_color.hex_color
        CONFIG.label_size = self.text_size.value()

        # update text information
        for layer in [self.fixed_points_layer, self.moving_points_layer]:
            if len(layer.data) == 0:
                continue

            if block:
                with layer.text.events.blocker():
                    layer.text = _get_text_format()
            else:
                layer.text = _get_text_format()

    def on_lock(self) -> None:
        """Lock transformation."""
        self.on_set_focus()
        hp.disable_widgets(self.x_center, self.y_center, self.zoom, disabled=self.lock_btn.locked, min_opacity=0.75)

    def on_set_focus(self) -> None:
        """Lock current focus to specified range."""
        self.zoom.setValue(self.view_fixed.viewer.camera.zoom)
        _, y, x = self.view_fixed.viewer.camera.center
        self.x_center.setValue(x)
        self.y_center.setValue(y)

    def on_apply_focus(self) -> None:
        """Apply focus to the current image range."""
        if all(v == 1.0 for v in [self.zoom.value(), self.x_center.value(), self.y_center.value()]):
            logger.warning("Please specify zoom and center first.")
            return
        self.view_fixed.viewer.camera.center = (0.0, self.y_center.value(), self.x_center.value())
        self.view_fixed.viewer.camera.zoom = self.zoom.value()
        logger.trace(
            f"Applied focus center=({self.y_center.value():.1f}, {self.x_center.value():.1f})"
            f" zoom={self.zoom.value():.3f}"
        )

    @qdebounced(timeout=50)
    def on_toggle_transformed_visibility(self) -> None:
        """Toggle visibility of transformed image."""
        if self.transformed_moving_image_layer:
            READER_CONFIG.show_transformed = not READER_CONFIG.show_transformed
            self.transformed_moving_image_layer.visible = READER_CONFIG.show_transformed

    @qdebounced(timeout=50)
    def on_toggle_transformed_image(self) -> None:
        """Toggle visibility of transformed image."""
        if self.transformed_moving_image_layer:
            self._moving_widget.toggle_transformed()

    @qdebounced(timeout=50)
    def on_zoom_on_point(self, increment: int):
        """Zoom-in on point."""
        n_max = self.fiducials_dlg.n_points
        current = self._current_index
        current += increment
        if current < 0:
            current = n_max - 1
        elif current >= n_max:
            current = 0
        self._current_index = current
        self.fiducials_dlg.on_select_point(current)

    def on_update_settings(self):
        """Update config."""
        CONFIG.sync_views = self.synchronize_zoom.isChecked()

    @qdebounced(timeout=200)
    def on_sync_views_fixed(self, _event: Event | None = None) -> None:
        """Synchronize views."""
        self._on_sync_views("fixed")

    @qdebounced(timeout=200)
    def on_sync_views_moving(self, _event: Event | None = None) -> None:
        """Synchronize views."""
        self._on_sync_views("moving")

    def _on_sync_views(self, which: str) -> None:
        if not CONFIG.sync_views or self.transform is None or self._zooming:
            return
        with self.zooming():
            before_func = lambda x: x  # noqa
            after_func = lambda x: x  # noqa
            if which == "fixed":
                func = self.transform_model.inverse
                after_func = partial(self.transform_model.apply_moving_initial_transform, inverse=False)
                camera = self.view_fixed.viewer.camera
                callback = self.on_sync_views_moving
                other_view = self.view_moving
                ratio = self.transform_model.moving_to_fixed_ratio
            else:
                func = self.transform_model
                before_func = partial(self.transform_model.apply_moving_initial_transform, inverse=True)
                callback = self.on_sync_views_fixed
                camera = self.view_moving.viewer.camera
                other_view = self.view_fixed
                ratio = self.transform_model.fixed_to_moving_ratio

            # predict where to zoom
            center = camera.center[1::]
            zoom = camera.zoom * ratio  # zoom must be multiplied by the ratio of pixel sizes
            transformed_center = before_func(center)
            transformed_center = func(transformed_center)
            transformed_center = after_func(transformed_center)[0]
            with other_view.viewer.camera.events.blocker(callback):
                other_view.viewer.camera.zoom = zoom
                other_view.viewer.camera.center = (0.0, transformed_center[0], transformed_center[1])

    # noinspection PyAttributeOutsideInit
    def _setup_ui(self) -> None:
        """Create panel."""
        view_layout = self._make_image_layout()
        side_widget = QWidget()
        side_widget.setMinimumWidth(375)
        side_widget.setMaximumWidth(375)
        self.import_project_btn = hp.make_btn(
            side_widget,
            "Import project",
            tooltip="Import previously computed transformation.",
            func=self.on_load_from_project,
        )

        self._fixed_widget = FixedWidget(
            self,
            self.view_fixed,
            allow_swap=False,
            project_extension=[".i2r.json", ".i2r.toml"],
        )
        self._moving_widget = MovingWidget(
            self,
            self.view_moving,
            allow_swap=False,
            project_extension=[".i2r.json", ".i2r.toml"],
        )

        self.transform_error = hp.make_label(
            side_widget,
            "<need more points>",
            bold=True,
            object_name="reg_error",
            tooltip="Error is estimated by computing the square root of the sum of squared errors. Value is <b>red</b>"
            " if the error is larger than half of the moving image resolution (off by half a pixel).",
        )
        self.transform_info = hp.make_label(side_widget, "", wrap=True)

        self.initial_btn = hp.make_btn(
            side_widget,
            "Orient moving image...",
            func=self.on_show_initial,
            tooltip="You can optionally rotate or flip the moving image so that it's easier to align with the fixed"
            " image. This button will be disabled if there are ANY points in the moving image.",
        )
        self.fiducials_btn = hp.make_btn(
            side_widget,
            "Show fiducials table...",
            tooltip="Show fiducial markers table where you can view and edit the markers",
            func=self.on_show_fiducials,
        )
        self.export_project_btn = hp.make_btn(
            side_widget,
            "Export to file...",
            tooltip="Export transformation to file. XML format is usable by MATLAB fusion.",
            func=self.on_save_to_project,
        )

        side_layout = hp.make_form_layout(side_widget)
        hp.style_form_layout(side_layout)
        side_layout.addRow(self.import_project_btn)
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._fixed_widget)
        side_layout.addRow(hp.make_h_line_with_text("+"))
        side_layout.addRow(self._moving_widget)
        side_layout.addRow(hp.make_h_line_with_text("Area of interest"))
        side_layout.addRow(self._make_focus_layout())
        side_layout.addRow(hp.make_h_line_with_text("Transformation"))
        side_layout.addRow(self.initial_btn)
        side_layout.addRow(self.fiducials_btn)
        side_layout.addRow(hp.make_btn(side_widget, "Compute transformation", func=self.on_run))
        side_layout.addRow(self.export_project_btn)
        side_layout.addRow(hp.make_label(self, "Estimated error"), self.transform_error)
        side_layout.addRow(hp.make_label(self, "About transformation"), self.transform_info)
        side_layout.addRow(hp.make_spacer_widget())
        side_layout.addRow(hp.make_h_line_with_text("Settings"))
        side_layout.addRow(self._make_settings_layout())

        layout = QHBoxLayout()
        layout.setSpacing(1)
        layout.addLayout(view_layout, stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addWidget(side_widget)

        widget = QWidget()  # noqa
        self.setCentralWidget(widget)
        main_layout = QVBoxLayout(widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(layout, stretch=True)

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
            "Import configuration file (.json, .toml)...",
            "Ctrl+C",
            menu=menu_file,
            func=self.on_load_from_project,
        )
        hp.make_menu_item(
            self,
            "Add fixed image (.tiff, .png, .jpg, .imzML, .tdf, .tsf, + others)...",
            "Ctrl+F",
            menu=menu_file,
            func=self._fixed_widget.on_select_dataset,
        )
        hp.make_menu_item(
            self,
            "Add moving image (.tiff, .png, .jpg, .imzML, .tdf, .tsf, + others)...",
            "Ctrl+M",
            menu=menu_file,
            func=self._moving_widget.on_select_dataset,
        )
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(
            self, "Select fixed channels...", menu=menu_tools, func=self._fixed_widget._on_select_channels
        )
        hp.make_menu_item(
            self, "Select moving channels...", menu=menu_tools, func=self._moving_widget._on_select_channels
        )
        hp.make_menu_item(
            self, "Show fiducials table...", "Ctrl+F", menu=menu_tools, func=self.on_show_fiducials, icon="fiducial"
        )
        hp.make_menu_item(
            self, "Show shortcuts...", "Ctrl+S", menu=menu_tools, func=self.on_show_shortcuts, icon="shortcut"
        )
        menu_tools.addSeparator()
        hp.make_menu_item(self, "Show IPython console...", "Ctrl+T", menu=menu_tools, func=self.on_show_console)

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def _make_focus_layout(self) -> QFormLayout:
        self.lock_btn = hp.make_lock_btn(
            self,
            func=self.on_lock,
            normal=True,
            tooltip="Lock the area of interest. Press <b>L</b> on your keyboard to lock.",
        )
        # self.set_current_focus_btn = hp.make_btn(self, "Set current range", func=self.on_set_focus)
        self.x_center = hp.make_double_spin_box(
            self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the x-axis."
        )
        self.y_center = hp.make_double_spin_box(
            self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the y-axis."
        )
        self.zoom = hp.make_double_spin_box(self, -1e5, 1e5, step_size=0.5, n_decimals=4, tooltip="Zoom factor.")
        self.use_focus_btn = hp.make_btn(
            self,
            "Zoom-in",
            func=self.on_apply_focus,
            tooltip="Zoom-in to an area of interest. Press <b>Z</b> on your keyboard to zoom-in.",
        )

        layout = hp.make_form_layout()
        hp.style_form_layout(layout)
        layout.addRow(hp.make_label(self, "Center (x)"), self.x_center)
        layout.addRow(hp.make_label(self, "Center (y)"), self.y_center)
        layout.addRow(hp.make_label(self, "Zoom"), self.zoom)
        layout.addRow(hp.make_h_layout(self.lock_btn, self.use_focus_btn, stretch_id=1))
        return layout

    def _make_settings_layout(self) -> QFormLayout:
        self.synchronize_zoom = hp.make_checkbox(
            self,
            "",
            "Synchronize zoom between views. It only starts taking effect once transformation model has been"
            " calculated.",
            value=CONFIG.sync_views,
            func=self.on_update_settings,
        )
        self.fixed_point_size = hp.make_int_spin_box(
            self, value=CONFIG.size_fixed, tooltip="Size of the points shown in the fixed image."
        )
        self.fixed_point_size.valueChanged.connect(partial(self.on_update_layer, "fixed"))  # noqa

        self.moving_point_size = hp.make_int_spin_box(
            self, value=CONFIG.size_moving, tooltip="Size of the points shown in the moving image."
        )
        self.moving_point_size.valueChanged.connect(partial(self.on_update_layer, "moving"))  # noqa

        self.fixed_opacity = hp.make_int_spin_box(
            self, value=CONFIG.opacity_fixed, step_size=10, tooltip="Opacity of the fixed image"
        )
        self.fixed_opacity.valueChanged.connect(partial(self.on_update_layer, "fixed"))  # noqa

        self.moving_opacity = hp.make_int_spin_box(
            self,
            value=CONFIG.opacity_moving,
            step_size=10,
            tooltip="Opacity of the moving image in the fixed view",
        )
        self.moving_opacity.valueChanged.connect(partial(self.on_update_layer, "moving"))  # noqa

        self.text_size = hp.make_int_spin_box(
            self, value=CONFIG.label_size, minimum=4, maximum=60, tooltip="Size of the text associated with each label."
        )
        self.text_size.valueChanged.connect(self.on_update_text)  # noqa

        self.text_color = hp.make_swatch(
            self, default=CONFIG.label_color, tooltip="Color of the text associated with each label."
        )
        self.text_color.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)  # type: ignore[attr-defined]
        self.text_color.evt_color_changed.connect(self.on_update_text)  # noqa

        layout = hp.make_form_layout()
        hp.style_form_layout(layout)
        layout.addRow(hp.make_label(self, "Synchronize views"), self.synchronize_zoom)
        layout.addRow(hp.make_label(self, "Marker size (fixed)"), self.fixed_point_size)
        layout.addRow(hp.make_label(self, "Marker size (moving)"), self.moving_point_size)
        layout.addRow(hp.make_label(self, "Image opacity (fixed)"), self.fixed_opacity)
        layout.addRow(hp.make_label(self, "Image opacity (moving)"), self.moving_opacity)
        layout.addRow(hp.make_label(self, "Label size"), self.text_size)
        layout.addRow(hp.make_label(self, "Label color"), self.text_color)
        return layout

    def _make_image_layout(self) -> QVBoxLayout:
        info = hp.make_h_line_with_text(
            "Please select at least <b>3 points</b> in each image to compute transformation.<br>"
            "It's advised to use <b>as many</b> anchor points as reasonable!",
            self,
            object_name="tip_label",
            alignment=Qt.AlignmentFlag.AlignHCenter,
        )

        view_layout = QVBoxLayout()
        view_layout.setContentsMargins(5, 0, 0, 0)
        view_layout.setSpacing(1)
        view_layout.addLayout(self._make_fixed_view())
        view_layout.addLayout(info)  # type: ignore[attr-defined]
        view_layout.addLayout(self._make_moving_view())
        return view_layout

    def _make_fixed_view(self) -> QHBoxLayout:
        self.view_fixed = self._make_image_view(self, add_toolbars=False, disable_new_layers=True)
        self.view_fixed.viewer.text_overlay.text = "Fixed"
        self.view_fixed.viewer.text_overlay.position = "top_left"
        self.view_fixed.viewer.text_overlay.font_size = 10
        self.view_fixed.viewer.text_overlay.visible = True
        self.view_fixed.widget.canvas.events.key_press.connect(self.keyPressEvent)

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=True)  # type: ignore[attr-defined]
        _fixed_clear_btn = toolbar.insert_qta_tool(
            "remove_all",
            func=lambda *args: self.on_clear("fixed", force=False),
            tooltip="Remove all points from the fixed image (need to confirm).",
        )
        _fixed_remove_selected_btn = toolbar.insert_qta_tool(
            "remove_multiple",
            func=lambda *args: self.on_remove_selected("fixed"),
            tooltip="Remove selected points from the fixed image.",
        )
        _fixed_remove_btn = toolbar.insert_qta_tool(
            "remove_single",
            func=lambda *args: self.on_remove("fixed"),
            tooltip="Remove last point from the fixed image.",
        )
        self.fixed_move_btn = toolbar.insert_qta_tool(
            "select_points",
            func=lambda *args: self.on_move("fixed"),
            tooltip="Move points in the fixed image. Press <b>3</b> on your keyboard to activate...",
            checkable=True,
        )
        self.fixed_add_btn = toolbar.insert_qta_tool(
            "add",
            func=lambda *args: self.on_add("fixed"),
            tooltip="Add new point to the fixed image. Press <b>2</b> on your keyboard to activate...",
            checkable=True,
        )
        self.fixed_zoom_btn = toolbar.insert_qta_tool(
            "pan_zoom",
            func=lambda *args: self.on_panzoom("fixed"),
            tooltip="Switch to zoom-only mode. Press <b>1</b> on your keyboard to activate...",
            checkable=True,
        )
        hp.make_radio_btn_group(self, [self.fixed_zoom_btn, self.fixed_add_btn, self.fixed_move_btn])
        _fixed_bring_to_top = toolbar.insert_qta_tool(
            "bring_to_top",
            func=lambda *args: self._select_layer("fixed"),
            tooltip="Bring points layer to the top.",
        )
        _fixed_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_fixed.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )
        self.fixed_indicator, _ = hp.make_loading_gif(self, "square", size=(24, 24))
        self.fixed_indicator.hide()
        toolbar.insert_widget(self.fixed_indicator)
        self.fixed_toolbar = toolbar

        layout = QHBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(5, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self.view_fixed.widget, stretch=True)
        return layout

    def _make_moving_view(self) -> QHBoxLayout:
        self.view_moving = self._make_image_view(self, add_toolbars=False, disable_new_layers=True)
        self.view_moving.viewer.text_overlay.text = "Moving"
        self.view_moving.viewer.text_overlay.position = "top_left"
        self.view_moving.viewer.text_overlay.font_size = 10
        self.view_moving.viewer.text_overlay.visible = True
        self.view_moving.widget.canvas.events.key_press.connect(self.keyPressEvent)

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=True)  # type: ignore[attr-defined]
        _moving_clear_btn = toolbar.insert_qta_tool(
            "remove_all",
            func=lambda *args: self.on_clear("moving", force=False),
            tooltip="Remove all points from the moving image (need to confirm).",
        )
        _moving_remove_selected_btn = toolbar.insert_qta_tool(
            "remove_multiple",
            func=lambda *args: self.on_remove_selected("moving"),
            tooltip="Remove selected points from the moving image.",
        )
        _moving_remove_btn = toolbar.insert_qta_tool(
            "remove_single",
            func=lambda *args: self.on_remove("moving"),
            tooltip="Remove last point from the moving image.",
        )
        self.moving_move_btn = toolbar.insert_qta_tool(
            "select_points",
            func=lambda *args: self.on_move("moving"),
            tooltip="Move points in the fixed image. Press <b>3</b> on your keyboard to activate...",
            checkable=True,
        )
        self.moving_add_btn = toolbar.insert_qta_tool(
            "add",
            func=lambda *args: self.on_add("moving"),
            tooltip="Add new point to the moving image. Press <b>2</b> on your keyboard to activate..",
            checkable=True,
        )
        self.moving_zoom_btn = toolbar.insert_qta_tool(
            "pan_zoom",
            func=lambda *args: self.on_panzoom("moving"),
            tooltip="Switch to zoom-only mode. Press <b>1</b> on your keyboard to activate...",
            checkable=True,
        )
        hp.make_radio_btn_group(self, [self.moving_zoom_btn, self.moving_add_btn, self.moving_move_btn])
        _moving_bring_to_top = toolbar.insert_qta_tool(
            "bring_to_top",
            func=lambda *args: self._select_layer("moving"),
            tooltip="Bring points layer to the top.",
        )
        _moving_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_moving.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )
        self.moving_indicator, _ = hp.make_loading_gif(self, "square", size=(24, 24))
        self.moving_indicator.hide()
        toolbar.insert_widget(self.moving_indicator)
        self.moving_toolbar = toolbar

        layout = QHBoxLayout()
        layout.setSpacing(1)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self.view_moving.widget, stretch=True)
        return layout

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        super()._make_statusbar()
        self.shortcuts_btn.show()

    def keyPressEvent(self, evt: ty.Any) -> None:
        """Key press event."""
        if hasattr(evt, "native"):
            evt = evt.native
        key = evt.key()
        if key == Qt.Key.Key_Escape:
            evt.ignore()
        elif key == Qt.Key.Key_1:
            self.on_mode("fixed", mode=Mode.PAN_ZOOM)
            self.on_panzoom("fixed")
            self.on_mode("moving", mode=Mode.PAN_ZOOM)
            self.on_panzoom("moving")
            evt.ignore()
        elif key == Qt.Key.Key_2:
            self.on_mode("fixed", mode=Mode.ADD)
            self.on_add("fixed")
            self.on_mode("moving", mode=Mode.ADD)
            self.on_add("moving")
            evt.ignore()
        elif key == Qt.Key.Key_3:
            self.on_mode("fixed", mode=Mode.SELECT)
            self.on_move("fixed")
            self.on_mode("moving", mode=Mode.SELECT)
            self.on_move("moving")
            evt.ignore()
        elif key == Qt.Key.Key_Z:
            self.on_apply_focus()
            evt.ignore()
        elif key == Qt.Key.Key_L:
            self.on_set_focus()
            evt.ignore()
        elif key == Qt.Key.Key_T:
            self.on_toggle_transformed_image()  # type: ignore[call-arg]
            evt.ignore()
        elif key == Qt.Key.Key_V:
            self.on_toggle_transformed_visibility()  # type: ignore[call-arg]
            evt.ignore()
        elif key == Qt.Key.Key_A:
            self.on_zoom_on_point(-1)
            evt.ignore()
        elif key == Qt.Key.Key_D:
            self.on_zoom_on_point(1)
            evt.ignore()
        else:
            super().keyPressEvent(evt)

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not CONFIG.confirm_close_register
            or QtConfirmCloseDialog(
                self, "confirm_close_register", self.on_save_to_project, CONFIG
            ).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and CONFIG.confirm_close_register
            and self.transform_model.is_valid()
            and QtConfirmCloseDialog(
                self, "confirm_close_register", self.on_save_to_project, CONFIG
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

    def dropEvent(self, event):
        """Override Qt method."""
        from qtextra.widgets.qt_pick_option import QtPickOption

        if self.allow_drop:
            hp.update_property(self.centralWidget(), "drag", False)

            dlg = QtPickOption(
                self,
                "Please select which view would you like to add the image(s) to?",
                {"fixed": "Fixed image", "moving": "Moving image"},
            )
            which = None
            if dlg.exec_() == QDialog.DialogCode.Accepted:  # type: ignore[attr-defined]
                which = dlg.option
            if which == "fixed":
                self.evt_fixed_dropped.emit(event)
            elif which == "moving":
                self.evt_moving_dropped.emit(event)
            else:
                event.ignore()

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        from image2image.qt._dialogs._tutorial import show_register_tutorial

        show_register_tutorial(self)
        CONFIG.first_time_register = False


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="register", level=0)
