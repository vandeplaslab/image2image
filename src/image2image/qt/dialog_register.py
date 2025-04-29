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
from napari.layers import Image, Points
from napari.layers.points.points import Mode
from napari.layers.utils._link_layers import link_layers
from napari.utils.events import Event
from qtextra.config import THEMES
from qtextra.dialogs.qt_close_window import QtConfirmCloseDialog
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_toolbar_mini import QtMiniToolbar
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QMenuBar, QSizePolicy, QVBoxLayout, QWidget
from superqt.utils import ensure_main_thread, qdebounced

import image2image.constants as C
from image2image import __version__
from image2image.config import RegisterConfig, get_register_config
from image2image.enums import ALLOWED_PROJECT_EXPORT_REGISTER_FORMATS, ALLOWED_PROJECT_IMPORT_REGISTER_FORMATS
from image2image.models.data import DataModel
from image2image.models.transformation import Transformation
from image2image.qt._dialog_base import Window
from image2image.qt._dialogs._select import FixedWidget, MovingWidget
from image2image.utils.utilities import (
    _get_text_data,
    _get_text_format,
    ensure_extension,
    get_colormap,
    get_simple_contrast_limits,
    init_points_layer,
)

if ty.TYPE_CHECKING:
    from qtextra.widgets.qt_button_icon import QtImagePushButton
    from skimage.transform import ProjectiveTransform

    from image2image.qt._dialogs import FiducialsDialog

FIXED_POINTS = "Fixed (points)"
FIXED_TMP_POINTS = "Temporary fixed (points)"
FIXED_TMP_ZOOM = "Temporary fixed (zoom)"
MOVING_POINTS = "Moving (points)"
MOVING_TMP_POINTS = "Temporary moving (points)"
MOVING_TMP_ZOOM = "Temporary moving (zoom)"


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

    APP_NAME = "register"

    view_fixed: NapariImageView
    view_moving: NapariImageView
    fixed_image_layer: list[Image] | None = None
    moving_image_layer: list[Image] | None = None
    is_predicting: bool = False

    _fiducials_dlg = None
    _zooming = False
    _current_index = -1

    # events
    evt_predicted = Signal()
    evt_fixed_dropped = Signal("QEvent")
    evt_moving_dropped = Signal("QEvent")

    def __init__(self, parent: QWidget | None, run_check_version: bool = True, **kwargs: ty.Any):
        self.CONFIG: RegisterConfig = get_register_config()
        super().__init__(
            parent,
            f"image2image: Registration app (v{__version__})",
            delay_events=True,
            run_check_version=run_check_version,
        )
        self.transform_model = Transformation(
            fixed_model=self.fixed_model,
            moving_model=self.moving_model,
            fixed_points=self.fixed_points_layer.data,
            moving_points=self.moving_points_layer.data,
        )
        if self.CONFIG.first_time:
            hp.call_later(self, self.on_show_tutorial, 10_000)
        self._setup_config()

    @staticmethod
    def _setup_config() -> None:
        READER_CONFIG.auto_pyramid = True
        READER_CONFIG.init_pyramid = True
        READER_CONFIG.split_roi = True
        READER_CONFIG.split_rgb = False
        READER_CONFIG.only_last_pyramid = False

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

    def on_layer_removed(self, event: Event, which: str) -> None:
        """Layer removed."""
        layer = event.value
        try:
            connect(layer.events.data, self.on_run, state=False)
            connect(layer.events.add_point, partial(self.on_predict, which), state=False)
        except AttributeError:
            pass

    @property
    def fixed_points_layer(self) -> Points:
        """Fixed points layer."""
        if FIXED_POINTS not in self.view_fixed.layers:
            layer = self.view_fixed.viewer.add_points(  # noqa
                None,
                size=self.fixed_point_size.value(),
                name=FIXED_POINTS,
                face_color="green",
                border_color="white",
                symbol="ring",
            )
            visual = self.view_fixed.widget.canvas.layer_to_visual[layer]

            init_points_layer(layer, visual, snap=False)
            connect(layer.events.data, self.on_run, state=True)
            connect(layer.events.add_point, partial(self.on_predict, "fixed"), state=True)
        return self.view_fixed.layers[FIXED_POINTS]

    @property
    def temporary_fixed_points_layer(self) -> Points:
        """Fixed points layer."""
        if FIXED_TMP_POINTS not in self.view_fixed.layers:
            layer = self.view_fixed.viewer.add_points(  # noqa
                None,
                size=self.moving_point_size.value(),
                name=FIXED_TMP_POINTS,
                face_color="green",
                border_color="white",
                symbol="ring",
            )
            visual = self.view_fixed.widget.canvas.layer_to_visual[layer]
            init_points_layer(layer, visual, snap=True)
        return self.view_fixed.layers[FIXED_TMP_POINTS]

    @property
    def moving_points_layer(self) -> Points:
        """Fixed points layer."""
        if MOVING_POINTS not in self.view_moving.layers:
            layer = self.view_moving.viewer.add_points(  # noqa
                None,
                size=self.moving_point_size.value(),
                name=MOVING_POINTS,
                face_color="green",
                border_color="white",
                symbol="ring",
            )
            visual = self.view_moving.widget.canvas.layer_to_visual[layer]
            init_points_layer(layer, visual, True)
            connect(layer.events.data, self.on_run, state=True)
            connect(layer.events.add_point, partial(self.on_predict, "moving"), state=True)
        return self.view_moving.layers[MOVING_POINTS]

    @property
    def temporary_moving_points_layer(self) -> Points:
        """Fixed points layer."""
        if MOVING_TMP_POINTS not in self.view_moving.layers:
            layer = self.view_moving.viewer.add_points(  # noqa
                None,
                size=self.moving_point_size.value(),
                name=MOVING_TMP_POINTS,
                face_color="green",
                border_color="white",
                symbol="ring",
            )
            visual = self.view_moving.widget.canvas.layer_to_visual[layer]
            init_points_layer(layer, visual, True)
        return self.view_moving.layers[MOVING_TMP_POINTS]

    def setup_events(self, state: bool = True) -> None:
        """Additional setup."""
        # fixed widget
        connect(self._fixed_widget.dset_dlg.evt_import_project, self._on_load_from_project, state=state)
        connect(self._fixed_widget.dset_dlg.evt_loaded, self.on_load_fixed, state=state)
        connect(self._fixed_widget.dset_dlg.evt_closed, self.on_close_fixed, state=state)
        connect(self._fixed_widget.dset_dlg.evt_channel, partial(self.on_toggle_channel, which="fixed"), state=state)
        connect(
            self._fixed_widget.dset_dlg.evt_channel_all,
            partial(self.on_toggle_all_channels, which="fixed"),
            state=state,
        )

        # moving widget
        connect(self._moving_widget.dset_dlg.evt_import_project, self._on_load_from_project, state=state)
        connect(self._moving_widget.dset_dlg.evt_loaded, self.on_load_moving, state=state)
        connect(self._moving_widget.dset_dlg.evt_closed, self.on_close_moving, state=state)
        connect(self._moving_widget.evt_show_fixed_channel, self.on_toggle_fixed_transformed_channel, state=state)
        connect(self._moving_widget.evt_show_moving_channel, self.on_toggle_moving_image, state=state)
        connect(self._moving_widget.evt_dataset_select, self.on_toggle_dataset, state=state)
        connect(
            self._moving_widget.dset_dlg.evt_channel,
            partial(self.on_toggle_channel, which="moving"),
            state=state,
        )
        connect(
            self._moving_widget.dset_dlg.evt_channel_all,
            partial(self.on_toggle_all_channels, which="moving"),
            state=state,
        )
        connect(self._moving_widget.evt_view_type, self.on_change_view_type, state=state)
        connect(self._moving_widget.dset_dlg.evt_iter_next, self.on_plot_temporary, state=state)
        connect(self._moving_widget.dset_dlg.evt_iter_remove, self.on_remove_temporary, state=state)
        connect(self._moving_widget.dset_dlg.evt_iter_add, self.on_add_temporary_to_viewer, state=state)

        # views
        connect(self.view_fixed.viewer.camera.events, self.on_sync_views_fixed, state=state)
        connect(self.view_fixed.viewer.events.status, self._status_changed, state=state)
        connect(
            self.view_fixed.viewer.layers.events.removed, partial(self.on_layer_removed, which="fixed"), state=state
        )
        connect(self.view_moving.viewer.camera.events, self.on_sync_views_moving, state=state)
        connect(self.view_moving.viewer.events.status, self._status_changed, state=state)
        connect(
            self.view_moving.viewer.layers.events.removed, partial(self.on_layer_removed, which="moving"), state=state
        )

    def on_plot_temporary(self, key: str, channel_index: int) -> None:
        """Plot temporary layer."""
        self._plot_temporary_layer(self.moving_model, self.view_moving, key, channel_index, True)

    def on_remove_temporary(self, _: ty.Any = None) -> None:
        """Remove temporary layer."""
        for view in [self.view_fixed, self.view_moving]:
            for layer in view.layers:
                if layer.name.startswith("temporary"):
                    view.remove_layer(layer.name)
                    logger.trace(f"Removed temporary layer '{layer.name}'.")

    def on_add_temporary_to_viewer(self, key: str, channel_index: int) -> None:
        """Add temporary layer to viewer."""
        reader = self.data_model.get_reader_for_key(key)
        layer = self.temporary_layers.get(key, None)
        if layer and reader:
            channel_name = reader.channel_names[channel_index]
            layer_name = f"{channel_name} | {key}"
            if layer_name in self.view_moving.layers:
                logger.warning(f"Temporary layer '{key}' is already added to viewer.")
                return
            layer = self.temporary_layers.pop(key, None)
            layer.name = layer_name
            logger.trace(f"Added image {channel_index} for '{key}' to viewer.")

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

    def _on_load_fixed(self, model: DataModel, channel_list: list[str] | None = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading fixed data with {model.n_paths} paths...")
            need_reset = len(self.view_fixed.layers) == 0
            self._plot_fixed_layers(channel_list)
            if need_reset:
                self.view_fixed.viewer.reset_view()
        logger.info(f"Loaded fixed data in {timer()}")

    def _plot_fixed_layers(self, channel_list: list[str] | None = None) -> None:
        self.fixed_image_layer, _, _ = self._plot_reader_layers(
            self.fixed_model, self.view_fixed, channel_list, "fixed view"
        )
        if isinstance(self.fixed_image_layer, list) and len(self.fixed_image_layer) > 1:
            link_layers(self.fixed_image_layer, attributes=("opacity",))

    def on_toggle_channel(self, state: bool, name: str, which: str) -> None:
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
        self._ensure_consistent_moving_dataset()

    def _ensure_consistent_moving_dataset(self) -> None:
        datasets = self._get_currently_visible_moving_datasets()
        current = self._moving_widget.dataset_choice.currentText()
        if current not in datasets and datasets:
            with hp.qt_signals_blocked(self._moving_widget):
                self._moving_widget.dataset_choice.setCurrentText(datasets[0])

    def _on_load_moving(self, model: DataModel, channel_list: list[str] | None = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading moving data with {model.n_paths} paths...")
            need_reset = len(self.view_moving.layers) == 0
            self._plot_moving_layers(channel_list)
            self.on_apply(update_data=True)
            if need_reset:
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
                    colormap = self.view_moving.layers[name].colormap if is_overlay else "turbo"
                    del self.view_moving.layers[name]
                if is_overlay and ((hasattr(colormap, "name") and colormap.name == "turbo") or colormap == "turbo"):
                    colormap = get_colormap(index, self.view_moving.layers)
                contrast_limits, contrast_limits_range = None, None
                if is_overlay:
                    contrast_limits, contrast_limits_range = get_simple_contrast_limits(array)
                moving_image_layer.append(
                    self.view_moving.viewer.add_image(
                        array,
                        name=name,
                        blending="additive",
                        colormap=colormap,
                        visible=is_visible or name in channel_list,
                        affine=initial_affine,
                        contrast_limits=contrast_limits,
                    )
                )
            if contrast_limits_range:
                moving_image_layer[-1].contrast_limits_range = contrast_limits_range
            logger.trace(f"Added '{name}' to fixed view with {initial_affine.flatten()} in {timer()}.")
        # hide away other layers if the user selected 'random' view
        if READER_CONFIG.view_type == ViewType.RANDOM:
            for index, layer in enumerate(moving_image_layer):
                if index > 0:
                    layer.visible = False
        self.moving_image_layer = moving_image_layer

    def on_update_contrast_limits(self) -> None:
        """Update contrast limits in all visible layers in the moving image."""
        for layer in self.view_moving.layers:
            if isinstance(layer, Image) and layer.visible:
                contrast_limits = get_simple_contrast_limits(layer.data)
                layer.contrast_limits = contrast_limits
                layer.contrast_limits_range = contrast_limits

    def on_change_view_type(self, _view_type: str) -> None:
        """Change view type."""
        if self.moving_model.n_paths:
            channel_list = self._moving_widget.channel_list()
            self._plot_moving_layers(channel_list)
            self.on_apply(update_data=True)

    def on_toggle_dataset(self, value: str) -> None:
        """Toggle dataset."""
        if value:
            layers = self.view_moving.get_layers_of_type(Image)
            if layers:
                layer_names = [layer.name for layer in layers]
                self.view_moving.remove_layers(layer_names)
            channel_names = self.moving_model.get_channel_names_for_keys([value])
            if channel_names:
                self._plot_moving_layers([channel_names[0]])
                self.view_moving.viewer.reset_view()

    def get_current_moving_reader(self) -> ty.Any:
        """Get current reader."""
        try:
            current = self._moving_widget.dataset_choice.currentText()
            reader = self.moving_model.get_reader_for_key(current)
            return reader
        except AttributeError:
            return None

    def get_current_fixed_reader(self) -> ty.Any:
        """Get current reader."""
        try:
            reader = next(self.fixed_model.wrapper.reader_iter())
            return reader
        except AttributeError:
            return None

    def on_toggle_fixed_transformed_channel(self, value: str) -> None:
        """Toggle visibility of transformed moving image."""
        self.on_apply(update_data=True, name=value)

    def on_toggle_moving_image(self, value: str) -> None:
        """Change displayed image in the moving image."""
        for layer in self.moving_image_layer:
            layer.visible = False
        self._plot_moving_layers([value])
        print([layer.visible for layer in self.moving_image_layer])

    def _select_point_layer(self, which: str) -> Points:
        """Select layer."""
        view, layer = (
            (self.view_fixed, self.fixed_points_layer)
            if which == "fixed"
            else (self.view_moving, self.moving_points_layer)
        )
        self._move_layer(view, layer)
        return layer

    def _get_points_mode_button(self, which: str, mode: Mode) -> QtImagePushButton | None:
        if which == "fixed":
            widgets = {
                Mode.ADD: self.fixed_add_btn,
                Mode.SELECT: self.fixed_move_btn,
                Mode.PAN_ZOOM: self.fixed_pan_btn,
            }
        else:
            widgets = {
                Mode.ADD: self.moving_add_btn,
                Mode.SELECT: self.moving_move_btn,
                Mode.PAN_ZOOM: self.moving_pan_btn,
            }
        return widgets.get(mode, None)

    def on_points_mode(self, which: str, evt: ty.Any = None, mode: Mode | None = None) -> None:
        """Update mode."""
        widget = self._get_points_mode_button(which, mode or evt.mode)
        if widget:
            widget.setChecked(True)
            logger.trace(f"Set mode to '{mode}' for '{which}'.")

    def _get_layer(self, which: str | ty.Literal["fixed", "moving", "both"]) -> list[Points]:
        """Get layer."""
        if which == "fixed":
            return [self.fixed_points_layer]
        if which == "moving":
            return [self.moving_points_layer]
        return [self.fixed_points_layer, self.moving_points_layer]

    def on_panzoom(self, which: str, _evt: ty.Any = None) -> None:
        """Switch to `panzoom` tool."""
        self._select_point_layer(which)
        for layer in self._get_layer(which):
            layer.mode = "pan_zoom"

    def on_add(self, which: str, _evt: ty.Any = None) -> None:
        """Add point to the image."""
        layer = self._select_point_layer(which)
        # extract button and layer based on the appropriate mode
        widget = self.fixed_add_btn if which == "fixed" else self.moving_add_btn
        # make sure the 'add' mode is active
        layer.mode = "add" if widget.isChecked() else "pan_zoom"

    def on_move(self, which: str, _evt: ty.Any = None) -> None:
        """Move points."""
        layer = self._select_point_layer(which)
        widget = self.fixed_move_btn if which == "fixed" else self.moving_move_btn
        layer.mode = "select" if widget.isChecked() else "pan_zoom"
        layer.selected_data = []

    def on_activate_initial(self) -> None:
        """Activate initial button."""
        hp.disable_widgets(self.initial_btn, self.guess_btn, disabled=has_any_points(self.moving_points_layer))

    def on_remove(self, which: str, _evt: ty.Any = None) -> None:
        """Remove point to the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # there is no data to remove
        if layer.data.shape[0] == 0:
            return

        data = layer.data
        layer.data = np.delete(data, -1, 0)
        # self.on_run()

    def on_remove_selected(self, which: str, _evt: ty.Any = None) -> None:
        """Remove selected points from the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # there is no data to remove
        if layer.data.shape[0] == 0:
            return
        try:
            layer.remove_selected()
        except IndexError:
            logger.warning(f"Failed to remove selected points from '{which}'.")
        # self.on_run()

    def on_clear(self, which: str, force: bool = True) -> None:
        """Remove point to the image."""
        if which == "both":
            self.on_clear("fixed", force)
            self.on_clear("moving", force)
            return

        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        view = self.view_fixed if which == "fixed" else self.view_moving
        if layer.data.size == 0:
            logger.warning(f"No data points to remove from '{which}'.")
            return
        if force or hp.confirm(self, "Are you sure you want to remove all data points from the points layer?"):
            layer.data = np.zeros((0, 2))
            self.evt_predicted.emit()  # noqa
            self.on_clear_transformation()
            # self.on_run()
            view.remove_layer(layer)

    def on_clear_modality(self, which: str) -> None:
        """Clear specified modality."""
        if hp.confirm(self, f"Are you sure you want to clear the <b>{which}</b> modality data?"):
            self.on_clear(which, force=True)
            if which == "fixed":
                self._fixed_widget.on_close_dataset(force=True)
            else:
                self._moving_widget.on_close_dataset(force=True)

    def on_clear_transformation(self) -> None:
        """Clear transformation and remove image."""
        if self.transform_model.is_valid():
            self.transform_model.clear(clear_model=False, clear_initial=False)
        if self.transformed_moving_image_layer:
            with suppress(ValueError):
                self.view_fixed.layers.remove(self.transformed_moving_image_layer)

    @qdebounced(timeout=200)
    @ensure_main_thread
    def on_run(self, _evt: ty.Any = None) -> None:
        """Compute transformation."""
        self._on_run()
        self.fiducials_dlg.on_load()

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

    def _get_moving_key(self, warn: bool = True) -> str | None:
        visible_datasets = self._get_currently_visible_moving_datasets()
        moving_key = self._moving_widget.dataset_choice.currentText()
        if not visible_datasets and not moving_key:
            if warn:
                hp.warn_pretty(
                    self,
                    "No visible datasets in the moving view. Please select from the list of available options.",
                    "No visible datasets",
                )
            return None
        if moving_key and moving_key not in visible_datasets:
            visible_datasets.append(moving_key)
        if moving_key in visible_datasets and len(visible_datasets) > 1:
            moving_key = hp.choose(
                self, visible_datasets, "Please select a single dataset from the list.", orientation="vertical"
            )
        if not moving_key:
            if warn:
                hp.warn_pretty(
                    self,
                    "No dataset selected. Please select a dataset to save configuration file.",
                    "No dataset selected",
                )
            return None
        return moving_key

    def _get_transform_model(self) -> Transformation | None:
        transform = self.transform_model
        if not transform.is_valid():
            logger.warning("Cannot save transformation - no transformation has been computed.")
            hp.warn_pretty(self, "Cannot save transformation - no transformation has been computed.")
            return None
        is_valid, reason = transform.is_error()
        if not is_valid:
            hp.warn_pretty(
                self,
                f"Cannot save transformations in this state.<br><br><b>Reason</b><br>{reason}"
                f"<br><br><b>Please fix these issues and try again.</b>",
            )
            return None
        is_recommended, reason = transform.is_recommended()
        if not is_recommended and not hp.confirm(
            self,
            f"Saving transformations in this state is not recommended.<br><br><b>Reason</b><br>{reason}"
            f"<br><br>Do you wish to continue?",
        ):
            return None
        return transform

    def on_save_to_project(self, _evt: ty.Any = None) -> None:
        """Export transformation."""
        transform_model = self._get_transform_model()
        if not transform_model:
            return
        # get currently select dataset
        moving_key = self._get_moving_key()
        if not moving_key:
            return

        # get filename which is based on the moving dataset
        filename = self.moving_model.get_filename(moving_key) + "_transform.i2r.json"
        path_ = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=self.CONFIG.output_dir,
            file_filter=ALLOWED_PROJECT_EXPORT_REGISTER_FORMATS,
            base_filename=filename,
        )
        if path_:
            path = Path(path_)
            path = ensure_extension(path, "i2r")
            self.CONFIG.update(output_dir=str(path.parent))
            transform_model.to_file(path, moving_key=moving_key)
            hp.long_toast(
                self,
                "Exported transformation",
                f"Saved project to<br><b>{hp.hyper(path)}</b>",
                icon="success",
                position="top_left",
            )

    def on_transform_moving(self) -> None:
        """Transform image."""
        from image2image_io.utils.warp import ImageWarper

        transform_model = self._get_transform_model()
        if not transform_model:
            return
        # get currently select dataset
        moving_key = self._get_moving_key()
        if not moving_key:
            return

        # get warper
        reader = self.moving_model.get_reader_for_key(moving_key)
        if not reader:
            hp.warn_pretty(self, "Cannot transform image - reader instance was not found.")
            return

        if reader and reader.allow_extraction:
            hp.warn_pretty(
                self,
                "Cannot transform dataset of this type. This can only be applied to images (e.g. OME-TIFF, CZI) and"
                " not IMS datasets.",
            )
            return

        warper = ImageWarper(transform_model.to_dict(moving_key=moving_key), inv=True)
        base_dir = reader.path.parent
        filename = f"{reader.path.stem}-transformed".replace(".ome", "") + ".ome.tiff"
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
        filename = reader.to_ome_tiff(
            filename,
            as_uint8=self.CONFIG.as_uint8,
            tile_size=self.CONFIG.tile_size,
            transformer=warper,
        )
        hp.toast(self, "Image saved", f"Saved image {hp.hyper(filename, moving_key)} as OME-TIFF.", icon="info")

    def _get_currently_visible_moving_datasets(self) -> list[str]:
        """Get currently visible datasets from the moving view."""
        datasets = []
        for layer in self.view_moving.get_layers_of_type(Image):
            name = layer.name
            if " | " in name:
                channel_name, dataset = name.split(" | ")
                datasets.append(dataset)
        return list(set(datasets))

    def on_load_from_project(self, _evt: ty.Any = None) -> None:
        """Import transformation."""
        path_ = hp.get_filename(
            self,
            "Load transformation",
            base_dir=self.CONFIG.output_dir,
            file_filter=ALLOWED_PROJECT_IMPORT_REGISTER_FORMATS,
        )
        self._on_load_from_project(path_)

    def _on_load_from_project(self, path_: str) -> None:
        if path_:
            from image2image.models.transformation import load_transform_from_file
            from image2image.models.utilities import _remove_missing_from_dict
            from image2image.qt._dialogs import LocateFilesDialog
            from image2image.qt._register._select import ImportSelectDialog

            # load transformation
            path = Path(path_)
            self.CONFIG.update(output_dir=str(path.parent), last_dir=str(path.parent))

            # get info on which settings should be imported
            dlg = ImportSelectDialog(self)
            if dlg.exec_():  # type: ignore[attr-defined]
                config = dlg.config
                logger.trace(f"Loaded configuration from {path}: {config}")

                # reset all widgets
                if config["fixed_image"]:
                    self._fixed_widget.dset_dlg.on_close_dataset(force=True)
                if config["moving_image"]:
                    self._moving_widget.dset_dlg.on_close_dataset(force=True)

                # load data from config file
                try:
                    (
                        transformation_type,
                        fixed_paths,
                        fixed_paths_missing,
                        fixed_points,
                        _fixed_resolution,
                        fixed_reader_kws,
                        moving_paths,
                        moving_paths_missing,
                        moving_points,
                        _moving_resolution,
                        moving_reader_kws,
                    ) = load_transform_from_file(path, **config)
                except (ValueError, KeyError) as e:
                    hp.warn_pretty(self, f"Failed to load config from {path}\n{e}", "Failed to load config")
                    logger.exception(e)
                    return

                # locate paths that are missing
                if fixed_paths_missing or moving_paths_missing:
                    locate_dlg = LocateFilesDialog(
                        self,
                        self.CONFIG,
                        fixed_paths_missing + moving_paths_missing or [],  # type: ignore[arg-type]
                    )
                    if locate_dlg.exec_():  # type: ignore[attr-defined]
                        if fixed_paths_missing:
                            fixed_paths = locate_dlg.fix_missing_paths(  # type: ignore[assignment]
                                fixed_paths_missing,
                                fixed_paths,  # type: ignore[arg-type]
                            )
                            fixed_reader_kws = _remove_missing_from_dict(fixed_reader_kws, fixed_paths)
                        if moving_paths_missing:
                            moving_paths = locate_dlg.fix_missing_paths(  # type: ignore[assignment]
                                moving_paths_missing,
                                moving_paths,  # type: ignore[arg-type]
                            )
                            moving_reader_kws = _remove_missing_from_dict(moving_reader_kws, moving_paths)
                # reset initial transform
                self.transform_model.clear(clear_data=False, clear_model=False, clear_initial=True)
                # set new paths
                if fixed_paths:
                    self._fixed_widget.on_set_path(fixed_paths, reader_kws=fixed_reader_kws)
                if moving_paths:
                    self._moving_widget.on_set_path(moving_paths, reader_kws=moving_reader_kws)
                # update points
                if moving_points is not None:
                    self._update_layer_points(self.moving_points_layer, moving_points, block=False)
                if fixed_points is not None:
                    self._update_layer_points(self.fixed_points_layer, fixed_points, block=False)
                if moving_points is not None or fixed_points is not None:
                    self.fiducials_dlg.on_load()  # update table
                    if moving_points is not None and fixed_points is not None:
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

    def on_show_fiducials(self) -> None:
        """View fiducials table."""
        self.fiducials_dlg.show()

    def on_show_initial(self) -> None:
        """Show initial transform dialog."""
        from image2image.qt._register._preprocess import PreprocessMovingDialog

        initial = PreprocessMovingDialog(self)
        initial.show_below_widget(self.initial_btn)
        hp.disable_widgets(self.initial_btn, disabled=True)

    def on_show_generate(self) -> None:
        """Show initial transform dialog."""
        from image2image.qt._register._guess import GuessDialog

        initial = GuessDialog(self)
        initial.show_left_of_widget(self.guess_btn, x_offset=-100)
        hp.disable_widgets(self.guess_btn, disabled=True)

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
            try:
                index = [layer.name for layer in self.moving_image_layer].index(name)
                moving_image_layer = self.moving_image_layer[index]
            except ValueError:
                logger.warning(f"Layer '{name}' not found in the moving image.")
                try:
                    moving_image_layer = self.moving_image_layer[0]
                except IndexError:
                    return

        # retrieve affine matrix which might be composite of initial + transform or just initial
        affine = self.transform.params  # if self.transform is not None else self.transform_model.moving_initial_affine

        # add image and apply transformation
        if READER_CONFIG.view_type == ViewType.OVERLAY:
            colormap = get_colormap(0, self.view_fixed.layers, moving_image_layer.colormap)
        else:
            colormap = moving_image_layer.colormap
        contrast_limits = moving_image_layer.contrast_limits
        update = self.transformed_moving_image_layer is not None
        if update:
            try:
                self.transformed_moving_image_layer.affine = affine
                if update_data:
                    self.transformed_moving_image_layer.data = moving_image_layer.data
                    self.transformed_moving_image_layer.colormap = colormap
                    self.transformed_moving_image_layer.reset_contrast_limits()
                    self.transformed_moving_image_layer.contrast_limits = contrast_limits
            except (ValueError, TypeError, KeyError):
                update = False
                self.view_fixed.remove_layer(self.transformed_moving_image_layer)
        if not update:
            self.view_fixed.viewer.add_image(
                moving_image_layer.data,
                name="Transformed",
                blending="translucent",
                affine=affine,
                colormap=colormap,
                opacity=self.CONFIG.opacity_moving / 100,
                contrast_limits=contrast_limits,
            )
        try:
            self.transformed_moving_image_layer.visible = READER_CONFIG.show_transformed
        except TypeError:
            logger.warning("Failed to apply transformation to the moving image.")
        self._move_layer(self.view_fixed, self.transformed_moving_image_layer, -1, False)
        self._select_point_layer("fixed")

    def on_add_transformed_moving(self, name: str | None = None) -> None:
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
                opacity=self.CONFIG.opacity_moving / 100,
            )

    @qdebounced(timeout=200, leading=True)
    @ensure_main_thread
    def on_predict(self, which: str, evt: ty.Any = None) -> None:
        """Predict transformation from either image."""
        self._on_predict(which, evt)

    @contextmanager
    def disable_prediction(self) -> ty.Generator:
        """Disable prediction."""
        self.is_predicting = True
        yield self
        self.is_predicting = False

    def _on_predict(self, which: str, _evt: ty.Any = None) -> None:
        n_fixed = len(self.fixed_points_layer.data)
        n_moving = len(self.moving_points_layer.data)
        if (
            not self.CONFIG.enable_prediction  # user disabled
            or self.is_predicting  # already predicting
            or n_fixed == n_moving  # no need to predict
            or abs(n_fixed - n_moving) > 1  # unreliable prediction
        ):
            return
        if self.transform is None:
            logger.warning("Cannot predict - no transformation has been computed.")
            return

        with self.disable_prediction():
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
            with self.fixed_points_layer.events.data.blocker(), self.moving_points_layer.events.data.blocker():
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

        def _update_points_layer(layer: Points, size: float) -> None:
            with suppress(IndexError):
                layer.size = size
            with suppress(IndexError):
                layer.current_size = size

        self.CONFIG.update(
            size_fixed=self.fixed_point_size.value(),
            size_moving=self.moving_point_size.value(),
            opacity_fixed=self.fixed_opacity.value(),
            opacity_moving=self.moving_opacity.value(),
        )

        # update point size
        if which == "fixed":
            if self.fixed_points_layer:
                _update_points_layer(self.fixed_points_layer, self.CONFIG.size_fixed)
            if FIXED_TMP_POINTS in self.view_fixed.layers:
                _update_points_layer(self.view_fixed.layers[FIXED_TMP_POINTS], self.CONFIG.size_fixed)
            if self.fixed_image_layer:
                self.fixed_image_layer[0].opacity = self.CONFIG.opacity_fixed / 100
        if which == "moving":
            if self.moving_points_layer:
                _update_points_layer(self.moving_points_layer, self.CONFIG.size_moving)
            if MOVING_TMP_POINTS in self.view_moving.layers:
                _update_points_layer(self.view_moving.layers[MOVING_TMP_POINTS], self.CONFIG.size_moving)
            if self.transformed_moving_image_layer:
                self.transformed_moving_image_layer.opacity = self.CONFIG.opacity_moving / 100

    def on_update_text(self, _: ty.Any = None, block: bool = False, refresh: bool = False) -> None:
        """Update text data in each layer."""

        def _update_points_text(layer_: Points) -> None:
            if len(layer_.data) == 0:
                return
            with suppress(IndexError, ValueError):
                if block:
                    with layer_.text.events.blocker():
                        layer_.text = _get_text_format()
                else:
                    layer_.text = _get_text_format()
                if refresh:
                    layer_.refresh_text()

        self.CONFIG.update(label_color=self.text_color.hex_color, label_size=self.text_size.value())

        # update text information
        for layer in [self.fixed_points_layer, self.moving_points_layer]:
            _update_points_text(layer)
        # if FIXED_TMP_POINTS in self.view_fixed.layers:
        #     _update_points_text(self.view_fixed.layers[FIXED_TMP_POINTS])
        # if MOVING_TMP_POINTS in self.view_moving.layers:
        #     _update_points_text(self.view_moving.layers[MOVING_TMP_POINTS])

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
    def on_toggle_transformed_image(self, _: ty.Any = None) -> None:
        """Toggle visibility of transformed image."""
        if self.transformed_moving_image_layer:
            self._moving_widget.toggle_transformed()
            logger.trace(
                f"Transformed image is {'visible' if self.transformed_moving_image_layer.visible else 'hidden'}."
            )

    @qdebounced(timeout=50)
    def on_toggle_synchronization(self, _: ty.Any = None) -> None:
        """Toggle synchronization of views."""
        self.CONFIG.update(sync_views=not self.CONFIG.sync_views)
        with hp.qt_signals_blocked(self.synchronize_zoom):
            self.synchronize_zoom.setChecked(self.CONFIG.sync_views)
        logger.trace(f"Synchronization of views is {'enabled' if self.CONFIG.sync_views else 'disabled'}.")

    @qdebounced(timeout=50)
    def on_zoom_on_point(self, increment: int) -> None:
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

    @qdebounced(timeout=50)
    def on_adjust_transformed_opacity(self, increase_by: int) -> None:
        """Toggle visibility of transformed image."""
        self.moving_opacity.setValue(self.moving_opacity.value() + increase_by)
        hp.notification(self, "Opacity", f"Adjusted opacity to {self.moving_opacity.value()}.", position="top_right")

    def on_update_settings(self) -> None:
        """Update config."""
        self.CONFIG.update(
            sync_views=self.synchronize_zoom.isChecked(), enable_prediction=self.enable_prediction_checkbox.isChecked()
        )
        self.on_sync_views_fixed()

    @qdebounced(timeout=200, leading=False)
    def on_sync_views_fixed(self, _event: Event | None = None) -> None:
        """Synchronize views."""
        if not self._zooming:
            self._on_sync_views("fixed")

    @qdebounced(timeout=200, leading=False)
    def on_sync_views_moving(self, _event: Event | None = None) -> None:
        """Synchronize views."""
        if not self._zooming:
            self._on_sync_views("moving")

    @qdebounced(timeout=200, leading=False)
    def _on_sync_views(self, from_which: str):
        self.__on_sync_views(from_which)

    def __on_sync_views(self, from_which: str, force: bool = False) -> None:
        if not self.CONFIG.sync_views:
            return
        if self._zooming and not force:
            logger.trace("Zooming in progress, skipping synchronization.")
            return
        if self.transform is None:
            logger.trace("No transformation computed, skipping synchronization.")
            return
        to_which = "moving" if from_which == "fixed" else "fixed"
        with self.zooming():
            before_func = lambda x: x  # noqa
            after_func = lambda x: x  # noqa
            if from_which == "fixed":
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
            logger.trace(f"Synchronized views: {from_which} -> {to_which}")

    def on_update_config(self, _: ty.Any = None) -> None:
        """Update values in config."""
        self.CONFIG.as_uint8 = self.as_uint8.isChecked()
        self.CONFIG.tile_size = int(self.tile_size.currentText())

    # noinspection PyAttributeOutsideInit
    def _setup_ui(self) -> None:
        """Create panel."""
        view_layout = self._make_image_layout()

        side_widget = QWidget()
        side_widget.setMinimumWidth(400)
        side_widget.setMaximumWidth(400)

        self.import_project_btn = hp.make_btn(
            side_widget,
            "Import project",
            tooltip="Import previously computed transformation.",
            func=self.on_load_from_project,
        )

        self._fixed_widget = FixedWidget(
            self,
            self.view_fixed,
            self.CONFIG,
            project_extension=[".i2r.json", ".i2r.toml"],
        )
        self._moving_widget = MovingWidget(
            self,
            self.view_moving,
            self.CONFIG,
            project_extension=[".i2r.json", ".i2r.toml"],
            allow_iterate=True,
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
            "Initialize moving image...",
            func=self.on_show_initial,
            tooltip="You can optionally rotate or flip the moving image so that it's easier to align with the fixed"
            " image. This button will be disabled if there are ANY points in the moving image.",
        )
        self.guess_btn = hp.make_btn(
            side_widget,
            "Generate fiducials...",
            func=self.on_show_generate,
            tooltip="You can generate fiducial markers based on the image. This will use the 'outline' of the image to "
            " determine the contour.",
        )
        self.fiducials_btn = hp.make_btn(
            side_widget,
            "Show fiducials table...",
            tooltip="Show fiducial markers table where you can view and edit the markers",
            func=self.on_show_fiducials,
        )
        self.enable_prediction_checkbox = hp.make_checkbox(
            self,
            tooltip="Enable prediction of points based on the transformation.",
            func=self.on_update_settings,
            value=self.CONFIG.enable_prediction,
        )
        self.close_btn = hp.make_qta_btn(
            side_widget,
            "delete",
            tooltip="Close project.<br>Right-click to open menu.",
            func=self.on_close,
            func_menu=self.on_close_menu,
            standout=True,
            normal=True,
        )
        self.export_project_btn = hp.make_btn(
            side_widget,
            "Export project...",
            tooltip="Export transformation to file. XML format is usable by MATLAB fusion.",
            func=self.on_save_to_project,
        )

        self.as_uint8 = hp.make_checkbox(
            self,
            "",
            tooltip=C.UINT8_TIP,
            value=self.CONFIG.as_uint8,
            func=self.on_update_config,
        )
        self.tile_size = hp.make_combobox(
            self,
            ["256", "512", "1024", "2048", "4096"],
            tooltip="Specify size of the tile. Default is 512",
            default="512",
            value=f"{self.CONFIG.tile_size}",
            func=self.on_update_config,
        )

        self.hidden_settings = hidden_settings = hp.make_advanced_collapsible(
            side_widget,
            "Export transformed image",
            allow_checkbox=False,
            allow_icon=False,
        )
        hidden_settings.addRow(
            hp.make_label(
                self,
                "Here you can apply the affine transformation to the image (e.g. CZI or OME-TIFF) in case elastix or"
                " valis methods are not providing desired results.",
                wrap=True,
            )
        )
        hidden_settings.addRow(hp.make_label(self, "Tile size"), self.tile_size)
        hidden_settings.addRow(
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
        hidden_settings.addRow(
            hp.make_btn(
                side_widget,
                "Export transformed image...",
                tooltip="Transform image using the specified transformation matrix and save it as OME-TIFF.",
                func=self.on_transform_moving,
            )
        )

        side_layout = hp.make_form_layout(parent=side_widget)
        side_layout.addRow(self.import_project_btn)
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._fixed_widget)
        side_layout.addRow(hp.make_h_line_with_text("+"))
        side_layout.addRow(self._moving_widget)
        side_layout.addRow(hp.make_h_line_with_text("Area of interest"))
        side_layout.addRow(self._make_focus_layout())
        side_layout.addRow(hp.make_h_line_with_text("Transformation"))
        side_layout.addRow(hp.make_h_layout(self.initial_btn, self.guess_btn, spacing=2, stretch_id=(0, 1)))
        side_layout.addRow(self.fiducials_btn)
        side_layout.addRow("Enable prediction", self.enable_prediction_checkbox)
        side_layout.addRow(hp.make_btn(side_widget, "Compute transformation", func=self.on_run))
        side_layout.addRow(hp.make_h_layout(self.close_btn, self.export_project_btn, stretch_id=(1,), spacing=2))
        side_layout.addRow(hidden_settings)
        side_layout.addRow(hp.make_h_line_with_text("About transformation"))
        side_layout.addRow(hp.make_label(self, "Estimated error"), self.transform_error)
        side_layout.addRow(self.transform_info)
        side_layout.addRow(hp.make_spacer_widget())

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

    def on_close(self, force: bool = False) -> None:
        """Close project and clear all data."""
        if force or hp.confirm(self, "Are you sure you want to close the project?", "Close project?"):
            self._moving_widget.on_close_dataset(force=True)
            self.on_clear("moving", force=True)
            self._fixed_widget.on_close_dataset(force=True)
            self.on_clear("fixed", force=True)
            self.transform_model.clear(clear_data=False, clear_model=False, clear_initial=True)
            self.on_run()
            # self.fiducials_dlg.on_load()
            # self.on_activate_initial()

    def on_close_menu(self) -> None:
        """Save menu."""
        menu = hp.make_menu(self.close_btn)
        hp.make_menu_item(self, "Clear fixed modality", menu=menu, func=lambda: self.on_clear_modality("fixed"))
        hp.make_menu_item(self, "Clear moving modality", menu=menu, func=lambda: self.on_clear_modality("moving"))
        menu.addSeparator()
        hp.make_menu_item(
            self,
            "Clear all fixed and moving points (without confirmation).",
            menu=menu,
            func=lambda *args: self.on_clear("both", force=True),
        )
        menu.addSeparator()
        hp.make_menu_item(self, "Close project", menu=menu, func=self.on_close, icon="delete")
        hp.make_menu_item(
            self, "Close project (without confirmation)", menu=menu, func=lambda: self.on_close(True), icon="delete"
        )
        hp.show_right_of_mouse(menu)

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
        menu_tools = self._make_tools_menu()
        hp.make_menu_item(
            self,
            "Show fiducials table...",
            "Ctrl+F",
            menu=menu_tools,
            func=self.on_show_fiducials,
            icon="fiducial",
            insert=True,
        )

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def _make_focus_layout(self) -> QFormLayout:
        self.lock_btn = hp.make_lock_btn(
            self,
            func=self.on_lock,
            tooltip="Lock the area of interest. Press <b>L</b> on your keyboard to lock.",
            normal=True,
            standout=True,
        )
        # self.set_current_focus_btn = hp.make_btn(self, "Set current range", func=self.on_set_focus)
        self.x_center = hp.make_double_spin_box(
            self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the x-axis.", prefix="x = "
        )
        self.y_center = hp.make_double_spin_box(
            self, -1e5, 1e5, step_size=500, tooltip="Center of the area of interest along the y-axis.", prefix="y = "
        )
        self.zoom = hp.make_double_spin_box(self, -1e5, 1e5, step_size=0.5, n_decimals=4, tooltip="Zoom factor.")
        self.use_focus_btn = hp.make_btn(
            self,
            "Zoom-in",
            func=self.on_apply_focus,
            tooltip="Zoom-in to an area of interest. Press <b>Z</b> on your keyboard to zoom-in.",
        )

        layout = hp.make_form_layout()
        layout.addRow(hp.make_label(self, "Center"), hp.make_h_layout(self.x_center, self.y_center))
        layout.addRow(hp.make_label(self, "Zoom"), self.zoom)
        layout.addRow(hp.make_h_layout(self.lock_btn, self.use_focus_btn, stretch_id=1, spacing=2))
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
        view_layout.addWidget(info)
        view_layout.addLayout(self._make_moving_view())
        return view_layout

    def _make_fixed_view(self) -> QHBoxLayout:
        self.view_fixed = self._make_image_view(self, add_toolbars=False, disable_new_layers=True)
        self.view_fixed.viewer.text_overlay.text = "Fixed"
        self.view_fixed.viewer.text_overlay.position = "top_left"
        self.view_fixed.viewer.text_overlay.font_size = 10
        self.view_fixed.viewer.text_overlay.visible = True
        self.view_fixed.viewer.scale_bar.unit = "um"
        self.view_fixed.widget.canvas.events.key_press.connect(self.keyPressEvent)

        toolbar = QtMiniToolbar(self, Qt.Orientation.Vertical, add_spacer=True, icon_size="normal")
        _fixed_clear_btn = toolbar.insert_qta_tool(
            "remove_all",
            func=lambda *args: self.on_clear("fixed", force=False),
            func_menu=lambda *args: self.on_clear("fixed", force=True),
            tooltip="Remove all points from the fixed image (need to confirm)."
            "\nRight-click to clear all points without confirmation.",
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
        self.fixed_pan_btn = toolbar.insert_qta_tool(
            "pan_zoom",
            func=lambda *args: self.on_panzoom("fixed"),
            tooltip="Switch to zoom-only mode. Press <b>1</b> on your keyboard to activate...",
            checkable=True,
        )
        hp.make_radio_btn_group(self, [self.fixed_pan_btn, self.fixed_add_btn, self.fixed_move_btn])
        _fixed_bring_to_top = toolbar.insert_qta_tool(
            "bring_to_top",
            func=lambda *args: self._select_point_layer("fixed"),
            tooltip="Bring points layer to the top.",
        )
        _fixed_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_fixed.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )
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
        self.view_moving.viewer.scale_bar.unit = "um"
        self.view_moving.widget.canvas.events.key_press.connect(self.keyPressEvent)

        toolbar = QtMiniToolbar(self, Qt.Orientation.Vertical, add_spacer=True, icon_size="normal")
        _moving_clear_btn = toolbar.insert_qta_tool(
            "remove_all",
            func=lambda *args: self.on_clear("moving", force=False),
            func_menu=lambda *args: self.on_clear("moving", force=True),
            tooltip="Remove all points from the moving image (need to confirm)."
            "\nRight-click to clear all points without confirmation.",
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
        self.moving_pan_btn = toolbar.insert_qta_tool(
            "pan_zoom",
            func=lambda *args: self.on_panzoom("moving"),
            tooltip="Switch to zoom-only mode. Press <b>1</b> on your keyboard to activate...",
            checkable=True,
        )
        hp.make_radio_btn_group(self, [self.moving_pan_btn, self.moving_add_btn, self.moving_move_btn])
        _moving_bring_to_top = toolbar.insert_qta_tool(
            "bring_to_top",
            func=lambda *args: self._select_point_layer("moving"),
            tooltip="Bring points layer to the top.",
        )
        _moving_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_moving.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )
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

        self.synchronize_zoom = hp.make_checkbox(
            self,
            "Sync views",
            "Synchronize zoom between views. It only starts taking effect once transformation model has been"
            " calculated.",
            value=self.CONFIG.sync_views,
            func=self.on_toggle_synchronization,
        )
        self.synchronize_zoom.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.statusbar.insertPermanentWidget(0, self.synchronize_zoom)
        self.statusbar.insertPermanentWidget(1, hp.make_v_line())

        self.fixed_point_size = hp.make_int_spin_box(
            self,
            value=self.CONFIG.size_fixed,
            tooltip="Size of the points shown in the fixed image.",
            minimum=1,
            maximum=40,
        )
        self.fixed_point_size.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.fixed_point_size.valueChanged.connect(partial(self.on_update_layer, "fixed"))  # noqa
        self.statusbar.insertPermanentWidget(2, hp.make_label(self, "Marker size (F)"))
        self.statusbar.insertPermanentWidget(3, self.fixed_point_size)

        self.moving_point_size = hp.make_int_spin_box(
            self,
            value=self.CONFIG.size_moving,
            tooltip="Size of the points shown in the moving image.",
            minimum=1,
            maximum=40,
        )
        self.moving_point_size.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.moving_point_size.valueChanged.connect(partial(self.on_update_layer, "moving"))  # noqa
        self.statusbar.insertPermanentWidget(4, hp.make_label(self, "(M)"))
        self.statusbar.insertPermanentWidget(5, self.moving_point_size)
        self.statusbar.insertPermanentWidget(6, hp.make_v_line())

        self.text_size = hp.make_int_spin_box(
            self,
            value=self.CONFIG.label_size,
            minimum=4,
            maximum=60,
            tooltip="Size of the text associated with each label.",
        )
        self.text_size.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.text_size.valueChanged.connect(self.on_update_text)  # noqa
        self.statusbar.insertPermanentWidget(7, hp.make_label(self, "Label size"))
        self.statusbar.insertPermanentWidget(8, self.text_size)

        self.text_color = hp.make_swatch(
            self, default=self.CONFIG.label_color, tooltip="Color of the text associated with each label."
        )
        self.text_color.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)  # type: ignore[attr-defined]
        self.text_color.evt_color_changed.connect(self.on_update_text)  # noqa
        self.statusbar.insertPermanentWidget(9, self.text_color)
        self.statusbar.insertPermanentWidget(10, hp.make_v_line(), stretch=1)

        self.fixed_opacity = hp.make_int_spin_box(
            self,
            value=self.CONFIG.opacity_fixed,
            step_size=10,
            tooltip="Opacity of the fixed image",
        )
        self.fixed_opacity.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.fixed_opacity.valueChanged.connect(partial(self.on_update_layer, "fixed"))  # noqa
        self.statusbar.insertPermanentWidget(11, hp.make_label(self, "Image opacity (F)"))
        self.statusbar.insertPermanentWidget(12, self.fixed_opacity)

        self.moving_opacity = hp.make_int_spin_box(
            self,
            value=self.CONFIG.opacity_moving,
            step_size=10,
            tooltip="Opacity of the moving image in the fixed view",
        )
        self.moving_opacity.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.moving_opacity.valueChanged.connect(partial(self.on_update_layer, "moving"))  # noqa
        self.statusbar.insertPermanentWidget(13, hp.make_label(self, "(M)"))
        self.statusbar.insertPermanentWidget(14, self.moving_opacity)
        self.statusbar.insertPermanentWidget(15, hp.make_v_line())

    def on_toggle_mode(self, which: str | ty.Literal["fixed", "moving", "both"], mode: str | Mode) -> None:
        """Toggle mode."""
        which = ["fixed", "moving"] if which == "both" else [which]
        for w in which:
            self.on_points_mode(w, mode=mode)
            if mode == Mode.PAN_ZOOM:
                self.on_panzoom(w)
            elif mode == Mode.ADD:
                self.on_add(w)
            elif mode == Mode.SELECT:
                self.on_move(w)

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
        if key == Qt.Key.Key_Escape:
            ignore = True
        elif key == Qt.Key.Key_1:
            self.on_toggle_mode("both", mode=Mode.PAN_ZOOM)
            ignore = True
        elif key == Qt.Key.Key_2:
            self.on_toggle_mode("both", mode=Mode.ADD)
            ignore = True
        elif key == Qt.Key.Key_3:
            self.on_toggle_mode("both", mode=Mode.SELECT)
        elif key == Qt.Key.Key_Z:
            self.on_apply_focus()
            ignore = True
        elif key == Qt.Key.Key_L:
            self.on_set_focus()
            ignore = True
        elif key == Qt.Key.Key_T:
            self.on_toggle_transformed_image()
            ignore = True
        elif key == Qt.Key.Key_V:
            self.on_toggle_transformed_visibility()
        elif key == Qt.Key.Key_R:
            self.on_toggle_transformed_view_type()
            ignore = True
        elif key == Qt.Key.Key_A:
            self.on_zoom_on_point(-1)
            ignore = True
        elif key == Qt.Key.Key_D:
            self.on_zoom_on_point(1)
            ignore = True
        elif key == Qt.Key.Key_S:
            self.on_toggle_synchronization()
        elif key == Qt.Key.Key_N:
            self.on_increment_dataset(1)
        elif key == Qt.Key.Key_P:
            self.on_increment_dataset(-1)
            ignore = True
        elif key == Qt.Key.Key_E:  # increase opacity
            self.on_adjust_transformed_opacity(25)
            ignore = True
        elif key == Qt.Key.Key_Q:  # decrease opacity
            self.on_adjust_transformed_opacity(-25)
            ignore = True
        return ignore

    def on_increment_dataset(self, increment: int) -> None:
        """Increase the dataset index."""
        hp.increment_combobox(self._moving_widget.dataset_choice, increment)
        hp.notification(
            self,
            "Dataset",
            f"Switched to dataset {self._moving_widget.dataset_choice.currentText()}.",
            position="top_right",
        )

    def on_toggle_transformed_view_type(self) -> None:
        """Toggle between image and random."""
        reader = self.get_current_moving_reader()
        is_overlay = READER_CONFIG.view_type == "overlay"
        if (
            is_overlay
            and reader
            and max(reader.image_shape) > 10_000
            and not hp.confirm(
                self,
                "The image is quite large to be displayed as random image.<br>Do you wish to <b>continue</b>?",
                "Warning",
            )
        ):
            READER_CONFIG.view_type = "random"
            return
        READER_CONFIG.view_type = "random" if is_overlay else "overlay"
        self._moving_widget.view_type_choice.value = "Random" if is_overlay else "Overlay"
        self.on_change_view_type(READER_CONFIG.view_type)
        hp.notification(
            self,
            "View type",
            f"Switched to {self._moving_widget.view_type_choice.value}.",
            position="top_right",
        )

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not self.CONFIG.confirm_close
            or QtConfirmCloseDialog(self, "confirm_close", self.on_save_to_project, self.CONFIG).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and self.CONFIG.confirm_close
            and self.transform_model.is_valid()
            and QtConfirmCloseDialog(self, "confirm_close", self.on_save_to_project, self.CONFIG).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return

        if self._console:
            self._console.close()
        self.CONFIG.save()
        READER_CONFIG.save()
        evt.accept()

    def dropEvent(self, event):
        """Override Qt method."""
        from qtextra.widgets.qt_select_one import QtScrollablePickOption

        self._setup_config()
        hp.update_property(self.centralWidget(), "drag", False)

        which = None
        files = event.mimeData().urls()
        if files and len(files) == 1:
            url = files[0]
            file = Path(url.toLocalFile() if url.isLocalFile() else url.toString())
            if len(file.suffixes) == 2 and file.suffixes[0] == ".i2r":
                which = "fixed"

        if files and which is None:
            dlg = QtScrollablePickOption(
                self,
                "Please select which view would you like to add the image(s) to?",
                {"Fixed image": "fixed", "Moving image": "moving"},
                orientation="vertical",
            )
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

        if show_register_tutorial(self):
            self.CONFIG.update(first_time=False)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="register", level=0)
