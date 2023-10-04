"""Registration dialog."""
import typing as ty
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image
from napari.layers.points.points import Mode, Points
from napari.layers.utils._link_layers import link_layers
from qtextra._napari.image.viewer import NapariImageView
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_image_button import QtImagePushButton
from qtextra.widgets.qt_mini_toolbar import QtMiniToolbar
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout, QHBoxLayout, QMenuBar, QSizePolicy, QVBoxLayout, QWidget
from superqt import ensure_main_thread

# need to load to ensure all assets are loaded properly
import image2image.assets  # noqa: F401
from image2image import __version__
from image2image._select import FixedWidget, MovingWidget
from image2image.config import CONFIG
from image2image.dialog_base import Window
from image2image.enums import (
    ALLOWED_EXPORT_REGISTER_FORMATS,
    ALLOWED_IMPORT_REGISTER_FORMATS,
    TRANSFORMATION_TRANSLATIONS,
    ViewType,
)
from image2image.models.data import DataModel
from image2image.models.transformation import Transformation
from image2image.utilities import (
    _get_text_data,
    _get_text_format,
    get_colormap,
    init_points_layer,
    style_form_layout,
)

if ty.TYPE_CHECKING:
    from skimage.transform import ProjectiveTransform


class ImageRegistrationWindow(Window):
    """Image registration dialog."""

    view_fixed: NapariImageView
    view_moving: NapariImageView
    fixed_image_layer: ty.Optional[ty.List["Image"]] = None
    moving_image_layer: ty.Optional[ty.List["Image"]] = None
    _table, _console = None, None

    # events
    evt_predicted = Signal()

    def __init__(self, parent: ty.Optional[QWidget]):
        super().__init__(parent, f"image2image: Simple image registration tool (v{__version__})")
        self.transform_model = Transformation(
            fixed_model=self.fixed_model,
            moving_model=self.moving_model,
            fixed_points=self.fixed_points_layer.data,
            moving_points=self.moving_points_layer.data,
        )

    @property
    def transform(self) -> ty.Optional["ProjectiveTransform"]:
        """Retrieve transform."""
        transform = self.transform_model
        if transform.is_valid():
            return transform.transform
        return None

    @property
    def transformed_moving_image_layer(self) -> ty.Optional["Image"]:
        """Return transformed, moving image layer."""
        if "Transformed" in self.view_fixed.layers:
            return self.view_fixed.layers["Transformed"]
        return None

    @property
    def fixed_points_layer(self) -> Points:
        """Fixed points layer."""
        if "Fixed (points)" not in self.view_fixed.layers:
            layer = self.view_fixed.viewer.add_points(
                None,
                size=self.fixed_point_size.value(),
                name="Fixed (points)",
                face_color="green",
                edge_color="white",
                symbol="ring",
            )
            visual = self.view_fixed.widget.layer_to_visual[layer]
            init_points_layer(layer, visual)
            connect(self.fixed_points_layer.events.data, self.on_run, state=True)
            connect(self.fixed_points_layer.events.add_point, partial(self.on_predict, "fixed"), state=True)
        return self.view_fixed.layers["Fixed (points)"]

    @property
    def moving_points_layer(self) -> Points:
        """Fixed points layer."""
        if "Moving (points)" not in self.view_moving.layers:
            layer = self.view_moving.viewer.add_points(
                None,
                size=self.moving_point_size.value(),
                name="Moving (points)",
                face_color="red",
                edge_color="white",
                symbol="ring",
            )
            visual = self.view_moving.widget.layer_to_visual[layer]
            init_points_layer(layer, visual)
            connect(self.moving_points_layer.events.data, self.on_run, state=True)
            connect(self.moving_points_layer.events.add_point, partial(self.on_predict, "moving"), state=True)
        return self.view_moving.layers["Moving (points)"]

    def setup_events(self, state: bool = True) -> None:
        """Additional setup."""
        # fixed widget
        connect(self._fixed_widget.dataset_dlg.evt_loading, partial(self.on_indicator, which="fixed"), state=state)
        connect(self._fixed_widget.dataset_dlg.evt_loaded, self.on_load_fixed, state=state)
        connect(self._fixed_widget.dataset_dlg.evt_closed, self.on_close_fixed, state=state)
        connect(self._fixed_widget.evt_toggle_channel, partial(self.on_toggle_channel, which="fixed"), state=state)
        connect(
            self._fixed_widget.evt_toggle_all_channels, partial(self.on_toggle_all_channels, which="fixed"), state=state
        )
        # imaging widget
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
        connect(self._moving_widget.evt_view_type, self.on_change_view_type, state=state)

    def on_indicator(self, which: str, state: bool = True) -> None:
        """Set indicator."""
        indicator = self.moving_indicator if which == "moving" else self.fixed_indicator
        indicator.setVisible(state)

    def on_close_fixed(self, model: DataModel) -> None:
        """Close fixed image."""
        self._close_model(model, self.view_fixed, "fixed view")

    @property
    def fixed_model(self) -> "DataModel":
        """Return transform model."""
        return self._fixed_widget.model

    @property
    def moving_model(self) -> "DataModel":
        """Return transform model."""
        return self._moving_widget.model

    @ensure_main_thread
    def on_load_fixed(self, model: DataModel, channel_list: ty.List[str]) -> None:
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

    def _on_load_fixed(self, model: DataModel, channel_list: ty.Optional[ty.List[str]] = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading fixed data with {model.n_paths} paths...")
            self._plot_fixed_layers(channel_list)
            self.view_fixed.viewer.reset_view()
        logger.info(f"Loaded fixed data in {timer()}")

    def _plot_fixed_layers(self, channel_list: ty.Optional[ty.List[str]] = None) -> None:
        self.fixed_image_layer = self._plot_image_layers(self.fixed_model, self.view_fixed, channel_list, "fixed view")
        if isinstance(self.fixed_image_layer, list) and len(self.fixed_image_layer) > 1:
            link_layers(self.fixed_image_layer, attributes=("opacity",))

    def on_toggle_channel(self, name: str, state: bool, which: str) -> None:
        """Toggle channel."""
        view = self.view_fixed if which == "fixed" else self.view_moving
        if name not in view.layers:
            logger.warning(f"Layer '{name}' not found in the {which} view.")
            return
        view.layers[name].visible = state

    def on_toggle_all_channels(self, state: bool, which: str) -> None:
        """Toggle channel."""
        view = self.view_fixed if which == "fixed" else self.view_moving
        for layer in view.layers:
            if isinstance(layer, Image):
                layer.visible = state

    def on_close_moving(self, model: DataModel) -> None:
        """Close moving image."""
        self._close_model(model, self.view_moving, "moving view")

    @ensure_main_thread
    def on_load_moving(self, model: DataModel, channel_list: ty.List[str]) -> None:
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

    def _on_load_moving(self, model: DataModel, channel_list: ty.Optional[ty.List[str]] = None) -> None:
        with MeasureTimer() as timer:
            logger.info(f"Loading moving data with {model.n_paths} paths...")
            self._plot_moving_layers(channel_list)
            self.on_apply(update_data=True)
            self.view_moving.viewer.reset_view()
        logger.info(f"Loaded moving data in {timer()}")

    def _plot_moving_layers(self, channel_list: ty.Optional[ty.List[str]] = None) -> None:
        CONFIG.view_type = ViewType(CONFIG.view_type)
        is_overlay = CONFIG.view_type == ViewType.OVERLAY
        wrapper = self.moving_model.get_wrapper()
        if not wrapper:
            return
        if channel_list is None:
            channel_list = wrapper.channel_names()

        moving_image_layer = []
        for index, (name, array) in enumerate(wrapper.channel_image_iter()):
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
                    )
                )
            logger.trace(f"Added '{name}' to fixed view in {timer()}.")
        # hide away other layers if user selected 'random' view
        if CONFIG.view_type == ViewType.RANDOM:
            for index, layer in enumerate(moving_image_layer):
                if index > 0:
                    layer.visible = False
        self.moving_image_layer = moving_image_layer

    def on_change_view_type(self, _view_type: str) -> None:
        """Change view type."""
        if self.moving_model.n_paths:
            self._plot_moving_layers()
            self.on_apply(update_data=True)

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

    def _get_mode_button(self, which: str, mode: Mode) -> ty.Optional[QtImagePushButton]:
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

    def on_mode(self, which: str, evt: ty.Any = None, mode: ty.Optional[Mode] = None) -> None:
        """Update mode."""
        widget = self._get_mode_button(which, mode or evt.mode)
        if widget:
            widget.setChecked(True)

    def on_panzoom(self, which: str, _evt: ty.Any = None) -> None:
        """Switch to `panzoom` tool."""
        self._select_layer(which)
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "pan_zoom"

    def on_move(self, which: str, _evt: ty.Any = None) -> None:
        """Move points."""
        self._select_layer(which)
        widget = self.fixed_move_btn if which == "fixed" else self.moving_move_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "select" if widget.isChecked() else "pan_zoom"

    def on_add(self, which: str, _evt: ty.Any = None) -> None:
        """Add point to the image."""
        self._select_layer(which)
        # extract button and layer based on the appropriate mode
        widget = self.fixed_add_btn if which == "fixed" else self.moving_add_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # make sure the 'add' mode is active
        layer.mode = "add" if widget.isChecked() else "pan_zoom"

    def on_remove(self, which: str, _evt: ty.Any = None) -> None:
        """Remove point to the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # there is no data to remove
        if layer.data.shape[0] == 0:
            return

        data = layer.data
        layer.data = np.delete(data, -1, 0)

    def on_remove_selected(self, which: str, _evt: ty.Any = None) -> None:
        """Remove selected points from the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # there is no data to remove
        if layer.data.shape[0] == 0:
            return
        layer.remove_selected()

    def on_clear_all(self) -> None:
        """Clear arr data."""
        if hp.confirm(self, "Are you sure you want to remove <b>all</b> images and data points?"):
            self.on_clear("fixed", force=True)
            self.on_clear("moving", force=True)
            self._moving_widget.dataset_dlg._on_close_dataset(force=True)
            self._fixed_widget.dataset_dlg._on_close_dataset(force=True)

    def on_clear(self, which: str, force: bool = True) -> None:
        """Remove point to the image."""
        if force or hp.confirm(self, "Are you sure you want to remove all data points from the points layer?"):
            layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
            layer.data = np.zeros((0, 2))
            self.on_clear_transformation()

    def on_clear_transformation(self) -> None:
        """Clear transformation and remove image."""
        if self.transform_model.is_valid():
            self.transform_model.clear(clear_model=False)
        if self.transformed_moving_image_layer:
            with suppress(ValueError):
                self.view_fixed.layers.remove(self.transformed_moving_image_layer)

    @ensure_main_thread
    def on_run(self, _evt: ty.Any = None) -> None:
        """Compute transformation."""
        from image2image.utilities import compute_transform

        if not self.fixed_points_layer or not self.moving_points_layer:
            self.on_notify_warning("There must be at least three points before we can compute the transformation.")
            return

        # execute transform calculation
        n_fixed = len(self.fixed_points_layer.data)
        n_moving = len(self.moving_points_layer.data)
        if 3 <= n_fixed == n_moving >= 3:
            method = self.transform_choice.currentText().lower()
            transform = compute_transform(
                self.moving_points_layer.data,  # source
                self.fixed_points_layer.data,  # destination
                method,
            )
            self.transform_model.update(
                transform=transform,
                transformation_type=method,
                time_created=datetime.now(),
                fixed_points=self.fixed_points_layer.data,
                moving_points=self.moving_points_layer.data,
            )
            error = self.transform_model.error()
            self.transform_error.setText(f"{error:.2f}")
            hp.update_widget_style(
                self.transform_error, "error" if error > self.transform_model.moving_model.resolution / 2 else "success"
            )
            logger.info(self.transform_model.about())
            self.on_apply()
        else:
            if n_fixed <= 3 or n_moving <= 3:
                logger.warning("There must be at least three points before we can compute the transformation.")
            elif n_fixed != n_moving:
                logger.warning("The number of `fixed` and `moving` points must be the same.")

    def on_save(self, _evt: ty.Any = None) -> None:
        """Export transformation."""
        transform = self.transform_model
        if transform.is_valid() is None:
            logger.warning("Cannot save transformation - no transformation has been computed.")
            hp.warn(self, "Cannot save transformation - no transformation has been computed.")
            return
        # get filename which is based on the moving dataset
        filename = self.moving_model.get_filename() + "_transform.i2r.json"
        path = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_EXPORT_REGISTER_FORMATS,
            base_filename=filename,
        )
        if path:
            path = Path(path)
            CONFIG.output_dir = str(path.parent)
            transform.to_file(path)
            hp.toast(
                self,
                "Exported transformation",
                f"Saved transformation to<br><b>{path}</b>",
                icon="success",
                position="top_left",
            )

    def on_load(self, _evt: ty.Any = None) -> None:
        """Import transformation."""
        path = hp.get_filename(
            self, "Load transformation", base_dir=CONFIG.output_dir, file_filter=ALLOWED_IMPORT_REGISTER_FORMATS
        )
        if path:
            from image2image._dialogs import ImportSelectDialog
            from image2image.models.transformation import load_transform_from_file

            # load transformation
            path = Path(path)
            CONFIG.output_dir = str(path.parent)

            # get info on which settings should be imported
            dlg = ImportSelectDialog(self)
            if dlg.exec_():  # noqa
                config = dlg.config
                logger.trace(f"Loaded configuration from {path}\n{config}")

                # reset all widgets
                if config["fixed_image"]:
                    self._fixed_widget.dataset_dlg._on_close_dataset(force=True)
                if config["moving_image"]:
                    self._moving_widget.dataset_dlg._on_close_dataset(force=True)

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
                    ) = load_transform_from_file(path, **config)
                except (ValueError, KeyError) as e:
                    hp.warn(self, f"Failed to load transformation from {path}\n{e}", "Failed to load transformation")
                    return

                # locate paths that are missing
                if fixed_paths_missing or moving_paths_missing:
                    from image2image._dialogs import LocateFilesDialog

                    locate_dlg = LocateFilesDialog(self, fixed_paths_missing, moving_paths_missing)
                    if locate_dlg.exec_():  # type: ignore[attr-defined]
                        if fixed_paths_missing:
                            fixed_paths = locate_dlg.fix_missing_paths(fixed_paths_missing, fixed_paths)
                        if moving_paths_missing:
                            moving_paths = locate_dlg.fix_missing_paths(moving_paths_missing, moving_paths)

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
                self.transform_choice.setCurrentText(transformation_type)
                # force update of the text
                self.on_update_text(block=False)

    def on_show_fiducials(self):
        """View fiducials table."""
        if self._table is None:
            from image2image._dialogs import FiducialsDialog

            self._table = FiducialsDialog(self)
        self._table.show()

    def on_show_shortcuts(self) -> None:
        """View shortcuts table."""
        from image2image._dialogs._shortcuts import RegisterShortcutsDialog

        dlg = RegisterShortcutsDialog(self)
        dlg.show()

    def _get_console_variables(self) -> ty.Dict:
        return {
            "transform_model": self.transform_model,
            "fixed_viewer": self.view_fixed.viewer,
            "fixed_model": self.fixed_model,
            "moving_viewer": self.view_moving.viewer,
            "moving_model": self.moving_model,
        }

    @ensure_main_thread
    def on_apply(self, update_data: bool = False, name: ty.Optional[str] = None):
        """Apply transformation."""
        if self.transform is None or self.moving_image_layer is None:
            logger.warning("Cannot apply transformation - no transformation has been computed.")
            return

        if name is None or name == "None":
            moving_image_layer = self.moving_image_layer[0]
        else:
            index = [layer.name for layer in self.moving_image_layer].index(name)
            moving_image_layer = self.moving_image_layer[index]

        # add image and apply transformation
        if self.transformed_moving_image_layer:
            self.transformed_moving_image_layer.affine = self.transform.params
            if update_data:
                self.transformed_moving_image_layer.data = moving_image_layer.data
                self.transformed_moving_image_layer.colormap = moving_image_layer.colormap
                self.transformed_moving_image_layer.reset_contrast_limits()
        else:
            self.view_fixed.viewer.add_image(
                moving_image_layer.data,
                name="Transformed",
                blending="translucent",
                affine=self.transform.params,
                colormap=moving_image_layer.colormap,
            )
        self.transformed_moving_image_layer.visible = CONFIG.show_transformed
        self._move_layer(self.view_fixed, self.transformed_moving_image_layer, -1, False)
        self._select_layer("fixed")

    @ensure_main_thread
    def on_predict(self, which: str, _evt: ty.Any = None) -> None:
        """Predict transformation from either image."""
        self.on_update_text()
        if self.transform is None:
            logger.warning("Cannot predict - no transformation has been computed.")
            return

        if which == "fixed":
            # predict point position in the moving image -> inverse transform
            layer = self.moving_points_layer
            transformed_data = self.transform.inverse(self.fixed_points_layer.data)
        else:
            # predict point position in the fixed image -> transform
            layer = self.fixed_points_layer
            transformed_data = self.transform(self.moving_points_layer.data)

        # don't predict positions if the number of points is lower than the number already present in the image
        if layer.data.shape[0] > len(transformed_data):
            return

        self._update_layer_points(layer, transformed_data)
        self.evt_predicted.emit()  # noqa

    @staticmethod
    def _update_layer_points(layer: Points, data: np.ndarray, block: bool = True) -> None:
        """Update points layer."""
        if block:
            with layer.events.data.blocker():
                layer.data = data
                layer.properties = _get_text_data(data)
        else:
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
            self.fixed_points_layer.current_size = CONFIG.size_fixed
        if self.moving_points_layer and which == "moving":
            self.moving_points_layer.size = CONFIG.size_moving
            self.moving_points_layer.current_size = CONFIG.size_moving
        if self.fixed_image_layer and which == "fixed":
            self.fixed_image_layer[0].opacity = CONFIG.opacity_fixed / 100
        if self.transformed_moving_image_layer and which == "moving":
            self.transformed_moving_image_layer.opacity = CONFIG.opacity_moving / 100

    def on_update_text(self, _: ty.Any = None, block: bool = True) -> None:
        """Update text data in each layer."""
        CONFIG.label_color = self.text_color.hex_color
        CONFIG.label_size = self.text_size.value()

        # update text information
        for layer in [self.fixed_points_layer, self.moving_points_layer]:
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

    # noinspection PyAttributeOutsideInit
    def _setup_ui(self) -> None:
        """Create panel."""
        view_layout = self._make_image_layout()

        self._fixed_widget = FixedWidget(self, self.view_fixed)
        self._moving_widget = MovingWidget(self, self.view_moving)

        self.transform_choice = hp.make_combobox(self, tooltip="Type of transformation to compute.")
        hp.set_combobox_data(self.transform_choice, TRANSFORMATION_TRANSLATIONS, "Affine")
        self.transform_choice.currentTextChanged.connect(self.on_run)  # type: ignore[arg-type]

        self.transform_error = hp.make_label(
            self, "", bold=True, tooltip="Error is estimated by computing the square root of the sum of squared errors."
        )

        side_layout = hp.make_form_layout()
        style_form_layout(side_layout)
        side_layout.addRow(
            hp.make_btn(self, "Import project", tooltip="Import previously computed transformation.", func=self.on_load)
        )
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._fixed_widget)
        side_layout.addRow(hp.make_h_line_with_text("+"))
        side_layout.addRow(self._moving_widget)
        side_layout.addRow(hp.make_h_line_with_text("Area of interest"))
        side_layout.addRow(self._make_focus_layout())
        side_layout.addRow(hp.make_h_line_with_text("Transformation"))
        side_layout.addRow(hp.make_label(self, "Type of transformation"), self.transform_choice)
        side_layout.addRow(hp.make_label(self, "Estimated error"), self.transform_error)
        side_layout.addRow(
            hp.make_btn(
                self,
                "Show fiducials table...",
                tooltip="Show fiducial markers table where you can view and edit the markers",
                func=self.on_show_fiducials,
            )
        )
        side_layout.addRow(
            hp.make_btn(
                self,
                "Export to file...",
                tooltip="Export transformation to file. XML format is usable by MATLAB fusion.",
                func=self.on_save,
            )
        )
        side_layout.addRow(hp.make_spacer_widget())
        side_layout.addRow(hp.make_h_line_with_text("Settings"))
        side_layout.addRow(self._make_settings_layout())

        widget = QWidget()
        self.setCentralWidget(widget)
        layout = QHBoxLayout()
        layout.addLayout(view_layout, stretch=True)
        layout.addWidget(hp.make_v_line())
        layout.addLayout(side_layout)
        main_layout = QVBoxLayout(widget)
        main_layout.addLayout(layout)

        # extra settings
        self._make_menu()
        self._make_icon()

    def _make_menu(self) -> None:
        """Make menu items."""
        # File menu
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(
            self, "Import configuration file (.json, .toml)...", "Ctrl+C", menu=menu_file, func=self.on_load
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
        menu_file.addSeparator()
        hp.make_menu_item(
            self,
            "Clear all data...",
            menu=menu_file,
            func=self.on_clear_all,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(
            self, "Select fixed channels...", menu=menu_tools, func=self._fixed_widget._on_select_channels
        )
        hp.make_menu_item(
            self, "Select moving channels...", menu=menu_tools, func=self._moving_widget._on_select_channels
        )
        hp.make_menu_item(self, "Show fiducials table...", menu=menu_tools, func=self.on_show_fiducials)
        hp.make_menu_item(self, "Show shortcuts...", menu=menu_tools, func=self.on_show_shortcuts)
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
        style_form_layout(layout)
        layout.addRow(hp.make_label(self, "Center (x)"), self.x_center)
        layout.addRow(hp.make_label(self, "Center (y)"), self.y_center)
        layout.addRow(hp.make_label(self, "Zoom"), self.zoom)
        layout.addRow(hp.make_h_layout(self.lock_btn, self.use_focus_btn, stretch_id=1))
        return layout

    def _make_settings_layout(self) -> QFormLayout:
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
        style_form_layout(layout)
        layout.addRow(hp.make_label(self, "Size (fixed)"), self.fixed_point_size)
        layout.addRow(hp.make_label(self, "Size (moving)"), self.moving_point_size)
        layout.addRow(hp.make_label(self, "Opacity (fixed)"), self.fixed_opacity)
        layout.addRow(hp.make_label(self, "Opacity (moving)"), self.moving_opacity)
        layout.addRow(hp.make_label(self, "Label size"), self.text_size)
        layout.addRow(hp.make_label(self, "Label color"), self.text_color)
        return layout

    def _make_image_layout(self) -> QVBoxLayout:
        self.info = hp.make_label(
            self,
            "Please select at least <b>3 points</b> in each image to compute transformation.",
            tooltip="Information regarding registration.",
            object_name="tip_label",
        )

        view_layout = QVBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        view_layout.addWidget(self.info, alignment=Qt.AlignCenter)  # type: ignore[attr-defined]
        view_layout.addLayout(self._make_fixed_view())
        view_layout.addWidget(hp.make_v_line())
        view_layout.addLayout(self._make_moving_view())
        return view_layout

    def _make_fixed_view(self) -> QHBoxLayout:
        self.view_fixed = self._make_image_view(self, add_toolbars=False)
        self.view_fixed.viewer.text_overlay.text = "Fixed"
        self.view_fixed.viewer.text_overlay.font_size = 8
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
            tooltip="Remove last point from the fixed image.",
        )
        _fixed_remove_btn = toolbar.insert_qta_tool(
            "remove_single",
            func=lambda *args: self.on_remove("fixed"),
            tooltip="Remove last point from the fixed image.",
        )
        self.fixed_move_btn = toolbar.insert_qta_tool(
            "move",
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
            "zoom",
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
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toolbar)
        layout.addWidget(self.view_fixed.widget, stretch=True)
        return layout

    def _make_moving_view(self) -> QHBoxLayout:
        self.view_moving = self._make_image_view(self, add_toolbars=False)
        self.view_moving.viewer.text_overlay.text = "Moving"
        self.view_moving.viewer.text_overlay.font_size = 8
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
            tooltip="Remove last point from the moving image.",
        )
        _moving_remove_btn = toolbar.insert_qta_tool(
            "remove_single",
            func=lambda *args: self.on_remove("moving"),
            tooltip="Remove last point from the moving image.",
        )
        self.moving_move_btn = toolbar.insert_qta_tool(
            "move",
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
            "zoom",
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

    def closeEvent(self, evt: ty.Any) -> None:
        """Close."""
        CONFIG.save()
        if self.transform_model.is_valid():
            if hp.confirm(self, "There might be unsaved changes. Would you like to save it?"):
                self.on_save()

    def keyPressEvent(self, evt: ty.Any) -> None:
        """Key press event."""
        if hasattr(evt, "native"):
            evt = evt.native
        key = evt.key()
        if key == Qt.Key_Escape:  # type: ignore[attr-defined]
            evt.ignore()
        elif key == Qt.Key_1:  # type: ignore[attr-defined]
            self.on_mode("fixed", mode=Mode.PAN_ZOOM)
            self.on_mode("moving", mode=Mode.PAN_ZOOM)
            evt.ignore()
        elif key == Qt.Key_2:  # type: ignore[attr-defined]
            self.on_mode("fixed", mode=Mode.ADD)
            self.on_mode("moving", mode=Mode.ADD)
            evt.ignore()
        elif key == Qt.Key_3:  # type: ignore[attr-defined]
            self.on_mode("fixed", mode=Mode.SELECT)
            self.on_mode("moving", mode=Mode.SELECT)
            evt.ignore()
        elif key == Qt.Key_Z:  # type: ignore[attr-defined]
            self.on_apply_focus()
            evt.ignore()
        elif key == Qt.Key_L:  # type: ignore[attr-defined]
            self.on_set_focus()
            evt.ignore()
        else:
            super().keyPressEvent(evt)


if __name__ == "__main__":  # pragma: no cover
    from image2image.main import run

    run(dev=True, tool="register", level=0)
