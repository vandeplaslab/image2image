"""Registration dialog."""

import typing as ty
from contextlib import suppress
from datetime import datetime
from functools import partial

import numpy as np
from napari.layers.utils._link_layers import link_layers

import qtextra.helpers as hp
from napari.layers.points.points import Mode, Points
from napari.utils.events import Event
from qtextra.mixins import ConfigMixin, IndicatorMixin
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_mini_toolbar import QtMiniToolbar
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from ims2micro.enums import TRANSFORMATION_TRANSLATIONS
from ims2micro.models import Transformation, DataModel
from ims2micro._select import IMSWidget, MicroscopyWidget
from qtextra._napari.mixins import ImageViewMixin
from superqt import ensure_main_thread

from ims2micro.utilities import add, select, _get_text_data
import ims2micro.assets  # noqa: F401

if ty.TYPE_CHECKING:
    from napari.layers import Image

from loguru import logger


class ImageRegistrationDialog(QtDialog, ConfigMixin, IndicatorMixin, ImageViewMixin):
    """Image registration dialog."""

    fixed_image_layer: ty.Optional[ty.List["Image"]] = None
    fixed_points_layer: ty.Optional[Points] = None
    moving_image_layer: ty.Optional["Image"] = None
    moving_points_layer: ty.Optional[Points] = None
    transformed_moving_image_layer: ty.Optional["Image"] = None
    temporary_transform = None

    def __init__(self, parent):
        QtDialog.__init__(self, parent, title="Image registration")
        self.setMouseTracking(True)
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(1200, 800)
        self.setup_events()

    @property
    def transform(self):
        """Retrieve transform."""
        transform = self.temporary_transform
        if transform:
            return transform.transform
        return None

    def setup_events(self, state: bool = True):
        """Additional setup."""
        connect(self._micro_widget.evt_loaded, self.on_load_fixed, state=state)
        connect(self._micro_widget.evt_toggle_channel, self.on_toggle_fixed_channel, state=state)
        connect(self._ims_widget.evt_loaded, self.on_load_moving, state=state)

        # add fixed points layer
        if state:
            self.on_add("fixed")
        connect(self.fixed_points_layer.events.data, self.on_run, state=state)
        connect(self.fixed_points_layer.events.add_point, partial(self.on_predict, "fixed"), state=state)
        # connect(self.fixed_points_layer.events.mode, partial(self.on_sync_mode, "fixed"), state=state)

        # add moving points layer
        if state:
            self.on_add("moving")
        connect(self.moving_points_layer.events.data, self.on_run, state=state)
        connect(self.moving_points_layer.events.add_point, partial(self.on_predict, "moving"), state=state)
        # connect(self.moving_points_layer.events.mode, partial(self.on_sync_mode, "moving"), state=state)

    @ensure_main_thread
    def on_load_fixed(self, model: DataModel):
        """Load fixed image."""
        self._on_load_fixed(model)

    def _on_load_fixed(self, model: DataModel):
        wrapper = model.get_reader()
        fixed_data = wrapper.image()
        channel_names = wrapper.channel_names()
        for name in channel_names:
            if name in self.view_fixed.layers:
                del self.view_moving.layers[name]
        fixed_image_layer = self.view_fixed.viewer.add_image(**wrapper.image())
        self.fixed_image_layer = fixed_image_layer if isinstance(fixed_image_layer, list) else [fixed_image_layer]
        if isinstance(self.fixed_image_layer, list) and len(self.fixed_image_layer) > 1:
            link_layers(self.fixed_image_layer, attributes=("opacity",))
        self.view_fixed.viewer.reset_view()

    def on_toggle_fixed_channel(self, name: str, state: bool):
        """Toggle fixed channel."""
        self.view_fixed.layers[name].visible = state

    @ensure_main_thread
    def on_load_moving(self, model: DataModel):
        """Open modality."""
        self._on_load_moving(model)

    def _on_load_moving(self, model: DataModel):
        wrapper = model.get_reader()
        moving_data = wrapper.image()
        if moving_data["name"] in self.view_moving.layers:
            del self.view_moving.layers[moving_data["name"]]
        self.moving_image_layer = self.view_moving.viewer.add_image(**moving_data)
        self.on_clear("fixed", True)
        self.on_clear("moving", True)
        self.view_moving.viewer.reset_view()

    def _select_layer(self, which: str):
        """Select layer."""
        if which == "fixed":
            self.view_fixed.layers.move(self.view_fixed.layers.index(self.fixed_points_layer), -1)
            self.view_fixed.layers.selection.select_only(self.fixed_points_layer)
        else:
            self.view_moving.layers.move(self.view_moving.layers.index(self.moving_points_layer), -1)
            self.view_moving.layers.selection.select_only(self.moving_points_layer)

    def _get_mode_button(self, which: str, mode):
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

    def on_sync_mode(self, which: str, evt=None):
        """Update mode."""
        widget = self._get_mode_button(which, evt.mode)
        if widget is not None:
            with hp.qt_signals_blocked(widget):
                widget.setChecked(True)

    def on_mode(self, which: str, evt=None):
        """Update mode."""
        widget = self._get_mode_button(which, evt.mode)
        if widget is not None:
            widget.setChecked(True)

    @ensure_main_thread
    def on_move(self, which: str, evt=None):
        """Move points."""
        widget = self.fixed_move_btn if which == "fixed" else self.moving_move_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "select" if widget.isChecked() else "pan_zoom"
        self._select_layer(which)

    def on_panzoom(self, which: str, evt=None):
        """Switch to `panzoom` tool."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "pan_zoom"
        self._select_layer(which)

    def on_add(self, which: str, evt=None):
        """Add point to the image."""

        def _init(layer):
            layer._drag_modes[Mode.ADD] = add
            layer._drag_modes[Mode.SELECT] = select
            layer.events.add(move=Event, add_point=Event)

        # make sure the points layer is present
        if which == "fixed" and self.fixed_points_layer is None:
            self.fixed_points_layer = self.view_fixed.viewer.add_points(
                None,
                size=self.fixed_point_size.value(),
                name="Fixed (points)",
                face_color="green",
                edge_color="black",
                symbol="cross",
            )
            _init(self.fixed_points_layer)
        elif which == "moving" and self.moving_points_layer is None:
            self.moving_points_layer = self.view_moving.viewer.add_points(
                None,
                size=self.moving_point_size.value(),
                name="Moving (points)",
                face_color="red",
                edge_color="black",
                symbol="x",
            )
            _init(self.moving_points_layer)

        self._select_layer(which)
        # extract button and layer based on the appropriate mode
        widget = self.fixed_add_btn if which == "fixed" else self.moving_add_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # make sure the add mode is active
        layer.mode = "add" if widget.isChecked() else "pan_zoom"

    def on_remove(self, which: str, evt=None):
        """Remove point to the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer

        # there is no data to remove
        if layer.data.shape[0] == 0:
            return

        data = layer.data
        layer.data = np.delete(data, -1, 0)
        # layer.text.remove([-1])

    def on_clear(self, which: str, force: bool = True):
        """Remove point to the image."""
        if force or hp.confirm(self, "Are you sure you want to remove all data points from the image?"):
            layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
            layer.data = np.zeros((0, 2))
            # layer.text.values = np.empty(0)
            self.on_clear_transformation()

    def on_clear_transformation(self):
        """Clear transformation and remove image."""
        if self.temporary_transform:
            self.temporary_transform = None
        if self.transformed_moving_image_layer:
            with suppress(ValueError):
                self.view_fixed.layers.remove(self.transformed_moving_image_layer)
            self.transformed_moving_image_layer = None

    @ensure_main_thread
    def on_run(self, _evt=None):
        """Compute transformation."""
        from ims2micro.utilities import compute_transform

        if not self.fixed_points_layer or not self.moving_points_layer:
            self.on_notify_warning("There must be at least three points before we can compute the transformation.")
            return

        # execute transform calculation
        method = self.transform_choice.currentText().lower()
        n_fixed = len(self.fixed_points_layer.data)
        n_moving = len(self.moving_points_layer.data)
        if 3 <= n_fixed == n_moving >= 3:
            transform = compute_transform(
                self.moving_points_layer.data,  # source
                self.fixed_points_layer.data,  # destination
                method,
            )
            self.temporary_transform = Transformation(
                transform=transform,
                transformation_type=method,
                micro_model=self._micro_widget.model,
                ims_model=self._ims_widget.model,
                time_created=datetime.now(),
                fixed_points=self.fixed_points_layer.data,
                moving_points=self.moving_points_layer.data,
            )
            logger.info(self.temporary_transform.about())
            self.on_apply()
        else:
            if n_fixed <= 3 or n_moving <= 3:
                logger.warning("There must be at least three points before we can compute the transformation.")
            elif n_fixed != n_moving:
                logger.warning("The number of `fixed` and `moving` points must be the same.")

    @ensure_main_thread
    def on_apply(self):
        """Apply transformation."""
        if self.transform is None and self.moving_image_layer is None:
            logger.warning("Cannot apply transformation - no transformation has been computed.")
            return

        # add image and apply transformation
        self.transformed_moving_image_layer = self.view_fixed.add_image(
            self.moving_image_layer.data,
            name="Transformed",
            blending="translucent",
            opacity=self.moving_opacity.value() / 100,
            affine=self.transform.params,
        )
        # self.transformed_moving_image_layer.affine = self.fixed_image_layer.affine.affine_matrix @ self.transform.params
        self._select_layer("fixed")

    @ensure_main_thread
    def on_predict(self, which: str, _evt=None):
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

    @staticmethod
    def _update_layer_points(layer: Points, data: np.ndarray):
        """Update points layer."""
        with layer.events.data.blocker():
            layer.data = data
            layer.properties = _get_text_data(data)

    def on_update_layer(self, which: str, _value=None):
        """Update points layer."""
        fixed_size = self.fixed_point_size.value()
        moving_size = self.moving_point_size.value()
        fixed_opacity = self.fixed_opacity.value()
        moving_opacity = self.moving_opacity.value()

        # update point size
        if self.fixed_points_layer and which == "fixed":
            self.fixed_points_layer.size = fixed_size
            self.fixed_point_size.current_size = fixed_size
        if self.moving_points_layer and which == "moving":
            self.moving_points_layer.size = moving_size
            self.moving_points_layer.current_size = moving_size

        if self.fixed_image_layer and which == "fixed":
            self.fixed_image_layer[0].opacity = fixed_opacity / 100
        if self.transformed_moving_image_layer and which == "moving":
            self.transformed_moving_image_layer.opacity = moving_opacity / 100

    def on_update_text(self, _=None):
        """Update text data in each layer."""
        text_color = self.text_color.color
        text_size = self.text_size.value()

        # update text information
        for layer in [self.fixed_points_layer, self.moving_points_layer]:
            with layer.text.events.blocker():
                layer.text.color = text_color
                layer.text.size = text_size

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QHBoxLayout:
        """Create panel."""
        view_layout = self._make_image_layout()

        self._micro_widget = MicroscopyWidget(self)
        self._ims_widget = IMSWidget(self)

        self.run_btn = hp.make_btn(
            self,
            "Compute transformation",
            tooltip="Compute transformation between the fixed and moving image.",
        )
        self.run_btn.clicked.connect(self.on_run)

        self.transform_choice = hp.make_combobox(self)
        hp.set_combobox_data(self.transform_choice, TRANSFORMATION_TRANSLATIONS, "Affine")
        self.transform_choice.currentTextChanged.connect(self.on_run)

        layout = hp.make_form_layout()
        layout.addRow(self._micro_widget)
        layout.addRow(self._ims_widget)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(hp.make_label(self, "Type of transformation"), self.transform_choice)
        layout.addRow(self.run_btn)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self._make_settings_layout())

        widget = QWidget()
        widget.setMinimumWidth(400)
        settings_layout = QVBoxLayout(widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addLayout(layout)
        settings_layout.addWidget(hp.make_v_line())

        main_layout = QHBoxLayout()
        main_layout.addLayout(view_layout, stretch=True)
        main_layout.addWidget(widget)
        return main_layout

    def _make_settings_layout(self):
        # functionality
        self.fixed_point_size = hp.make_int_spin_box(
            self, value=3, tooltip="Size of the points shown in the fixed image."
        )
        self.fixed_point_size.valueChanged.connect(partial(self.on_update_layer, "fixed"))

        self.moving_point_size = hp.make_int_spin_box(
            self, value=1, tooltip="Size of the points shown in the moving image."
        )
        self.moving_point_size.valueChanged.connect(partial(self.on_update_layer, "moving"))

        self.fixed_opacity = hp.make_int_spin_box(self, value=75, step_size=10, tooltip="Opacity of the fixed image")
        self.fixed_opacity.valueChanged.connect(partial(self.on_update_layer, "fixed"))

        self.moving_opacity = hp.make_int_spin_box(
            self,
            value=100,
            step_size=10,
            tooltip="Opacity of the fixed image",
        )
        self.moving_opacity.valueChanged.connect(partial(self.on_update_layer, "moving"))

        self.text_size = hp.make_int_spin_box(
            self, value=12, minimum=4, maximum=60, tooltip="Size of the text associated with each label."
        )
        self.text_size.valueChanged.connect(self.on_update_text)

        self.text_color = hp.make_swatch(
            self, default="#00ff00", tooltip="Color of the text associated with each label."
        )
        self.text_color.evt_color_changed.connect(self.on_update_text)

        layout = hp.make_form_layout()
        layout.addRow(hp.make_label(self, "Size (fixed)"), self.fixed_point_size)
        layout.addRow(hp.make_label(self, "Size (moving)"), self.moving_point_size)
        layout.addRow(hp.make_label(self, "Opacity (fixed)"), self.fixed_opacity)
        layout.addRow(hp.make_label(self, "Opacity (moving)"), self.moving_opacity)
        layout.addRow(hp.make_label(self, "Label size"), self.text_size)
        layout.addRow(hp.make_label(self, "Label color"), self.text_color)
        return layout

    def _make_image_layout(self):
        self.info = hp.make_label(
            self,
            "Please select at least <b>3 points</b> in either image to compute transformation.",
            tooltip="Information regarding registration.",
        )

        view_layout = QVBoxLayout()
        view_layout.addWidget(self.info, alignment=Qt.AlignCenter)
        view_layout.addLayout(self._make_fixed_view())
        view_layout.addWidget(hp.make_v_line())
        view_layout.addLayout(self._make_moving_view())
        return view_layout

    def _make_fixed_view(self):
        self.view_fixed = self._make_image_view(self, add_toolbars=False)
        self.view_fixed.viewer.text_overlay.text = "Fixed"
        self.view_fixed.viewer.text_overlay.visible = True

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=True)
        self.fixed_clear_btn = toolbar.insert_qta_tool(
            "cross_full",
            # func=partial(self.on_clear, "fixed")
            func=lambda *args: self.on_clear("fixed"),
            tooltip="Remove last point from the fixed image.",
        )
        self.fixed_remove_btn = toolbar.insert_qta_tool(
            "remove",
            # func=partial(self.on_remove, "fixed"),
            func=lambda *args: self.on_remove("fixed"),
            tooltip="Remove last point from the fixed image.",
        )
        self.fixed_move_btn = toolbar.insert_qta_tool(
            "move",
            # func=partial(self.on_move, "fixed"),
            func=lambda *args: self.on_move("fixed"),
            tooltip="Move points in the fixed image...",
            checkable=True,
        )
        self.fixed_add_btn = toolbar.insert_qta_tool(
            "add",
            # func=partial(self.on_add, "fixed"),
            func=lambda *args: self.on_add("fixed"),
            tooltip="Add new point to the fixed image...",
            checkable=True,
        )
        self.fixed_zoom_btn = toolbar.insert_qta_tool(
            "zoom",
            # func=partial(self.on_panzoom, "fixed"),
            func=lambda *args: self.on_panzoom("fixed"),
            tooltip="Switch to zoom-only mode...",
            checkable=True,
        )
        hp.make_radio_btn_group(self, [self.fixed_zoom_btn, self.fixed_add_btn, self.fixed_move_btn])
        self.fixed_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_fixed.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )

        fixed_layout = QHBoxLayout()
        fixed_layout.addWidget(toolbar)
        fixed_layout.addWidget(self.view_fixed.widget, stretch=True)
        return fixed_layout

    def _make_moving_view(self):
        self.view_moving = self._make_image_view(self, add_toolbars=False)
        self.view_moving.viewer.text_overlay.text = "Moving"
        self.view_moving.viewer.text_overlay.visible = True

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=True)
        self.moving_clear_btn = toolbar.insert_qta_tool(
            "cross_full",
            # func=partial(self.on_clear, "moving"),
            func=lambda *args: self.on_clear("moving"),
            tooltip="Remove last point from the moving image.",
        )
        self.moving_remove_btn = toolbar.insert_qta_tool(
            "remove",
            # func=partial(self.on_remove, "moving"),
            func=lambda *args: self.on_remove("moving"),
            tooltip="Remove last point from the moving image.",
        )
        self.moving_move_btn = toolbar.insert_qta_tool(
            "move",
            # func=partial(self.on_move, "moving"),
            func=lambda *args: self.on_move("moving"),
            tooltip="Move points in the fixed image...",
            checkable=True,
        )
        self.moving_add_btn = toolbar.insert_qta_tool(
            "add",
            # func=partial(self.on_add, "moving"),
            func=lambda *args: self.on_add("moving"),
            tooltip="Add new point to the moving image...",
            checkable=True,
        )
        self.moving_zoom_btn = toolbar.insert_qta_tool(
            "zoom",
            # func=partial(self.on_panzoom, "moving"),
            func=lambda *args: self.on_panzoom("moving"),
            tooltip="Switch to zoom-only mode...",
            checkable=True,
        )
        hp.make_radio_btn_group(self, [self.moving_zoom_btn, self.moving_add_btn, self.moving_move_btn])
        self.moving_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_moving.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )

        moving_layout = QHBoxLayout()
        moving_layout.addWidget(toolbar)
        moving_layout.addWidget(self.view_moving.widget, stretch=True)
        return moving_layout
