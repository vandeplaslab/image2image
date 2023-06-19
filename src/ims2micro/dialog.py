"""Registration dialog."""

import typing as ty
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import ionglow.gui_elements.helpers as hp
import numpy as np
from ionglow._qt.mixins import ConfigMixin, IndicatorMixin
from ionglow._qt.qt_dialog import QtDialog
from ionglow._qt.qt_mini_toolbar import QtMiniToolbar
from ionglow.config import ENV, get_settings
from ionglow.gui_elements.mixins import ImageViewMixin
from ionglow.gui_elements.qt_image_manager import ImageSelectionManager, ImageSelectionMixin
from ionglow.models.registration import RegistrationModel, Transformation
from ionglow.plugins.registration.qt_registration_list import QtRegistrationList
from ionglow.plugins.registration.utilities import add, select
from ionglow.utils.enums import TRANSFORMATION_TRANSLATIONS
from ionglow.utils.misc import SchemaWrapper
from ionglow.utils.utilities import connect
from napari.layers.points.points import Mode, Points
from napari.utils.events import Event
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

if ty.TYPE_CHECKING:
    from napari.layers import Image

from loguru import logger


class ImageRegistrationDialog(QtDialog, ConfigMixin, IndicatorMixin, ImageViewMixin, ImageSelectionMixin):
    """Image registration dialog."""

    fixed_image_layer: ty.Optional["Image"] = None
    fixed_points_layer: ty.Optional[Points] = None
    moving_image_layer: ty.Optional["Image"] = None
    moving_points_layer: ty.Optional[Points] = None
    transformed_moving_image_layer: ty.Optional["Image"] = None
    filename: ty.Optional[str] = None
    _transform: ty.Optional[RegistrationModel] = None

    def __init__(self, parent):
        QtDialog.__init__(self, parent, title="Image registration")
        self.setMouseTracking(True)
        self.setWindowFlags(Qt.Window)
        self.setMinimumSize(1200, 800)

        self.setup_events()
        self.on_load_dataset()

    def _on_teardown(self):
        """Teardown."""
        self.setup_events(False)
        self.image_manager.teardown()
        super()._on_teardown()

    @property
    def transform_model(self) -> ty.Optional[RegistrationModel]:
        """Current transform model."""
        return self._transform

    @transform_model.setter
    def transform_model(self, registration: RegistrationModel):
        ENV.registration_map[registration.name] = registration
        self._transform = registration

    @property
    def transform(self):
        """Retrieve transform."""
        transform = self.transform_model
        if transform:
            return transform.transform
        return None

    def on_load_dataset(self):
        """Load dataset."""
        if not ENV.is_loaded:
            return

        for model in ENV.registration_map.values():
            self.registration_list.append_item(model)
        # force update of ion image
        self.image_manager._set_used_last("im_ion")

    def setup_events(self, state: bool = True):
        """Additional setup."""
        connect(self.load_moving_btn.clicked, self.on_open_moving, state=state)
        connect(self.image_manager.evt_update, self.on_plot_fixed_image, state=state)

        # add fixed points layer
        if state:
            self.on_add("fixed")
        connect(self.fixed_points_layer.events.data, self.on_run, state=state)
        connect(self.fixed_points_layer.events.add_point, partial(self.on_predict, "fixed"), state=state)
        connect(self.fixed_points_layer.events.mode, partial(self.on_mode, "fixed"), state=state)

        # add moving points layer
        if state:
            self.on_add("moving")
        connect(self.moving_points_layer.events.data, self.on_run, state=state)
        connect(self.moving_points_layer.events.add_point, partial(self.on_predict, "moving"), state=state)
        connect(self.moving_points_layer.events.mode, partial(self.on_mode, "moving"), state=state)

    def _on_open_fixed(self, filename: str):
        self.fixed_image_layer = self.view_fixed.widget.viewer.open(filename, name="Fixed")[0]
        self.view_fixed.layers.move(self.view_fixed.layers.index(self.fixed_image_layer), 0)
        self.view_fixed.layers.selection.select_only(self.fixed_points_layer)

    def on_plot_fixed_image(self, array: np.ndarray):
        """Update fixed image."""
        reset = self.fixed_image_layer is None

        def _check_existing(n: int):
            nonlocal reset
            if reset:
                return
            if len(self.fixed_image_layer.data.shape) != n:
                self.view_fixed.remove_layer("Fixed", True)
                self.fixed_image_layer = None
                reset = True

        if len(array.shape) == 3:
            _check_existing(3)
            self.fixed_image_layer = self.view_fixed.plot_rgb(array, "Fixed")
        else:
            _check_existing(2)
            self.fixed_image_layer = self.view_fixed.add_image(array, "Fixed")
        self.view_fixed.layers.move(self.view_fixed.layers.index(self.fixed_image_layer), 0)
        self.view_fixed.layers.selection.select_only(self.fixed_points_layer)
        if reset:
            self.view_fixed.viewer.reset_view()

    def on_open_moving(self):
        """Open modality."""
        filename = hp.open_filename(self, file_filter="*")
        if filename:
            self._on_open_moving(filename, force=True)
            self.on_clear("fixed", True)
            self.on_clear("moving", True)

    def _on_open_moving(self, filename: str, reset: bool = True, force: bool = False):
        self.filename = filename
        if self.moving_image_layer is not None:
            if force or hp.confirm(
                self, "An image is already present in the canvas (moving). Would you like to replace it with new image?"
            ):
                self.view_moving.layers.remove(self.moving_image_layer)
                self.moving_image_layer = None
            else:
                return

        self.moving_image_layer = self.view_moving.widget.viewer.open(filename, name="Moving")[0]
        self.view_moving.layers.move(self.view_moving.layers.index(self.moving_image_layer), 0)
        self.view_moving.layers.selection.select_only(self.moving_points_layer)

        if reset:
            path = Path(filename)

            with hp.qt_signals_blocked(self.registration_list):
                self.transform_model = RegistrationModel(
                    name=path.stem, image_path=filename, time_created=datetime.now(), is_exported=False
                )
            widget = self.registration_list.get_widget_for_item_model(self.transform_model)
            if widget:
                widget.on_select()

    def _on_open_mis(self, filename: str):
        """Open Bruker .mis file."""

    def on_plot_moving_image(self, array: np.ndarray):
        """Update moving image."""
        reset = self.moving_image_layer is None
        self.moving_image_layer = self.view_moving.add_image(array, "Moving")
        self.view_moving.layers.move(self.view_moving.layers.index(self.moving_image_layer), 0)
        self.view_moving.layers.selection.select_only(self.moving_points_layer)
        if reset:
            self.view_moving.viewer.reset_view()

    def _select_layer(self, which: str):
        """Select layer."""
        if which == "fixed":
            self.view_fixed.layers.move(self.view_fixed.layers.index(self.fixed_points_layer), -1)
            self.view_fixed.layers.selection.select_only(self.fixed_points_layer)
        else:
            self.view_moving.layers.move(self.view_moving.layers.index(self.moving_points_layer), -1)
            self.view_moving.layers.selection.select_only(self.moving_points_layer)

    def on_mode(self, which: str, evt=None):
        """Update mode."""
        mode = evt.mode
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
        widget = widgets.get(mode, None)
        if widget is not None:
            widget.setChecked(True)

    def on_move(self, which: str):
        """Move points."""
        widget = self.fixed_move_btn if which == "fixed" else self.moving_move_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "select" if widget.isChecked() else "pan_zoom"
        self._select_layer(which)

    def on_panzoom(self, which: str):
        """Switch to `panzoom` tool."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "pan_zoom"
        self._select_layer(which)

    def on_add(self, which: str):
        """Add point to the image."""

        def _init(layer):
            layer._drag_modes[Mode.ADD] = add
            layer._drag_modes[Mode.SELECT] = select
            layer.events.add(move=Event, add_point=Event)

        # make sure the points layer is present
        if which == "fixed" and self.fixed_points_layer is None:
            self.fixed_points_layer = self.view_fixed.add_points_layer(
                None, [], size=self.fixed_point_size.value(), name="Fixed (points)"
            )
            _init(self.fixed_points_layer)
        elif which == "moving" and self.moving_points_layer is None:
            self.moving_points_layer = self.view_moving.add_points_layer(
                None, [], size=self.moving_point_size.value(), name="Moving (points)"
            )
            _init(self.moving_points_layer)

        self._select_layer(which)
        # extract button and layer based on the appropriate mode
        widget = self.fixed_add_btn if which == "fixed" else self.moving_add_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # make sure the add mode is active
        layer.mode = "add" if widget.isChecked() else "pan_zoom"

    def on_remove(self, which: str):
        """Remove point to the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer

        # there is no data to remove
        if layer.data.shape[0] == 0:
            return

        data = layer.data
        layer.data = np.delete(data, -1, 0)
        layer.text.remove([-1])

    def on_clear(self, which: str, force: bool = True):
        """Remove point to the image."""
        if force or hp.confirm(self, "Are you sure you want to remove all data points from the image?"):
            layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
            layer.data = np.zeros((0, 2))
            layer.text.values = np.empty(0)
            self.on_clear_transformation()

    def on_clear_transformation(self):
        """Clear transformation and remove image."""
        if self.transform_model:
            self.transform_model.temporary_transform = None
        if self.transformed_moving_image_layer:
            self.view_fixed.layers.remove(self.transformed_moving_image_layer)
            self.transformed_moving_image_layer = None

    def on_run(self, _evt=None):
        """Compute transformation."""
        from ionglow.plugins.registration.utilities import compute_transform

        if not self.fixed_points_layer or not self.moving_points_layer:
            self.on_notify_warning("There must be at least three points before we can compute the transformation.")
            return

        get_settings().registration.transformation = self.transform_choice.currentData()

        # execute transform calculation
        n_fixed = len(self.fixed_points_layer.data)
        n_moving = len(self.moving_points_layer.data)
        if 3 <= n_fixed == n_moving >= 3:
            transform = compute_transform(
                self.moving_points_layer.data,  # source
                self.fixed_points_layer.data,  # destination
                self.transform_choice.currentText().lower(),
            )
            temporary_transform = Transformation(
                transform,
                self.transform_choice.currentText().lower(),
                self.filename,
                datetime.now(),
                self.fixed_points_layer.data,
                self.moving_points_layer.data,
            )
            self.transform_model.temporary_transform = temporary_transform
            if self.transform_model.path and self.transform_model.is_exported:
                self.transform_model.is_exported = False
                self.registration_list.refresh()
            self.on_apply()
        else:
            if n_fixed <= 3 or n_moving <= 3:
                logger.warning("There must be at least three points before we can compute the transformation.")
            elif n_fixed != n_moving:
                logger.warning("The number of `fixed` and `moving` points must be the same.")

    def on_apply(self):
        """Apply transformation."""
        if self.transform is None:
            return

        # add image and apply transformation
        self.transformed_moving_image_layer = self.view_fixed.add_image(
            self.moving_image_layer.data,
            name="Transformed",
            blending="translucent",
            opacity=self.moving_opacity.value() / 100,
        )
        self.transformed_moving_image_layer.affine = self.fixed_image_layer.affine.affine_matrix @ self.transform.params
        self._select_layer("fixed")

    def on_predict(self, which: str, _evt=None):
        """Predict transformation from either image."""
        self.on_update_text()
        if self.transform is None:
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
        label = np.asarray([str(v + 1) for v in range(data.shape[0])])
        layer.text.values = label
        with layer.events.data.blocker():
            layer.data = data

    def on_update_layer(self, which: str, _value=None):
        """Update points layer."""
        settings = get_settings()
        settings.registration.fixed_size = self.fixed_point_size.value()
        settings.registration.moving_size = self.moving_point_size.value()
        settings.registration.fixed_opacity = self.fixed_opacity.value()
        settings.registration.moving_opacity = self.moving_opacity.value()

        # update point size
        if self.fixed_points_layer and which == "fixed":
            self.fixed_points_layer.size = settings.registration.fixed_size
            self.fixed_point_size.current_size = settings.registration.fixed_size
        if self.moving_points_layer and which == "moving":
            self.moving_points_layer.size = settings.registration.moving_size
            self.moving_points_layer.current_size = settings.registration.moving_size

        if self.fixed_image_layer and which == "fixed":
            self.fixed_image_layer.opacity = settings.registration.fixed_opacity / 100
        if self.transformed_moving_image_layer and which == "moving":
            self.transformed_moving_image_layer.opacity = settings.registration.moving_opacity / 100

    def on_update_text(self, _=None):
        """Update text data in each layer."""
        settings = get_settings()
        color = self.text_color.color
        settings.registration.text_color = color
        settings.registration.text_size = self.text_size.value()

        # update text information
        for layer in [self.fixed_points_layer, self.moving_points_layer]:
            # with layer.text.events.blocker():
            layer.text.color = color
            layer.text.size = settings.registration.text_size

    def on_remove_registration(self, item: RegistrationModel):
        """Remove registration."""
        self._on_unload_registration(item)

    def on_lock_registration(self, item: RegistrationModel, _state: bool = False):
        """Lock/unload registration."""
        if item is not None:
            hp.disable_with_opacity(
                self,
                [
                    self.fixed_add_btn,
                    self.fixed_move_btn,
                    self.fixed_zoom_btn,
                    self.fixed_remove_btn,
                    self.fixed_clear_btn,
                    self.moving_add_btn,
                    self.moving_move_btn,
                    self.moving_zoom_btn,
                    self.moving_remove_btn,
                    self.moving_clear_btn,
                    self.run_btn,
                    self.transform_choice,
                ],
                item.locked,
            )
            self.on_panzoom("fixed")
            self.on_panzoom("moving")

    def on_load_registration(self, item: RegistrationModel, state: bool):
        """Load/unload registration."""
        if state:
            self._on_load_registration(item)
        else:
            self._on_unload_registration(item)

    def _on_load_registration(self, item: RegistrationModel):
        with self.measure_time("Loaded registration in"):
            self._transform = item
            if item is not None and item.image_path_exists:
                item.from_hdf5()
                # load data
                self._on_open_moving(item.image_path, False, force=True)

                # This is bloody annoying but I can't figure it out
                for _ in range(2):
                    with suppress(Exception):
                        self._update_layer_points(self.fixed_points_layer, item.fixed_points)
                    with suppress(Exception):
                        self._update_layer_points(self.moving_points_layer, item.moving_points)
                self.on_apply()
                self.on_lock_registration(item)

    def _on_unload_registration(self, item: RegistrationModel):
        """Unload registration."""
        with self.measure_time("Unloaded registration in"):
            if self._transform == item:
                self.on_clear("fixed", True)
                self.on_clear("moving", True)
                if self.moving_image_layer:
                    self.view_moving.layers.remove(self.moving_image_layer)
                    self.moving_image_layer = None
                self._transform = None

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QHBoxLayout:
        """Create panel."""
        settings = get_settings().registration
        schema = SchemaWrapper(settings)

        self.info = hp.make_label(
            self,
            "Please select at least <b>3 points</b> in either image to compute transformation.",
            tooltip="Information regarding registration.",
        )

        self.view_fixed = self._make_image_view(self, add_toolbars=False)
        self.view_fixed.viewer.text_overlay.text = "Fixed"
        self.view_fixed.viewer.text_overlay.visible = True

        # fixed section
        self.image_manager = ImageSelectionManager(
            self,
            viewer=self.view_fixed,
            allow_multiple=True,
            allow_action=False,
            allow_controls=True,
            title_bold=True,
            allow_about=False,
            allow_supervised=False,
            show_range=True,
            reshape=True,
        )
        self.image_manager.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=False)
        self.fixed_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_fixed.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )
        toolbar.insert_spacer()
        self.fixed_clear_btn = toolbar.insert_qta_tool(
            "cross_full", func=partial(self.on_clear, "fixed"), tooltip="Remove last point from the fixed image."
        )
        self.fixed_remove_btn = toolbar.insert_qta_tool(
            "remove",
            func=partial(self.on_remove, "fixed"),
            tooltip="Remove last point from the fixed image.",
        )
        self.fixed_move_btn = toolbar.insert_qta_tool(
            "move",
            func=partial(self.on_move, "fixed"),
            tooltip="Move points in the fixed image...",
            checkable=True,
        )
        self.fixed_add_btn = toolbar.insert_qta_tool(
            "add",
            func=partial(self.on_add, "fixed"),
            tooltip="Add new point to the fixed image...",
            checkable=True,
        )
        self.fixed_zoom_btn = toolbar.insert_qta_tool(
            "zoom", func=partial(self.on_panzoom, "fixed"), tooltip="Switch to zoom-only mode...", checkable=True
        )

        hp.make_radio_btn_group(self, [self.fixed_zoom_btn, self.fixed_add_btn, self.fixed_move_btn])

        fixed_layout = QHBoxLayout()
        fixed_layout.addWidget(toolbar)
        fixed_layout.addWidget(self.view_fixed.widget, stretch=True)

        # moving section
        self.load_moving_btn = hp.make_btn(
            self,
            "Load moving image...",
            tooltip="This image will be registered onto the static image. It could be microscopy or optical image.",
        )
        self.view_moving = self._make_image_view(self, add_toolbars=False)
        self.view_moving.viewer.text_overlay.text = "Moving"
        self.view_moving.viewer.text_overlay.visible = True

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=False)
        self.moving_layers_btn = toolbar.insert_qta_tool(
            "layers",
            func=self.view_moving.widget.on_open_controls_dialog,
            tooltip="Open layers control panel.",
        )
        toolbar.insert_spacer()
        self.moving_clear_btn = toolbar.insert_qta_tool(
            "cross_full",
            func=partial(self.on_clear, "moving"),
            tooltip="Remove last point from the moving image.",
        )
        self.moving_remove_btn = toolbar.insert_qta_tool(
            "remove",
            func=partial(self.on_remove, "moving"),
            tooltip="Remove last point from the moving image.",
        )
        self.moving_move_btn = toolbar.insert_qta_tool(
            "move",
            func=partial(self.on_move, "moving"),
            tooltip="Move points in the fixed image...",
            checkable=True,
        )
        self.moving_add_btn = toolbar.insert_qta_tool(
            "add",
            func=partial(self.on_add, "moving"),
            tooltip="Add new point to the moving image...",
            checkable=True,
        )
        self.moving_zoom_btn = toolbar.insert_qta_tool(
            "zoom",
            func=partial(self.on_panzoom, "moving"),
            tooltip="Switch to zoom-only mode...",
            checkable=True,
        )

        hp.make_radio_btn_group(self, [self.moving_zoom_btn, self.moving_add_btn, self.moving_move_btn])

        moving_layout = QHBoxLayout()
        moving_layout.addWidget(toolbar)
        moving_layout.addWidget(self.view_moving.widget, stretch=True)

        # functionality
        self.fixed_point_size = hp.make_int_spin_box(self, value=settings.fixed_size, **schema.fixed_size)
        self.fixed_point_size.valueChanged.connect(partial(self.on_update_layer, "fixed"))

        self.moving_point_size = hp.make_int_spin_box(self, value=settings.moving_size, **schema.moving_size)
        self.moving_point_size.valueChanged.connect(partial(self.on_update_layer, "moving"))

        self.fixed_opacity = hp.make_int_spin_box(self, value=settings.fixed_opacity, **schema.fixed_opacity)
        self.fixed_opacity.valueChanged.connect(partial(self.on_update_layer, "fixed"))

        self.moving_opacity = hp.make_int_spin_box(self, value=settings.moving_opacity, **schema.moving_opacity)
        self.moving_opacity.valueChanged.connect(partial(self.on_update_layer, "moving"))

        self.text_size = hp.make_int_spin_box(self, value=settings.text_size, **schema.text_size)
        self.text_size.valueChanged.connect(self.on_update_text)

        self.text_color = hp.make_swatch(self, value=settings.text_color, **schema.text_color)
        self.text_color.evt_color_changed.connect(self.on_update_text)

        self.run_btn = hp.make_btn(
            self,
            "Compute transformation",
            tooltip="Compute transformation between the fixed and moving image.",
        )
        self.run_btn.clicked.connect(self.on_run)

        self.transform_choice = hp.make_combobox(self)
        hp.set_combobox_data(
            self.transform_choice, TRANSFORMATION_TRANSLATIONS, get_settings().registration.transformation
        )
        self.transform_choice.currentTextChanged.connect(self.on_run)

        self.registration_list = QtRegistrationList(self)
        self.registration_list.evt_lock.connect(self.on_lock_registration)
        self.registration_list.evt_checked.connect(self.on_load_registration)
        self.registration_list.evt_remove.connect(self.on_remove_registration)

        layout = hp.make_form_layout()
        layout.addRow(self.load_moving_btn)
        layout.addRow(self.image_manager)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(hp.make_label(self, "Size (fixed)"), self.fixed_point_size)
        layout.addRow(hp.make_label(self, "Size (moving)"), self.moving_point_size)
        layout.addRow(hp.make_label(self, "Opacity (fixed)"), self.fixed_opacity)
        layout.addRow(hp.make_label(self, "Opacity (moving)"), self.moving_opacity)
        layout.addRow(hp.make_label(self, "Label size"), self.text_size)
        layout.addRow(hp.make_label(self, "Label color"), self.text_color)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(hp.make_label(self, "Type of transformation"), self.transform_choice)
        layout.addRow(self.run_btn)

        widget = QWidget()
        widget.setMinimumWidth(400)
        settings_layout = QVBoxLayout(widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.addLayout(layout)
        settings_layout.addWidget(hp.make_v_line())
        settings_layout.addWidget(self.registration_list, stretch=True)

        view_layout = QVBoxLayout()
        view_layout.addWidget(self.info, alignment=Qt.AlignCenter)
        view_layout.addLayout(fixed_layout)
        view_layout.addWidget(hp.make_v_line())
        view_layout.addLayout(moving_layout)

        main_layout = QHBoxLayout()
        main_layout.addLayout(view_layout, stretch=True)
        main_layout.addWidget(widget)
        return main_layout
