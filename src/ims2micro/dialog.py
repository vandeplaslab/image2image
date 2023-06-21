"""Registration dialog."""

import typing as ty
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from napari.layers.points.points import Mode, Points
from napari.layers.utils._link_layers import link_layers
from qtextra._napari.mixins import ImageViewMixin
from qtextra.mixins import IndicatorMixin
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_mini_toolbar import QtMiniToolbar
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QHBoxLayout, QMainWindow, QSizePolicy, QVBoxLayout, QWidget, QGridLayout
from superqt import ensure_main_thread

# need to load to ensure all assets are loaded properly
import ims2micro.assets  # noqa: F401
from ims2micro import __version__
from ims2micro._select import IMSWidget, MicroscopyWidget
from ims2micro.config import CONFIG
from ims2micro.enums import TRANSFORMATION_TRANSLATIONS, ALLOWED_EXPORT_FORMATS
from ims2micro.models import DataModel, Transformation
from ims2micro.utilities import _get_text_data, _get_text_format, init_points_layer

if ty.TYPE_CHECKING:
    from napari.layers import Image

from loguru import logger


class ImageRegistrationDialog(QMainWindow, IndicatorMixin, ImageViewMixin):
    """Image registration dialog."""

    fixed_image_layer: ty.Optional[ty.List["Image"]] = None
    moving_image_layer: ty.Optional["Image"] = None
    temporary_transform: ty.Optional[Transformation] = None

    def __init__(self, parent):
        super().__init__()
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(f"ims2micro: Simple image registration tool (v{__version__})")
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setMouseTracking(True)
        self.setMinimumSize(1200, 800)

        # load configuration
        CONFIG.load()

        self._setup_ui()
        self.setup_events()

    @property
    def transform(self):
        """Retrieve transform."""
        transform = self.temporary_transform
        if transform:
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
                edge_color="black",
                symbol="cross",
            )
            init_points_layer(layer)
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
                edge_color="black",
                symbol="x",
            )
            init_points_layer(layer)
        return self.view_moving.layers["Moving (points)"]

    def setup_events(self, state: bool = True):
        """Additional setup."""
        connect(self._micro_widget.evt_loading, partial(self.on_indicator, which="fixed"), state=state)
        connect(self._micro_widget.evt_loaded, self.on_load_fixed, state=state)
        connect(self._micro_widget.evt_toggle_channel, self.on_toggle_fixed_channel, state=state)
        connect(self._micro_widget.evt_closed, self.on_close_fixed, state=state)

        connect(self._ims_widget.evt_show_transformed, self.on_toggle_transformed_moving, state=state)
        connect(self._ims_widget.evt_loading, partial(self.on_indicator, which="moving"), state=state)
        connect(self._ims_widget.evt_loaded, self.on_load_moving, state=state)
        connect(self._ims_widget.evt_closed, self.on_close_moving, state=state)
        connect(self._ims_widget.evt_view_type, self.on_change_view_type, state=state)

        # add fixed points layer
        connect(self.fixed_points_layer.events.data, self.on_run, state=state)
        connect(self.fixed_points_layer.events.add_point, partial(self.on_predict, "fixed"), state=state)

        # add moving points layer
        connect(self.moving_points_layer.events.data, self.on_run, state=state)
        connect(self.moving_points_layer.events.add_point, partial(self.on_predict, "moving"), state=state)

    def on_indicator(self, which: str, state: bool = True):
        """Set indicator."""
        indicator = self.moving_indicator if which == "moving" else self.fixed_indicator
        indicator.setVisible(state)

    def on_close_fixed(self, model: DataModel):
        """Close fixed image."""
        try:
            channel_names = model.channel_names()
            for name in channel_names:
                if name in self.view_fixed.layers:
                    del self.view_fixed.layers[name]
            self.fixed_points_layer.data = None
        except Exception as e:
            logger.error(e)

    @ensure_main_thread
    def on_load_fixed(self, model: DataModel):
        """Load fixed image."""
        self._on_load_fixed(model)
        self.on_indicator("fixed", False)

    def _on_load_fixed(self, model: DataModel):
        wrapper = model.get_reader()
        channel_names = wrapper.channel_names()
        for name in channel_names:
            if name in self.view_fixed.layers:
                del self.view_fixed.layers[name]
        fixed_image_layer = self.view_fixed.viewer.add_image(**wrapper.image())
        self.fixed_image_layer = fixed_image_layer if isinstance(fixed_image_layer, list) else [fixed_image_layer]
        if isinstance(self.fixed_image_layer, list) and len(self.fixed_image_layer) > 1:
            link_layers(self.fixed_image_layer, attributes=("opacity",))
        self.view_fixed.viewer.reset_view()

    def on_toggle_fixed_channel(self, name: str, state: bool):
        """Toggle fixed channel."""
        self.view_fixed.layers[name].visible = state

    def on_close_moving(self, model: DataModel):
        """Close moving image."""
        try:
            channel_names = model.channel_names()
            for name in channel_names:
                if name in self.view_moving.layers:
                    del self.view_moving.layers[name]
        except Exception as e:
            logger.error(e)
            self.moving_points_layer.data = None

    def on_change_view_type(self, view_type: str):
        """Change view type."""
        if self._ims_widget.model:
            self.moving_image_layer.data = self._ims_widget.model.get_reader().get_image(view_type)
            self.moving_image_layer.reset_contrast_limits()

    @ensure_main_thread
    def on_load_moving(self, model: DataModel):
        """Open modality."""
        self._on_load_moving(model)
        self.on_indicator("moving", False)

    def _on_load_moving(self, model: DataModel):
        wrapper = model.get_reader()
        moving_data = wrapper.image(str(CONFIG.view_type).lower())
        if moving_data["name"] in self.view_moving.layers:
            del self.view_moving.layers[moving_data["name"]]
        self.moving_image_layer = self.view_moving.viewer.add_image(**moving_data, colormap="turbo")
        # self.on_clear("fixed", True)
        # self.on_clear("moving", True)
        self.on_apply()
        self.view_moving.viewer.reset_view()

    def on_toggle_transformed_moving(self, state: bool):
        """Toggle visibility of transformed moving image."""
        if self.transformed_moving_image_layer:
            self.transformed_moving_image_layer.visible = state

    def _select_layer(self, which: str):
        """Select layer."""
        view, layer = (
            (self.view_fixed, self.fixed_points_layer)
            if which == "fixed"
            else (self.view_moving, self.moving_points_layer)
        )
        self._move_layer(view, layer)

    @staticmethod
    def _move_layer(view, layer):
        """Move a layer and select it."""
        view.layers.move(view.layers.index(layer), -1)
        view.layers.selection.select_only(layer)

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

    def on_mode(self, which: str, evt=None):
        """Update mode."""
        widget = self._get_mode_button(which, evt.mode)
        if widget is not None:
            widget.setChecked(True)

    def on_panzoom(self, which: str, evt=None):
        """Switch to `panzoom` tool."""
        self._select_layer(which)
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "pan_zoom"

    def on_move(self, which: str, evt=None):
        """Move points."""
        self._select_layer(which)
        widget = self.fixed_move_btn if which == "fixed" else self.moving_move_btn
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        layer.mode = "select" if widget.isChecked() else "pan_zoom"

    def on_add(self, which: str, evt=None):
        """Add point to the image."""
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

    def on_remove_selected(self, which: str, evt=None):
        """Remove selected points from the image."""
        layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
        # there is no data to remove
        if layer.data.shape[0] == 0:
            return
        layer.remove_selected()

    def on_clear(self, which: str, force: bool = True):
        """Remove point to the image."""
        if force or hp.confirm(self, "Are you sure you want to remove all data points from the points layer?"):
            layer = self.fixed_points_layer if which == "fixed" else self.moving_points_layer
            layer.data = np.zeros((0, 2))
            self.on_clear_transformation()

    def on_clear_transformation(self):
        """Clear transformation and remove image."""
        if self.temporary_transform:
            self.temporary_transform = None
        if self.transformed_moving_image_layer:
            with suppress(ValueError):
                self.view_fixed.layers.remove(self.transformed_moving_image_layer)

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

    def on_save(self, _evt=None):
        """Export transformation."""
        transform = self.temporary_transform
        if transform is None:
            logger.warning("Cannot save transformation - no transformation has been computed.")
            return
        # get filename which is based on the IMS dataset
        filename = transform.ims_model.path.parent.stem + "_i2m_transform.json"
        path = hp.get_save_filename(
            self,
            "Save transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_EXPORT_FORMATS,
            base_filename=filename,
        )
        if path:
            path = Path(path)
            CONFIG.output_dir = str(path.parent)
            transform.to_file(path)
            hp.toast(self, "Exported transformation", f"Saved transformation to\n\n<b>{path}</b>")

    def on_load(self, _evt=None):
        """Import transformation."""
        path = hp.get_filename(
            self,
            "Load transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_EXPORT_FORMATS,
        )
        if path:
            from ims2micro.models import load_from_file

            # reset all widgets
            self._micro_widget._on_close_dataset()
            self._ims_widget._on_close_dataset()

            # load transformation
            path = Path(path)
            CONFIG.output_dir = str(path.parent)
            transformation_type, micro_path, fixed_points, ims_path, moving_points = load_from_file(path)
            self.transform_choice.setCurrentText(transformation_type)
            if micro_path:
                self._micro_widget.on_set_path(str(micro_path))
            if ims_path:
                self._ims_widget.on_set_path(str(ims_path))
            self._update_layer_points(self.moving_points_layer, moving_points, block=False)
            self._update_layer_points(self.fixed_points_layer, fixed_points, block=False)
            self.on_update_text(block=False)

    @ensure_main_thread
    def on_apply(self):
        """Apply transformation."""
        if self.transform is None or self.moving_image_layer is None:
            logger.warning("Cannot apply transformation - no transformation has been computed.")
            return

        # add image and apply transformation
        if self.transformed_moving_image_layer:
            self.transformed_moving_image_layer.affine = self.transform.params
        else:
            self.view_fixed.viewer.add_image(
                self.moving_image_layer.data,
                name="Transformed",
                blending="translucent",
                opacity=self.moving_opacity.value() / 100,
                affine=self.transform.params,
                visible=CONFIG.show_transformed,
                colormap="turbo",
            )
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
    def _update_layer_points(layer: Points, data: np.ndarray, block: bool = True):
        """Update points layer."""
        if block:
            with layer.events.data.blocker():
                layer.data = data
                layer.properties = _get_text_data(data)
        else:
            layer.data = data
            layer.properties = _get_text_data(data)

    def on_update_layer(self, which: str, _value=None):
        """Update points layer."""
        CONFIG.size_fixed = self.fixed_point_size.value()
        CONFIG.size_moving = self.moving_point_size.value()
        CONFIG.opacity_fixed = self.fixed_opacity.value()
        CONFIG.opacity_moving = self.moving_opacity.value()

        # update point size
        if self.fixed_points_layer and which == "fixed":
            self.fixed_points_layer.size = CONFIG.size_fixed
            self.fixed_point_size.current_size = CONFIG.size_fixed
        if self.moving_points_layer and which == "moving":
            self.moving_points_layer.size = CONFIG.size_moving
            self.moving_points_layer.current_size = CONFIG.size_moving

        if self.fixed_image_layer and which == "fixed":
            self.fixed_image_layer[0].opacity = CONFIG.opacity_fixed / 100
        if self.transformed_moving_image_layer and which == "moving":
            self.transformed_moving_image_layer.opacity = CONFIG.opacity_moving / 100

    def on_update_text(self, _=None, block: bool = True):
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

    def on_set_focus(self):
        """Lock current focus to specified range."""
        self.zoom.setValue(self.view_fixed.viewer.camera.zoom)
        _, y, x = self.view_fixed.viewer.camera.center
        self.x_center.setValue(x)
        self.y_center.setValue(y)

    def on_apply_focus(self):
        """Apply focus to the current image range."""
        if all([v == 1.0] for v in [*self.view_fixed.viewer.camera.center, self.view_fixed.viewer.camera.zoom]):
            logger.warning("Please specify zoom and center first.")
            return
        self.view_fixed.viewer.camera.center = (0.0, self.y_center.value(), self.x_center.value())
        self.view_fixed.viewer.camera.zoom = self.zoom.value()

    def on_viewer_orientation_changed(self, value=None):
        """Change viewer orientation."""
        self.viewer_orientation.currentText()

    # noinspection PyAttributeOutsideInit
    def _setup_ui(self) -> QHBoxLayout:
        """Create panel."""
        view_layout = self._make_image_layout()

        self.load_btn = hp.make_btn(
            self,
            "Import from file",
            tooltip="Import previously computed transformation.",
            func=self.on_load,
        )

        self._micro_widget = MicroscopyWidget(self)
        self._ims_widget = IMSWidget(self)

        self.transform_choice = hp.make_combobox(self)
        hp.set_combobox_data(self.transform_choice, TRANSFORMATION_TRANSLATIONS, "Affine")
        self.transform_choice.currentTextChanged.connect(self.on_run)

        self.run_btn = hp.make_btn(
            self,
            "Compute transformation",
            tooltip="Compute transformation between the fixed and moving image.",
            func=self.on_run,
        )
        self.save_btn = hp.make_btn(
            self,
            "Export to file",
            tooltip="Export transformation to file.",
            func=self.on_save,
        )

        side_layout = hp.make_form_layout()
        side_layout.addRow(self.load_btn)
        side_layout.addRow(hp.make_h_line_with_text("or"))
        side_layout.addRow(self._micro_widget)
        side_layout.addRow(hp.make_h_line_with_text("+"))
        side_layout.addRow(self._ims_widget)
        side_layout.addRow(hp.make_h_line(self))
        side_layout.addRow(self._make_focus_layout())
        side_layout.addRow(hp.make_h_line(self))
        side_layout.addRow(hp.make_label(self, "Type of transformation"), self.transform_choice)
        side_layout.addRow(self.run_btn)
        side_layout.addRow(self.save_btn)
        side_layout.addRow(hp.make_spacer_widget())
        side_layout.addRow(hp.make_h_line_with_text("Settings"))
        side_layout.addRow(self._make_settings_layout())

        widget = QWidget()
        self.setCentralWidget(widget)
        main_layout = QHBoxLayout(widget)
        main_layout.addLayout(view_layout, stretch=True)
        main_layout.addWidget(hp.make_v_line())
        main_layout.addLayout(side_layout)

    def _make_focus_layout(self):
        self.set_current_focus = hp.make_btn(self, "Set current range", func=self.on_set_focus)
        self.x_center = hp.make_double_spin_box(self, -1e5, 1e5, step_size=500)  # , func=self.on_apply_focus)
        self.y_center = hp.make_double_spin_box(self, -1e5, 1e5, step_size=500)  # , func=self.on_apply_focus)
        self.zoom = hp.make_double_spin_box(self, -1e5, 1e5, step_size=0.5)  # , func=self.on_apply_focus)
        self.use_focus = hp.make_btn(self, "Use focus", func=self.on_apply_focus)

        layout = hp.make_form_layout()
        layout.addRow(self.set_current_focus)
        layout.addRow(hp.make_label(self, "Center (x)"), self.x_center)
        layout.addRow(hp.make_label(self, "Center (y)"), self.y_center)
        layout.addRow(hp.make_label(self, "Zoom"), self.zoom)
        layout.addRow(self.use_focus)
        return layout

    def _make_settings_layout(self):
        self.fixed_point_size = hp.make_int_spin_box(
            self, value=CONFIG.size_fixed, tooltip="Size of the points shown in the fixed image."
        )
        self.fixed_point_size.valueChanged.connect(partial(self.on_update_layer, "fixed"))

        self.moving_point_size = hp.make_int_spin_box(
            self, value=CONFIG.size_moving, tooltip="Size of the points shown in the moving image."
        )
        self.moving_point_size.valueChanged.connect(partial(self.on_update_layer, "moving"))

        self.fixed_opacity = hp.make_int_spin_box(
            self, value=CONFIG.opacity_fixed, step_size=10, tooltip="Opacity of the fixed image"
        )
        self.fixed_opacity.valueChanged.connect(partial(self.on_update_layer, "fixed"))

        self.moving_opacity = hp.make_int_spin_box(
            self,
            value=CONFIG.opacity_moving,
            step_size=10,
            tooltip="Opacity of the moving image in the fixed view",
        )
        self.moving_opacity.valueChanged.connect(partial(self.on_update_layer, "moving"))

        self.text_size = hp.make_int_spin_box(
            self, value=CONFIG.label_size, minimum=4, maximum=60, tooltip="Size of the text associated with each label."
        )
        self.text_size.valueChanged.connect(self.on_update_text)

        self.text_color = hp.make_swatch(
            self, default=CONFIG.label_color, tooltip="Color of the text associated with each label."
        )
        self.text_color.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Minimum)
        self.text_color.evt_color_changed.connect(self.on_update_text)

        # self.viewer_orientation = hp.make_combobox(self, tooltip="Orientation of the viewer.")
        # hp.set_combobox_data(self.viewer_orientation, ORIENTATION_TRANSLATIONS, str(CONFIG.viewer_orientation))
        # self.viewer_orientation.currentTextChanged.connect(self.on_viewer_orientation_changed)

        layout = hp.make_form_layout()
        layout.addRow(hp.make_label(self, "Size (fixed)"), self.fixed_point_size)
        layout.addRow(hp.make_label(self, "Size (moving)"), self.moving_point_size)
        layout.addRow(hp.make_label(self, "Opacity (fixed)"), self.fixed_opacity)
        layout.addRow(hp.make_label(self, "Opacity (moving)"), self.moving_opacity)
        layout.addRow(hp.make_label(self, "Label size"), self.text_size)
        layout.addRow(hp.make_label(self, "Label color"), self.text_color)
        # layout.addRow(hp.make_label(self, "Viewer orientation"), self.viewer_orientation)
        return layout

    def _make_image_layout(self):
        self.info = hp.make_label(
            self,
            "Please select at least <b>3 points</b> in either image to compute transformation.",
            tooltip="Information regarding registration.",
        )

        view_layout = QVBoxLayout()
        view_layout.setContentsMargins(0, 0, 0, 0)
        view_layout.setSpacing(0)
        view_layout.addWidget(self.info, alignment=Qt.AlignCenter)
        view_layout.addLayout(self._make_fixed_view())
        view_layout.addWidget(hp.make_v_line())
        view_layout.addLayout(self._make_moving_view())
        return view_layout

    def _make_fixed_view(self):
        self.view_fixed = self._make_image_view(self, add_toolbars=False)
        self.view_fixed.viewer.text_overlay.text = "Fixed"
        self.view_fixed.viewer.text_overlay.font_size = 8
        self.view_fixed.viewer.text_overlay.visible = True

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=True)
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
            tooltip="Move points in the fixed image...",
            checkable=True,
        )
        self.fixed_add_btn = toolbar.insert_qta_tool(
            "add",
            func=lambda *args: self.on_add("fixed"),
            tooltip="Add new point to the fixed image...",
            checkable=True,
        )
        self.fixed_zoom_btn = toolbar.insert_qta_tool(
            "zoom",
            func=lambda *args: self.on_panzoom("fixed"),
            tooltip="Switch to zoom-only mode...",
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

    def _make_moving_view(self):
        self.view_moving = self._make_image_view(self, add_toolbars=False)
        self.view_moving.viewer.text_overlay.text = "Moving"
        self.view_moving.viewer.text_overlay.font_size = 8
        self.view_moving.viewer.text_overlay.visible = True

        toolbar = QtMiniToolbar(self, Qt.Vertical, add_spacer=True)
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
            tooltip="Move points in the fixed image...",
            checkable=True,
        )
        self.moving_add_btn = toolbar.insert_qta_tool(
            "add",
            func=lambda *args: self.on_add("moving"),
            tooltip="Add new point to the moving image...",
            checkable=True,
        )
        self.moving_zoom_btn = toolbar.insert_qta_tool(
            "zoom",
            func=lambda *args: self.on_panzoom("moving"),
            tooltip="Switch to zoom-only mode...",
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

    def closeEvent(self, evt):
        """Close."""
        CONFIG.save()
        return super().closeEvent(evt)
