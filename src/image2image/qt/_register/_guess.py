"""Guess dialog."""

from __future__ import annotations

import typing as ty

import numpy as np
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtCore import Qt  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout
from superqt.utils import qdebounced

from image2image.utils.fiducials import (
    contour_formatter,
    contour_to_points,
    filter_contours,
    find_contours,
    simplify_contour,
)
from image2image.utils.utilities import open_docs

if ty.TYPE_CHECKING:
    from image2image.qt.dialog_register import ImageRegistrationWindow


logger = logger.bind(src="InitialTransformDialog")


class GuessDialog(QtFramelessTool):
    """Dialog to pre-process moving image."""

    contours: list[np.ndarray] | None = None
    simplified_contours: list[np.ndarray] | None = None
    current_index: int = -1
    current_contour: np.ndarray | None = None
    current_transformed_contour: np.ndarray | None = None
    _fixed_point_set: np.ndarray | None = None

    def __init__(self, parent: ImageRegistrationWindow):
        self.CONFIG = parent.CONFIG
        super().__init__(parent)
        self.on_detect()
        connect(parent.temporary_fixed_points_layer.events.data, self.on_point_selected, state=True)
        self.fixed_point_set = None  # disables the OK button

    @property
    def contour_index(self) -> int:
        """Get region index."""
        current = self.region_choice.currentText()
        return int(current.split(" ")[1])

    @property
    def fixed_point_set(self) -> None:
        """Get fixed point set."""
        return self._fixed_point_set

    @fixed_point_set.setter
    def fixed_point_set(self, value):
        """Set fixed point set."""
        self._fixed_point_set = value
        hp.disable_widgets(self.ok_btn, disabled=value is None)

    def on_detect(self) -> None:
        """Detect fiducials."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        reader = parent.get_current_moving_reader()
        if reader is None:
            return
        # get image and detect contours
        image = reader.get_channel(0)
        if image is not None:
            self.contours = find_contours(image)
            logger.trace(f"Found {len(self.contours)} contours...")
            self.on_update_region_choices()

    def on_update_region_choices(self) -> None:
        """Detect fiducials."""
        if not self.contours:
            return
        # update available contours
        self.simplified_contours = filter_contours(simplify_contour(self.contours, self.distance.value()))

        # update list of contours
        options = contour_formatter(self.simplified_contours)
        hp.combobox_setter(self.region_choice, clear=True, items=options)
        # update number of available point indices
        contour_index = self.contour_index
        contour = contour_to_points(self.simplified_contours[contour_index])
        with hp.qt_signals_blocked(self.point_index):
            self.point_index.setMinimum(-1)
            self.point_index.setMaximum(len(contour) - 1)
            self.point_index.setValue(-1)
        self.on_region_change()

    def on_region_change(self) -> None:
        """Update region."""
        if not self.contours:
            return
        contour_index = self.contour_index
        logger.trace(f"Selected contour {contour_index}...")
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        # add contour to the moving image
        contour = contour_to_points(self.simplified_contours[contour_index])
        contour = contour[:, [1, 0]]
        contour = parent.transform_model.apply_moving_initial_transform(contour, inverse=False)
        self.current_contour = parent.temporary_moving_points_layer.data = contour
        # select the top left point
        point_index = self.point_index.value()
        if point_index == -1:
            point_index = np.argmin(contour[:, 0] + contour[:, 1])
        self.current_index = point_index
        # update size
        sizes = parent.temporary_moving_points_layer.size
        min_size = np.min(sizes)
        sizes = np.full_like(sizes, min_size)
        sizes[point_index] = min_size * 2
        parent.temporary_moving_points_layer.size = sizes
        # update color so it's easier to see which point they are trying to edit
        colors = ["yellow"] * len(contour)
        colors[point_index] = "cyan"
        parent.temporary_moving_points_layer.border_color = colors

        if self.zoom_in.isChecked():
            y, x = contour[point_index]
            parent.view_moving.camera.center = (x, y, 0)
            parent.view_moving.viewer.camera.zoom = (
                parent.CONFIG.zoom_factor * parent.transform_model.moving_to_fixed_ratio * 0.05
            )
        self.fixed_point_set = None

    def on_enable_selection(self) -> None:
        """Detect fiducials."""
        from napari.layers.points.points import Mode

        if not self.contours:
            return
        contour_index = self.contour_index
        logger.trace(f"Selected contour {contour_index}...")
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        parent.temporary_fixed_points_layer.mode = Mode.ADD
        parent.view_fixed.select_one_layer(parent.temporary_fixed_points_layer)

    @qdebounced(timeout=100)
    def on_point_selected(self, _evt: ty.Any) -> None:
        """Point was added."""
        self._on_point_selected()

    def _on_point_selected(self) -> None:
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        moving_reader = parent.get_current_moving_reader()
        fixed_reader = parent.get_current_fixed_reader()
        if not moving_reader or not fixed_reader:
            logger.warning("No moving or fixed image loaded...")
            return
        data = parent.temporary_fixed_points_layer.data
        if len(data) == 0:  # or len(data) > 1:
            logger.warning("You need to add exactly one point in the fixed image...")
            return
        # get the last point from the fixed layer
        # points are in the physical space (in um) so let's convert it to pixel space
        self.fixed_point_set = data[-1]
        fixed_point = data[-1] * fixed_reader.resolution
        # get the corresponding point in the moving image
        # points are in the physical space (in um) so let's convert it to pixel space
        moving_point = self.current_contour[self.current_index] * moving_reader.resolution
        # transform the moving contour from um to pixel space
        contours = np.copy(self.current_contour) * moving_reader.resolution
        # subtract the moving point (normalize coordinates)
        contours -= moving_point
        # add the fixed point so it's origin is corrected
        contours += fixed_point
        # transform the points to the original space
        contours = contours * fixed_reader.inv_resolution
        self.current_transformed_contour = contours
        with parent.temporary_fixed_points_layer.events.data.blocker():
            parent.temporary_fixed_points_layer.data = contours

    def accept(self) -> None:
        """Accept changes."""
        if self.current_contour is None or self.current_transformed_contour is None:
            hp.warn(self, "No contour selected...")
            return None
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        parent.view_moving.remove_layers([parent.temporary_moving_points_layer.name, parent.moving_points_layer.name])
        parent.view_fixed.remove_layers([parent.temporary_fixed_points_layer.name, parent.fixed_points_layer.name])
        parent._update_layer_points(parent.moving_points_layer, self.current_contour, block=False)
        parent._update_layer_points(parent.fixed_points_layer, self.current_transformed_contour, block=False)
        parent.on_update_text(block=False)
        hp.disable_widgets(parent.guess_btn, disabled=False)
        return super().accept()

    def reject(self) -> None:
        """Reject changes."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        parent.view_moving.remove_layers([parent.temporary_moving_points_layer.name, parent.moving_points_layer.name])
        parent.view_fixed.remove_layers([parent.temporary_fixed_points_layer.name, parent.fixed_points_layer.name])
        hp.disable_widgets(parent.guess_btn, disabled=False)
        return super().reject()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_close_handle("Find fiducials...")

        self.detect_btn = hp.make_btn(
            self, "Find fiducials", func=self.on_detect, tooltip="Find fiducial markers in the moving image."
        )
        self.distance = hp.make_double_spin_box(
            self,
            minimum=0.0,
            maximum=10,
            value=self.CONFIG.simplify_contours_distance,
            step_size=0.25,
            n_decimals=2,
            tooltip="Distance to simplify contours.",
            func=self.on_update_region_choices,
        )
        self.region_choice = hp.make_combobox(self, func=self.on_region_change, tooltip="Select region of interest.")
        self.point_index = hp.make_int_spin_box(
            self, minimum=-1, maximum=-1, value=-1, func=self.on_region_change, tooltip="Select point index."
        )
        self.zoom_in = hp.make_checkbox(
            self,
            "",
            tooltip="Zoom-in on the selected point.",
            func=self.on_region_change,
            value=self.CONFIG.zoom_on_point,
        )

        self.select_btn = hp.make_btn(
            self,
            "Select point in fixed image",
            func=self.on_enable_selection,
            tooltip="Select a point in the fixed image.",
        )

        self.ok_btn = hp.make_btn(self, "OK", func=self.accept)

        layout = hp.make_form_layout(parent=self, margin=6)
        layout.addRow(header_layout)
        layout.addRow(
            hp.make_label(
                self,
                "You can try to to automatically find fiducial markers in the <b>moving</b> image.<br><br>"
                "<b>Instructions</b><br>"
                "1. Specify how many fiducials you wish to retain.<br>"
                "2. Click on the <b>Select point in the fixed image</b> button, to select the first matching point in"
                " the fixed image.<br>Make sure this matches the fiducial marker in the <b>moving</b> image, exactly."
                "<br>3. Click on <b>OK</b> to continue.<br><br>"
                "At this point you might be satisfied, however, it's advised to manually check <b>each</b> fiducial"
                " marker to ensure they are correctly placed.",
                wrap=True,
                alignment=Qt.AlignmentFlag.AlignLeft,
            )
        )
        layout.addRow(self.detect_btn)
        layout.addRow("Simplify contours", self.distance)
        layout.addRow("Region", self.region_choice)
        layout.addRow("Point index", self.point_index)
        layout.addRow("Zoom-in on point", self.zoom_in)
        layout.addRow(self.select_btn)
        layout.addRow(
            hp.make_h_layout(
                self.ok_btn,
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_url_btn(self, func=lambda: open_docs(dialog="generate-fiducials")),
                stretch_before=True,
                spacing=2,
                margin=2,
                alignment=Qt.AlignmentFlag.AlignVCenter,
            )
        )
        return layout
