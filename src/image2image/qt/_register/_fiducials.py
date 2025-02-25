"""Fiducials marker."""

from __future__ import annotations

import typing as ty

import numpy as np
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view_check import QtCheckableTableView
from qtpy.QtCore import QModelIndex, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QFormLayout

from image2image.config import get_register_config
from image2image.utils.utilities import open_docs

if ty.TYPE_CHECKING:
    from image2image.qt.dialog_register import ImageRegistrationWindow


logger = logger.bind(src="FiducialsDialog")


class FiducialsDialog(QtFramelessTool):
    """Dialog to display fiducial marker information."""

    HIDE_WHEN_CLOSE = True

    # event emitted when the popup closes
    evt_close = Signal()
    evt_update = Signal()

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 0, no_sort=True, hidden=True)
        .add("index", "index", "int", 50)
        .add("y-f(px)", "y_px_fixed", "float", 50)
        .add("x-f(px)", "x_px_fixed", "float", 50)
        .add("y-m(px)", "y_px_moving", "float", 50)
        .add("x-m(px)", "x_px_moving", "float", 50)
    )

    last_point: int = 0

    def __init__(self, parent: ImageRegistrationWindow):
        super().__init__(parent)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.points_data: np.ndarray | None = None
        self.on_load()

    @property
    def n_points(self) -> int:
        """Number of points."""
        return 0 if self.points_data is None else len(self.points_data)

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        # change of model events
        connect(parent.evt_predicted, self.on_load, state=state)
        # table events
        connect(self.table.doubleClicked, self.on_double_click, state=state)

    def keyPressEvent(self, evt: QKeyEvent) -> None:
        """Key press event."""
        if evt.key() == Qt.Key_Escape:  # type: ignore[attr-defined]
            evt.ignore()
        elif evt.key() == Qt.Key_Backspace or evt.key() == Qt.Key_Delete:  # type: ignore[attr-defined]
            self.on_delete_row()
            evt.accept()
        else:
            super().keyPressEvent(evt)

    def on_delete_row(self) -> None:
        """Delete row."""
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        sel_model = self.table.selectionModel()
        if sel_model.hasSelection():
            indices = [index.row() for index in sel_model.selectedRows()]
            indices = sorted(indices, reverse=True)
            for index in indices:
                fixed_points = parent.fixed_points_layer.data
                moving_points = parent.moving_points_layer.data
                if index < len(fixed_points):
                    fixed_points = np.delete(fixed_points, index, axis=0)
                    with parent.fixed_points_layer.events.data.blocker(self.on_load):
                        parent.fixed_points_layer.data = fixed_points
                if index < len(moving_points):
                    moving_points = np.delete(moving_points, index, axis=0)
                    with parent.fixed_points_layer.events.data.blocker(self.on_load):
                        parent.moving_points_layer.data = moving_points
                logger.debug(f"Deleted index '{index}' from fiducial table")
            self.table.scrollTo(sel_model.currentIndex())
            self.evt_update.emit()

    def on_double_click(self, index: QModelIndex) -> None:
        """Zoom in."""
        row = index.row()
        self.on_select_point(row)

    def on_select_last_point(self) -> None:
        """Zoom in on last point."""
        if self.last_point is not None:
            row = self.last_point
            self.on_select_point(row)

    def on_select_point(self, row: int):
        """Zoom in on point."""
        get_register_config().zoom_factor = self.zoom_factor.value()

        # zoom-in
        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        if self.points_data is not None:
            try:
                y_fixed, x_fixed, y_moving, x_moving = self.points_data[row]
                self.last_point = row
            except IndexError:
                return

            # zoom-in on fixed data
            if not np.isnan(x_fixed):
                view = parent.view_fixed
                with view.viewer.camera.events.blocker():
                    view.viewer.camera.center = (0.0, y_fixed, x_fixed)
                    view.viewer.camera.zoom = get_register_config().zoom_factor
                logger.debug(
                    f"Applied focus center=({y_fixed:.1f}, {x_fixed:.1f}) zoom={view.viewer.camera.zoom:.3f}"
                    " on fixed data"
                )
            else:
                logger.debug("Fixed point was NaN - can't zoom-in on the point.")

            # sync views will take care of the rest
            if parent.transform is None:
                return

            # zoom-in on moving data
            if not np.isnan(x_moving):
                view = parent.view_moving
                with view.viewer.camera.events.blocker():
                    view.viewer.camera.center = (0.0, y_moving, x_moving)
                    view.viewer.camera.zoom = (
                        get_register_config().zoom_factor * parent.transform_model.moving_to_fixed_ratio
                    )
                logger.debug(
                    f"Applied focus center=({y_moving:.1f}, {x_moving:.1f}) zoom={view.viewer.camera.zoom:.3f}"
                    " on moving data"
                )
            else:
                logger.debug("Moving point was NaN - can't zoom-in on the point.")
        else:
            logger.debug("No fiducial points to zoom-in on.")

    def on_load(self, _evt: ty.Any = None) -> None:
        """On load."""

        def _str_fmt(value: float) -> str:
            if np.isnan(value):
                return ""
            return f"{value:.3f}"

        parent: ImageRegistrationWindow = self.parent()  # type: ignore[assignment]
        fixed_points_layer = parent.fixed_points_layer
        moving_points_layer = parent.moving_points_layer
        n = max([len(fixed_points_layer.data), len(moving_points_layer.data)])
        array = np.full((n, 4), fill_value=np.nan)
        array[0 : len(fixed_points_layer.data), 0:2] = fixed_points_layer.data
        array[0 : len(moving_points_layer.data), 2:] = moving_points_layer.data

        data = []
        for index, row in enumerate(array, start=1):
            data.append([True, str(index), *map(_str_fmt, row)])

        # get the current selection
        model_index = self.table.selectionModel().currentIndex()
        # reset table data
        self.table.reset_data()
        self.table.add_data(data)
        if model_index.isValid() and model_index.row() < self.table.n_rows:
            self.table.scrollTo(model_index)
        self.points_data = array

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle("Fiducial markers")

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
        self.zoom_factor = hp.make_double_spin_box(
            self,
            1,
            100,
            n_decimals=2,
            default=get_register_config().zoom_factor,
            func=self.on_select_last_point,
            step_size=0.25,
        )

        layout = hp.make_form_layout(parent=self, margin=6)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow("Zoom factor", self.zoom_factor)
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(
                    self,
                    "<b>Tip.</b> Double-click on a row to zoom in on the point.<br>"
                    "<b>Tip.</b> Press  <b>Delete</b> or <b>Backspace</b> to delete a point.",
                    alignment=Qt.AlignmentFlag.AlignHCenter,
                    object_name="tip_label",
                    enable_url=True,
                ),
                hp.make_url_btn(self, func=lambda: open_docs(dialog="fiducials-table")),
                stretch_id=(0,),
                spacing=2,
                margin=2,
                alignment=Qt.AlignmentFlag.AlignVCenter,
            )
        )
        return layout
