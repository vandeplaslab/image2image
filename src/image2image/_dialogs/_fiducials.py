import numpy as np
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout

from image2image.utilities import style_form_layout


class FiducialsDialog(QtFramelessTool):
    """Dialog to display fiducial marker information."""

    HIDE_WHEN_CLOSE = True

    # event emitted when the popup closes
    evt_close = Signal()

    TABLE_CONFIG = (
        TableConfig()
        .add("", "check", "bool", 0, no_sort=True, hidden=True)
        .add("index", "index", "int", 50)
        .add("y-m(px)", "y_px_micro", "float", 50)
        .add("x-m(px)", "x_px_micro", "float", 50)
        .add("y-i(px)", "y_px_ims", "float", 50)
        .add("x-i(px)", "x_px_ims", "float", 50)
    )

    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.points_data = None
        self.on_load()

    def connect_events(self, state: bool = True):
        """Connect events."""
        # change of model events
        connect(self.parent().fixed_points_layer.events.data, self.on_load, state=state)
        connect(self.parent().moving_points_layer.events.data, self.on_load, state=state)
        connect(self.parent().evt_predicted, self.on_load, state=state)
        # table events
        connect(self.table.doubleClicked, self.on_double_click, state=state)

    def keyPressEvent(self, evt):
        """Key press event."""
        if evt.key() == Qt.Key_Escape:
            evt.ignore()
        elif evt.key() == Qt.Key_Backspace or evt.key() == Qt.Key_Delete:
            self.on_delete_row()
            evt.accept()
        else:
            super().keyPressEvent(evt)

    def on_delete_row(self):
        """Delete row."""
        sel_model = self.table.selectionModel()
        if sel_model.hasSelection():
            indices = [index.row() for index in sel_model.selectedRows()]
            indices = sorted(indices, reverse=True)
            for index in indices:
                fixed_points = self.parent().fixed_points_layer.data
                moving_points = self.parent().moving_points_layer.data
                if index < len(fixed_points):
                    fixed_points = np.delete(fixed_points, index, axis=0)
                    self.parent().fixed_points_layer.data = fixed_points
                if index < len(moving_points):
                    moving_points = np.delete(moving_points, index, axis=0)
                    self.parent().moving_points_layer.data = moving_points
                logger.debug(f"Deleted {index} from fiducial table")

    def on_double_click(self, index):
        """Zoom in."""
        row = index.row()
        y_micro, x_micro, y_ims, x_ims = self.points_data[row]
        # zoom-in on fixed data
        if not np.isnan(x_micro):
            view_fixed = self.parent().view_fixed
            view_fixed.viewer.camera.center = (0.0, y_micro, x_micro)
            view_fixed.viewer.camera.zoom = 5
            logger.debug(
                f"Applied focus center=({y_micro:.1f}, {x_micro:.1f}) zoom={view_fixed.viewer.camera.zoom:.3f} on micro"
                f" data"
            )
        # zoom-in on moving data
        if not np.isnan(x_ims):
            view_moving = self.parent().view_moving
            view_moving.viewer.camera.center = (0.0, y_ims, x_ims)
            view_moving.viewer.camera.zoom = 50
            logger.debug(
                f"Applied focus center=({y_ims:.1f}, {x_ims:.1f}) zoom={view_moving.viewer.camera.zoom:.3f} on IMS data"
            )

    def on_load(self, _evt=None):
        """On load."""

        def _str_fmt(value):
            if np.isnan(value):
                return ""
            return f"{value:.3f}"

        fixed_points_layer = self.parent().fixed_points_layer
        moving_points_layer = self.parent().moving_points_layer
        n = max([len(fixed_points_layer.data), len(moving_points_layer.data)])
        array = np.full((n, 4), fill_value=np.nan)
        array[0 : len(fixed_points_layer.data), 0:2] = fixed_points_layer.data
        array[0 : len(moving_points_layer.data), 2:] = moving_points_layer.data

        data = []
        for index, row in enumerate(array, start=1):
            data.append([True, str(index), *map(_str_fmt, row)])
        self.table.reset_data()
        self.table.add_data(data)
        self.points_data = array

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Fiducial markers")

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Double-click on a row to zoom in on the point."
                "<b>Tip.</b> Press  <b>Delete</b> or <b>Backspace</b> to delete a point.",
                alignment=Qt.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        return layout
