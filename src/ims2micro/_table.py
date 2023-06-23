"""Table selection."""
import typing as ty

import numpy as np
import qtextra.helpers as hp
from loguru import logger
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFormLayout

from ims2micro.utilities import style_form_layout

OverlayConfig = (
    TableConfig()
    .add("", "check", "bool", 25, no_sort=True)
    .add("channel name", "channel_name", "str", 125)
    .add("dataset", "dataset", "str", 250)
)

FiducialConfig = (
    TableConfig()
    .add("", "check", "bool", 0, no_sort=True)
    .add("index", "index", "int", 50)
    .add("y-m(px)", "y_px_micro", "float", 50)
    .add("x-m(px)", "x_px_micro", "float", 50)
    .add("y-i(px)", "y_px_ims", "float", 50)
    .add("x-i(px)", "x_px_ims", "float", 50)
)
if ty.TYPE_CHECKING:
    from ims2micro._select import LoadWidget
    from ims2micro.models import DataModel


class FiducialTableDialog(QtFramelessTool):
    """Dialog to display fiducial marker information."""

    HIDE_WHEN_CLOSE = True

    shown_once = False

    # event emitted when the popup closes
    evt_close = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.points_data = None
        self.on_load()

    def connect_events(self, state: bool = True):
        """Connect events."""
        # TODO: connect event that updates checkbox state when user changes visibility in layer list
        # change of model events
        connect(self.parent().fixed_points_layer.events.data, self.on_load, state=state)
        connect(self.parent().moving_points_layer.events.data, self.on_load, state=state)
        connect(self.parent().evt_predicted, self.on_load, state=state)
        # table events
        connect(self.table.doubleClicked, self.on_zoom_in, state=state)

    def on_zoom_in(self, index):
        """Zoom in."""
        row = index.row()
        ym, xm, yi, xi = self.points_data[row]
        # zoom-in on IMS data
        if not np.isnan(xi):
            view_moving = self.parent().view_moving
            view_moving.viewer.camera.center = (0.0, yi, xi)
            view_moving.viewer.camera.zoom = 15
            logger.debug(f"Applied focus center=({yi:.1f}, {xi:.1f}) zoom={15:.3f} on IMS data")
        if not np.isnan(xm):
            view_fixed = self.parent().view_fixed
            view_fixed.viewer.camera.center = (0.0, ym, xm)
            view_fixed.viewer.camera.zoom = 20
            logger.debug(f"Applied focus center=({ym:.1f}, {xm:.1f}) zoom={20:.3f} on micro data")

    def on_load(self, evt=None):
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

        self.table = QtCheckableTableView(self, config=FiducialConfig, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(FiducialConfig.header, FiducialConfig.no_sort_columns, FiducialConfig.hidden_columns)
        self.get_all_unchecked = self.table.get_all_unchecked
        self.get_all_checked = self.table.get_all_checked

        self.info_label = hp.make_label(self, "", tooltip="Information about the fiducial markers.")

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow(self.info_label)
        return layout


class OverlayTableDialog(QtFramelessTool):
    """Dialog to enable creation of overlays."""

    HIDE_WHEN_CLOSE = True

    shown_once = False

    # event emitted when the popup closes
    evt_close = Signal()

    def __init__(self, parent: "LoadWidget", model: "DataModel", view):
        super().__init__(parent)
        self.model = model
        self.view = view

        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    def connect_events(self, state: bool = True):
        """Connect events."""
        # TODO: connect event that updates checkbox state when user changes visibility in layer list
        # change of model events
        connect(self.parent().evt_loaded, self.on_load, state=state)
        connect(self.parent().evt_closed, self.on_load, state=state)
        # table events
        connect(self.table.evt_checked, self.on_toggle_channel, state=state)

    def on_toggle_channel(self, index: int, state: bool):
        """Toggle channel."""
        channel_name = self.table.get_value(OverlayConfig.channel_name, index)
        dataset = self.table.get_value(OverlayConfig.dataset, index)
        self.parent().evt_toggle_channel.emit(f"{channel_name} | {dataset}", state)

    def on_load(self, model: "DataModel"):
        """On load."""
        self.model = model
        data = []
        for name in self.model.get_reader().channel_names():
            channel_name, dataset = name.split(" | ")
            data.append([True, channel_name, dataset])
        existing_data = self.table.get_data()
        if existing_data:
            for exist_row in existing_data:
                for new_row in data:
                    if (
                        exist_row[OverlayConfig.channel_name] == new_row[OverlayConfig.channel_name]
                        and exist_row[OverlayConfig.dataset] == new_row[OverlayConfig.dataset]
                    ):
                        new_row[OverlayConfig.check] = exist_row[OverlayConfig.check]
        self.table.reset_data()
        self.table.add_data(data)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Channel Selection")

        self.table = QtCheckableTableView(self, config=OverlayConfig, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(OverlayConfig.header, OverlayConfig.no_sort_columns, OverlayConfig.hidden_columns)
        self.get_all_unchecked = self.table.get_all_unchecked
        self.get_all_checked = self.table.get_all_checked

        self.info_label = hp.make_label(self, "", tooltip="Information about currently overlaid items.")

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow(self.info_label)
        return layout
