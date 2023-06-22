"""Table selection."""
import typing as ty

import qtextra.helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QFormLayout

from ims2micro.utilities import style_form_layout

Config = (
    TableConfig()
    .add("", "check", "bool", 25, no_sort=True)
    .add("channel name", "channel_name", "str", 125)
    .add("dataset", "dataset", "str", 250)
)

if ty.TYPE_CHECKING:
    from ims2micro._select import LoadWidget
    from ims2micro.models import DataModel


class TableDialog(QtFramelessTool):
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
        connect(self.parent().evt_closed, self.on_clear, state=state)
        # table events
        connect(self.table.evt_checked, self.on_toggle_channel, state=state)
        # synchronize view events
        # connect(self.view.viewer.)

    def on_toggle_channel(self, index: int, state: bool):
        """Toggle channel."""
        channel_name = self.table.get_value(Config.channel_name, index)
        dataset = self.table.get_value(Config.dataset, index)
        self.parent().evt_toggle_channel.emit(f"{channel_name} | {dataset}", state)

    def on_clear(self):
        """On clear."""
        self.table.reset_data()

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
                        exist_row[Config.channel_name] == new_row[Config.channel_name]
                        and exist_row[Config.dataset] == new_row[Config.dataset]
                    ):
                        new_row[Config.check] = exist_row[Config.check]
        self.table.reset_data()
        self.table.add_data(data)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Channel Selection")

        self.table = QtCheckableTableView(self, config=Config, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(Config.header, Config.no_sort_columns, Config.hidden_columns)
        self.get_all_unchecked = self.table.get_all_unchecked
        self.get_all_checked = self.table.get_all_checked

        self.info_label = hp.make_label(self, "", tooltip="Information about currently overlaid items.")

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow(self.info_label)
        return layout
