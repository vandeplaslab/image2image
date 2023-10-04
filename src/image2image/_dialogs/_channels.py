"""Select channels."""
import typing as ty

from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout

from image2image.utilities import style_form_layout

if ty.TYPE_CHECKING:
    from image2image._select import LoadWidget
    from image2image.models.data import DataModel


class OverlayChannelsDialog(QtFramelessTool):
    """Dialog to enable creation of overlays."""

    HIDE_WHEN_CLOSE = True

    # event emitted when the popup closes
    evt_close = Signal()

    TABLE_CONFIG = (
        TableConfig()
        .add("", "check", "bool", 25, no_sort=True)
        .add("channel name", "channel_name", "str", 125, no_sort=True)
        .add("dataset", "dataset", "str", 250, no_sort=True)
    )

    def __init__(self, parent: "LoadWidget", model: "DataModel", view):
        self.model = model
        self.view = view
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    def connect_events(self, state: bool = True):
        """Connect events."""
        # TODO: connect event that updates checkbox state when user changes visibility in layer list
        # change of model events
        connect(self.parent().dataset_dlg.evt_loaded, self.on_update_data_list, state=state)  # noqa
        connect(self.parent().dataset_dlg.evt_closed, self.on_update_data_list, state=state)  # noqa
        # table events
        connect(self.table.evt_checked, self.on_toggle_channel, state=state)

    def on_toggle_channel(self, index: int, state: bool):
        """Toggle channel."""
        if index == -1:
            self.parent().evt_toggle_all_channels.emit(state)  # noqa
        else:
            channel_name = self.table.get_value(self.TABLE_CONFIG.channel_name, index)
            dataset = self.table.get_value(self.TABLE_CONFIG.dataset, index)
            self.parent().evt_toggle_channel.emit(f"{channel_name} | {dataset}", state)  # noqa
        self.on_update_info()

    def on_update_info(self):
        """Update information about selected/total channels."""
        n_total = self.table.n_rows
        n_selected = len(self.table.get_all_checked())
        verb = "is" if n_selected == 1 else "are"
        self.info.setText(
            f"Total number of channels: <b>{n_total}</b> out of which <b>{n_selected}</b> {verb} selected."
        )

    def on_update_data_list(self, model: "DataModel"):
        """On load."""
        if not model:
            return

        self.model = model
        data = []
        reader = self.model.get_wrapper()
        if reader:
            for name in reader.channel_names():
                channel_name, dataset = name.split(" | ")
                data.append([True, channel_name, dataset])
        existing_data = self.table.get_data()
        if existing_data:
            for exist_row in existing_data:
                for new_row in data:
                    if (
                        exist_row[self.TABLE_CONFIG.channel_name] == new_row[self.TABLE_CONFIG.channel_name]
                        and exist_row[self.TABLE_CONFIG.dataset] == new_row[self.TABLE_CONFIG.dataset]
                    ):
                        new_row[self.TABLE_CONFIG.check] = exist_row[self.TABLE_CONFIG.check]
        self.table.reset_data()
        self.table.add_data(data)
        self.on_update_info()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        which = "Fixed" if self.model.is_fixed else "Moving"
        self._title_label.setText(f"'{which}' Channel Selection")

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )

        self.info = hp.make_label(self, "", enable_url=True)

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow(self.info)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Check/uncheck a row to toggle visibility of the channel.",
                alignment=Qt.AlignHCenter,  # noqa
                object_name="tip_label",
                enable_url=True,
            )
        )
        return layout

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        print(key)
        if key == Qt.Key_Escape:  # noqa
            evt.ignore()
        else:
            super().keyPressEvent(evt)
