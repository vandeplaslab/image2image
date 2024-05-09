"""Select channels."""

import typing as ty
from collections import Counter
from contextlib import contextmanager

from loguru import logger
from napari.utils.events import Event
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import FilterProxyModel, QtCheckableTableView
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout, QWidget
from superqt.utils import ensure_main_thread

from image2image.config import STATE

if ty.TYPE_CHECKING:
    from qtextra._napari.image.wrapper import NapariImageView

    from image2image.models.data import DataModel
    from image2image.qt._dialogs._select import LoadWidget


logger = logger.bind(src="ChannelsDialog")


class OverlayChannelsDialog(QtFramelessTool):
    """Dialog to enable creation of overlays."""

    HIDE_WHEN_CLOSE = True

    _editing: bool = False

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("index", "index", "int", 50, sizing="contents")
        .add("channel name", "channel_name", "str", 125)
        .add("dataset", "dataset", "str", 250)
        .add("key", "key", "str", 0, hidden=True)
    )

    def __init__(
        self,
        parent: "LoadWidget",
        model: "DataModel",
        view: "NapariImageView",
        is_fixed: ty.Optional[bool] = False,
        allow_iterate: bool = False,
    ):
        self.model = model
        self.view = view
        self.is_fixed = is_fixed
        self.allow_iterate = allow_iterate
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    @contextmanager
    def editing(self) -> ty.Generator[None, None, None]:
        """Context manager to set editing."""
        self._editing = True
        yield
        self._editing = False

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        parent: "LoadWidget" = self.parent()  # type: ignore[assignment]
        # change of model events
        connect(parent.dataset_dlg.evt_loaded, self.on_update_data_list, state=state)
        connect(parent.dataset_dlg.evt_closed, self.on_update_data_list, state=state)
        # table events
        connect(self.table.evt_checked, self.on_toggle_channel, state=state)
        connect(self.view.layers.events, self.sync_layers, state=state)
        connect(self.iterate_widget.evt_update, parent.evt_update_temp.emit, state=state)
        connect(self.iterate_widget.evt_add, parent.evt_add_channel.emit, state=state)
        connect(self.iterate_widget.evt_close, parent.evt_remove_temp.emit, state=state)
        connect(self.evt_hide, self.iterate_widget.on_close, state=state)

    @ensure_main_thread
    def sync_layers(self, event: Event) -> None:
        """Synchronize layers."""
        self._sync_layers(event)

    def _sync_layers(self, event: Event) -> None:
        if event.type in ["visible", "inserted"]:
            self._sync_layer_visibility(event)
            self.on_update_info()
        elif event.type == "removed":
            self._sync_layer_presence(event)
            self.on_update_info()

    def _sync_layer_presence(self, event: Event) -> None:
        with hp.qt_signals_blocked(self), self.editing():
            layer = event.value
            name = layer.name
            if " | " not in name:
                return
            row_id = self.table.get_row_id(self.TABLE_CONFIG.key, name)
            if row_id != -1:
                self.table.set_value(self.TABLE_CONFIG.check, row_id, False)

    def _sync_layer_visibility(self, _event: Event) -> None:
        with hp.qt_signals_blocked(self), self.editing():
            for layer in self.view.layers:
                name = layer.name
                if " | " not in name:
                    continue
                row_id = self.table.get_row_id(self.TABLE_CONFIG.key, name)
                if row_id != -1:
                    self.table.set_value(self.TABLE_CONFIG.check, row_id, layer.visible)

    def channel_list(self) -> list[str]:
        """Return list of currently selected channels."""
        checked = self.table.get_all_checked()
        channel_names = [self.table.get_value(self.TABLE_CONFIG.key, index) for index in checked]
        return channel_names

    def on_toggle_channel(self, index: int, state: bool) -> None:
        """Toggle channel."""
        if self._editing:
            return
        parent: "LoadWidget" = self.parent()  # type: ignore[assignment]
        with self.view.layers.events.blocker(self.sync_layers):
            if index == -1:
                channel_names = self.channel_list()
                parent.evt_toggle_all_channels.emit(state, channel_names)  # noqa
            else:
                channel_name = self.table.get_value(self.TABLE_CONFIG.channel_name, index)
                dataset = self.table.get_value(self.TABLE_CONFIG.dataset, index)
                parent.evt_toggle_channel.emit(f"{channel_name} | {dataset}", state)  # noqa
        self.on_update_info()

    def on_update_info(self) -> None:
        """Update information about selected/total channels."""
        n_total = self.table.n_rows
        n_selected = len(self.table.get_all_checked())
        verb = "is" if n_selected == 1 else "are"
        self.info.setText(
            f"Total number of channels: <b>{n_total}</b> out of which <b>{n_selected}</b> {verb} selected."
        )

    def on_update_data_list(self, model: "DataModel") -> None:
        """On load."""
        if not model:
            return

        self.model = model
        data = []
        wrapper = self.model.wrapper
        if wrapper:
            counter = Counter()
            for name in wrapper.channel_names():
                channel_name, dataset = name.split(" | ")
                data.append([False, counter[dataset], channel_name, dataset, name])
                counter[dataset] += 1
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
        self.table.enable_all_check = len(data) < 20
        self.on_update_info()
        logger.trace(f"Updated channel table - {len(data)} rows.")

    # def on_open_iterate_dlg(self):
    #     """Open iterate dialog."""
    #     self.iterate_dialog = None
    #     if self.iterate_dialog is None:
    #         parent: "LoadWidget" = self.parent()  # type: ignore[assignment]
    #         self.iterate_dialog = IterateDialog(self)

    #     self.iterate_dialog.show()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        if self.is_fixed:
            which = "Fixed" if self.is_fixed else "Moving"
            title = f"'{which}' Channel Selection"
        else:
            title = "Channel Selection"
        self._title_label.setText(title)

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
        if STATE.allow_filters:
            self.table_proxy = FilterProxyModel(self)
            self.table_proxy.setSourceModel(self.table.model())
            self.table.model().table_proxy = self.table_proxy
            self.table.setModel(self.table_proxy)
            self.filter_by_dataset = hp.make_line_edit(
                self,
                placeholder="Type in dataset name...",
                func_changed=lambda text, col=self.TABLE_CONFIG.dataset: self.table_proxy.setFilterByColumn(text, col),
            )
            self.filter_by_name = hp.make_line_edit(
                self,
                placeholder="Type in channel name...",
                func_changed=lambda text, col=self.TABLE_CONFIG.channel_name: self.table_proxy.setFilterByColumn(
                    text, col
                ),
            )
        self.iterate_widget = IterateWidget(self)
        if not self.allow_iterate:
            self.iterate_widget.hide()
        self.info = hp.make_label(self, "", enable_url=True, alignment=Qt.AlignmentFlag.AlignHCenter)

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        if STATE.allow_filters:
            layout.addRow(hp.make_h_layout(self.filter_by_name, self.filter_by_dataset, stretch_id=(0, 1), spacing=1))
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Check/uncheck a row to toggle visibility of the channel.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        if self.allow_iterate:
            layout.addRow(hp.make_h_line_with_text("Iterate through images"))
        layout.addRow(self.iterate_widget)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(self.info)
        return layout

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        if key == Qt.Key.Key_Escape:  # noqa
            evt.ignore()
        else:
            super().keyPressEvent(evt)


class IterateWidget(QWidget):
    """Widget for iterating through data."""

    HIDE_WHEN_CLOSE = True

    evt_update = Signal(tuple)
    evt_add = Signal(tuple)
    evt_close = Signal(tuple)

    current_index: int = 0

    def __init__(self, parent: OverlayChannelsDialog):
        """Init."""
        super().__init__(parent)
        self.setup_ui()

    # noinspection PyTestUnpassedFixture,PyAttributeOutsideInit
    def setup_ui(self):
        """Setup UI."""
        self.dataset_combo = hp.make_combobox(self, func=self.on_change_source)
        self.index_spinbox = hp.make_int_spin_box(self, min=0, max=1_000, func=self.on_change_index)
        self.channel_label = hp.make_label(self, "", bold=True, alignment=Qt.AlignmentFlag.AlignCenter)

        layout = hp.make_form_layout(self)
        layout.addRow(hp.make_h_layout(self.dataset_combo, self.index_spinbox, stretch_id=(0,)))
        layout.addRow(self.channel_label)
        layout.addRow(hp.make_btn(self, "Add to viewer", func=self.on_add_to_viewer, tooltip="Add image to viewer."))

        parent: OverlayChannelsDialog = self.parent()
        parent.parent().dataset_dlg.evt_loaded.connect(self.on_update_sources)
        self.on_update_sources()

    def on_update_sources(self):
        """Update sources."""
        parent: OverlayChannelsDialog = self.parent()
        dataset_names = parent.model.dataset_names(reader_type=("image",))
        current = self.dataset_combo.currentText()
        with hp.qt_signals_blocked(self.dataset_combo):
            hp.combobox_setter(self.dataset_combo, items=dataset_names, set_item=current)
        current = self.dataset_combo.currentText()
        if current:
            reader = parent.model.get_reader_for_key(current)
            with hp.qt_signals_blocked(self.index_spinbox):
                self.index_spinbox.setMaximum(reader.n_channels - 1)

    def on_change_index(self, value: int) -> None:
        """Change index."""
        self.current_index = value
        self._update_current()

    def on_change_source(self, value: str) -> None:
        """Change source."""
        parent: OverlayChannelsDialog = self.parent()
        current = self.dataset_combo.currentText()
        if current:
            reader = parent.model.get_reader_for_key(current)
            with hp.qt_signals_blocked(self.index_spinbox):
                self.index_spinbox.setMaximum(reader.n_channels - 1)
        self.current_index = 0
        self._update_current()

    def _update_current(self):
        """Emit update event."""
        parent: OverlayChannelsDialog = self.parent()
        reader = parent.model.get_reader_for_key(self.dataset_combo.currentText())
        if reader:
            name = reader.channel_names[self.current_index]
            self.channel_label.setText(name)
        with hp.qt_signals_blocked(self.index_spinbox):
            self.index_spinbox.setValue(self.current_index)
        self.evt_update.emit((self.dataset_combo.currentText(), self.current_index))

    def on_next(self) -> None:
        """Next image."""
        self.current_index += 1
        self._update_current()

    def on_previous(self) -> None:
        """Previous image."""
        self.current_index -= 1
        self._update_current()

    def on_add_to_viewer(self) -> None:
        """Add image to viewer."""
        self.evt_add.emit((self.dataset_combo.currentText(), self.current_index))
        logger.trace("Added temporary image to the viewer.")

    def on_close(self):
        """Close event."""
        self.evt_close.emit((self.dataset_combo.currentText(), self.current_index))
        logger.trace("Removed temporary image.")
