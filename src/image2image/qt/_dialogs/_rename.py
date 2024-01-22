"""Rename channels."""
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import FilterProxyModel, QtCheckableTableView
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout, QWidget

from image2image.config import STATE

logger = logger.bind(src="RenameDialog")


class ChannelRenameDialog(QtFramelessTool):
    """Dialog to enable creation of overlays."""

    HIDE_WHEN_CLOSE = True

    # event emitted when the popup closes
    evt_close = Signal()

    changed: bool = False

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("scene", "scene_index", "int", 65, sizing="fixed")
        .add("channel index", "channel_index", "int", 100, sizing="fixed")
        .add("channel name", "channel_name", "str", 200)
    )

    def __init__(self, parent: QWidget, reader_metadata: dict[int, dict[str, list[int]]]):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.reader_metadata = reader_metadata
        self.on_populate_table()

    def accept(self):
        """Accept dialog."""
        reader_metadata = {}
        for row in range(self.table.n_rows):
            scene_index = self.table.get_value(self.TABLE_CONFIG.scene_index, row)
            if scene_index not in reader_metadata:
                reader_metadata[scene_index] = {"keep": [], "channel_ids": [], "channel_names": []}
            keep = self.table.get_value(self.TABLE_CONFIG.check, row)
            channel_index = self.table.get_value(self.TABLE_CONFIG.channel_index, row)
            channel_name = self.table.get_value(self.TABLE_CONFIG.channel_name, row)
            reader_metadata[scene_index]["keep"].append(keep)
            reader_metadata[scene_index]["channel_ids"].append(channel_index)
            reader_metadata[scene_index]["channel_names"].append(channel_name)
        self.reader_metadata = reader_metadata
        return super().accept()

    def on_replace(self) -> None:
        """Replace channel names."""
        search_for = self.search_for.text()
        replace_with = self.replace_with.text()
        if not search_for or not replace_with:
            return
        for row in range(self.table.n_rows):
            channel_name = self.table.get_value(self.TABLE_CONFIG.channel_name, row)
            new_channel_name = channel_name.replace(search_for, replace_with)
            self.table.set_value(self.TABLE_CONFIG.channel_name, row, new_channel_name)

    def on_edit(self, row: int):
        """Edit channel name."""
        old_value = self.table.get_value(self.TABLE_CONFIG.channel_name, row)
        new_value = hp.get_text(self, "Edit channel name", "Edit channel name", old_value)
        if new_value is None or new_value == old_value:
            return
        self.table.set_value(self.TABLE_CONFIG.channel_name, row, new_value)
        self.changed = True

    def on_edit_state(self, _row: int, _state: bool):
        """Edit channel name."""
        self.changed = True

    def on_populate_table(self):
        """Populate table."""
        data = []
        for scene_index, scene_metadata in self.reader_metadata.items():
            for keep, channel_index, channel_name in zip(
                scene_metadata["keep"], scene_metadata["channel_ids"], scene_metadata["channel_names"]
            ):
                data.append([keep, scene_index, channel_index, channel_name])
        # set data in table
        self.table.reset_data()
        self.table.add_data(data)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Select and rename channels")

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
        self.table.evt_double_clicked.connect(self.on_edit)
        self.table.evt_checked.connect(self.on_edit_state)

        self.search_for = hp.make_line_edit(self, placeholder="Search for...")
        self.replace_with = hp.make_line_edit(self, placeholder="Replace with...")
        self.replace_btn = hp.make_btn(self, "Replace", func=self.on_replace)
        if STATE.allow_filters:
            self.table_proxy = FilterProxyModel(self)
            self.table_proxy.setSourceModel(self.table.model())
            self.table.model().table_proxy = self.table_proxy
            self.table.setModel(self.table_proxy)
            self.filter_by_name = hp.make_line_edit(
                self,
                placeholder="Filter by...",
                func_changed=lambda text, col=self.TABLE_CONFIG.channel_name: self.table_proxy.setFilterByColumn(
                    text, col
                ),
            )

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
        layout.addRow(header_layout)
        if STATE.allow_filters:
            layout.addRow(hp.make_h_layout(self.filter_by_name, spacing=1))
        layout.addRow(self.table)
        layout.addRow(hp.make_h_line_with_text("Rename channels"))
        layout.addRow(self.search_for)
        layout.addRow(self.replace_with)
        layout.addRow(self.replace_btn)
        layout.addRow(hp.make_h_line())
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Accept", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.on_reject),
                stretch_id=(0, 1),
            )
        )

        return layout

    def on_reject(self):
        """Reject dialog."""
        if self.changed:
            if not hp.confirm(
                self, "Any changes you made will not be saved - are you sure you wish to continue?", "Reject changes?"
            ):
                return None
        return super().reject()
