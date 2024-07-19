"""Rename channels."""

from __future__ import annotations

from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import FilterProxyModel, QtCheckableTableView
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QDialog, QFormLayout, QWidget

from image2image.config import STATE

logger = logger.bind(src="RenameDialog")


class MixinDialog(QtFramelessTool):
    HIDE_WHEN_CLOSE = False

    changed: bool = False

    # event emitted when the popup closes
    evt_close = Signal()

    def reject(self) -> int:
        """Reject dialog."""
        if self.changed and not hp.confirm(
            self,
            "Any changes you made will not be saved.<b><b>Are you sure you wish to continue</b>?",
            "Reject changes?",
        ):
            return None
        return super().reject()


class ChannelRenameDialog(MixinDialog):
    """Dialog to enable creation of overlays."""

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("scene", "scene_index", "int", 65, sizing="fixed")
        .add("channel index", "channel_index", "int", 100, sizing="fixed")
        .add("channel name", "channel_name", "str", 200)
    )

    def __init__(
        self, parent: QWidget, reader_metadata: dict[int, dict[str, dict | list[int]]], allow_merge: bool = True
    ):
        self.allow_merge = allow_merge
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.reader_metadata = reader_metadata
        self.on_populate_table()

    def accept(self):
        """Accept dialog."""
        reader_metadata = self.reader_metadata
        # reset metadata
        for _, scene_metadata in reader_metadata.items():
            scene_metadata["keep"] = []
            scene_metadata["channel_ids"] = []
            scene_metadata["channel_names"] = []

        # iterate over all rows and update metadata
        for row in range(self.table.n_rows):
            scene_index = self.table.get_value(self.TABLE_CONFIG.scene_index, row)
            if scene_index not in reader_metadata:
                reader_metadata[scene_index] = {"keep": [], "channel_ids": [], "channel_names": []}
            reader_metadata[scene_index]["keep"].append(self.table.get_value(self.TABLE_CONFIG.check, row))
            reader_metadata[scene_index]["channel_ids"].append(
                self.table.get_value(self.TABLE_CONFIG.channel_index, row)
            )
            reader_metadata[scene_index]["channel_names"].append(
                self.table.get_value(self.TABLE_CONFIG.channel_name, row)
            )
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

    def on_merge(self) -> None:
        """Merge channels."""
        dlg = MergeDialog(self, self.reader_metadata)
        if dlg.exec_() == QDialog.DialogCode.Accepted:
            self.reader_metadata = dlg.reader_metadata

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
        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
        self.table.evt_double_clicked.connect(self.on_edit)
        self.table.evt_checked.connect(self.on_edit_state)

        if self.allow_merge:
            self.merge_btn = hp.make_btn(self, "Merge...", func=self.on_merge)

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
        layout.addRow(self._make_hide_handle("Select and rename channels")[1])
        if STATE.allow_filters:
            layout.addRow(hp.make_h_layout(self.filter_by_name, spacing=1))
        layout.addRow(self.table)
        if self.allow_merge:
            layout.addRow(hp.make_h_line_with_text("Merge channels"))
            layout.addRow(self.merge_btn)
        layout.addRow(hp.make_h_line_with_text("Rename channels"))
        layout.addRow(self.search_for)
        layout.addRow(self.replace_with)
        layout.addRow(self.replace_btn)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Accept", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
                stretch_id=(0, 1),
            )
        )
        return layout


class MergeDialog(MixinDialog):
    """Merge channels."""

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True, sizing="fixed")
        .add("channel index", "channel_index", "int", 75, sizing="fixed")
        .add("channel name", "channel_name", "str", 100, sizing="fixed")
        .add("merge name", "merge_name", "str", 200, sizing="stretch")
    )

    def __init__(self, parent: ChannelRenameDialog, reader_metadata: dict[int, dict[str, dict | list[int]]]):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.reader_metadata = reader_metadata
        self.on_populate_table()

    def on_edit_state(self, _row: int, _state: bool):
        """Edit channel name."""
        self.changed = True
        self.parent().changed = True

    def accept(self):
        """Accept dialog."""
        self.parent().changed = True
        return super().accept()

    def on_select_scene(self) -> None:
        """Select scene."""
        scene_index = int(self.scene_index.currentText())
        scene_metadata = self.reader_metadata[scene_index]

        data = []
        for channel_index, channel_name in zip(scene_metadata["channel_ids"], scene_metadata["channel_names"]):
            channel_id_to_merge = scene_metadata.get("channel_id_to_merge", {})
            merge_name = channel_id_to_merge.get(channel_index, "")
            data.append([False, channel_index, channel_name, merge_name])
        # set data in table
        self.table.reset_data()
        self.table.add_data(data)

    def on_populate_table(self) -> None:
        """Populate table."""
        scenes = [str(k) for k in self.reader_metadata]
        with hp.qt_signals_blocked(self.scene_index):
            self.scene_index.clear()
            self.scene_index.addItems(scenes)
        self.on_select_scene()

    def on_merge(self) -> None:
        """Merge channels."""
        scene_index = int(self.scene_index.currentText())
        channel_ids = self.table.get_all_checked()
        new_name = self.new_name.text()
        if not channel_ids or len(channel_ids) < 2:
            hp.warn_pretty(self, "Please select channels to merge", "No channels selected")
            return

        for channel_index in channel_ids:
            if not new_name:
                self.reader_metadata[scene_index]["channel_id_to_merge"].pop(channel_index, None)
            else:
                self.reader_metadata[scene_index]["channel_id_to_merge"][channel_index] = new_name
            self.table.set_value(self.TABLE_CONFIG.merge_name, channel_index, new_name)
        self.reader_metadata[scene_index]["merge_and_keep"] = self.merge_and_keep.isChecked()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.scene_index = hp.make_combobox(self, func=self.on_populate_table)

        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
        self.table.evt_checked.connect(self.on_edit_state)

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
        self.new_name = hp.make_line_edit(self, placeholder="New channel name...")
        self.merge_and_keep = hp.make_checkbox(
            self,
            tooltip="Keep channels that are being merged together. This will create new 'merged' channel and keep the"
            " unmerged channels as well.",
            checked=False,
        )
        self.merge_btn = hp.make_btn(self, "Create merged channel", func=self.on_merge)

        layout = hp.make_form_layout(self)
        hp.style_form_layout(layout)
        layout.addRow(self._make_hide_handle("Select and merge channels")[1])
        layout.addRow(hp.make_label(self, "Scene index"), self.scene_index)
        if STATE.allow_filters:
            layout.addRow(hp.make_h_layout(self.filter_by_name, spacing=1))
        layout.addRow(self.table)
        layout.addRow(self.new_name)
        layout.addRow("Keep merged channels", self.merge_and_keep)
        layout.addRow(self.merge_btn)
        layout.addRow(hp.make_h_line())
        link = hp.hyper(
            "https://en.wikipedia.org/wiki/Maximum_intensity_projection", "maximum-intensity projection", ""
        )
        layout.addRow(
            hp.make_label(
                self,
                f"<b>Note.</b> Multiple channels will be merged by performing {link}.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Accept", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
                stretch_id=(0, 1),
            )
        )
        return layout

    def reject(self) -> int:
        """Reject dialog."""
        if self.changed and not hp.confirm(
            self,
            "Any changes you made will not be saved.<b><b>Are you sure you wish to continue</b>?",
            "Reject changes?",
        ):
            return None
        return super().reject()
