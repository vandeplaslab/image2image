"""Rename channels."""

from __future__ import annotations

from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view_check import MultiColumnSingleValueProxyModel, QtCheckableTableView
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

    merge_info = None

    def __init__(
        self, parent: QWidget, scene_index: int, scene_metadata: dict[str, dict | list[int]], allow_merge: bool = True
    ):
        self.allow_merge = allow_merge
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.scene_index = scene_index
        self.scene_metadata = scene_metadata
        self.on_populate_table()

    def accept(self) -> int:
        """Accept dialog."""
        scene_metadata = self.scene_metadata
        # reset metadata
        scene_metadata["keep"] = []
        scene_metadata["channel_ids"] = []
        scene_metadata["channel_names"] = []

        # iterate over all rows and update metadata
        for row in range(self.table.n_rows):
            scene_metadata["keep"].append(self.table.get_value(self.TABLE_CONFIG.check, row))
            scene_metadata["channel_ids"].append(self.table.get_value(self.TABLE_CONFIG.channel_index, row))
            scene_metadata["channel_names"].append(self.table.get_value(self.TABLE_CONFIG.channel_name, row))
        self.scene_metadata = scene_metadata
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
        dlg = MergeDialog(self, self.scene_index, self.scene_metadata)
        if dlg.exec_() == QDialog.DialogCode.Accepted:
            self.scene_metadata = dlg.scene_metadata
            self.on_populate_table()

    def on_edit(self, row: int) -> None:
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

    def on_populate_table(self) -> None:
        """Populate table."""
        scene_metadata = self.scene_metadata
        data = []
        # for scene_index, scene_metadata in self.reader_metadata.items():
        for keep, channel_index, channel_name in zip(
            scene_metadata["keep"], scene_metadata["channel_ids"], scene_metadata["channel_names"]
        ):
            data.append([keep, self.scene_index, channel_index, channel_name])
        # set data in table
        self.table.reset_data()
        self.table.add_data(data)
        self.on_populate_merge()

    def on_populate_merge(self) -> None:
        """Populate merge."""
        if self.merge_info is not None or "channel_id_to_merge" in self.scene_metadata:
            n = len(list(set(self.scene_metadata["channel_id_to_merge"].values())))
            self.merge_info.setText(f"+ {n} merged channel(s)")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True)
        hp.set_font(self.table)
        self.table.setup_model_from_config(self.TABLE_CONFIG)
        self.table.evt_double_clicked.connect(self.on_edit)
        self.table.evt_checked.connect(self.on_edit_state)

        self.search_for = hp.make_line_edit(self, placeholder="Search for...")
        self.replace_with = hp.make_line_edit(self, placeholder="Replace with...")
        self.replace_btn = hp.make_btn(self, "Replace", func=self.on_replace)
        if STATE.allow_filters:
            self.table_proxy = MultiColumnSingleValueProxyModel(self)
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

        layout = hp.make_form_layout(parent=self)
        layout.addRow(self._make_hide_handle("Select and rename channels")[1])
        if STATE.allow_filters:
            layout.addRow(hp.make_h_layout(self.filter_by_name, spacing=1))
        layout.addRow(self.table)
        if self.allow_merge:
            self.merge_btn = hp.make_btn(self, "Merge...", func=self.on_merge)
            self.merge_info = hp.make_label(self, "")

            layout.addRow(hp.make_h_line_with_text("Merge channels"))
            layout.addRow(self.merge_btn)
            layout.addRow(self.merge_info)
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

    def __init__(self, parent: ChannelRenameDialog, scene_index: int, scene_metadata: dict[str, dict | list[int]]):
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.scene_index = scene_index
        self.scene_metadata = scene_metadata
        self.on_populate_table()

    def on_edit_state(self, _row: int, _state: bool) -> None:
        """Edit channel name."""
        self.changed = True
        self.parent().changed = True

    def accept(self) -> int:
        """Accept dialog."""
        self.parent().changed = True
        return super().accept()  # type: ignore[no-any-return]

    def on_populate_table(self) -> None:
        """Populate table."""
        scene_metadata = self.scene_metadata

        data = []
        for channel_index, channel_name in zip(scene_metadata["channel_ids"], scene_metadata["channel_names"]):
            channel_id_to_merge = scene_metadata.get("channel_id_to_merge", {})
            merge_name = channel_id_to_merge.get(channel_index, "")
            data.append([False, channel_index, channel_name, merge_name])
        # set data in table
        self.table.reset_data()
        self.table.add_data(data)

    def on_merge(self) -> None:
        """Merge channels."""
        channel_ids = self.table.get_all_checked()
        new_name = self.new_name.text()
        if not channel_ids or len(channel_ids) < 2:
            hp.warn_pretty(self, "Please select channels to merge", "No channels selected")
            return

        merge_and_keep = self.merge_and_keep.isChecked()
        for channel_index in channel_ids:
            if not new_name:
                self.scene_metadata["channel_id_to_merge"].pop(channel_index, None)
            else:
                self.scene_metadata["channel_id_to_merge"][channel_index] = new_name
            # update dictionary
            if not merge_and_keep:
                self.scene_metadata["keep"][channel_index] = False

            # update table
            self.table.set_value(self.TABLE_CONFIG.merge_name, channel_index, new_name)
        self.scene_metadata["merge_and_keep"] = merge_and_keep

    def on_update_new_name(self, _: str | None = None) -> None:
        """Update new name."""
        value = self.new_name.text()
        hp.set_object_name(self.new_name, object_name="success" if value else "warning")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
        self.table.evt_checked.connect(self.on_edit_state)

        if STATE.allow_filters:
            self.table_proxy = MultiColumnSingleValueProxyModel(self)
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
        self.new_name = hp.make_line_edit(
            self, placeholder="New channel name...", object_name="warning", func_changed=self.on_update_new_name
        )
        self.merge_and_keep = hp.make_checkbox(
            self,
            tooltip="Keep channels that are being merged together. This will create new 'merged' channel and keep the"
            " unmerged channels as well.",
            checked=False,
        )
        self.merge_btn = hp.make_btn(self, "Create merged channel", func=self.on_merge)

        layout = hp.make_form_layout(parent=self)
        layout.addRow(self._make_hide_handle("Select and merge channels")[1])
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
