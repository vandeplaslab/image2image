"""Iterate dialog."""

from __future__ import annotations

import typing as ty

from loguru import logger
from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtCore import Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout

if ty.TYPE_CHECKING:
    from image2image.qt._dialogs._list import QtDatasetItem


class IterateDialog(QtFramelessTool):
    """Iterate dialog."""

    evt_iter_remove = Signal(str, int)
    evt_iter_next = Signal(str, int)
    evt_iter_add = Signal(str, int)

    current_index: int = 0

    def __init__(self, parent: QtDatasetItem):
        self.key = parent.key
        super().__init__(parent)
        self._update_source()
        self.evt_close.connect(self.on_close)
        hp.call_later(self, self._update_current, 200)

    # noinspection PyTestUnpassedFixture,PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Setup UI."""
        _, header_layout = self._make_close_handle(title="Datasets")

        self.index_spinbox = hp.make_int_spin_box(self, min=0, max=1_000, func=self.on_change_index)
        self.channel_combo = hp.make_searchable_combobox(self, func=self.on_change_channel)

        layout = hp.make_form_layout(margin=6)
        layout.addRow(header_layout)
        layout.addRow(self.channel_combo)
        layout.addRow(self.index_spinbox)
        layout.addRow(hp.make_btn(self, "Add to viewer", func=self.on_add_to_viewer, tooltip="Add image to viewer."))
        return layout

    def on_change_index(self, value: int) -> None:
        """Change index."""
        self.current_index = value
        self._update_current()

    def on_change_channel(self) -> None:
        """Change channel."""
        self.current_index = self.channel_combo.currentIndex()
        self._update_current()

    def _update_source(self) -> None:
        parent: QtDatasetItem = self.parent()
        reader = parent.get_model()
        if reader:
            with hp.qt_signals_blocked(self.index_spinbox):
                self.index_spinbox.setMaximum(reader.n_channels - 1)
            hp.combobox_setter(self.channel_combo, items=reader.channel_names, clear=True)

    def _update_current(self) -> None:
        """Emit update event."""
        with hp.qt_signals_blocked(self.index_spinbox):
            self.index_spinbox.setValue(self.current_index)
        with hp.qt_signals_blocked(self.channel_combo):
            self.channel_combo.setCurrentIndex(self.current_index)
        self.evt_iter_next.emit(self.key, self.current_index)

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
        self.evt_iter_add.emit(self.key, self.current_index)
        logger.trace("Added temporary image to the viewer.")

    def on_close(self) -> None:
        """Close event."""
        self.evt_iter_remove.emit(self.key, self.current_index)
