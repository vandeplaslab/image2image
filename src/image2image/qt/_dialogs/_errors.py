"""Errors dialog."""

from __future__ import annotations

import qtextra.helpers as hp
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtWidgets import QWidget


class ErrorsDialog(QtFramelessTool):
    """Show errors dialog.."""

    TITLE = "Errors"

    def __init__(self, parent: QWidget, errors: str | list[str]):
        super().__init__(parent)
        if isinstance(errors, list):
            errors = "\n".join(errors)
        self.errors.setText(errors)
        # sh = self.errors.sizeHint()
        # width = min(sh.width() + 250, 600)
        # height = min(sh.height() + 100, 400)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

    # noinspection PyAttributeOutsideInit
    def make_panel(self):
        """Make panel."""
        self.errors = hp.make_scrollable_label(self, "", object_name="errors", wrap=True)
        layout = hp.make_v_layout(
            self._make_close_handle(self.TITLE)[1],
            self.errors,
        )
        layout.setContentsMargins(6, 6, 6, 6)
        return layout
