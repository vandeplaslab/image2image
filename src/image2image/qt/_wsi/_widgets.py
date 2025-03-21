"""Label widget."""

from __future__ import annotations

import typing as ty

from qtextra.widgets.qt_label_icon import QtQtaLabel


class QtModalityLabel(QtQtaLabel):
    """Modality label."""

    STATES = ("image", "geojson", "points", "shapes")

    def __init__(self, *args: ty.Any, color: str | None = None, **kwargs: ty.Any):
        super().__init__(*args, **kwargs)
        if color:
            self._icon_color = color
        self._state: str = "image"
        self.state = "image"
        self.set_normal()

    @property
    def state(self) -> str:
        """Get state."""
        return self._state

    @state.setter
    def state(self, state: str) -> None:
        self._state = state
        self.set_qta(state)
        self.setToolTip(f"Modality: {state}")
