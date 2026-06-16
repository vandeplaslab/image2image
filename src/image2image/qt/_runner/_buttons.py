"""Button class."""

from __future__ import annotations

import typing as ty

import qtextra.helpers as hp
from qtpy.QtWidgets import QWidget

if ty.TYPE_CHECKING:
    from qtextra.widgets.qt_toggle_group import QtQtaToggleGroup


def make_toggle_group(widget: QWidget, func: ty.Callable) -> QtQtaToggleGroup:
    """Make toggle group."""
    return hp.make_icon_toggle_group(
        widget, "thumbs_up", "thumbs_down", tooltip="Decide whether the preview is looking good or bad.", func=func
    )
