"""Asset and stylesheet tests."""

from __future__ import annotations

import pytest
from napari._qt.widgets.qt_mode_buttons import QtModePushButton
from napari.layers import Points
from qtextra.config import THEMES
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

import image2image.assets  # noqa: F401


@pytest.mark.xfail("flaky")
def test_napari_mode_push_buttons_receive_qt_icons(qtbot) -> None:
    """Test that napari push-button mode icons are assigned by the app stylesheet."""
    parent = QWidget()
    qtbot.addWidget(parent)

    button = QtModePushButton(Points([[0, 0]], name="points"), "delete_shape")
    layout = QVBoxLayout(parent)
    layout.addWidget(button)

    parent.setStyleSheet(THEMES.get_theme_stylesheet("light"))
    QApplication.processEvents()

    assert not button.icon().isNull(), "The napari delete button should have a Qt icon."
    assert button.iconSize().width() == 18, "The napari delete button icon should use the app icon size."
