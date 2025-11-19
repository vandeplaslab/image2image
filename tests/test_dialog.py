"""Test various dialog windows."""

from image2image.qt._dialogs._about import AboutDialog
from image2image.qt._dialogs._errors import ErrorsDialog
from image2image.qt._dialogs._shortcuts import RegisterShortcutsDialog, WsiPrepShortcutsDialog


def test_about(qtbot) -> None:
    """Test ImageViewerWindow."""
    widget = AboutDialog(None)
    qtbot.addWidget(widget)
    assert isinstance(widget, AboutDialog), "Window is not an instance of AboutDialog."


def test_errors(qtbot) -> None:
    """Test ImageViewerWindow."""
    widget = ErrorsDialog(None, "Error message")
    qtbot.addWidget(widget)
    assert isinstance(widget, ErrorsDialog), "Window is not an instance of ErrorsDialog."

    widget = ErrorsDialog(None, ["Error message"])
    qtbot.addWidget(widget)
    assert isinstance(widget, ErrorsDialog), "Window is not an instance of ErrorsDialog."


def test_shortcuts(qtbot) -> None:
    """Test ImageViewerWindow."""
    widget = RegisterShortcutsDialog(None)
    qtbot.addWidget(widget)
    assert isinstance(widget, RegisterShortcutsDialog), "Window is not an instance of RegisterShortcutsDialog."

    widget = WsiPrepShortcutsDialog(None)
    qtbot.addWidget(widget)
    assert isinstance(widget, WsiPrepShortcutsDialog), "Window is not an instance of WsiPrepShortcutsDialog."
