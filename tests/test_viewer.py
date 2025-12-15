"""Elastix app"""

from image2image.qt._viewer._mask import MasksDialog
from image2image.qt.dialog_viewer import ImageViewerWindow


def test_masks(qtbot) -> None:
    """Test ImageViewerWindow."""
    window = ImageViewerWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageViewerWindow), "Window is not an instance of ImageViewerWindow."

    widget = MasksDialog(window)
    qtbot.addWidget(widget)
    assert isinstance(widget, MasksDialog), "Window is not an instance of MasksDialog."
