"""Elastix app"""

from image2image.qt._wsi._network import NetworkViewer
from image2image.qt.dialog_elastix import ImageElastixWindow


def test_network_viewer(qtbot) -> None:
    """Test ImageElastixWindow."""
    window = ImageElastixWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageElastixWindow), "Window is not an instance of ImageElastixWindow."

    widget = NetworkViewer(window)
    qtbot.addWidget(widget)
    assert isinstance(widget, NetworkViewer), "Window is not an instance of NetworkViewer."
    widget.on_plot()
