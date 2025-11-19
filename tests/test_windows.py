"""Viewer test utilities."""

from image2image.qt.dialog_convert import ImageConvertWindow
from image2image.qt.dialog_crop import ImageCropWindow
from image2image.qt.dialog_elastix import ImageElastixWindow
from image2image.qt.dialog_elastix3d import ImageElastix3dWindow
from image2image.qt.dialog_fusion import ImageFusionWindow
from image2image.qt.dialog_merge import ImageMergeWindow
from image2image.qt.dialog_register import ImageRegistrationWindow
from image2image.qt.dialog_valis import ImageValisWindow
from image2image.qt.dialog_viewer import ImageViewerWindow
from image2image.qt.launcher import Launcher


def test_window_launcher(qtbot) -> None:
    """Test Launcher."""
    window = Launcher(None)
    qtbot.addWidget(window)
    assert isinstance(window, Launcher), "Window is not an instance of Launcher."


def test_window_window(qtbot) -> None:
    """Test ImageViewerWindow."""
    window = ImageViewerWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageViewerWindow), "Window is not an instance of ImageViewerWindow."


def test_window_crop(qtbot) -> None:
    """Test ImageCropWindow."""
    window = ImageCropWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageCropWindow), "Window is not an instance of ImageCropWindow."


def test_window_register(qtbot) -> None:
    """Test ImageRegistrationWindow."""
    window = ImageRegistrationWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageRegistrationWindow), "Window is not an instance of ImageRegistrationWindow."


def test_window_elastix(qtbot) -> None:
    """Test ImageElastixWindow."""
    window = ImageElastixWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageElastixWindow), "Window is not an instance of ImageElastixWindow."


def test_window_valis(qtbot) -> None:
    """Test ImageValisWindow."""
    window = ImageValisWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageValisWindow), "Window is not an instance of ImageValisWindow."


def test_window_elastix3d(qtbot) -> None:
    """Test ImageElastix3dWindow."""
    window = ImageElastix3dWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageElastix3dWindow), "Window is not an instance of ImageElastix3dWindow."


def test_window_merge(qtbot) -> None:
    """Test ImageMergeWindow."""
    window = ImageMergeWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageMergeWindow), "Window is not an instance of ImageMergeWindow."


def test_window_fusion(qtbot) -> None:
    """Test ImageFusionWindow."""
    window = ImageFusionWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageFusionWindow), "Window is not an instance of ImageFusionWindow."


def test_window_convert(qtbot) -> None:
    """Test ImageConvertWindow."""
    window = ImageConvertWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageConvertWindow), "Window is not an instance of ImageConvertWindow."
