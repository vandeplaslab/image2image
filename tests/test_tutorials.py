import pytest
from qtextra.widgets.qt_tutorial import QtTutorial

from image2image.qt._dialogs._tutorial import (
    show_convert_tutorial,
    show_crop_tutorial,
    show_elastix_tutorial,
    show_fusion_tutorial,
    show_merge_tutorial,
    show_register_tutorial,
    show_valis_tutorial,
    show_viewer_tutorial,
)
from image2image.qt.dialog_convert import ImageConvertWindow
from image2image.qt.dialog_crop import ImageCropWindow
from image2image.qt.dialog_elastix import ImageElastixWindow
from image2image.qt.dialog_fusion import ImageFusionWindow
from image2image.qt.dialog_merge import ImageMergeWindow
from image2image.qt.dialog_register import ImageRegistrationWindow
from image2image.qt.dialog_valis import ImageValisWindow
from image2image.qt.dialog_viewer import ImageViewerWindow


@pytest.mark.parametrize(
    "window_class,tutorial_function",
    [
        (ImageConvertWindow, show_convert_tutorial),
        (ImageCropWindow, show_crop_tutorial),
        (ImageElastixWindow, show_elastix_tutorial),
        (ImageFusionWindow, show_fusion_tutorial),
        (ImageMergeWindow, show_merge_tutorial),
        (ImageValisWindow, show_valis_tutorial),
        (ImageRegistrationWindow, show_register_tutorial),
        (ImageViewerWindow, show_viewer_tutorial),
    ],
)
def test_tutorial(qtbot, monkeypatch, window_class, tutorial_function) -> None:
    window = window_class(None)
    qtbot.addWidget(window)
    assert isinstance(window, window_class), f"Window is not an instance of {window_class}."

    def _mock_show(self, *args, **kwargs):
        qtbot.addWidget(self)

    monkeypatch.setattr(QtTutorial, "show", _mock_show)
    tutorial_function(window)
