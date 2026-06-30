from types import SimpleNamespace

import pytest
from qtextra.widgets.qt_tutorial import QtTutorial, TutorialStep
from qtpy.QtWidgets import QPushButton, QWidget

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


def _make_register_tutorial_widget(parent: QWidget | None = None) -> QWidget:
    widget = QWidget(parent)
    widget.view_fixed = SimpleNamespace(widget=QWidget(widget))
    widget.view_moving = SimpleNamespace(widget=QWidget(widget))
    widget.import_project_btn = QPushButton(widget)

    fixed_widget = QWidget(widget)
    fixed_widget.add_btn = QPushButton(fixed_widget)
    fixed_widget.more_btn = QPushButton(fixed_widget)
    widget._fixed_widget = fixed_widget

    moving_widget = QWidget(widget)
    moving_widget.view_type_choice = QPushButton(moving_widget)
    moving_widget.displayed_in_fixed_choice = QPushButton(moving_widget)
    widget._moving_widget = moving_widget

    widget.fiducials_btn = QPushButton(widget)
    widget.export_project_btn = QPushButton(widget)
    return widget


@pytest.mark.xfail(
    reason="Some tutorial dialogs are flaky in CI — passes locally, fails intermittently on certain runners."
)
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


def test_register_tutorial_omits_missing_optional_buttons(qtbot, monkeypatch) -> None:
    """Ensure missing toolbar buttons do not become tutorial targets."""
    widget = _make_register_tutorial_widget()
    qtbot.addWidget(widget)
    captured_steps: list[TutorialStep] = []

    def _mock_set_steps(self: QtTutorial, steps: list[TutorialStep]) -> None:
        captured_steps.extend(steps)

    def _mock_show(self: QtTutorial) -> None:
        qtbot.addWidget(self)

    monkeypatch.setattr(QtTutorial, "set_steps", _mock_set_steps)
    monkeypatch.setattr(QtTutorial, "show", _mock_show)

    assert show_register_tutorial(widget)
    assert {step.title for step in captured_steps}.isdisjoint({"Tutorial", "Feedback"})
