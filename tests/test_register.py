"""Register app."""

from types import SimpleNamespace

from napari.layers.base import ActionType
from pytest import MonkeyPatch
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QKeyEvent

from image2image.qt._register._fiducials import FiducialsDialog
from image2image.qt._register._guess import GuessDialog
from image2image.qt._register._preprocess import PreprocessMovingDialog
from image2image.qt._register._select import ImportSelectDialog
from image2image.qt.dialog_register import (
    ImageRegistrationPlugin,
    ImageRegistrationWindow,
    _is_changing_points_data_event,
)


class CanvasKeyEvent:
    """Minimal napari-style event wrapper with a native Qt key event."""

    def __init__(self, native: QKeyEvent) -> None:
        self.native = native


class DummyFiducialsDialog:
    """Minimal fiducials dialog for on_run tests."""

    def __init__(self) -> None:
        self.load_count = 0

    def on_load(self) -> None:
        """Track fiducials table refreshes."""
        self.load_count += 1


def make_register_plugin_for_key_tests() -> ImageRegistrationPlugin:
    """Return a minimal plugin instance for shortcut handling tests."""
    plugin = ImageRegistrationPlugin.__new__(ImageRegistrationPlugin)
    plugin._last_canvas_shortcut = None
    return plugin


def test_fiducials(qtbot) -> None:
    """Test FiducialsDialog."""
    window = ImageRegistrationWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageRegistrationWindow), "Window is not an instance of ImageRegistrationWindow."

    widget = FiducialsDialog(window)
    qtbot.addWidget(widget)
    assert isinstance(widget, FiducialsDialog), "Window is not an instance of FiducialsDialog."


def test_preprocess(qtbot) -> None:
    """Test PreprocessMovingDialog."""
    window = ImageRegistrationWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageRegistrationWindow), "Window is not an instance of ImageRegistrationWindow."

    widget = PreprocessMovingDialog(window)
    qtbot.addWidget(widget)
    assert isinstance(widget, PreprocessMovingDialog), "Window is not an instance of PreprocessMovingDialog."
    assert widget.initial_model.rotate == 0, "Initial rotation should be 0."
    assert widget.initial_model.flip_lr is False, "Initial flip_lr should be False."

    # flip left-right
    widget.on_flip_lr()
    assert widget.initial_model.flip_lr is True, "Initial flip_lr should be True."

    # rotate right
    widget.on_rotate("right")
    assert widget.initial_model.rotate == 15, "Rotation should be 15 after rotating right."
    # rotate left
    widget.on_rotate("left")
    assert widget.initial_model.rotate == 0, "Rotation should be 0 after rotating left."
    widget.on_rotate("left")
    assert widget.initial_model.rotate == -15, "Rotation should be 345 after rotating left again."

    # set rotate to 350 and rotate right
    widget.on_set_rotate(300)
    assert widget.initial_model.rotate == 300, "Rotation should be 300 after setting it."

    # reset transform
    widget.on_reset()
    assert widget.initial_model.rotate == 0, "Initial rotation should be 0."
    assert widget.initial_model.flip_lr is False, "Initial flip_lr should be False."


def test_guess(qtbot) -> None:
    """Test GuessDialog."""
    window = ImageRegistrationWindow(None)
    qtbot.addWidget(window)
    assert isinstance(window, ImageRegistrationWindow), "Window is not an instance of ImageRegistrationWindow."

    widget = GuessDialog(window)
    qtbot.addWidget(widget)
    assert isinstance(widget, GuessDialog), "Window is not an instance of GuessDialog."


def test_select(qtbot) -> None:
    """Test GuessDialog."""
    widget = ImportSelectDialog(None)
    qtbot.addWidget(widget)
    assert isinstance(widget, ImportSelectDialog), "Window is not an instance of ImportSelectDialog."
    assert widget.fixed_image_check.isVisibleTo(widget), "Fixed image check should be visible."
    assert widget.moving_image_check.isVisibleTo(widget), "Moving image check should be visible."
    assert widget.fixed_fiducial_check.isVisibleTo(widget), "Fixed fiducial check should be visible."
    assert widget.moving_fiducial_check.isVisibleTo(widget), "Moving fiducial check should be visible."
    config = widget.get_config()
    assert isinstance(config, dict), "Config is not an instance of config."
    assert len(config) == 4, "Config does not have 4 entries."

    widget = ImportSelectDialog(None, disable=("fixed_image", "moving_image", "fixed_points", "moving_points"))
    qtbot.addWidget(widget)
    assert isinstance(widget, ImportSelectDialog), "Window is not an instance of ImportSelectDialog."
    assert not widget.fixed_image_check.isVisibleTo(widget), "Fixed image check should be visible."
    assert not widget.moving_image_check.isVisibleTo(widget), "Moving image check should be visible."
    assert not widget.fixed_fiducial_check.isVisibleTo(widget), "Fixed fiducial check should be visible."
    assert not widget.moving_fiducial_check.isVisibleTo(widget), "Moving fiducial check should be visible."
    config = widget.get_config()
    assert len(config) == 4, "Config should be empty."
    assert all(value is False for value in config.values()), "All config values should be False."


def test_register_points_data_changing_event_is_deferred(monkeypatch: MonkeyPatch) -> None:
    """Test that in-progress point edits are deferred before recomputing."""
    plugin = ImageRegistrationPlugin.__new__(ImageRegistrationPlugin)
    fiducials_dlg = DummyFiducialsDialog()
    run_count = 0

    def on_run() -> None:
        nonlocal run_count
        run_count += 1

    plugin._fiducials_dlg = fiducials_dlg
    monkeypatch.setattr(plugin, "_on_run", on_run)

    changing_event = SimpleNamespace(action=ActionType.CHANGING)
    changed_event = SimpleNamespace(action=ActionType.CHANGED)
    on_run_without_decorators = ImageRegistrationPlugin.on_run.__wrapped__.__wrapped__

    assert _is_changing_points_data_event(changing_event) is True, "Changing point edits should be deferred."
    assert _is_changing_points_data_event(changed_event) is False, "Completed point edits should be accepted."
    assert _is_changing_points_data_event(None) is False, "Manual recompute calls should be accepted."
    on_run_without_decorators(plugin, changing_event)
    assert run_count == 0, "Changing point edits should not recompute the transform."
    assert fiducials_dlg.load_count == 0, "Changing point edits should not refresh the fiducials table."

    on_run_without_decorators(plugin, changed_event)
    assert run_count == 1, "Completed point edits should recompute the transform."
    assert fiducials_dlg.load_count == 1, "Completed point edits should refresh the fiducials table."

    on_run_without_decorators(plugin)
    assert run_count == 2, "Manual recompute calls should still recompute the transform."
    assert fiducials_dlg.load_count == 2, "Manual recompute calls should refresh the fiducials table."


def test_register_shortcut_is_not_double_fired(monkeypatch: MonkeyPatch) -> None:
    """Test that canvas and Qt delivery of one key only run one shortcut."""
    plugin = make_register_plugin_for_key_tests()
    modes: list[tuple[str, object]] = []

    def on_toggle_mode(which: str, mode: object) -> None:
        modes.append((which, mode))

    monkeypatch.setattr(plugin, "on_toggle_mode", on_toggle_mode)
    event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier)

    plugin._on_canvas_key_press(CanvasKeyEvent(event))
    plugin.keyPressEvent(event)

    assert len(modes) == 1, "Shortcut should only be handled once."
    assert event.isAccepted(), "Handled shortcut event should be accepted."


def test_register_unhandled_key_is_not_marked_handled() -> None:
    """Test that unknown keys are reported as unhandled."""
    plugin = make_register_plugin_for_key_tests()

    assert plugin._handle_key_press(Qt.Key.Key_B) is False, "Unknown shortcut should be unhandled."
