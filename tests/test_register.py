"""Register app."""

from image2image.qt._register._fiducials import FiducialsDialog
from image2image.qt._register._guess import GuessDialog
from image2image.qt._register._preprocess import PreprocessMovingDialog
from image2image.qt._register._select import ImportSelectDialog
from image2image.qt.dialog_register import ImageRegistrationWindow


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
