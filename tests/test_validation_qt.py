"""Qt controls, native entry points, and asynchronous result ownership."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from image2image.cli import cli
from image2image.qt.dialog_validate import ImageValidationWindow
from image2image.utils.validation import analyze_validation_session, inspect_validation_image, save_validation_session
from test_validation import make_validation_session


@pytest.fixture
def validation_window(qtbot: QtBot) -> ImageValidationWindow:
    window = ImageValidationWindow(run_check_version=False)

    def before_close(widget: ImageValidationWindow) -> None:
        widget._dirty = False
        widget._cancel.set()
        qtbot.waitUntil(lambda: not widget._workers, timeout=15000)

    qtbot.addWidget(window, before_close_func=before_close)
    return window


def test_controls_annotations_and_zoom(validation_window: ImageValidationWindow, tmp_path: Path) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(session.candidates[0].path)
    assert [window.tabs.tabText(i) for i in range(4)] == ["Inputs", "Landmarks", "Masks", "Results"]
    window.begin_pair()
    window._pair_click(0, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 24, 20)))
    window._pair_click(1, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 28, 23)))
    assert window.session.landmarks[0].fixed == (20, 24)
    assert window.session.landmarks[0].moving == (23, 28)
    np.testing.assert_array_equal(window.point_layers[0].data, [[24, 20]])
    window.landmark_table.item(0, 2).setText("21")
    assert window.session.landmarks[0].fixed == (21, 24)
    window.landmark_table.item(0, 0).setCheckState(Qt.CheckState.Unchecked)
    assert not window.session.landmarks[0].enabled
    window.point_layers[1].data = np.array([[30, 25]])
    assert window.session.landmarks[0].moving == (25, 30)
    window.views[0].viewer.camera.zoom = 3
    assert window.views[1].viewer.camera.zoom == 3
    window.landmark_table.selectRow(0)
    window.delete_pair()
    assert window.session.landmarks == []
    assert window.point_layers[0].data.shape == (0, 2)


def test_analysis_views_and_invalidation(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(save_validation_session(tmp_path / "session", session))
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis is not None, window.status.text()
    assert window.analysis.candidates[0].validation["rms"] == 5
    assert window.export_btn.isEnabled()
    for index in range(window.result_mode.count()):
        window.result_mode.setCurrentIndex(index)
    old_directory = window._result_directory.name
    window.downsample.setValue(2)
    assert window.analysis is None
    assert not window.export_btn.isEnabled()
    assert not Path(old_directory).exists()
    assert not window.result_view.viewer.layers


def test_stale_results_and_cancel(validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(save_validation_session(tmp_path / "session", session))
    with TemporaryDirectory() as directory:
        result = analyze_validation_session(session, directory)
        with TemporaryDirectory() as stale:
            # Input revisions must reject successful but obsolete worker output.
            window.accept_analysis(window._generation - 1, stale, result)
        assert window.analysis is None
    window.start_analysis()
    window.cancel_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis is None
    assert not window.export_btn.isEnabled()


def test_mask_controls_and_failed_worker(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(session.candidates[0].path)
    for index, name in enumerate(("fixed", "moving")):
        path = tmp_path / f"{name}-mask.tiff"
        tifffile.imwrite(path, np.ones((32, 40), dtype=np.uint8) * (index + 1))
        window.set_mask(index, inspect_validation_image(path, (2, 2)))
    assert window.session.mask_pairs == []
    window.add_mask_pair()
    window.mask_table.item(0, 2).setText("2")
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis.candidates[0].masks[0]["dice"] == 1
    assert window.analysis.candidates[0].masks[0]["joint_coverage"] == 1
    window.clear_masks()
    assert window.analysis is None
    assert window.session.fixed_mask is None and window.session.moving_mask is None
    Path(session.candidates[0].path).unlink()
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis is None
    assert "Registration missing" in window.status.text()
    assert not window.export_btn.isEnabled()


def test_cli_validate_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "session.i2validate.json"
    path.write_text("{}")
    calls = []
    monkeypatch.setattr("image2image.cli._cli_setup", lambda *args, **kwargs: (10, False))
    monkeypatch.setattr("image2image.main.run", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(cli, ["-t", "validate", "-p", str(path)])
    assert result.exit_code == 0, result.output
    assert calls[0]["tool"] == "validate"
    assert calls[0]["project_dir"] == str(path)


def test_launcher_validate_tile(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    from qtextra.widgets.qt_tile import QtTileWidget

    from image2image.qt.launcher import Launcher

    calls = []
    monkeypatch.setattr("image2image.qt._dialog_base.create_new_window", lambda *args: calls.append(args))
    launcher = Launcher(None)
    qtbot.addWidget(launcher)
    tile = next(tile for tile in launcher.findChildren(QtTileWidget) if "Validate" in tile._tile.title)
    tile._tile.func()
    assert calls == [("validate",)]


def test_runner_validation_action(qtbot: QtBot, tmp_path: Path) -> None:
    from image2image.qt._runner._card import QtRunnerProjectCard
    from image2image.qt._runner._constants import RunnerProject

    project = RunnerProject(
        "elastix", tmp_path, SimpleNamespace(name="demo", modalities={}, output_dir=tmp_path, project_dir=tmp_path)
    )
    card = QtRunnerProjectCard(project)
    qtbot.addWidget(card)
    assert not card.validate_btn.isEnabled()
    (tmp_path / "registered-project.config.json").write_text("{}")
    card.refresh_actions()
    assert card.validate_btn.isEnabled()
    with qtbot.waitSignal(card.evt_validate) as signal:
        qtbot.mouseClick(card.validate_btn, Qt.MouseButton.LeftButton)
    assert signal.args == [tmp_path]
    card.set_status("Running")
    assert not card.validate_btn.isEnabled()


def test_register_validation_menu(qtbot: QtBot) -> None:
    from image2image.qt.dialog_register import ImageRegistrationWindow

    window = ImageRegistrationWindow(None, run_check_version=False)
    qtbot.addWidget(window)
    menus = [action.menu() for action in window.menubar.actions() if action.menu()]
    assert any(action.text() == "Validate registration..." for menu in menus for action in menu.actions())
