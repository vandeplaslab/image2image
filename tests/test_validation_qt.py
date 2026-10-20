"""Qt controls, native entry points, and asynchronous result ownership."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QDialog, QInputDialog

from image2image.cli import cli
from image2image.models.validation import (
    ImageGeometry,
    ValidationImage,
    ValidationLandmark,
    ValidationPair,
    ValidationSession,
    ValidationStudy,
)
from image2image.qt.dialog_validate import ImageValidationWindow
from image2image.utils.validation import (
    analyze_validation_session,
    discover_validation_candidates,
    inspect_validation_image,
    landmark_exclusions,
    load_validation_transform,
    save_validation_session,
    save_validation_study,
)
from test_validation import make_elastix_project, make_validation_session, make_validation_study


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
    assert [window.tabs.tabText(i) for i in range(4)] == [
        "Inputs",
        "Landmarks",
        "Region overlap (optional)",
        "Analysis",
    ]
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
    window.tabs.setCurrentIndex(2)
    assert all(view.viewer.layers["Region labels"].visible for view in window.views)
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
    assert all("Region labels" not in view.viewer.layers for view in window.views)
    Path(session.candidates[0].path).unlink()
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis is None
    assert "Registration missing" in window.status.text()
    assert window.export_btn.isEnabled()
    assert "Registration missing" in window._failures[window._pair_id]


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


def test_native_pyramid_calibration_and_display_state(validation_window: ImageValidationWindow, tmp_path: Path) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(session.candidates[0].path)
    image = window.session.fixed
    image.geometry.spacing, image.geometry.origin = (2, 3), (10, 20)
    window._load_views()
    assert window.native_splitter.orientation() == Qt.Orientation.Vertical
    assert not window.views[0].toolbar.toolbar_right.isHidden()
    layer = window.views[0].viewer.layers["Image"]
    np.testing.assert_array_equal(layer.scale, (3, 2))
    np.testing.assert_array_equal(layer.translate, (20, 10))
    np.testing.assert_array_equal(layer.data, np.arange(32 * 40, dtype=np.uint16).reshape(32, 40))
    layer.gamma, layer.opacity, layer.contrast_limits = 1.7, 0.4, (20, 200)
    generation = window._generation
    window.views[0].viewer.camera.zoom = 2
    reader = window._readers[0]
    window._load_views()
    updated = window.views[0].viewer.layers["Image"]
    assert updated.gamma == 1.7 and updated.opacity == 0.4
    assert updated.contrast_limits == [20, 200]
    assert window._readers[0] is reader
    assert window.views[0].viewer.camera.zoom == 2 and window._generation == generation
    window.begin_pair()
    window._pair_click(0, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 29, 18)))
    window._pair_click(1, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 28, 23)))
    assert window.session.landmarks[-1].fixed == (18, 29)
    assert image.path == window.image_labels[0].toolTip()
    window.views[0].viewer.cursor.position = (0, 29, 18)
    assert window.coordinate_labels[0].text() == "X: 18 um, Y: 29 um"


def test_study_switching_keeps_results_and_invalidates_only_edited_pair(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = validation_window
    study = make_validation_study(tmp_path)
    window.open_path(save_validation_study(tmp_path / "study", study))
    first_reader = window._readers[1]
    window.start_analysis_all()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert len(window._results) == 2, window.status.text()
    assert window.analysis.candidates[0].validation["rms"] == 5
    directory = Path(window._result_directory.name)
    window.pair_selector.setCurrentIndex(1)
    assert window.analysis.candidates[0].validation["rms"] == 10
    assert first_reader.fh is None
    window.downsample.setValue(2)
    assert window.analysis is None and len(window._results) == 1 and directory.exists()
    window.pair_selector.setCurrentIndex(0)
    assert window.analysis.candidates[0].validation["rms"] == 5
    window.downsample.setValue(2)
    assert not window._results and not directory.exists()


def test_results_without_independent_evidence(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(session.candidates[0].path)
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis is not None, window.status.text()
    assert "add held-out landmark" in window.result_detail.text()
    assert window.summary_table.item(0, 4).text() == "Unavailable"
    assert window.fitting_table.item(0, 1).text() == "3"
    assert window.deformation_table.item(0, 1).text() == "100.0%"
    assert "No segmentation masks supplied" in window.overlap_table.item(0, 6).text()
    generation = window._generation
    window.views[0].viewer.layers["Image"].gamma = 1.5
    assert window.analysis is not None and window._generation == generation


def test_calibration_confirmation_and_cancellation(
    validation_window: ImageValidationWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = make_validation_session(tmp_path)
    image = inspect_validation_image(session.fixed.path)
    monkeypatch.setattr(QDialog, "exec", lambda _dialog: QDialog.DialogCode.Rejected)
    assert not validation_window._confirm_calibration(image)
    assert not image.calibration_confirmed
    monkeypatch.setattr(QDialog, "exec", lambda _dialog: QDialog.DialogCode.Accepted)
    assert validation_window._confirm_calibration(image)
    assert image.calibration_confirmed and image.calibration_source == "user"


def test_rgb_pyramid_channel_selection(validation_window: ImageValidationWindow, tmp_path: Path) -> None:
    path = tmp_path / "rgb.ome.tiff"
    pixels = np.zeros((32, 40, 3), dtype=np.uint8)
    pixels[..., 1], pixels[..., 2] = 55, 77
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(pixels, subifds=1, photometric="rgb", metadata={"axes": "YXS"})
        writer.write(pixels[::2, ::2], subfiletype=1, photometric="rgb")
    image = inspect_validation_image(path, (2, 3))
    image.geometry.origin = (10, 20)
    study = ValidationStudy(pairs=[ValidationPair(session=ValidationSession(fixed=image))])
    window = validation_window
    window.open_path(save_validation_study(tmp_path / "rgb-study", study))
    assert window.channels[0].currentText() == "RGB"
    rgb_layer = window.views[0].viewer.layers["Image"]
    assert rgb_layer.rgb and rgb_layer.data[0].shape == (32, 40, 3)
    window.channels[0].setCurrentText("G")
    layer = window.views[0].viewer.layers["Image"]
    assert layer.multiscale and window.channels[0].currentText() == "G"
    assert [level.shape for level in layer.data] == [(32, 40), (16, 20)]
    np.testing.assert_array_equal(layer.data[0].compute(), 55)
    np.testing.assert_array_equal(layer.scale, (3, 2))
    np.testing.assert_array_equal(layer.translate, (20, 10))
    layer.gamma = 1.8
    window.channels[0].setCurrentText("B")
    np.testing.assert_array_equal(window.views[0].viewer.layers["Image"].data[0].compute(), 77)
    window.channels[0].setCurrentText("G")
    assert window.views[0].viewer.layers["Image"].gamma == 1.8
    legacy = window.study.model_dump()
    saved_image = legacy["pairs"][0]["session"]["fixed"]
    saved_image.pop("display_channel")
    saved_image.pop("is_rgb")
    saved_image["channel"] = 1
    legacy_path = tmp_path / "legacy-rgb.i2validate.json"
    legacy_path.write_text(json.dumps(legacy))
    window.open_path(legacy_path)
    assert window.channels[0].currentText() == "G" and window.channels[0].count() == 4
    assert not window.views[0].viewer.layers["Image"].rgb and window.session.fixed.display_channel is None


def test_reader_reopens_when_scene_changes_and_cancelled_add_keeps_study(
    validation_window: ImageValidationWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(session.candidates[0].path)
    reader = window._readers[1]
    window.session.moving.reader_kws = {"scene_index": 0}
    window._load_views()
    assert window._readers[1] is not reader and reader.fh is None
    monkeypatch.setattr("image2image.qt.dialog_validate.QFileDialog.getOpenFileName", lambda *_args: ("", ""))
    pair_id = window._pair_id
    window.add_moving_image()
    assert len(window.study.pairs) == 1 and window._pair_id == pair_id


def test_linked_camera_centers_and_image_selection(validation_window: ImageValidationWindow, tmp_path: Path) -> None:
    window = validation_window
    session = make_validation_session(tmp_path)
    window.open_path(session.candidates[0].path)
    window._reset_native_views()
    fixed, moving = (view.viewer for view in window.views)
    assert fixed.camera.center == moving.camera.center
    fixed.camera.center = (0, 34, 25)
    fixed.camera.zoom = 7
    assert moving.camera.center == fixed.camera.center and moving.camera.zoom == 7
    moving.camera.center = (0, 17, 14)
    assert fixed.camera.center == moving.camera.center
    window.link_views.setChecked(False)
    fixed.camera.center = (0, 20, 20)
    assert moving.camera.center != fixed.camera.center
    window.link_views.setChecked(True)
    assert moving.camera.center == fixed.camera.center
    assert all(view.viewer.layers.selection.active is view.viewer.layers["Image"] for view in window.views)
    fixed.layers.selection.active = window.point_layers[0]
    assert fixed.layers.selection.active is fixed.layers["Image"]
    window.begin_pair()
    assert all(
        view.viewer.layers.selection.active is point
        for view, point in zip(window.views, window.point_layers, strict=True)
    )
    window._pair_click(0, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 20, 20)))
    window._pair_click(1, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 24, 23)))
    assert all(view.viewer.layers.selection.active is view.viewer.layers["Image"] for view in window.views)
    window.edit_landmarks.setChecked(True)
    assert fixed.layers.selection.active is window.point_layers[0]
    window.tabs.setCurrentIndex(0)
    assert fixed.layers.selection.active is fixed.layers["Image"]


def test_direct_image_analysis_and_section_controls(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = validation_window
    source = make_validation_session(tmp_path)
    Path(source.candidates[0].path).unlink()
    paths = iter((source.fixed.path, source.moving.path))
    monkeypatch.setattr("image2image.qt.dialog_validate.QFileDialog.getOpenFileName", lambda *_args: (next(paths), ""))
    monkeypatch.setattr(QDialog, "exec", lambda _dialog: QDialog.DialogCode.Accepted)
    window.choose_image(0)
    window.choose_image(1)
    assert window.session.candidates == []
    window.begin_pair()
    window._pair_click(0, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 24, 20)))
    window._pair_click(1, None, SimpleNamespace(type="mouse_press", button=1, position=(0, 28, 23)))
    assert window.pair_selector.parent() is window.image_sections[1]
    assert all(
        button.parent() is section
        for button, section in zip(window.geometry_buttons, window.image_sections, strict=True)
    )
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis is not None, window.status.text()
    assert window.analysis.candidates[0].validation["rms"] == 5
    assert window.summary_table.item(0, 4).text() == "5"
    assert "original registration deformation is unknown" in window.result_detail.text()


def test_geometry_edit_remaps_native_annotations_and_invalidates_shared_results(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = validation_window
    window.open_path(save_validation_study(tmp_path / "study", make_validation_study(tmp_path)))
    window.start_analysis_all()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    old_fixed = window.session.fixed.geometry.model_copy()
    original = [pair.session.landmarks[0].fixed for pair in window.study.pairs]
    window.views[0].viewer.layers["Image"].gamma = 1.7

    def confirm(image: ValidationImage, force: bool = False) -> bool:
        image.geometry = ImageGeometry(shape=image.geometry.shape, spacing=(3, 5), origin=(7, 11))
        image.calibration_source, image.calibration_confirmed = "user", True
        return True

    with monkeypatch.context() as patcher:
        patcher.setattr(window, "_confirm_calibration", confirm)
        window.geometry_buttons[0].click()
    assert not window._results
    assert window.views[0].viewer.layers["Image"].gamma == 1.7
    for pair, coordinates in zip(window.study.pairs, original, strict=True):
        geometry = pair.session.fixed.geometry
        np.testing.assert_allclose(
            pair.session.landmarks[0].fixed, geometry.to_physical(old_fixed.to_pixels(np.asarray(coordinates)))
        )
        assert geometry.spacing == (3, 5) and geometry.origin == (7, 11)
    window.start_analysis_all()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert len(window._results) == 2 and not window._failures, window.status.text()


def test_remove_last_moving_and_shared_fixed_images(
    validation_window: ImageValidationWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = validation_window
    source = make_validation_session(tmp_path)
    window.open_path(source.candidates[0].path)
    monkeypatch.setattr("image2image.qt.dialog_validate.hp.confirm", lambda *_args: True)
    fixed_reader, moving_reader = window._readers
    window.remove_moving_btn.click()
    assert window.session.moving is None and window.session.fixed is not None
    assert window._readers[1] is None and moving_reader.fh is None
    assert window._readers[0] is fixed_reader
    assert not window.session.candidates and not window.session.landmarks
    assert "Image" not in window.views[1].viewer.layers
    window.open_path(source.candidates[0].path)
    window.remove_image_buttons[0].click()
    assert window.session.fixed is None and window.session.moving is not None
    assert window._readers[0] is None and fixed_reader.fh is None
    assert not window.session.candidates and not window.session.landmarks
    assert Path(source.fixed.path).exists() and Path(source.moving.path).exists()
    window.open_path(save_validation_study(tmp_path / "multiple", make_validation_study(tmp_path)))
    window.remove_fixed_image()
    assert all(pair.session.fixed is None and pair.session.moving is not None for pair in window.study.pairs)
    assert all(not pair.session.candidates and not pair.session.landmarks for pair in window.study.pairs)


def test_analysis_layout_help_and_escaped_messages(validation_window: ImageValidationWindow) -> None:
    from qtextra.config import THEMES

    window = validation_window
    THEMES.set_theme_stylesheet(window)
    window.ensurePolished()
    analysis = window.tabs.widget(3)
    assert not hasattr(window, "run_btn")
    for widget in (window.run_all_btn, window.cancel_btn, window.export_btn, window.downsample, window.grid_label):
        assert analysis.isAncestorOf(widget)
    assert analysis.layout().itemAt(0).layout().itemAt(0).widget() is window.run_all_btn
    assert analysis.layout().itemAt(analysis.layout().count() - 1).widget() is window.export_btn
    landmarks = window.tabs.widget(1).layout()
    csv_controls = landmarks.itemAt(0).layout()
    assert [csv_controls.itemAt(i).widget().text() for i in range(3)] == ["Import CSV...", "Export CSV...", "CSV units"]
    assert landmarks.itemAt(2).layout().itemAt(0).widget().text() == "Add pair"
    assert landmarks.itemAt(3).widget() is window.edit_landmarks
    assert landmarks.itemAt(4).widget() is window.landmark_table
    for index in range(window.result_mode.count()):
        window.result_mode.setCurrentIndex(index)
        assert window.diagnostic_help.toolTip()
    assert "intentional reflections" in window.diagnostic_help.toolTip().lower()
    window._set_status("Cannot load <sample&1>.tiff\nTry another image.", "Input error")
    assert window.status.text().startswith("<b>Input error</b>")
    assert "&lt;sample&amp;1&gt;" in window.status.text() and "<sample" not in window.status.text()
    assert window.status.font().pointSize() == window.font().pointSize() + 2
    assert window.result_detail.font().pointSize() == window.status.font().pointSize()


def test_all_elastix_paths_navigation_and_duplicate_import(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = validation_window
    path = make_elastix_project(tmp_path, multiple=True)
    window.open_path(path)
    assert len(window.study.pairs) == window.candidate_list.count() == 2
    assert [len(pair.session.candidates) for pair in window.study.pairs] == [1, 1]
    window.open_path(path)
    assert len(window.study.pairs) == window.candidate_list.count() == 2
    assert "skipped duplicate" in window.status.text()
    window.start_analysis_all()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert len(window._results) == 2 and not window._failures, window.status.text()
    first_id = window._pair_id
    window.candidate_list.setCurrentRow(1)
    assert window.pair_selector.currentIndex() == 1 and window._pair_id != first_id
    assert window.result_candidate.currentData() == window.session.candidates[0].id
    assert window.analysis is window._results[window._pair_id]
    window.views[1].viewer.layers["Image"].gamma = 1.9
    window.pair_selector.setCurrentIndex(0)
    assert window.candidate_list.currentRow() == 0
    window.candidate_list.setCurrentRow(1)
    assert window.views[1].viewer.layers["Image"].gamma == 1.9
    window.remove_candidate()
    assert window.candidate_list.count() == 1 and not window.session.candidates
    assert first_id in window._results and len(window.study.pairs[0].session.candidates) == 1


def test_elastix_shared_reference_filter_and_failed_paths(
    validation_window: ImageValidationWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = make_elastix_project(tmp_path, multiple=True)
    config_path = path / "project.config.json"
    config = json.loads(config_path.read_text())
    other_fixed = tmp_path / "other-fixed.ome.tiff"
    other_fixed.write_bytes((tmp_path / "fixed.ome.tiff").read_bytes())
    config["modalities"]["other-fixed"] = {**config["modalities"]["fixed"], "path": str(other_fixed)}
    config["modalities"]["other-moving"] = {**config["modalities"]["moving"]}
    config["registration_paths"]["other"] = {"source": "other-moving", "target": "other-fixed"}
    config["registration_paths"]["missing"] = {"source": "missing", "target": "fixed"}
    config_path.write_text(json.dumps(config))
    prompts = []

    def choose_reference(*args: object, **kwargs: object) -> tuple[str, bool]:
        prompts.append(args[1])
        return args[3][0], True

    monkeypatch.setattr(QInputDialog, "getItem", choose_reference)
    window = validation_window
    window.open_path(path)
    assert prompts == ["Choose shared fixed reference"]
    assert len(window.study.pairs) == window.candidate_list.count() == 2
    assert "incompatible fixed reference" in window.status.text()
    assert "missing to fixed" in window.status.text()
    before = window.study.model_dump()
    window.open_path(path)
    assert window.study.model_dump() == before and len(prompts) == 1
    assert "skipped duplicate" in window.status.text()


def test_cancelled_bulk_import_preserves_live_study(
    validation_window: ImageValidationWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = validation_window
    path = make_elastix_project(tmp_path, multiple=True)
    before, generation = window.study.model_dump(), window._generation
    monkeypatch.setattr(window, "_confirm_calibration", lambda image: "moving-second" not in image.path)
    window.open_path(path)
    assert window.study.model_dump() == before and window._generation == generation
    assert not window._dirty and not window.candidate_list.count()


@pytest.mark.parametrize("accept", [False, True])
def test_landmark_copy_offer_preserves_flags_and_pair_isolation(
    validation_window: ImageValidationWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accept: bool
) -> None:
    study = make_validation_study(tmp_path)
    study.pairs[0].session.landmarks[0].enabled = False
    study.pairs[0].session.landmarks[0].note = "Check this location"
    study.pairs[1].session.landmarks = []
    window = validation_window
    window.open_path(save_validation_study(tmp_path / "copy-study", study))
    source = window.session
    window._results[window._pair_id] = analyze_validation_session(source, tmp_path / "source-result")
    source_id = window._pair_id
    prompts = []

    def offer(dialog: QDialog) -> QDialog.DialogCode:
        prompts.append(dialog.findChild(QComboBox).currentText())
        return QDialog.DialogCode.Accepted if accept else QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", offer)
    window.candidate_list.setCurrentRow(1)
    assert prompts == [study.pairs[0].name]
    assert window.pair_selector.currentIndex() == 1 and source_id in window._results
    if accept:
        assert [point.enabled for point in window.session.landmarks] == [False, True]
        assert window.session.landmarks[0].note == source.landmarks[0].note
        assert window.session.landmarks[0].id != source.landmarks[0].id
        transform = load_validation_transform(window.session.candidates[0], window.session)
        exclusions = landmark_exclusions(window.session, [transform])
        assert exclusions[window.session.landmarks[1].id] == "Reuses a known fitting landmark"
        window.session.landmarks[0].note = "Changed independently"
        assert source.landmarks[0].note == "Check this location"
        assert window._pair_id not in window._results and window._dirty
    else:
        assert not window.session.landmarks and not window._dirty


def test_landmark_copy_source_selection_and_no_overwrite(
    validation_window: ImageValidationWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = make_validation_study(tmp_path)
    target = ValidationPair(session=study.pairs[1].session.model_copy(deep=True, update={"landmarks": []}))
    study.pairs.append(target)
    window = validation_window
    window.open_path(save_validation_study(tmp_path / "copy-sources", study))
    window.pair_selector.setCurrentIndex(1)
    prompts = []

    def offer(dialog: QDialog) -> QDialog.DialogCode:
        selector = dialog.findChild(QComboBox)
        prompts.append((selector.count(), selector.currentIndex()))
        selector.setCurrentIndex(0)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", offer)
    window.pair_selector.setCurrentIndex(2)
    assert prompts == [(2, 1)]
    assert window.session.landmarks[0].moving == study.pairs[0].session.landmarks[0].moving
    window.pair_selector.setCurrentIndex(0)
    window.pair_selector.setCurrentIndex(2)
    assert len(prompts) == 1


def test_rgb_display_during_analysis_keeps_measurements_and_native_background(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    source = make_validation_session(tmp_path)
    path = tmp_path / "histology.ome.tiff"
    pixels = np.broadcast_to(np.array([11, 77, 203], dtype=np.uint8), (32, 40, 3)).copy()
    tifffile.imwrite(path, pixels, photometric="rgb", metadata={"axes": "YXS"})
    image = inspect_validation_image(path, (2, 3))
    source.fixed, source.moving, source.candidates = image, image.model_copy(deep=True), []
    window = validation_window
    window.open_path(save_validation_session(tmp_path / "rgb-analysis", source))
    qtbot.wait(30)
    reader = window._readers[0]
    window.views[0].viewer.camera.center = (0, 30, 22)
    window.views[0].viewer.camera.zoom = 2.5
    window.views[0].viewer.layers["Image"].gamma = 1.4
    window.start_analysis_all()
    generation = window._generation
    window.channels[0].setCurrentText("G")
    assert window._generation == generation and not window._cancel.is_set()
    assert window._readers[0] is reader and window.session.fixed.channel == 0
    assert window.views[0].viewer.camera.zoom == 2.5
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    assert window.analysis is not None
    # With no registration, both supplied landmarks are eligible: errors are 5 um and 0 um.
    assert window.analysis.candidates[0].validation["rms"] == pytest.approx(5 / np.sqrt(2))
    window.result_mode.setCurrentIndex(1)
    assert window.result_view.viewer.layers["Native moving"].rgb
    result, map_path = window.analysis, window.analysis.candidates[0].map_path
    window.channels[1].setCurrentText("B")
    assert not window.result_view.viewer.layers["Native moving"].rgb
    assert window.analysis is result and map_path.exists() and window.export_btn.isEnabled()
    window.channels[0].setCurrentText("RGB")
    assert window.views[0].viewer.layers["Image"].gamma == 1.4
    assert window.views[0].viewer.layers.selection.active.name == "Image"


def test_validation_error_legend_labels_and_shared_color_scale(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    session = make_validation_session(tmp_path)
    session.landmarks.append(ValidationLandmark(id="zero-error", fixed=(30, 35), moving=(30, 35)))
    translated = tmp_path / "translated.i2r.json"
    definition = json.loads(Path(session.candidates[0].path).read_text())
    definition["matrix_yx_px"][0][2] = 6
    translated.write_text(json.dumps(definition))
    session.candidates.append(discover_validation_candidates(translated)[0][0])
    window = validation_window
    window.open_path(save_validation_session(tmp_path / "legend", session))
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    window.result_mode.setCurrentIndex(1)
    assert not window.validation_legend.isHidden()
    summary = window.validation_summary.text()
    assert "2/2 eligible landmarks evaluated" in summary and "3 supplied, 1 excluded" in summary
    assert "Mean: 2.5 um" in summary and "Max: 5 um" in summary
    layer = window.result_view.viewer.layers["Validation error (um)"]
    assert layer.face_colorbar.visible and layer.face_colorbar.position == "top_right"
    maximum = max(
        point["error_um"]
        for result in window.analysis.candidates
        for point in result.landmarks
        if point["error_um"] is not None
    )
    assert layer.face_contrast_limits == (0, maximum)
    assert list(layer.features["landmark_id"]) == ["independent", "zero-error"]
    assert list(layer.text.values) == ["independent: 5 um", "zero-error: 0 um"]
    vectors = window.result_view.viewer.layers["Residual G(fixed)-moving"]
    assert vectors.vector_style == "arrow" and vectors.length == 1
    assert vectors.blending == "translucent_no_depth"
    np.testing.assert_array_equal(vectors.data[0, 0] + vectors.data[0, 1], [24, 20])
    analysis, generation = window.analysis, window._generation
    window.error_labels.setChecked(False)
    assert not layer.text.visible and window.analysis is analysis and window._generation == generation
    window.result_candidate.setCurrentIndex(1)
    layer = window.result_view.viewer.layers["Validation error (um)"]
    assert not layer.text.visible and layer.face_contrast_limits == (0, maximum)
    window.result_mode.setCurrentIndex(0)
    assert window.validation_legend.isHidden()
    assert "Validation error (um)" not in window.result_view.viewer.layers
    window.result_mode.setCurrentIndex(1)
    window.downsample.setValue(2)
    assert window.validation_legend.isHidden() and not window.validation_summary.text()


@pytest.mark.parametrize("evidence", ["missing", "fitting", "disabled"])
def test_validation_error_legend_explains_missing_measurements(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path, evidence: str
) -> None:
    session = make_validation_session(tmp_path)
    if evidence == "missing":
        session.landmarks = []
    elif evidence == "fitting":
        session.landmarks = session.landmarks[1:]
    else:
        session.landmarks = session.landmarks[:1]
        session.landmarks[0].enabled = False
        session.landmarks[0].note = "Excluded <sample&1>"
    window = validation_window
    window.open_path(save_validation_session(tmp_path / "unavailable-legend", session))
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    window.result_mode.setCurrentIndex(1)
    assert not window.validation_legend.isHidden() and not window.error_labels.isEnabled()
    assert "No independent validation errors to display" in window.validation_summary.text()
    assert "Validation error (um)" not in window.result_view.viewer.layers
    expected = {
        "missing": "Add independent landmark pairs",
        "fitting": "Reuses a known fitting landmark",
        "disabled": "Excluded &lt;sample&amp;1&gt;",
    }
    assert expected[evidence] in window.validation_summary.text()


def test_validation_zero_errors_have_real_values_and_visible_color_scale(
    validation_window: ImageValidationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    session = make_validation_session(tmp_path)
    session.landmarks[0].moving = session.landmarks[0].fixed
    window = validation_window
    window.open_path(save_validation_session(tmp_path / "zero-legend", session))
    window.start_analysis()
    qtbot.waitUntil(lambda: not window._workers, timeout=30000)
    window.result_mode.setCurrentIndex(1)
    layer = window.result_view.viewer.layers["Validation error (um)"]
    assert layer.face_colorbar.visible and layer.face_contrast_limits == (0, 1)
    assert "Mean: 0 um" in window.validation_summary.text()
    assert "All measured errors are zero" in window.result_detail.text()
    assert list(layer.text.values) == ["independent: 0 um"]
