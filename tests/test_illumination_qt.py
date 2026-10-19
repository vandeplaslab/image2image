"""Illumination UI, entry points, asynchronous previews, and worker cleanup."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from pytestqt.qtbot import QtBot
from qtpy.QtCore import Qt

from image2image.cli import cli
from image2image.qt.dialog_illumination import IlluminationConfig, ImageIlluminationWindow
from image2image.utils.illumination import CorrectionPreview
from test_illumination import make_source


@pytest.fixture
def illumination_window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ImageIlluminationWindow:
    monkeypatch.setattr(IlluminationConfig, "USER_CONFIG_DIR", tmp_path)
    monkeypatch.setattr("vispy.app.canvas.get_dpi", lambda **_kwargs: 96)
    window = ImageIlluminationWindow(run_check_version=False)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    window.output_directory.setText(str(tmp_path / "output"))

    def before_close(widget: ImageIlluminationWindow) -> None:
        widget._closing = True
        widget.cancel()
        qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)

    qtbot.addWidget(window, before_close_func=before_close)
    return window


def wait_preview(window: ImageIlluminationWindow, qtbot: QtBot) -> None:
    qtbot.waitUntil(
        lambda: window.worker is None and not window._preview_pending and not window.preview_timer.isActive(),
        timeout=20000,
    )


def test_load_preview_recipe_and_display_only_controls(
    illumination_window: ImageIlluminationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = illumination_window
    image = np.arange(35 * 39, dtype=np.float32).reshape(35, 39)
    source = make_source(tmp_path / "image.ome.tiff", image)
    window.add_paths([str(source.path)])
    wait_preview(window, qtbot)
    assert window.table.rowCount() == 1
    assert window.current_recipe().channels == ("A",)
    assert window.export_btn.isEnabled()
    assert window._preview is not None
    np.testing.assert_array_equal(window._preview.corrected, image)
    original = window._preview.corrected.copy()
    generation = window._generation
    window.display_max.setValue(10)
    window.shared_limits.setChecked(False)
    assert window._generation == generation
    np.testing.assert_array_equal(window._preview.corrected, original)
    window.method.setCurrentIndex(window.method.findData("gaussian"))
    window.sigma.setValue(3)
    wait_preview(window, qtbot)
    assert np.any(window._preview.corrected < 0)
    window.display_mode.setCurrentText("Estimated background")
    np.testing.assert_array_equal(window.views[1].viewer.layers[0].data, window._preview.background)
    saved = window.current_recipe()
    window.method.setCurrentIndex(window.method.findData("none"))
    window.preview_timer.stop()
    window.apply_recipe(saved)
    assert window.current_recipe() == saved
    wait_preview(window, qtbot)
    window._pick_region(None, SimpleNamespace(position=(0, 15, 20)))
    wait_preview(window, qtbot)
    assert window.preview_mode.currentIndex() == 1
    assert "Full-resolution" in window.statistics.text()


def test_explicit_scene_and_reference_mapping(
    illumination_window: ImageIlluminationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = illumination_window
    source = make_source(tmp_path / "image.ome.tiff", np.ones((2, 23, 25), dtype=np.float32), ("A", "B"))
    reference = make_source(tmp_path / "dark.ome.tiff", np.ones((23, 25), dtype=np.float32), ("Shared",))
    window._images_loaded(([(source, source.model_copy(update={"scene": 1}))], []))
    assert window.selected_source() is None
    assert not window.export_btn.isEnabled()
    window.table.cellWidget(0, 2).setCurrentIndex(1)
    window._reference_loaded("dark", (reference,))
    assert window.current_recipe().dark_channels == (-1, -1)
    assert not window.export_btn.isEnabled()
    for row in range(2):
        combo = window.mapping.cellWidget(row, 1)
        combo.setCurrentIndex(combo.findData(0))
    wait_preview(window, qtbot)
    assert window.export_btn.isEnabled()
    np.testing.assert_array_equal(window._preview.corrected, np.zeros((23, 25)))
    window.preview_timer.stop()
    window._reference_loaded("flat", (reference,))
    assert window.current_recipe().dark_channels == (0, 0)
    assert window.current_recipe().flat_channels == (-1, -1)


def test_batch_export_and_failure_report(
    illumination_window: ImageIlluminationWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = illumination_window
    source = make_source(tmp_path / "image.ome.tiff", np.ones((21, 29), dtype=np.float32))
    window.add_paths([str(source.path)])
    wait_preview(window, qtbot)
    window.start_export()
    assert not window.tabs.isEnabled()
    qtbot.waitUntil(lambda: window.worker is None, timeout=20000)
    assert len(window.results) == 1
    assert window.results[0].output.is_file()
    window.start_export()
    qtbot.waitUntil(lambda: window.worker is None, timeout=20000)
    assert "already exists" in window.batch_report.toPlainText()


def test_stale_preview_discarded_and_close_waits(
    illumination_window: ImageIlluminationWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = illumination_window
    source = make_source(tmp_path / "image.ome.tiff", np.ones((21, 29), dtype=np.float32))
    window._images_loaded(([(source,)], []))
    window.preview_timer.stop()
    started, release = Event(), Event()

    def slow_preview(*_args: object, **_kwargs: object) -> CorrectionPreview:
        started.set()
        release.wait(timeout=10)
        pixels = np.full((21, 29), 999.0)
        return CorrectionPreview(pixels, pixels, pixels, 1, (0, 0), {})

    monkeypatch.setattr("image2image.qt.dialog_illumination.preview_correction", slow_preview)
    window._start_preview()
    qtbot.waitUntil(started.is_set)
    window.radius.setValue(7)
    window.close()
    assert window.worker is not None
    assert window._closing
    release.set()
    qtbot.waitUntil(lambda: window.worker is None, timeout=15000)
    assert window._preview is None


def test_illumination_cli_and_apps_menu(
    monkeypatch: pytest.MonkeyPatch, illumination_window: ImageIlluminationWindow
) -> None:
    calls = []
    monkeypatch.setattr("image2image.cli._cli_setup", lambda *_args, **_kwargs: (10, False))
    monkeypatch.setattr("image2image.main.run", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(cli, ["--tool", "illumination"])
    assert result.exit_code == 0, result.output
    assert calls[0]["tool"] == "illumination"
    launches = []
    monkeypatch.setattr("image2image.qt._dialog_base.create_new_window", lambda *args: launches.append(args))
    menu = illumination_window._make_apps_menu()
    action = next(action for action in menu.actions() if "Illumination" in action.text())
    action.trigger()
    assert launches[0][0] == "illumination"
