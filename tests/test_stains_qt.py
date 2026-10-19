"""Stain UI workflows, background workers, and application entry points."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner
from pytestqt.qtbot import QtBot
from qtpy.QtCore import QMimeData, QPointF, Qt, QUrl
from qtpy.QtGui import QDropEvent

from image2image.cli import cli
from image2image.models.stains import StainEstimation, StainRecipe
from image2image.qt.dialog_stains import ImageStainsWindow, StainsConfig
from image2image.utils.stain_io import preview_stains
from test_stains import make_stain_pyramid, make_stain_source, mixture


@pytest.fixture
def stains_window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ImageStainsWindow:
    monkeypatch.setattr(StainsConfig, "USER_CONFIG_DIR", tmp_path)
    monkeypatch.setattr("vispy.app.canvas.get_dpi", lambda **_kwargs: 96)
    window = ImageStainsWindow(run_check_version=False)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    window.output_directory.setText(str(tmp_path / "out"))

    def before_close(widget: ImageStainsWindow) -> None:
        widget._closing = True
        widget.cancel()
        qtbot.waitUntil(lambda: widget.worker is None, timeout=20000)

    qtbot.addWidget(window, before_close_func=before_close)
    return window


def wait_preview(window: ImageStainsWindow, qtbot: QtBot) -> None:
    qtbot.waitUntil(
        lambda: window.worker is None and not window._preview_pending and not window.preview_timer.isActive(),
        timeout=30000,
    )


def test_drop_pyramidal_ome_tiff(stains_window: ImageStainsWindow, qtbot: QtBot, tmp_path: Path) -> None:
    pixels = mixture((1025, 1101))
    source = make_stain_pyramid(tmp_path / "pyramid.ome.tiff", pixels)
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(source.path))])
    event = QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    stains_window.dropEvent(event)
    assert event.isAccepted()
    wait_preview(stains_window, qtbot)
    assert stains_window._preview is not None, stains_window.status.text()
    np.testing.assert_array_equal(stains_window._preview.original, pixels[::2, ::2])
    assert stains_window.export_btn.isEnabled()
    assert all("Pixels" in view.viewer.layers for view in stains_window.views)


def test_load_edit_reference_display_and_export(stains_window: ImageStainsWindow, qtbot: QtBot, tmp_path: Path) -> None:
    window = stains_window
    source = make_stain_source(tmp_path / "source.ome.tiff")
    window.add_paths([str(source.path)])
    wait_preview(window, qtbot)
    assert window._preview is not None, window.status.text()
    assert window.export_btn.isEnabled()
    original = window._preview.channels.copy()
    generation = window._generation
    window.display_limit.setValue(4)
    window.pseudocolor.setChecked(False)
    assert window._generation == generation
    np.testing.assert_array_equal(window._preview.channels, original)
    window.vector_table.cellWidget(0, 1).setValue(0.7)
    window.apply_vector_edits()
    wait_preview(window, qtbot)
    assert source.key in window.overrides
    assert window._preview.source_fit.profile == window.overrides[source.key].profile
    window.use_selected_reference()
    wait_preview(window, qtbot)
    assert window.current_recipe().reference == source
    assert window.reference_profile == window.overrides[source.key].profile
    assert window._preview.normalized is not None
    assert window.normalization.currentIndex() == 0
    window.display_mode.setCurrentIndex(1)
    assert window.views[2].viewer.layers["Pixels"].rgb
    window.start_export()
    assert not window.tabs.isEnabled()
    qtbot.waitUntil(lambda: window.worker is None, timeout=30000)
    assert len(window.results) == 1
    assert len(window.results[0].outputs) == 2, window.batch_report.toPlainText()
    assert all(path.is_file() for path in window.results[0].outputs)
    window.start_export()
    qtbot.waitUntil(lambda: window.worker is None, timeout=30000)
    assert "already exists" in window.batch_report.toPlainText()


def test_recipe_restores_settings_and_explicit_scenes(
    stains_window: ImageStainsWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = stains_window
    source = make_stain_source(tmp_path / "source.ome.tiff")
    window._images_loaded(([(source, source.model_copy(update={"scene": 1}))], []))
    assert window.selected_source() is None
    assert not window.export_btn.isEnabled()
    window.table.cellWidget(0, 2).setCurrentIndex(1)
    wait_preview(window, qtbot)
    recipe = StainRecipe(
        vector_mode="shared",
        estimation=StainEstimation(seed=19, max_samples=128),
        export_channels=(0, 2),
        channel_dtype="uint16",
    )
    window.apply_recipe(recipe)
    assert window.current_recipe() == recipe
    wait_preview(window, qtbot)
    window._pick_region(None, SimpleNamespace(position=(0, 20, 30)))
    wait_preview(window, qtbot)
    assert window.native_region.isChecked()
    assert window._preview.scale == (1, 1)


def test_failed_auto_estimate_keeps_original_available(
    stains_window: ImageStainsWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = stains_window
    source = make_stain_source(tmp_path / "blank.ome.tiff", np.full((60, 70, 3), 255, dtype=np.uint8))
    window.add_paths([str(source.path)])
    wait_preview(window, qtbot)
    assert window._preview is None
    assert "Too little" in window.status.text()
    assert window.views[0].viewer.layers["Pixels"].rgb
    window.draw_region()
    assert "Sampling rectangle" in window.views[0].viewer.layers
    window.vector_mode.setCurrentIndex(1)
    wait_preview(window, qtbot)
    assert window._preview is not None
    np.testing.assert_allclose(window._preview.channels, 0)


def test_switch_to_unmapped_source_clears_stale_pixels(
    stains_window: ImageStainsWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = stains_window
    source = make_stain_source(tmp_path / "source.ome.tiff")
    second = make_stain_source(tmp_path / "second.ome.tiff").model_copy(update={"rgb_channels": None})
    window._images_loaded(([(source,), (second,)], []))
    wait_preview(window, qtbot)
    assert "Pixels" in window.views[0].viewer.layers
    window.table.selectRow(1)
    assert "Pixels" not in window.views[0].viewer.layers
    wait_preview(window, qtbot)
    assert "Map" in window.status.text()
    for index, combo in enumerate(window.mapping):
        combo.setCurrentIndex(combo.findData(index))
    wait_preview(window, qtbot)
    assert window._preview is not None


def test_roi_sampling_custom_stains_and_shared_vectors(
    stains_window: ImageStainsWindow, qtbot: QtBot, tmp_path: Path
) -> None:
    window = stains_window
    pixels = mixture((128, 128))
    pixels[:24, :24] = [245, 250, 253]
    source = make_stain_source(tmp_path / "source.ome.tiff", pixels)
    window.add_paths([str(source.path)])
    wait_preview(window, qtbot)
    window.draw_region()
    layer = window.views[0].viewer.layers["Sampling rectangle"]
    layer.add_rectangles([[[0, 0], [0, 24], [24, 24], [24, 0]]])
    assert window._regions[source.key] == (0, 0, 24, 24)
    window.sample_profile("background")
    wait_preview(window, qtbot)
    np.testing.assert_allclose(window.overrides[source.key].profile.white, np.array([245, 250, 253]) / 255)
    window.share_vectors()
    wait_preview(window, qtbot)
    assert not window.overrides
    assert window.current_recipe().vector_mode == "shared"
    window.preset.setCurrentText("Custom (3 stains)")
    wait_preview(window, qtbot)
    assert len(window.current_recipe().profile.names) == 3
    assert window.vector_table.rowCount() == 3
    window.vector_table.cellWidget(0, 0).setText("My stain")
    window.apply_vector_edits()
    wait_preview(window, qtbot)
    assert window.profile.names[0] == "My stain"


def test_stale_preview_and_close_wait_for_worker(
    stains_window: ImageStainsWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = stains_window
    source = make_stain_source(tmp_path / "source.ome.tiff")
    result = preview_stains(source, StainRecipe())
    window._images_loaded(([(source,)], []))
    window.preview_timer.stop()
    started, release = Event(), Event()

    def slow(*_args: object, **_kwargs: object):
        started.set()
        release.wait(timeout=10)
        return result

    monkeypatch.setattr("image2image.qt.dialog_stains.preview_stains", slow)
    window._start_preview()
    qtbot.waitUntil(started.is_set)
    window.min_od.setValue(0.2)
    window.close()
    assert window.worker is not None
    assert window._closing
    release.set()
    qtbot.waitUntil(lambda: window.worker is None, timeout=20000)
    assert window._preview is None


def test_stains_cli_and_apps_menu(stains_window: ImageStainsWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr("image2image.cli._cli_setup", lambda *_args, **_kwargs: (10, False))
    monkeypatch.setattr("image2image.main.run", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(cli, ["--tool", "stains", "--file", "source.tiff"])
    assert result.exit_code == 0, result.output
    assert calls[0]["tool"] == "stains"
    launches = []
    monkeypatch.setattr("image2image.qt._dialog_base.create_new_window", lambda *args: launches.append(args))
    menu = stains_window._make_apps_menu()
    action = next(action for action in menu.actions() if "Stain Separation" in action.text())
    action.trigger()
    assert launches[0][0] == "stains"
