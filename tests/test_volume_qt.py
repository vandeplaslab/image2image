"""Volume Builder editing, application entry points, and worker ownership."""

from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np
import pytest
from click.testing import CliRunner
from image2image_io.volume import open_volume
from pytestqt.qtbot import QtBot
from qtpy.QtCore import QMimeData, QPointF, Qt, QUrl
from qtpy.QtGui import QDropEvent

from image2image.cli import cli
from image2image.qt.dialog_volume import ImageVolumeWindow, VolumeConfig
from image2image.utils.volume import reconcile_channels
from test_volume import make_slide, write_slide


@pytest.fixture
def volume_window(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ImageVolumeWindow:
    monkeypatch.setattr(VolumeConfig, "USER_CONFIG_DIR", tmp_path)
    window = ImageVolumeWindow(run_check_version=False)
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

    def before_close(widget: ImageVolumeWindow) -> None:
        widget.cancel()
        qtbot.waitUntil(lambda: widget.worker is None, timeout=15000)

    qtbot.addWidget(window, before_close_func=before_close)
    window.output_directory.setText(str(tmp_path))
    window.z_spacing.setValue(5)
    return window


def test_tabs_editing_and_warnings(volume_window: ImageVolumeWindow, qtbot: QtBot, tmp_path: Path) -> None:
    window = volume_window
    assert [window.tabs.tabText(i) for i in range(3)] == ["Slides", "Channels", "Output"]
    window.slides = [make_slide(tmp_path / "slide1.tif", 1), make_slide(tmp_path / "slide3.tif", 3, (" dapi ",))]
    window.refresh()
    assert window.export_btn.isEnabled()
    assert "Excluded channels (1): FITC" in window.warning.toPlainText()
    assert "Blank Z positions (1): 2" in window.warning.toPlainText()
    window.channel_table.item(1, 0).setCheckState(Qt.CheckState.Checked)
    qtbot.waitUntil(lambda: window.channels[1].selected)
    window.policy.setCurrentIndex(1)
    qtbot.waitUntil(lambda: "Excluded slides (1)" in window.warning.toPlainText())
    assert window._plan.present == (True, False, False)
    window.mapping_slide.setCurrentIndex(1)
    mapping = window.mapping_table.cellWidget(1, 1)
    mapping.setCurrentIndex(mapping.findData(0))
    qtbot.waitUntil(lambda: "Excluded slides (0)" in window.warning.toPlainText())
    window.channel_table.item(1, 3).setText("Green")
    qtbot.waitUntil(lambda: window._plan.channel_names == ("DAPI", "Green"))
    window.move_channel(1, 0)
    assert window._plan.channel_names == ("Green", "DAPI")
    window.move_slide(0, 1)
    assert window.slides[0].path.name == "slide3.tif"
    assert [slide.position for slide in window.slides] == [1, 3]
    window.slide_table.item(1, 4).setText("1")
    qtbot.waitUntil(lambda: not window.export_btn.isEnabled())
    assert "Duplicate" in window.summary.text()
    window.slide_table.item(1, 4).setText("4")
    qtbot.waitUntil(lambda: window.export_btn.isEnabled())
    window.z_spacing.setValue(0)
    qtbot.waitUntil(lambda: not window.export_btn.isEnabled())


def test_async_drop_and_duplicate_load(
    volume_window: ImageVolumeWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = volume_window
    monkeypatch.setattr(
        "image2image.qt.dialog_volume.inspect_volume_file", lambda path, _pattern: make_slide(path, int(path.stem[-1]))
    )
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(tmp_path / "slide2.tif"))])
    event = QDropEvent(
        QPointF(5, 5), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    window.dropEvent(event)
    qtbot.waitUntil(lambda: window.worker is None)
    assert len(window.slides) == 1
    assert event.isAccepted()
    window.slide_table.item(0, 4).setText("6")
    qtbot.waitUntil(lambda: window.slides[0].position == 6)
    window.channel_table.item(0, 3).setText("Nucleus")
    window.add_paths([tmp_path / "slide2.tif", tmp_path / "slide4.tif"])
    qtbot.waitUntil(lambda: window.worker is None)
    assert len(window.slides) == 2
    assert window.channels[0].name == "Nucleus"
    assert next(slide for slide in window.slides if slide.path.name == "slide2.tif").position == 6


def test_real_worker_export(volume_window: ImageVolumeWindow, qtbot: QtBot, tmp_path: Path) -> None:
    slide = write_slide(tmp_path / "slide1.ome.tif", ("DAPI",), 1, np.full((1, 12, 14), 7, dtype="uint16"))
    window = volume_window
    window.add_paths([slide.path])
    qtbot.waitUntil(lambda: window.worker is None, timeout=15000)
    assert window.export_btn.isEnabled(), window.summary.text()
    window.start_export()
    assert not window.tabs.isEnabled()
    qtbot.waitUntil(lambda: window.worker is None, timeout=15000)
    assert window.tabs.isEnabled()
    result = tmp_path / "volume.ome.zarr"
    assert result.exists(), window.status.text()
    np.testing.assert_array_equal(open_volume(result).levels[0].compute(), 7)
    assert "Exported:" in window.status.text()
    assert not window.export_btn.isEnabled()
    assert "already exists" in window.summary.text()


def test_drag_reorders_existing_slots(volume_window: ImageVolumeWindow, tmp_path: Path) -> None:
    window = volume_window
    window.slides = [make_slide(tmp_path / "slide1.tif", 1), make_slide(tmp_path / "slide3.tif", 3)]
    window.refresh()
    table = window.slide_table
    table.selectRow(0)
    mime = QMimeData()
    mime.setData("application/x-image2image-volume-row", b"0")

    class RowDrop(QDropEvent):
        """Supply the source widget normally set by the Qt drag manager."""

        def source(self) -> object:
            return table

    event = RowDrop(
        QPointF(table.visualRect(table.model().index(1, 1)).center()),
        Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    table.dropEvent(event)
    assert event.isAccepted()
    assert [slide.path.name for slide in window.slides] == ["slide3.tif", "slide1.tif"]
    assert [slide.position for slide in window.slides] == [1, 3]
    assert table.cellWidget(0, 2) is not None
    assert table.item(0, 1).text() == "slide3.tif"


def test_scene_selection_and_import_failure(
    volume_window: ImageVolumeWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace

    window = volume_window
    slide = make_slide(tmp_path / "slide1.tif", 1)
    slide.sources = (slide.source, replace(slide.source, scene_index=1, shape=(20, 24)))
    slide.scene_index = None
    window.slides = [slide]
    window.refresh()
    assert not window.export_btn.isEnabled()
    combo = window.slide_table.cellWidget(0, 2)
    combo.setCurrentIndex(combo.findData(1))
    qtbot.waitUntil(lambda: window.export_btn.isEnabled())
    assert window._plan.slides[0].source.scene_index == 1
    assert window._plan.shape[2:] == (20, 24)

    def fail(*_args: object) -> None:
        raise ValueError("not a 2D image")

    monkeypatch.setattr("image2image.qt.dialog_volume.inspect_volume_file", fail)
    window.add_paths([tmp_path / "slide4.bad"])
    qtbot.waitUntil(lambda: window.worker is None)
    assert "not a 2D image" in window.status.text()
    assert window.tabs.isEnabled()
    assert len(window.slides) == 1


@pytest.mark.parametrize("close", [False, True])
def test_cancel_waits_for_cleanup(
    volume_window: ImageVolumeWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, close: bool
) -> None:
    window = volume_window
    window.slides = [write_slide(tmp_path / "slide1.ome.tif", ("DAPI",), 1, np.ones((1, 12, 14), dtype="uint16"))]
    window.channels = reconcile_channels(window.slides, [])
    entered, release = Event(), Event()

    def read(*_args: object, **kwargs: object) -> np.ndarray:
        entered.set()
        release.wait(10)
        return np.ones(kwargs["region"][2:], dtype="uint16")

    monkeypatch.setattr("image2image_reg.elastix.tiles.resample_region", read)
    window.start_export()
    qtbot.waitUntil(entered.is_set, timeout=15000)
    if close:
        assert not window.close()
        assert window._closing
    else:
        window.cancel()
    assert window.worker is not None
    release.set()
    qtbot.waitUntil(lambda: window.worker is None, timeout=15000)
    assert not (tmp_path / "volume.ome.zarr").exists()
    assert not list(tmp_path.glob("*.partial"))
    if not close:
        assert "Cancelled" in window.status.text()
        assert window.export_btn.isEnabled()


def test_launcher_and_cli(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    from qtextra.widgets.qt_tile import QtTileWidget

    from image2image.qt.launcher import Launcher

    calls = []
    monkeypatch.setattr("image2image.qt._dialog_base.create_new_window", lambda *args: calls.append(args))
    launcher = Launcher(None)
    qtbot.addWidget(launcher)
    tile = next(tile for tile in launcher.findChildren(QtTileWidget) if "Volume Builder" in tile._tile.title)
    tile._tile.func()
    assert calls == [("volume",)]
    monkeypatch.setattr("image2image.cli._cli_setup", lambda *_args, **_kwargs: (10, False))
    monkeypatch.setattr("image2image.main.run", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(cli, ["--tool", "volume"])
    assert result.exit_code == 0, result.output
    assert calls[-1]["tool"] == "volume"
