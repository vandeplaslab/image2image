"""Serial workflow and calibrated volume controls."""

from pathlib import Path

import numpy as np
import pytest
from pytestqt.qtbot import QtBot

from image2image.qt.dialog_elastix3d import ImageElastix3dWindow


def test_serial_tabs_and_settings(qtbot: QtBot, tmp_path: Path) -> None:
    from qtpy.QtCore import Qt

    from test_serial import make_project

    window = ImageElastix3dWindow(None, run_check_version=False, serial_mode=True)
    qtbot.addWidget(window)
    panel = window.serial_panel
    assert [panel.tabs.tabText(index) for index in range(3)] == ["Sections", "Registration", "Volume"]
    assert not panel.register_btn.isEnabled()
    window.registration = make_project(tmp_path)
    panel.refresh()
    assert panel.register_btn.isEnabled()
    assert not panel.reconstruct_btn.isEnabled()
    keys = panel.settings.ordered_keys(window.registration)
    window.registration.images[keys[-1]].is_reference = True
    assert window.check_if_can_update(window.registration.images[keys[-1]], silent=True)
    panel.set_section_number(keys[-1], 8)
    assert panel.settings.section_numbers[keys[-1]] == 8
    with pytest.raises(ValueError):
        panel.set_section_number(keys[-1], 1)
    panel.spacing.setValue(7.5)
    assert panel.settings.spacing_um == 7.5
    assert panel.settings.canvas_xywh is None
    panel.tabs.setCurrentIndex(2)
    assert not window._handle_key_press(Qt.Key.Key_Q)
    panel.dispose()
    window.serial_panel = None


def test_legacy_window_stays_available(qtbot: QtBot) -> None:
    window = ImageElastix3dWindow(None, run_check_version=False)
    qtbot.addWidget(window)
    assert window.serial_panel is None
    assert window.project_mode.count() == 4


@pytest.mark.parametrize("rgb", [False, True])
def test_volume_views_and_reopen(qtbot: QtBot, tmp_path: Path, rgb: bool) -> None:
    from image2image_io.volume import write_volume

    from image2image.qt._serial_volume import SerialVolumeWidget

    path = write_volume(
        tmp_path / "volume.ome.zarr",
        shape=(3 if rgb else 2, 5, 12, 14),
        dtype=np.dtype("uint8"),
        spacing=(7.5, 0.5, 0.5),
        origin=(0, -2, 3),
        channel_names=["R", "G", "B"] if rgb else ["a", "b"],
        present=[True, False, True, True, True],
        read_tile=lambda c, z, y, x, h, w: np.full((h, w), z + c, dtype=np.uint8),
        provenance={"rgb": rgb},
    )
    widget = SerialVolumeWidget()
    widget.PREVIEW_BYTES = 512
    qtbot.addWidget(widget)
    widget.open(path)
    qtbot.waitUntil(lambda: bool(widget.preview.viewer.layers), timeout=15000)
    first = widget.views[0][0].viewer
    first.dims.set_current_step((0, 1, 2), (1, 3, 4))
    assert all(view.viewer.dims.current_step == (1, 3, 4) for view, _order in widget.views)
    assert "Missing/excluded" in widget.coordinates.text()
    assert "7.500" in widget.coordinates.text()
    assert widget.preview.viewer.dims.ndisplay == 3
    assert widget.preview.viewer.layers[0].data.nbytes <= widget.PREVIEW_BYTES
    widget.contrast[0].setValue(1)
    widget.contrast[1].setValue(20)
    assert all(view.viewer.layers[0].contrast_limits == [1, 20] for view, _order in widget.views)
    assert widget.preview.viewer.layers[0].contrast_limits == [1, 20]
    assert widget.volume.levels[0][0, 0, 0, 0].compute() == 0
    widget.channel.setCurrentIndex(1)
    qtbot.waitUntil(lambda: bool(widget.preview.viewer.layers), timeout=15000)
    assert first.layers[0].name == ("G" if rgb else "b")
    widget.dispose()


def test_real_queue_reconstruction_and_cancellation(qtbot: QtBot, tmp_path: Path) -> None:
    from image2image_io.volume import open_volume
    from qtextra.typing import TaskState

    from test_serial import add_identity_registration, make_project

    window = ImageElastix3dWindow(None, run_check_version=False, serial_mode=True)
    qtbot.addWidget(window)
    window.registration = make_project(tmp_path)
    add_identity_registration(window.registration, tmp_path / "registered")
    panel = window.serial_panel
    panel.refresh()
    panel._queue("register", tmp_path)
    task = panel.task
    qtbot.waitUntil(lambda: task.state == TaskState.RUNNING, timeout=15000)
    panel.cancel()
    qtbot.waitUntil(lambda: panel.task is None, timeout=15000)
    assert "cancelled" in panel.status.text()
    output_dir = tmp_path / "output with spaces"
    output_dir.mkdir()
    output = output_dir / "volume.ome.zarr"
    panel._queue("reconstruct", output_dir, output)
    qtbot.waitUntil(lambda: panel.task is None, timeout=30000)
    assert output.is_dir(), panel.status.text()
    assert panel.settings.volume_path == str(output)
    assert panel.tabs.currentIndex() == 2
    assert open_volume(output).levels[0].shape == (2, 3, 24, 28)


def test_launcher_serial_tile(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    from qtextra.widgets.qt_tile import QtTileWidget

    from image2image.qt.launcher import Launcher

    calls = []
    monkeypatch.setattr("image2image.qt._dialog_base.create_new_window", lambda *args: calls.append(args))
    launcher = Launcher(None)
    qtbot.addWidget(launcher)
    tile = next(tile for tile in launcher.findChildren(QtTileWidget) if "Serial Sections" in tile._tile.title)
    tile._tile.func()
    assert calls == [("serial",)]


def test_serial_project_reopens_without_renumbering(qtbot: QtBot, tmp_path: Path) -> None:
    from image2image.utils.serial import save_serial_project
    from test_serial import make_project

    project = make_project(tmp_path)
    project.serial.section_numbers["section10.ome"] = 12
    project.images["section10.ome"].keep = False
    path = tmp_path / "reopen.i2wsiprep.json"
    save_serial_project(path, project)
    window = ImageElastix3dWindow(None, run_check_version=False, serial_mode=True, project_dir=path)
    qtbot.addWidget(window)
    qtbot.waitUntil(lambda: window.data_model.n_paths == 3, timeout=15000)
    qtbot.waitUntil(lambda: window.table.n_rows >= 3, timeout=15000)
    assert window.table.n_rows == 3, list(window.registration.images)
    from image2image_io.readers import get_key

    expected_numbers = {
        get_key(project.images[key].path): number for key, number in project.serial.section_numbers.items()
    }
    assert window.registration.serial.section_numbers == expected_numbers
    assert not window.registration.images[get_key(project.images["section10.ome"].path)].keep
    # Sorting the displayed rows does not alter physical order.
    window.table.sortByColumn(window.TABLE_CONFIG.key)
    assert window.registration.serial.ordered_keys(window.registration, False) == [
        get_key(project.images[key].path) for key in project.serial.ordered_keys(project, False)
    ]


def test_snapshot_and_queue_state(qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from qtextra.queue.task import Task

    from image2image.qt._serial import QUEUE
    from image2image.utils.serial import load_serial_project
    from test_serial import make_project

    queued = []
    monkeypatch.setattr(QUEUE, "add_task", queued.append)
    window = ImageElastix3dWindow(None, run_check_version=False, serial_mode=True)
    qtbot.addWidget(window)
    window.registration = make_project(tmp_path)
    panel = window.serial_panel
    panel.refresh()
    panel._queue("register", tmp_path)
    task = queued[0]
    assert isinstance(task, Task)
    assert task.commands[0][-4:-2] == ["serial", "register"]
    assert load_serial_project(panel.snapshot).serial.spacing_um == 5
    assert not panel.register_btn.isEnabled()
    assert panel.cancel_btn.isEnabled()
    panel.on_failed(task, "Synthetic processing failure")
    assert panel.register_btn.isEnabled()
    assert "Synthetic processing failure" in panel.status.text()
    panel._queue("register", tmp_path)
    panel.on_cancelled(queued[-1])
    assert "cancelled" in panel.status.text()
    assert not panel.cancel_btn.isEnabled()
