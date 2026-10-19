"""Editor integration, real drawing commands, project persistence, and worker lifecycle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from shapely.geometry import box

from image2image.cli import cli
from image2image.qt.dialog_annotate import ImageAnnotationWindow


@pytest.fixture
def editor(qtbot, tmp_path: Path, monkeypatch):
    window = ImageAnnotationWindow(run_check_version=False)
    monkeypatch.setattr(window, "_error", lambda message: window.status.setText(message))

    def before_close(widget):
        widget._closing = True
        widget.render_timer.stop()
        widget.jobs.close()
        qtbot.waitUntil(lambda: not widget._threads, timeout=30000)

    qtbot.addWidget(window, before_close_func=before_close)
    window.new_project(tmp_path / "test.i2annotate")
    path = tmp_path / "image.ome.tiff"
    tifffile.imwrite(
        path,
        np.arange(96 * 128, dtype=np.uint16).reshape(96, 128),
        metadata={"axes": "YX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.8},
    )
    window.add_slides([str(path)])
    qtbot.waitUntil(lambda: not window._threads and window.reader is not None, timeout=30000)
    window.render_timer.stop()
    yield window


def test_manual_edit_id_holes_and_undo(editor, qtbot):
    window = editor
    key = window.add_geometry(box(5, 5, 40, 45))
    public_id = window.store.get(key).object_id
    window._brush(box(12, 12, 20, 20), True)
    assert len(window.store.get(key).geometry.interiors) == 1
    window.tool.setCurrentText("Vertices")
    window._show_selected()
    assert len(window.draw_layer.data) == 2
    window.apply_drawing()
    assert window.store.get(key).object_id == public_id
    assert len(window.store.get(key).geometry.interiors) == 1
    window.undo()
    window.undo()
    assert not window.store.get(key).geometry.interiors
    window.redo()
    assert len(window.store.get(key).geometry.interiors) == 1


def test_search_training_class_lock_and_pagination(editor):
    window = editor
    key = window.add_geometry(box(10, 10, 20, 20))
    window.toggle_training()
    assert window.store.get(key).training
    window.search.setText(str(window.store.get(key).object_id))
    window.find_object()
    assert window.selected == [key]
    window.toggle_object_lock()
    with pytest.raises(ValueError, match="locked"):
        window.delete_selected()
    window.toggle_object_lock()
    window.delete_selected()
    assert window.store.count(window.set_id) == 0


def test_worker_import_and_export(editor, qtbot, tmp_path):
    window = editor
    source = tmp_path / "labels.tiff"
    labels = np.zeros((96, 128), dtype=np.uint16)
    labels[3:15, 6:20] = 1901
    tifffile.imwrite(source, labels)
    window.jobs.submit(
        {
            "operation": "import",
            "project": str(window.store.path),
            "slide_id": window.slide_id,
            "path": str(source),
            "format": "labels",
            "coordinates": {"units": "px"},
        }
    )
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    assert window.store.find_id(window.set_id, 1901) is not None
    output = tmp_path / "roundtrip.tiff"
    window.jobs.submit(
        {
            "operation": "export",
            "project": str(window.store.path),
            "set_id": window.set_id,
            "path": str(output),
            "format": "labels",
            "coordinates": {"units": "px"},
        }
    )
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    np.testing.assert_array_equal(tifffile.imread(output), labels)


def test_cli_annotate_project(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("image2image.cli._cli_setup", lambda *args, **kwargs: (10, False))
    monkeypatch.setattr("image2image.main.run", lambda **kwargs: calls.append(kwargs))
    result = CliRunner().invoke(cli, ["-t", "annotate", "-p", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls[0]["tool"] == "annotate"
    assert calls[0]["project_dir"] == str(tmp_path)


def test_rgb_slide_and_classify_review_workflow(editor, qtbot, tmp_path):
    window = editor
    path = tmp_path / "color.tiff"
    data = np.zeros((96, 128, 3), dtype=np.uint8)
    data[:, 64:, :] = [240, 160, 90]
    tifffile.imwrite(path, data, photometric="rgb")
    window.add_slides([str(path)])
    qtbot.waitUntil(lambda: not window._threads and window.reader.path == path, timeout=30000)
    assert window._image_layers[0].rgb
    assert window._image_layers[0].data.shape == data.shape
    window.classes.setCurrentRow(0)
    window.add_geometry(box(5, 5, 40, 85))
    window.toggle_training()
    window.classes.setCurrentRow(1)
    window.add_geometry(box(80, 5, 120, 85))
    window.toggle_training()
    window.train_downsample.setValue(1)
    window.feature_sigma.setValue(2)
    window.sample_count.setValue(100)
    window.train()
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    assert Path(window.model.text()).is_file(), window.status.text()
    window.predict_slide()
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    assert window.store.count(window.set_id, review="proposed") > 0, window.status.text()
    assert window.store.count(window.set_id) == 0
    window.review(True, True)
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    assert window.store.count(window.set_id) > 0, window.status.text()
    window.undo()
    assert window.store.count(window.set_id) == 0


def test_cut_merge_and_properties_dialog(editor, monkeypatch):
    from image2image.qt import dialog_annotate

    window = editor
    key = window.add_geometry(box(10, 10, 50, 50))
    identifier = window.store.get(key).object_id
    monkeypatch.setattr(
        dialog_annotate.QInputDialog, "getItem", lambda _parent, _title, _label, options, *_args: (options[0], True)
    )
    window.tool.setCurrentText("Cut")
    window.draw_layer.add(np.array([[0, 25], [60, 25]]), shape_type="path")
    window.apply_drawing()
    assert len(window.selected) == 2
    assert window.store.get(key).object_id == identifier
    window.merge_selected()
    assert len(window.selected) == 1
    assert window.store.get(window.selected[0]).geometry.area == 1600
    window.undo()
    assert window.store.count(window.set_id) == 2


def test_live_editor_holds_project_recovery_lock(editor):
    from qtpy.QtCore import QLockFile

    lock = QLockFile(str(editor.store.path / ".editor.lock"))
    lock.setStaleLockTime(0)
    assert not lock.tryLock(0)


def test_search_distinguishes_integer_and_string_ids(editor):
    from image2image.models.annotation import AnnotationObject, EditCommand

    window = editor
    keys = window.store.apply(
        EditCommand(
            "Typed IDs",
            [
                (None, AnnotationObject(window.set_id, 12, box(0, 0, 5, 5))),
                (None, AnnotationObject(window.set_id, "12", box(10, 10, 15, 15))),
            ],
        )
    )
    for text, key in [("int:12", keys[0]), ("str:12", keys[1])]:
        window.search.setText(text)
        window.find_object()
        assert window.selected == [key]
