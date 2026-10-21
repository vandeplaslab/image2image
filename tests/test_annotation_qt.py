"""Editor integration, real drawing commands, project persistence, and worker lifecycle."""

from __future__ import annotations

from pathlib import Path
from threading import Event
from time import sleep
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from pytestqt.qtbot import QtBot
from qtextra.widgets.qt_button_color import QtColorSwatch
from qtextra.widgets.qt_table_view_check import QtCheckableTableView
from qtpy.QtCore import QEvent, QLockFile, QMimeData, QPoint, QPointF, Qt, QUrl
from qtpy.QtGui import QDragEnterEvent, QDropEvent, QMouseEvent
from qtpy.QtTest import QTest
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHeaderView,
    QLineEdit,
    QStyle,
    QStyleOptionViewItem,
    QWidget,
)
from shapely.geometry import Point, box

from image2image.cli import cli
from image2image.models.annotation import (
    AnnotationClass,
    AnnotationObject,
    EditCommand,
    FeatureSettings,
    ImagePatch,
)
from image2image.qt._annotation_finetune import FineTuneDialog
from image2image.qt._annotation_model import ModelSelector
from image2image.qt.dialog_annotate import AnnotateConfig, ImageAnnotationWindow
from image2image.utils.annotation.images import PatchReader
from image2image.utils.annotation.ml import backend_available, train_classifier
from image2image.utils.annotation.model_cache import read_classifier_metadata
from image2image.utils.annotation.store import AnnotationStore
from image2image.utils.annotation.wand import WandSource


def _wand_slide(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    path = editor.reader.path.parent / "wand.ome.tiff"
    image = np.full((128, 192), 240, dtype=np.uint8)
    image[25:105, 25:100] = 35
    image[40:85, 135:175] = 35
    tifffile.imwrite(path, image, tile=(32, 32), metadata={"axes": "YX"})
    editor.add_slides([str(path)])
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    editor._image_layers[0].contrast_limits = (0, 255)
    editor.view.viewer.camera.zoom = 1
    editor.wand_radius.setValue(75)
    editor.tool.setCurrentText("Wand")


def test_wand_click_drag_identity_undo_and_quick_successive_strokes(
    editor: ImageAnnotationWindow, qtbot: QtBot
) -> None:
    _wand_slide(editor, qtbot)
    _brush_drag(editor, (55, 60), (55, 60))
    _brush_drag(editor, (150, 60), (150, 60))
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert editor.store.count(editor.set_id) == 2, editor.status.text()
    objects = list(editor.store.iter_objects(set_id=editor.set_id))
    assert all(obj.geometry.is_valid for obj in objects)
    first = next(obj for obj in objects if obj.geometry.covers(Point(55, 60)))
    assert 4500 < first.geometry.area < 6500
    assert not first.geometry.covers(Point(115, 60))
    editor.undo()
    assert editor.store.count(editor.set_id) == 1
    editor.undo()
    assert editor.store.count(editor.set_id) == 0
    editor.redo()
    assert editor.store.get(first.key).object_id == first.object_id


def test_brush_live_shift_and_single_undo(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    key = editor.add_geometry(box(10, 10, 90, 40))
    original = editor.store.get(key)
    editor.view.viewer.camera.zoom = 1
    editor.brush_radius.setValue(4)
    editor.tool.setCurrentText("Brush +")
    event = SimpleNamespace(button=1, modifiers=[], position=(20, 20), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture)
    event.position, event.type = (20, 40), "mouse_move"
    next(gesture)
    qtbot.keyPress(canvas, Qt.Key.Key_Shift)
    assert "subtracting" in editor.tool_hint.text()
    event.modifiers, event.position = ["Shift"], (20, 70)
    next(gesture)
    qtbot.keyRelease(canvas, Qt.Key.Key_Shift)
    assert "adding" in editor.tool_hint.text()
    event.modifiers, event.position, event.type = [], (20, 100), "mouse_release"
    list(gesture)
    edited = editor.store.get(key)
    assert edited.object_id == original.object_id
    assert edited.geometry.covers(Point(25, 20))
    assert not edited.geometry.covers(Point(55, 20))
    assert edited.geometry.covers(Point(95, 20))
    editor.undo()
    assert editor.store.get(key).geometry.equals(original.geometry)


@pytest.mark.parametrize("mode", ["Brush -", "Brush +", "Wand"])
def test_paint_full_erasure_and_locks(editor: ImageAnnotationWindow, qtbot: QtBot, mode: str) -> None:
    _wand_slide(editor, qtbot)
    key = editor.add_geometry(box(45, 50, 55, 60))
    original = editor.store.get(key)
    editor.brush_radius.setValue(20)
    editor.tool.setCurrentText(mode)
    event = SimpleNamespace(button=1, modifiers=["Shift"], position=(55, 50), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture, None)
    event.type = "mouse_release"
    list(gesture)
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert editor.store.count(editor.set_id) == 0, editor.status.text()
    assert not editor.selected
    editor.undo()
    assert editor.store.get(key).geometry.equals(original.geometry)
    editor.selected = [key]
    editor.toggle_object_lock()
    _brush_drag(editor, (50, 55), (75, 55))
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert "locked" in editor.status.text()
    assert editor.store.get(key).geometry.equals(original.geometry)


@pytest.mark.parametrize("cancel", ["escape", "mode", "zoom", "contrast", "review", "refresh", "deactivate"])
def test_wand_discards_delayed_work(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, cancel: str
) -> None:
    _wand_slide(editor, qtbot)
    entered, release = Event(), Event()
    region = WandSource.region

    def delayed(source, *args, **kwargs):
        entered.set()
        assert release.wait(10)
        return region(source, *args, **kwargs)

    monkeypatch.setattr(WandSource, "region", delayed)
    event = SimpleNamespace(button=1, modifiers=[], position=(60, 55), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture)
    qtbot.waitUntil(entered.is_set, timeout=10000)
    source = editor._wand_source
    closed = []
    close = source.reader.close

    def track_close() -> None:
        closed.append(1)
        close()

    monkeypatch.setattr(source.reader, "close", track_close)
    try:
        if cancel == "escape":
            editor.clear_drawing()
        elif cancel == "mode":
            editor.tool.setCurrentText("Brush +")
        elif cancel == "zoom":
            editor.view.viewer.camera.zoom = 2
        elif cancel == "contrast":
            editor._image_layers[0].contrast_limits = (0, 100)
        elif cancel == "review":
            editor.tabs.setCurrentIndex(3)
        elif cancel == "deactivate":
            QApplication.sendEvent(editor, QEvent(QEvent.Type.WindowDeactivate))
        else:
            editor.refresh_image()
        event.type = "mouse_release"
        list(gesture)
        assert not closed
    finally:
        release.set()
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert editor.store.count(editor.set_id) == 0
    assert not len(editor.brush_preview.data)
    assert closed == ([1] if cancel == "refresh" else [])
    assert editor._wand_source is (None if cancel == "refresh" else source)


def test_wand_visible_channels_settings_and_shortcut(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    qtbot.keyClick(canvas, Qt.Key.Key_W)
    assert editor.tool.currentText() == "Wand"
    editor.wand_radius.setValue(42)
    editor.wand_sensitivity.setValue(23)
    config = AnnotateConfig(_auto_load=True)
    assert config.wand_radius == 42 and config.wand_sensitivity == 23
    editor._image_layers[0].visible = False
    _brush_drag(editor, (50, 50), (50, 50))
    assert "Show at least one image channel" in editor.status.text()
    assert editor.store.count(editor.set_id) == 0
    assert editor._wand_source is None


def test_wand_live_shift_preview_flush_and_one_edit(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    _wand_slide(editor, qtbot)
    editor.filled_paint_preview.setChecked(True)
    event = SimpleNamespace(button=1, modifiers=[], position=(60, 55), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture)
    qtbot.waitUntil(lambda: len(editor.brush_preview.data) > 0, timeout=10000)
    assert editor.paint_fill_preview.visible
    event.type, event.modifiers = "mouse_move", ["Shift"]
    next(gesture)
    qtbot.waitUntil(lambda: not len(editor.brush_preview.data), timeout=10000)
    assert not editor.paint_fill_preview.visible
    event.position, event.modifiers = (60, 65), []
    next(gesture)
    qtbot.waitUntil(lambda: len(editor.brush_preview.data) > 0, timeout=10000)
    event.type = "mouse_release"
    list(gesture)
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert not editor.paint_fill_preview.visible
    assert editor.store.count(editor.set_id) == 1
    obj = editor.store.get(editor.selected[0])
    assert obj.geometry.covers(Point(55, 60))
    assert not obj.geometry.covers(Point(150, 60))
    editor.undo()
    assert editor.store.count(editor.set_id) == 0


def test_wand_read_failure_can_retry(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    _wand_slide(editor, qtbot)
    region = WandSource.region

    def fail(*args, **kwargs):
        raise ValueError("Synthetic image read failure")

    monkeypatch.setattr(WandSource, "region", fail)
    _brush_drag(editor, (55, 60), (55, 60))
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert "Synthetic image read failure" in editor.status.text()
    assert not len(editor.brush_preview.data)
    assert editor.store.count(editor.set_id) == 0
    assert editor._wand_source is None
    monkeypatch.setattr(WandSource, "region", region)
    _brush_drag(editor, (55, 60), (55, 60))
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert editor.store.count(editor.set_id) == 1


@pytest.mark.parametrize("finish", ["release", "refresh", "slide", "project", "close"])
def test_wand_reader_and_tile_cache_survive_strokes_and_close_on_context_change(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    editor.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
    _wand_slide(editor, qtbot)
    opened, closed, reads = [], [], []

    def track_reader(path: str | Path, *, series: int = 0) -> PatchReader:
        reader = PatchReader(path, series=series)
        opened.append(reader)
        original_read, original_close = reader.read, reader.close

        def read(*args: Any, **kwargs: Any) -> ImagePatch:
            reads.append(reader)
            return original_read(*args, **kwargs)

        def close() -> None:
            closed.append(reader)
            original_close()

        monkeypatch.setattr(reader, "read", read)
        monkeypatch.setattr(reader, "close", close)
        return reader

    monkeypatch.setattr("image2image.qt.dialog_annotate.PatchReader", track_reader)
    _brush_drag(editor, (55, 60), (55, 60))
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    source = editor._wand_source
    assert source is not None and opened == [source.reader] and not closed
    first_reads = len(reads)
    assert first_reads > 0 and source.cache_bytes > 0
    _brush_drag(editor, (55, 60), (55, 60))
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert editor._wand_source is source and len(opened) == 1
    assert len(reads) == first_reads and not closed
    editor.render_timer.stop()
    if finish == "release":
        editor.clear_drawing()
        editor.tool.setCurrentText("Interact")
        editor.view.viewer.camera.zoom = 2
        editor.tool.setCurrentText("Wand")
        assert editor._wand_source is source and not closed
        editor._image_layers[0].gamma = 2
        _brush_drag(editor, (55, 60), (55, 60))
        qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
        assert editor._wand_source is source and len(opened) == 1
        assert len(reads) > first_reads and not closed
        editor.close()
    elif finish == "refresh":
        editor.refresh_image()
    elif finish == "slide":
        editor.slides.setCurrentRow(0)
    elif finish == "project":
        editor.new_project(editor.store.path.parent / "other.i2annotate")
    else:
        editor.close()
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert closed.count(source.reader) == 1 and editor._wand_source is None


@pytest.fixture(autouse=True)
def annotation_preferences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AnnotateConfig, "USER_CONFIG_DIR", tmp_path)


@pytest.fixture
def editor(qtbot, tmp_path: Path, monkeypatch):
    window = ImageAnnotationWindow(run_check_version=False)
    monkeypatch.setattr(window, "_error", lambda message: window.status.setText(message))

    def before_close(widget):
        widget._closing = True
        widget._cancel_paint()
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
    window.classes.setCurrentRow(1)
    window.add_geometry(box(80, 5, 120, 85))
    window.train_downsample.setValue(1)
    window.feature_sigma.setValue(2)
    window.sample_count.setValue(100)
    window.train()
    qtbot.waitUntil(lambda: not window.jobs.busy and not window._threads, timeout=60000)
    assert Path(window._model_value()).is_file(), window.status.text()
    model_path = window._model_value()
    window._catalogue_project["cellpose"] = str(window.store.path)
    window._catalogues["cellpose"] = [{"name": "cpsam", "cached": False, "reason": ""}]
    window.backend.setCurrentText("cellpose")
    window.checkpoint.setText("/tmp/cellpose-checkpoint")
    window.predict_downsample.setValue(2)
    window.device.setCurrentText("cpu")
    window.backend.setCurrentText("forest")
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    assert window._model_value() == model_path
    assert "CPU" in window.backend_hint.text()
    assert "Tissue classifier" in window.model_summary.text()
    window.backend.setCurrentText("cellpose")
    assert window._model_value() == "cpsam"
    assert window.checkpoint.text() == "/tmp/cellpose-checkpoint"
    assert window.predict_downsample.value() == 2
    assert window.device.currentText() == "cpu"
    window.backend.setCurrentText("forest")
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    window.channels.setText("99")
    assert window._request().channels == (0, 1, 2)
    window.predict_slide()
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    assert window.store.count(window.set_id, review="proposed") > 0, window.status.text()
    assert window.store.count(window.set_id) == 0
    assert window.proposals.proxy.index(0, 1).data() in {"Tissue", "Stroma"}
    assert window.proposals.proxy.index(0, 3).data() == "Needs review"
    window.proposals.selectRow(0)
    assert "forest" in window.proposal_details.text()
    window.review(True, True)
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    assert window.store.count(window.set_id) > 0, window.status.text()
    assert all(not obj.training for obj in window.store.iter_objects(set_id=window.set_id))
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


def test_training_checkboxes_counts_and_existing_choices(editor: ImageAnnotationWindow) -> None:
    window = editor
    key = window.add_geometry(box(5, 5, 25, 25))
    cell = window.objects.proxy.index(0, 0)
    assert cell.data(Qt.ItemDataRole.CheckStateRole) == Qt.CheckState.Checked
    window.objects.proxy.setData(cell, Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert not window.store.get(key).training
    # Native Qt delegates send integer states, while direct callers can send enums.
    assert window.objects.proxy.setData(window.objects.proxy.index(0, 0), 2, Qt.ItemDataRole.CheckStateRole)
    assert window.store.get(key).training
    assert window.objects.proxy.setData(window.objects.proxy.index(0, 0), 0, Qt.ItemDataRole.CheckStateRole)
    assert not window.store.get(key).training
    window.tabs.setCurrentIndex(1)
    assert "0 included" in window.training_summary.text()
    window.undo()
    assert window.store.get(key).training
    window.set_training(False, keys=[key])
    assert not window.store.get(key).training
    path = window.store.path
    window.new_project(path.parent / "another.i2annotate")
    window.open_project(path)
    assert not window.store.get(key).training
    assert len(window.store.classes()) == 8
    ignore = next(item for item in window.store.classes() if item.is_ignore)
    window.classes.setCurrentRow(window.classes.count() - 3)
    assert window.class_id == ignore.id
    window.add_geometry(box(8, 8, 12, 12))
    cell = window.objects.proxy.index(1, 0)
    assert not cell.flags() & Qt.ItemFlag.ItemIsUserCheckable
    assert "Ignore mask" in cell.data(Qt.ItemDataRole.ToolTipRole)


def test_overview_palette_and_backend_guidance(
    editor: ImageAnnotationWindow, monkeypatch: pytest.MonkeyPatch, qtbot: QtBot
) -> None:
    window = editor
    palette = {item.name: item for item in window.store.classes()}
    assert set(palette) == {"Tissue", "Stroma", "Background", "Nucleus", "Cells", "Ignore", "Other", "Necrosis"}
    assert palette["Background"].color == "#ffd54f"
    window.show()
    qtbot.wait(350)
    qtbot.waitUntil(lambda: not window.overview.widget.visibleRegion().isEmpty())
    left_layout = window.slides.parentWidget().layout()
    assert 0 <= left_layout.indexOf(window.overview.widget) < left_layout.indexOf(window.slides)
    assert window.overview.widget.height() == 160
    monkeypatch.setattr("image2image.qt.dialog_annotate.backend_available", lambda name: name == "forest")
    window.tabs.setCurrentIndex(2)
    window.backend.setCurrentText("micro-sam")
    assert "Backend unavailable" in window.backend_hint.text()
    assert "annotate-sam" in window.backend_hint.text()
    assert window.backend_hint.isVisible()
    assert window.backend_hint.mapTo(window, QPoint()).y() < window.model.mapTo(window, QPoint()).y()
    assert not window.prompt_btn.isEnabled()
    assert not any(button.isEnabled() for button in window.predict_buttons)
    window.backend.setCurrentText("forest")
    assert window.device.isHidden()
    assert not window.predict_downsample.isEnabled()
    assert "Train or choose" in window.backend_hint.text()
    assert not any(button.isEnabled() for button in window.predict_buttons)
    # Opening an older/custom project must not insert or recolor default classes.
    store = window.store
    tissue = palette["Tissue"].model_copy(update={"color": "#123456"})
    store.save_class(tissue)
    store.save_class(AnnotationClass(name="Custom"))
    project_path = store.path
    window.new_project(project_path.parent / "palette.i2annotate")
    window.open_project(project_path)
    assert len(window.store.classes()) == 9
    assert next(item for item in window.store.classes() if item.id == tissue.id).color == "#123456"
    legacy_path = project_path.parent / "legacy.i2annotate"
    with AnnotationStore(legacy_path, create=True) as legacy:
        legacy.save_class(AnnotationClass(name="Background", color="#bdbdbd"))
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    window.open_project(legacy_path)
    assert [(item.name, item.color) for item in window.store.classes()] == [("Background", "#bdbdbd")]


@pytest.mark.parametrize("mode", ["Freehand", "Brush +", "Brush -", "Shift brush"])
def test_canvas_drags_commit_without_panning(editor: ImageAnnotationWindow, qtbot: QtBot, mode: str) -> None:
    window = editor
    window.show()
    qtbot.waitExposed(window)
    window.render_timer.stop()
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    original = box(5, 5, 120, 90)
    if mode in ("Brush -", "Shift brush"):
        key = window.add_geometry(original)
    window.tool.setCurrentText("Brush +" if mode == "Shift brush" else mode)
    canvas = window.view.widget.canvas.native
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    points = [center + QPoint(-25, -20), center + QPoint(25, -20), center + QPoint(25, 20), center + QPoint(-25, 20)]
    if mode == "Brush +":
        points.append(points[0])
    camera_center = tuple(window.view.viewer.camera.center)
    assert not window.view.viewer.camera.mouse_pan
    modifier = Qt.KeyboardModifier.ShiftModifier if mode == "Shift brush" else Qt.KeyboardModifier.NoModifier
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, modifier, pos=points[0])
    if mode != "Freehand":
        assert len(window.brush_preview.data) > 0
        assert window.draw_layer.cursor_size == window.brush_radius.value() * 2
        expected_color = [1, 99 / 255, 71 / 255, 1] if mode in ("Brush -", "Shift brush") else [0, 1, 0, 1]
        np.testing.assert_allclose(
            window.brush_preview.edge_color, np.tile(expected_color, (len(window.brush_preview.data), 1))
        )
        assert all(width == 3 for width in window.brush_preview.edge_width)
    for point in points[1:]:
        qtbot.mouseMove(canvas, point)
    assert window.store.count(window.set_id) == (1 if mode in ("Brush -", "Shift brush") else 0)
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, modifier, pos=points[-1])
    assert not len(window.brush_preview.data)
    assert tuple(window.view.viewer.camera.center) == camera_center
    assert window.store.count(window.set_id) == 1, window.status.text()
    obj = window.store.get(window.selected[0])
    assert obj.geometry.is_valid
    if mode in ("Brush -", "Shift brush"):
        assert obj.key == key
        assert obj.geometry.area < original.area
        window.undo()
        assert window.store.get(key).geometry.equals(original)
    else:
        assert obj.geometry.area > 0
        if mode == "Brush +":
            assert not obj.geometry.interiors
        window.undo()
        assert window.store.count(window.set_id) == 0
    window.tool.setCurrentText("Interact")
    assert window.view.viewer.camera.mouse_pan
    assert window.view.viewer.camera.mouse_zoom


def test_compact_training_layout_and_parameters(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = editor
    window.show()
    window.tabs.setCurrentIndex(1)
    for height in (800, 1100):
        window.resize(1650, height)
        qtbot.wait(50)
        assert window.random_seed.y() - window.sample_count.y() < 50
        assert window.max_depth.y() - window.random_seed.y() < 50
        assert window.training_counts.height() <= 190
    jobs = []
    monkeypatch.setattr(window.jobs, "submit", jobs.append)
    window.random_seed.setValue(123)
    window.max_depth.setValue(7)
    window.train()
    assert jobs[0]["settings"]["seed"] == 123
    assert jobs[0]["settings"]["max_depth"] == 7


def test_independent_pagination_and_boundaries(editor: ImageAnnotationWindow) -> None:
    window = editor
    window.store.apply(
        EditCommand(
            "Many objects",
            [(None, AnnotationObject(window.set_id, index, box(1, 1, 3, 3), window.class_id)) for index in range(225)]
            + [
                (None, AnnotationObject(window.set_id, index, box(4, 4, 7, 7), window.class_id, review="proposed"))
                for index in range(1000, 1130)
            ],
        )
    )
    window.refresh_tables()
    previous, next_, label = window._pagination["accepted"]
    assert "Page 1 of 3" in label.text() and "100 per page" in label.text()
    assert not previous.isEnabled() and next_.isEnabled()
    window.next_page("accepted")
    window.next_page("proposed")
    assert "Page 2 of 3" in label.text()
    assert window.proposals.n_rows == 30
    window.next_page("accepted")
    assert window.objects.n_rows == 25
    assert previous.isEnabled() and not next_.isEnabled()
    window.next_page("accepted")
    assert window.objects.n_rows == 25
    window.previous_page("accepted")
    assert window.objects.n_rows == 100
    assert "Page 2 of 2" in window._pagination["proposed"][2].text()


def test_external_classifier_is_copied_and_discovered(
    editor: ImageAnnotationWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, qtbot: QtBot
) -> None:
    window = editor
    window.add_geometry(box(1, 1, 25, 80))
    window.classes.setCurrentRow(1)
    window.add_geometry(box(70, 1, 120, 80))
    model = train_classifier(
        window.store,
        "Library",
        FeatureSettings(downsample=1, sigma_max=2, samples_per_class=100, trees=3, seed=23, max_depth=3),
    )
    external = tmp_path / "external.npz"
    external.write_bytes(model.read_bytes())
    monkeypatch.setattr("image2image.qt.dialog_annotate.QFileDialog.getOpenFileName", lambda *args: (str(external), ""))
    window.choose_model()
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    imported = Path(window._model_value())
    assert imported.parent == window.store.path / "models"
    assert imported != external and imported.read_bytes() == external.read_bytes()
    assert "seed: 23" in window.model_summary.text()
    assert "maximum depth: 3" in window.model_summary.text()
    window.choose_model()
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    assert window._model_value() != str(imported)
    assert window.model.count() == 3
    assert read_classifier_metadata(window._model_value())["name"] == "Library"
    window.show()
    window.tabs.setCurrentIndex(2)
    qtbot.wait(50)
    assert window.tabs.widget(2).horizontalScrollBar().maximum() == 0


def test_pretrained_selector_download_state_and_automatic_requests(
    editor: ImageAnnotationWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = editor
    monkeypatch.setattr("image2image.qt.dialog_annotate.backend_available", lambda _backend: True)
    window._catalogue_project["micro-sam"] = str(window.store.path)
    window._catalogues["micro-sam"] = [
        {"name": "vit_b_lm", "cached": False, "reason": ""},
        {"name": "vit_t", "cached": False, "reason": "Requires mobile_sam."},
    ]
    window.backend.setCurrentText("micro-sam")
    assert window.download_btn.isEnabled()
    assert not any(button.isEnabled() for button in window.predict_buttons)
    assert not window.model.model().item(window.model.findData("vit_t")).isEnabled()
    jobs = []
    monkeypatch.setattr(window.jobs, "submit", jobs.append)
    window.download_selected_model()
    assert jobs[0] == {"operation": "download_model", "backend": "micro-sam", "model": "vit_b_lm"}
    set_id = window.set_id
    window._job_done(jobs[0], {"name": "vit_b_lm", "cached": True, "reason": ""})
    assert window.set_id == set_id
    assert all(button.isEnabled() for button in window.predict_buttons)
    assert window.prompt_btn.isEnabled()
    assert not window.download_btn.isEnabled()
    assert "Model cached in <a href=" in window.model_status.text() and "file:///" in window.model_status.text()
    assert "micro-sam model directory" in window.model_status.text()
    assert window._request().sam_mode == "automatic"
    assert window._request(prompts=True).sam_mode == "prompts"
    assert not window._request().allow_download


def test_progress_eta_and_phase_reset(editor: ImageAnnotationWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = editor.jobs
    samples = iter([10.0, 11.0, 13.0, 14.0, 15.0, 15.0])
    monkeypatch.setattr("image2image.qt._annotation_jobs.monotonic", lambda: next(samples))
    jobs.current = {"operation": "predict", "queue_index": 2, "queue_total": 3}
    jobs._started = 10.0
    details = []
    jobs.details.connect(details.append)
    for completed in (0, 1, 3):
        jobs._report_progress({"phase": "Patches", "completed": completed, "total": 5, "message": "Predicting"})
    assert details[0]["eta"] is None
    assert details[1]["eta"] == 4
    assert details[2]["eta"] == 2
    assert "Slide 2/3" in editor.status.text()
    assert "ETA ~2 s" in editor.status.text()
    assert "Predicting" in editor.status.text()
    assert not hasattr(editor, "job_label") and not hasattr(editor, "job_bar")
    jobs._report_progress({"phase": "Publishing", "completed": 0, "total": None, "message": "Publishing"})
    assert details[-1]["eta"] is None
    assert details[-1]["elapsed"] == 4
    assert editor.progress_bar.maximum() == 0
    jobs._poll()
    assert details[-1]["elapsed"] == 5
    assert "Elapsed 5 s" in editor.status.text()
    jobs.current = None


def test_brush_preview_cancellation(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    window = editor
    window.show()
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    window.tool.setCurrentText("Brush +")
    canvas = window.view.widget.canvas.native
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center)
    qtbot.mouseMove(canvas, center + QPoint(25, 0))
    assert len(window.brush_preview.data)
    window.clear_drawing()
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(25, 0))
    assert not len(window.brush_preview.data)
    assert window.store.count(window.set_id) == 0


def _annotation_canvas(window: ImageAnnotationWindow, qtbot: QtBot) -> QWidget:
    window.show()
    qtbot.waitExposed(window)
    window.raise_()
    window.activateWindow()
    QApplication.setActiveWindow(window)
    qtbot.waitUntil(window.isActiveWindow)
    QTest.qWait(100)
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    window.render_timer.stop()
    return window.view.widget.canvas.native


def _canvas_mouse_event(
    canvas: QWidget,
    kind: QEvent.Type,
    point: QPoint,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    """Send canvas input without relying on queued movement of the system cursor."""
    moving = kind == QEvent.Type.MouseMove
    if moving:
        # Vispy drops mouse moves less than 10 ms apart.
        QTest.qWait(20)
    held = kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick)
    event = QMouseEvent(
        kind,
        QPointF(point),
        QPointF(canvas.mapToGlobal(point)),
        Qt.MouseButton.NoButton if moving else Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton if held else Qt.MouseButton.NoButton,
        modifiers,
    )
    QApplication.sendEvent(canvas, event)


def _draw_polygon(canvas: QWidget, points: list[QPoint]) -> None:
    for point in points:
        for kind in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
            _canvas_mouse_event(canvas, kind, point)
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonDblClick, points[-1])
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonRelease, points[-1])


def _brush_drag(window: ImageAnnotationWindow, start: tuple[float, float], end: tuple[float, float]) -> None:
    event = SimpleNamespace(button=1, modifiers=[], position=start[::-1], type="mouse_press")
    gesture = window._mouse_drag(window.view.viewer, event)
    next(gesture, None)
    event.position, event.type = end[::-1], "mouse_release"
    list(gesture)


@pytest.mark.parametrize("fill", [False, True])
def test_brush_optional_hole_filling(editor: ImageAnnotationWindow, fill: bool) -> None:
    editor.view.viewer.camera.zoom = 1
    editor.brush_radius.setValue(3)
    editor.brush_fill_holes.setChecked(fill)
    editor.tool.setCurrentText("Brush +")
    event = SimpleNamespace(button=1, modifiers=[], position=(20, 20), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture)
    for point in [(20, 80), (80, 80), (80, 20)]:
        event.position, event.type = point, "mouse_move"
        next(gesture)
    event.position, event.type = (20, 20), "mouse_release"
    list(gesture)
    geometry = editor.store.get(editor.selected[0]).geometry
    assert len(geometry.interiors) == (0 if fill else 1)
    assert geometry.covers(Point(50, 50)) == fill
    assert AnnotateConfig(_auto_load=True).brush_fill_holes == fill
    editor.undo()
    assert editor.store.count(editor.set_id) == 0


def test_freehand_crossing_saves_both_pieces_with_one_undo(editor: ImageAnnotationWindow) -> None:
    editor.view.viewer.camera.zoom = 1
    editor.tool.setCurrentText("Freehand")
    event = SimpleNamespace(button=1, modifiers=[], position=(20, 20), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture)
    for point in [(80, 80), (80, 20)]:
        event.position, event.type = point, "mouse_move"
        next(gesture)
    event.position, event.type = (20, 80), "mouse_release"
    list(gesture)
    obj = editor.store.get(editor.selected[0])
    assert obj.geometry.is_valid and len(obj.geometry.geoms) == 2
    assert obj.geometry.area == 1800 and obj.class_id == editor.class_id
    assert obj.training and obj.provenance["source"] == "manual"
    assert editor.store.count(editor.set_id) == 1
    editor.undo()
    assert editor.store.count(editor.set_id) == 0


def test_filled_preview_preserves_holes_and_is_bounded(
    editor: ImageAnnotationWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    editor.tool.setCurrentText("Brush +")
    monkeypatch.setattr(editor, "viewport", lambda: (0, 0, 100, 100))
    geometry = box(10, 10, 90, 90).difference(box(30, 30, 70, 70))
    editor._preview_geometry(geometry, False)
    assert len(editor.brush_preview.data) == 2 and not editor.paint_fill_preview.visible
    editor.filled_paint_preview.setChecked(True)
    editor._preview_geometry(geometry, False)
    layer = editor.paint_fill_preview
    assert layer.visible
    for point, alpha in [((20, 20), 64), ((50, 50), 0)]:
        row, col = np.rint(layer.world_to_data(point)).astype(int)
        assert layer.data[row, col, 3] == alpha
    assert tuple(layer.data[..., :3][layer.data[..., 3] > 0][0]) == (0, 255, 0)
    editor._preview_geometry(geometry, True)
    assert tuple(layer.data[..., :3][layer.data[..., 3] > 0][0]) == (255, 99, 71)
    editor.opacity_sliders["drawings"].setValue(60)
    assert layer.opacity == 0.6
    monkeypatch.setattr(editor, "viewport", lambda: (0, 0, 1e6, 1e6))
    editor._preview_geometry(box(0, 0, 1e6, 1e6), False)
    assert max(layer.data.shape[:2]) <= 512
    assert AnnotateConfig(_auto_load=True).filled_paint_preview
    editor.clear_drawing()
    assert not layer.visible and not len(editor.brush_preview.data)
    assert np.max(np.abs(layer.extent.world)) < 2


@pytest.mark.parametrize("mode", ["Freehand", "Brush +", "Brush -", "Wand"])
def test_alt_drag_uses_napari_box_zoom_without_drawing(editor: ImageAnnotationWindow, qtbot: QtBot, mode: str) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    editor.tool.setCurrentText(mode)
    QTest.qWait(20)
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    start, end = center - QPoint(30, 30), center + QPoint(30, 30)
    zoom = editor.view.viewer.camera.zoom
    modifier = Qt.KeyboardModifier.AltModifier
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonPress, start, modifier)
    _canvas_mouse_event(canvas, QEvent.Type.MouseMove, end, modifier)
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonRelease, end, modifier)
    assert editor.view.viewer.camera.zoom > zoom
    assert editor.store.count(editor.set_id) == 0
    assert editor._paint is None and not len(editor.brush_preview.data)


@pytest.mark.parametrize("auto_polygons", [False, True])
def test_rectangle_saves_once_on_release_and_can_predict_saved_selection(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, auto_polygons: bool
) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    editor.auto_add_polygons.setChecked(auto_polygons)
    editor.tool.setCurrentText("Rectangle")
    QTest.qWait(20)
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    start, end = center - QPoint(30, 25), center + QPoint(30, 25)
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonPress, start)
    _canvas_mouse_event(canvas, QEvent.Type.MouseMove, end)
    assert editor.store.count(editor.set_id) == 0
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonRelease, end)
    qtbot.waitUntil(lambda: editor.store.count(editor.set_id) == 1)
    obj = editor.store.get(editor.selected[0])
    assert obj.geometry.equals(box(*obj.geometry.bounds)) and obj.training
    assert not len(editor.draw_layer.data)
    editor.apply_drawing()
    assert editor.store.count(editor.set_id) == 1
    requests = []
    monkeypatch.setattr(editor, "_request", lambda bounds: bounds)
    monkeypatch.setattr(editor, "_submit_prediction", requests.append)
    editor.predict_roi()
    assert requests == [obj.geometry.bounds]
    editor.undo()
    assert editor.store.count(editor.set_id) == 0
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonPress, center)
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonRelease, center)
    QTest.qWait(20)
    assert editor.store.count(editor.set_id) == 0
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonPress, start)
    _canvas_mouse_event(canvas, QEvent.Type.MouseMove, end)
    _canvas_mouse_event(canvas, QEvent.Type.MouseButtonRelease, end)
    editor.clear_drawing()
    QTest.qWait(20)
    assert editor.store.count(editor.set_id) == 0


def test_brush_creates_three_objects_on_empty_space(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    editor.tool.setCurrentText("Brush +")
    editor.brush_radius.setValue(5)
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    for index, offset in enumerate((-90, 0, 90), 1):
        start = center + QPoint(offset, 0)
        qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        qtbot.mouseMove(canvas, start + QPoint(20, 0))
        qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=start + QPoint(20, 0))
        assert editor.store.count(editor.set_id) == index
        editor.render_timer.stop()
    assert editor.brush_preview.opacity == 1
    editor.undo()
    assert editor.store.count(editor.set_id) == 2


def test_brush_targets_stroke_start_and_keeps_target(editor: ImageAnnotationWindow) -> None:
    first = editor.add_geometry(box(10, 10, 20, 20))
    second = editor.add_geometry(box(40, 10, 50, 20))
    editor.tool.setCurrentText("Brush +")
    editor.brush_radius.setValue(2)
    editor.view.viewer.camera.zoom = 1
    event = SimpleNamespace(button=1, modifiers=[], position=(15, 15), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture)
    editor.selected = [second]
    event.position, event.type = (15, 23), "mouse_release"
    list(gesture)
    assert editor.selected == [first]
    assert editor.store.get(first).geometry.area > 100
    assert editor.store.get(second).geometry.equals(box(40, 10, 50, 20))
    editor.selected = [first, second]
    _brush_drag(editor, (70, 15), (75, 15))
    assert editor.store.count(editor.set_id) == 3


def test_brush_locked_target_and_subtraction_fallback(editor: ImageAnnotationWindow) -> None:
    first = editor.add_geometry(box(10, 10, 30, 30))
    editor.toggle_object_lock()
    editor.tool.setCurrentText("Brush +")
    editor.brush_radius.setValue(2)
    editor.view.viewer.camera.zoom = 1
    _brush_drag(editor, (20, 20), (35, 20))
    assert "locked" in editor.status.text()
    assert editor.store.count(editor.set_id) == 1
    assert editor.store.get(first).geometry.equals(box(10, 10, 30, 30))
    editor.toggle_object_lock()
    editor.tool.setCurrentText("Brush -")
    _brush_drag(editor, (8, 20), (20, 20))
    assert editor.store.get(first).geometry.area < 400
    editor.undo()
    assert editor.store.get(first).geometry.equals(box(10, 10, 30, 30))
    editor.selected = []
    _brush_drag(editor, (70, 70), (75, 70))
    assert "Start on an object" in editor.status.text()


@pytest.mark.parametrize("automatic", [True, False])
def test_polygon_double_click_saves_once(editor: ImageAnnotationWindow, qtbot: QtBot, automatic: bool) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    editor.auto_add_polygons.setChecked(automatic)
    editor.tool.setCurrentText("Polygon")
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    points = [center + QPoint(x, y) for x, y in ((-40, -40), (40, -40), (40, 40), (-40, 40))]
    _draw_polygon(canvas, points)
    QTest.qWait(20)
    assert editor.store.count(editor.set_id) == int(automatic), editor.status.text()
    if not automatic:
        editor.apply_drawing()
    assert editor.store.count(editor.set_id) == 1
    editor.apply_drawing()
    qtbot.mouseDClick(canvas, Qt.MouseButton.LeftButton, pos=points[-1])
    QTest.qWait(20)
    assert editor.store.count(editor.set_id) == 1
    editor.tool.setCurrentText("Interact")
    editor._show_selected()
    editor.tool.setCurrentText("Polygon")
    QTest.qWait(20)
    assert editor.store.count(editor.set_id) == 1


def test_invalid_polygon_and_cancelled_completion(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    editor.tool.setCurrentText("Polygon")
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    points = [center + QPoint(x, y) for x, y in ((-40, -40), (40, 40), (-40, 40), (40, -40))]
    _draw_polygon(canvas, points)
    QTest.qWait(20)
    assert editor.store.count(editor.set_id) == 0
    assert len(editor.draw_layer.data) == 1
    assert "geometry" in editor.status.text() or "Polygon" in editor.status.text()
    for clear in (editor.clear_drawing, lambda: editor.tool.setCurrentText("Interact")):
        editor.tool.setCurrentText("Polygon")
        editor.clear_drawing()
        for point in points[:3]:
            for kind in (QEvent.Type.MouseMove, QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease):
                _canvas_mouse_event(canvas, kind, point)
        editor._drawing_finished(editor.draw_layer, SimpleNamespace(button=1))
        clear()
        QTest.qWait(20)
        assert editor.store.count(editor.set_id) == 0


@pytest.mark.parametrize("reviewing", [False, True])
def test_area_selection_finishes_without_creating_objects(
    editor: ImageAnnotationWindow, qtbot: QtBot, reviewing: bool
) -> None:
    accepted = editor.add_geometry(box(0, 0, 128, 96))
    proposed = editor.store.apply(
        EditCommand(
            "Proposal", [(None, AnnotationObject(editor.set_id, "pred", box(0, 0, 128, 96), review="proposed"))]
        )
    )[0]
    editor.refresh_tables()
    canvas = _annotation_canvas(editor, qtbot)
    editor.tabs.setCurrentIndex(3 if reviewing else 0)
    editor.auto_add_polygons.setChecked(False)
    editor.tool.setCurrentText("Select by polygon")
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    _draw_polygon(canvas, [center + QPoint(x, y) for x, y in ((-30, -30), (30, -30), (0, 30))])
    qtbot.waitUntil(lambda: "selected by area" in editor.status.text())
    assert editor.selected == [proposed if reviewing else accepted]
    assert len(editor.draw_layer.data) == 1
    if reviewing:
        assert editor._review_selection == [proposed]
    assert editor.store.count(editor.set_id, review=None) == 2


def test_prompt_right_click_removes_points_then_latest_box(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    for mode, offset in (("Prompt +", -30), ("Prompt -", 0), ("Prompt +", 30)):
        editor.tool.setCurrentText(mode)
        QTest.qWait(20)
        center = QPoint(canvas.width() // 2, canvas.height() // 2)
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(offset, 0))
    editor.tool.setCurrentText("Prompt box")
    QTest.qWait(20)
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    for size in (60, 45):
        qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center - QPoint(size, size))
        qtbot.mouseMove(canvas, center + QPoint(size, size))
        qtbot.wait(10)
        qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(size, size))
    first_box = editor.prompt_box_layer.data[0].copy()
    assert len(editor.prompt_box_layer.data) == 2
    qtbot.mouseClick(canvas, Qt.MouseButton.RightButton, pos=center)
    assert editor._prompt_labels == [1, 1]
    assert len(editor.prompt_layer.data) == 2
    assert len(editor.prompt_box_layer.data) == 2
    np.testing.assert_allclose(editor.prompt_layer.face_color, [[0, 1, 0, 1]] * 2)
    qtbot.mouseClick(canvas, Qt.MouseButton.RightButton, pos=center)
    assert len(editor.prompt_box_layer.data) == 1
    np.testing.assert_allclose(editor.prompt_box_layer.data[0], first_box)
    for mode, offset in (("Prompt -", -30), ("Prompt +", 30)):
        editor.tool.setCurrentText(mode)
        QTest.qWait(20)
        center = QPoint(canvas.width() // 2, canvas.height() // 2)
        qtbot.mouseClick(canvas, Qt.MouseButton.RightButton, pos=center + QPoint(offset, 0))
    assert not len(editor.prompt_layer.data) and not editor._prompt_labels
    editor.draw_layer.add(np.array([[10, 10], [10, 20], [20, 20]]), shape_type="polygon")
    editor.clear_prompts()
    assert not len(editor.prompt_box_layer.data)
    assert len(editor.draw_layer.data) == 1


def test_display_preferences_survive_overlay_creation_and_reload(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    for attr in ("_overlay", "_proposal_overlay"):
        layer = getattr(editor, attr)
        if layer is not None:
            editor.view.viewer.layers.remove(layer)
            setattr(editor, attr, None)
    values = {"annotations": 25, "predictions": 50, "drawings": 75, "prompts": 0}
    for group, value in values.items():
        editor.opacity_sliders[group].setValue(value)
    assert editor.draw_layer.opacity == editor.brush_preview.opacity == 0.75
    assert editor.prompt_layer.opacity == editor.prompt_box_layer.opacity == 0
    editor._render()
    qtbot.waitUntil(lambda: not editor._threads and editor._overlay is not None, timeout=30000)
    assert editor._overlay.opacity == 0.25
    assert editor._proposal_overlay.opacity == 0.5
    editor._render()
    qtbot.waitUntil(lambda: not editor._threads, timeout=30000)
    assert editor._overlay.opacity == 0.25
    editor.show_predictions.setChecked(False)
    qtbot.waitUntil(lambda: not editor._proposal_overlay.visible, timeout=30000)
    editor.auto_add_polygons.setChecked(False)
    reloaded = AnnotateConfig(_auto_load=True)
    assert not reloaded.auto_add_polygons
    assert {group: getattr(reloaded, f"opacity_{group}") for group in values} == values


def test_statusbar_progress_and_cancel_are_available_in_annotate(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    _annotation_canvas(editor, qtbot)
    assert editor.statusBar() is editor.statusbar
    assert all(editor.statusbar.isAncestorOf(widget) for widget in (editor.queue_btn, editor.cancel_btn, editor.status))
    editor.jobs.current = {"operation": "predict", "id": "test"}
    editor._job_busy(True)
    editor._job_progress(
        {
            "phase": "Patches",
            "message": "Predicting",
            "completed": 1,
            "total": 4,
            "elapsed": 2,
            "eta": 6,
            "position": 1,
            "queue_total": 1,
        }
    )
    assert editor.progress_bar.isVisible()
    assert editor.progress_bar.value() == 250
    assert all(text in editor.status.text() for text in ("Patches", "Elapsed 2 s", "ETA ~6 s"))
    qtbot.mouseClick(editor.cancel_btn, Qt.MouseButton.LeftButton)
    assert editor.jobs.cancelling and editor.jobs._cancel.is_set()
    assert "Cancelling" in editor.status.text()
    editor.jobs.current, editor.jobs.cancelling = None, False
    editor._job_busy(False)
    assert editor.progress_bar.isHidden() and not editor.cancel_btn.isEnabled()
    editor.jobs.failed.emit({"operation": "predict"}, "Operation cancelled before publication.")
    assert editor.status.text() == editor.status.toolTip() == "Operation cancelled before publication."
    editor.jobs.completed.emit({"operation": "export"}, None)
    assert editor.status.text() == editor.status.toolTip() == "Completed export."
    editor.jobs.failed.emit({"operation": "predict"}, "Could not load the selected checkpoint.")
    assert editor.status.text() == editor.status.toolTip() == "Could not load the selected checkpoint."


def test_model_catalogue_before_project(qtbot: QtBot) -> None:
    if not backend_available("cellpose"):
        pytest.skip("Cellpose is optional")
    window = ImageAnnotationWindow(run_check_version=False)
    qtbot.addWidget(window)
    window.backend.setCurrentText("cellpose")
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=60000)
    assert window.store is None
    assert window.model.findData("cpsam") >= 0, window.backend_hint.text()
    assert "cellpose" in window._catalogues


def test_predict_segments_help_and_multiple_prompt_boxes(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = editor
    monkeypatch.setattr("image2image.qt.dialog_annotate.backend_available", lambda _backend: True)
    window._catalogue_project["micro-sam"] = str(window.store.path)
    window._catalogues["micro-sam"] = [{"name": "vit_b_lm", "cached": True, "reason": ""}]
    window.backend.setCurrentText("micro-sam")
    window.tabs.setCurrentIndex(2)
    window.show()
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    qtbot.wait(100)
    assert "light microscopy" in window.model_help.toolTip()
    assert window.download_btn.icon().isNull() is False
    assert window.download_btn.x() > window.model.x()
    assert window.download_btn.y() == window.model.y() or abs(window.download_btn.y() - window.model.y()) < 10
    form = window._predict_form
    assert form.getWidgetPosition(window.prompt_segment)[0] < form.getWidgetPosition(window.slide_separator)[0]
    assert form.getWidgetPosition(window.slide_separator)[0] < form.getWidgetPosition(window.slide_btn)[0]
    assert window.tabs.widget(2).horizontalScrollBar().maximum() == 0
    qtbot.mouseClick(window.draw_roi_btn, Qt.MouseButton.LeftButton)
    assert window.tool.currentText() == "Rectangle"
    qtbot.mouseClick(window.prompt_tool_buttons[2], Qt.MouseButton.LeftButton)
    assert window.tool.currentText() == "Prompt box"
    canvas = window.view.widget.canvas.native
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    for offset in (-70, 30):
        start = center + QPoint(offset, -30)
        finish = start + QPoint(35, 45)
        qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=start)
        qtbot.mouseMove(canvas, finish)
        qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=finish)
    assert len(window.prompt_box_layer.data) == 2
    for button, offset in ((window.prompt_tool_buttons[0], 0), (window.prompt_tool_buttons[1], 20)):
        qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(offset, 0))
    assert window.tool.currentText() == "Prompt -"
    request = window._request(prompts=True)
    assert len(request.boxes) == 2 and len(request.points) == 2
    assert [point[2] for point in request.points] == [1, 0]
    window.clear_drawing()
    assert not window._request(prompts=True).boxes
    assert not window._request(prompts=True).points


def test_prompt_points_can_be_restarted_after_clearing(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    window = editor
    window.show()
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    window.tool.setCurrentText("Prompt +")
    canvas = window.view.widget.canvas.native
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    for index in range(6):
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(index * 10, 0))
    assert window.prompt_layer.selected_data == {5}
    window.clear_drawing()
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center)
    assert len(window.prompt_layer.data) == 1
    assert window._prompt_labels == [1]
    window.tool.setCurrentText("Prompt -")
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(20, 0))
    assert window._prompt_labels == [1, 0]
    np.testing.assert_allclose(window.prompt_layer.face_color, [[0, 1, 0, 1], [1, 99 / 255, 71 / 255, 1]])
    window.clear_drawing()
    assert not window.prompt_layer.selected_data


def test_download_cancel_cleans_own_files_clears_queue_and_restarts_worker(
    editor: ImageAnnotationWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, jobs = editor, editor.jobs
    monkeypatch.setattr("image2image.utils.annotation.model_cache.USER_CACHE_DIR", tmp_path)
    monkeypatch.setattr("image2image.qt.dialog_annotate.backend_available", lambda _backend: True)
    window._catalogue_project["cellpose"] = str(window.store.path)
    window._catalogues["cellpose"] = [{"name": "cpsam", "cached": False, "reason": ""}]
    window.backend.setCurrentText("cellpose")
    directory = tmp_path / "models" / "cellpose"
    directory.mkdir(parents=True)
    own = directory / ".cancel123.cpsam.part"
    other = directory / ".other456.cpsam.part"
    own.write_bytes(b"partial")
    other.write_bytes(b"other job")
    worker = jobs._context.Process(target=sleep, args=(30,), daemon=True)
    worker.start()
    jobs._process = worker
    jobs.current = {"id": "cancel123", "operation": "download_model", "backend": "cellpose", "model": "cpsam"}
    jobs.pending.append({"id": "pending123", "operation": "model_catalogue", "backend": "micro-sam"})
    window._catalogue_pending.add("micro-sam")
    jobs.queue_changed.emit()
    window.show_task_queue()
    assert window.task_table.n_rows == 2
    assert "Download model" in window.task_table.proxy.index(0, 1).data()
    assert "Queued" in window.task_table.proxy.index(1, 0).data()
    assert "Cancel download" in window.download_btn.toolTip()
    qtbot.mouseClick(window.download_btn, Qt.MouseButton.LeftButton)
    assert jobs.cancelling
    assert window.task_table.proxy.index(0, 0).data() == "Cancelling"
    qtbot.waitUntil(lambda: not jobs.busy, timeout=5000)
    assert not worker.is_alive()
    assert not own.exists() and other.read_bytes() == b"other job"
    assert not jobs.pending and not window._catalogue_pending
    assert window.task_table.n_rows == 0
    assert "0 tasks" in window.queue_btn.toolTip()
    assert window.download_btn.isEnabled()
    # A real operation verifies that transport queues remain usable after stopping a blocked download.
    jobs.submit(
        {
            "operation": "train",
            "project": str(window.store.path),
            "name": "Empty",
            "settings": FeatureSettings().model_dump(),
        }
    )
    qtbot.waitUntil(lambda: not jobs.busy, timeout=30000)
    assert "two classes" in window.status.text()


def test_qtextra_tables_sort_filter_and_edit_stable_objects(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    window = editor
    for table in (window.objects, window.proposals, window.training_counts, window.task_table):
        assert isinstance(table, QtCheckableTableView)
    small = window.add_geometry(box(1, 1, 3, 3))
    large = window.add_geometry(box(10, 10, 20, 20))
    table = window.objects
    table.sortByColumn(3, Qt.SortOrder.DescendingOrder)
    assert table.key_at(0) == large
    table.selectRow(0)
    table.sortByColumn(3, Qt.SortOrder.AscendingOrder)
    assert table.selected_keys() == [large]
    assert table.key_at(0) == small
    target_id = str(window.store.get(large).object_id)
    table.filter_edit.setText(target_id)
    assert table.proxy.rowCount() == 1 and table.key_at(0) == large
    assert table.proxy.setData(table.proxy.index(0, 0), Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
    assert not window.store.get(large).training and window.store.get(small).training
    class_label = next(
        label for label, key in window._class_labels.items() if key not in (None, window.store.get(large).class_id)
    )
    window.show()
    window.activateWindow()
    table.edit(table.proxy.index(0, 2))
    editor = table.findChild(QComboBox)
    assert editor is not None
    qtbot.waitUntil(editor.view().isVisible)
    editor.hidePopup()
    editor.setCurrentText(class_label)
    qtbot.keyClick(editor, Qt.Key.Key_Tab)
    assert window.store.get(large).class_id == window._class_labels[class_label]
    assert not table.proxy.setData(table.proxy.index(0, 1), "changed ID")
    assert str(window.store.get(large).object_id) == target_id
    table.filter_edit.clear()
    assert table.proxy.rowCount() == 2
    # Sorting a page must not change the database cursor used to fetch the next page.
    window.store.apply(
        EditCommand(
            "Many",
            [(None, AnnotationObject(window.set_id, index, box(2, 2, 4, 4), window.class_id)) for index in range(250)],
        )
    )
    window.refresh_tables()
    table.sortByColumn(1, Qt.SortOrder.DescendingOrder)
    first_page = set(table.get_col_data(4))
    window.next_page("accepted")
    assert not first_page.intersection(table.get_col_data(4))
    assert len(first_page) == table.n_rows == 100


def test_training_checkbox_clicks_and_column_configuration(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    window = editor
    key = window.add_geometry(box(5, 5, 25, 25))
    window.show()
    qtbot.waitExposed(window)
    table = window.objects
    assert table.config.header[:4] == ["Training", "ID", "Class", "Area (px2)"]
    assert table.header.sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents
    assert table.header.sectionResizeMode(1) == QHeaderView.ResizeMode.Interactive
    assert table.header.sectionResizeMode(3) == QHeaderView.ResizeMode.Stretch
    assert table.isColumnHidden(4)
    assert not table.model().index(0, 4).flags() & Qt.ItemFlag.ItemIsSelectable
    assert "Double-click" in table.model().headerData(2, Qt.Orientation.Horizontal, Qt.ItemDataRole.ToolTipRole)
    table.setColumnWidth(1, 95)
    for expected in (False, True, False, True):
        cell = table.proxy.index(0, 0)
        option = QStyleOptionViewItem()
        option.initFrom(table)
        option.rect = table.visualRect(cell)
        option.features |= QStyleOptionViewItem.ViewItemFeature.HasCheckIndicator
        option.checkState = cell.data(Qt.ItemDataRole.CheckStateRole)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter
        checkbox = table.style().subElementRect(QStyle.SubElement.SE_ItemViewItemCheckIndicator, option, table)
        qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=checkbox.center())
        assert window.store.get(key).training is expected
        assert table.columnWidth(1) == 95


@pytest.mark.parametrize("review", [False, True])
def test_class_double_click_opens_choices_and_commits(
    editor: ImageAnnotationWindow, qtbot: QtBot, review: bool
) -> None:
    window = editor
    key = window.add_geometry(box(5, 5, 25, 25))
    if review:
        from dataclasses import replace

        window.store.apply(EditCommand("Propose", [(key, replace(window.store.get(key), review="proposed"))]))
        window.refresh_tables()
    table = window.proposals if review else window.objects
    window.tabs.setCurrentIndex(3 if review else 0)
    window.show()
    qtbot.waitExposed(window)
    cell = table.proxy.index(0, table.config.find_col_id("class"))
    position = table.visualRect(cell).center()
    qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    combobox = table.findChild(QComboBox)
    assert combobox is not None
    qtbot.waitUntil(combobox.view().isVisible)
    label = next(
        name
        for name, class_id in window._class_labels.items()
        if class_id not in (None, window.store.get(key).class_id)
    )
    choice = combobox.model().index(combobox.findText(label), 0)
    qtbot.mouseClick(
        combobox.view().viewport(), Qt.MouseButton.LeftButton, pos=combobox.view().visualRect(choice).center()
    )
    assert window.store.get(key).class_id == window._class_labels[label]
    assert window.store.get(key).review == ("proposed" if review else "accepted")


def test_review_selection_outlines_follow_sorted_filtered_rows(editor: ImageAnnotationWindow) -> None:
    window = editor
    geometries = [box(5, 5, 25, 25).difference(box(10, 10, 15, 15)), box(30, 30, 35, 35)]
    keys = window.store.apply(
        EditCommand(
            "Proposals",
            [
                (None, AnnotationObject(window.set_id, str(index), geometry, window.class_id, review="proposed"))
                for index, geometry in enumerate(geometries)
            ],
        )
    )
    window.refresh_tables()
    window.tabs.setCurrentIndex(3)
    table = window.proposals
    table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
    table.selectAll()
    assert set(window.selected) == set(keys)
    assert len(window.draw_layer.data) == 3
    np.testing.assert_allclose(window.draw_layer.edge_color, [[1, 1, 0, 1]] * 3)
    table.filter_column.setCurrentText("ID")
    table.filter_edit.setText("1")
    table.selectRow(0)
    assert window.selected == [keys[1]]
    np.testing.assert_allclose(window.draw_layer.data[0], np.asarray(geometries[1].exterior.coords)[:-1, ::-1])
    table.clearSelection()
    assert not window.selected and not len(window.draw_layer.data)


def test_set_counts_combine_copy_and_remove_undo_in_editor(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    window = editor
    first = window.set_id
    window.add_geometry(box(1, 1, 3, 3))
    assert "1 object" in window.sets.currentText()
    second = window.store.create_set(window.slide_id, "Second")
    window.store.apply(
        EditCommand("Second", [(None, AnnotationObject(second, 1, box(10, 10, 20, 20), window.class_id))])
    )
    window.refresh_sets()
    window.combine_sets([first, second], "Joined")
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=30000)
    combined = window.set_id
    assert combined not in (first, second)
    assert window.store.set_info(combined)["name"] == "Joined"
    assert "2 objects" in window.sets.currentText()
    assert window.sets.findData(first) >= 0 and window.sets.findData(second) >= 0
    window.remove_set()
    assert window.sets.findData(combined) == -1
    window.undo()
    assert window.sets.findData(combined) >= 0 and window.store.count(combined) == 2
    window.redo()
    assert window.sets.findData(combined) == -1


def test_review_merge_shortcuts_and_filter_typing(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = editor
    window.store.apply(
        EditCommand(
            "Proposals",
            [
                (None, AnnotationObject(window.set_id, index, geometry, window.class_id, review="proposed"))
                for index, geometry in enumerate((box(0, 0, 2, 2), box(2, 0, 4, 2), box(5, 0, 6, 1)))
            ],
        )
    )
    window.refresh_tables()
    window.tabs.setCurrentIndex(3)
    window.show()
    window.activateWindow()
    QApplication.setActiveWindow(window)
    table = window.proposals
    table.sortByColumn(2, Qt.SortOrder.DescendingOrder)
    qtbot.waitUntil(window.isActiveWindow)
    table.setFocus()
    qtbot.waitUntil(table.hasFocus)

    def press_review_key(key: Qt.Key) -> None:
        QApplication.setActiveWindow(window)
        table.setFocus()
        qtbot.keyClick(table, key)

    for row in (0, 1):
        qtbot.mouseClick(
            table.viewport(),
            Qt.MouseButton.LeftButton,
            modifier=Qt.KeyboardModifier.ControlModifier if row else Qt.KeyboardModifier.NoModifier,
            pos=table.visualRect(table.proxy.index(row, 0)).center(),
        )
    assert len(table.selected_keys()) == 2
    assert window.merge_proposals_btn.isEnabled()
    monkeypatch.setattr("image2image.qt.dialog_annotate.QInputDialog.getItem", lambda *args: (args[3][0], True))
    press_review_key(Qt.Key.Key_M)
    assert window.store.count(window.set_id, review="proposed") == 2, (
        window.status.text(),
        window.isActiveWindow(),
        table.hasFocus(),
        table.selected_keys(),
    )
    survivor = window.store.get(table.selected_keys()[0])
    assert survivor.geometry.area == 8 and survivor.review == "proposed"
    press_review_key(Qt.Key.Key_A)
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=30000)
    assert window.store.count(window.set_id) == 1
    table.selectRow(0)
    press_review_key(Qt.Key.Key_E)
    assert window.tool.currentText() == "Vertices"
    press_review_key(Qt.Key.Key_R)
    qtbot.waitUntil(lambda: not window.jobs.busy, timeout=30000)
    assert window.store.count(window.set_id, review="rejected") == 1
    window.undo()
    window.undo()
    window.undo()
    assert window.store.count(window.set_id, review="proposed") == 3
    table.selectRow(0)
    table.filter_edit.setFocus()
    qtbot.keyClicks(table.filter_edit, "AREM")
    assert table.filter_edit.text() == "AREM"
    assert not window.jobs.busy
    assert window.store.count(window.set_id, review="proposed") == 3


def test_training_class_checkboxes_preserve_object_flags(
    editor: ImageAnnotationWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = editor
    first = window.add_geometry(box(5, 5, 20, 20))
    first_class = window.store.get(first).class_id
    window.classes.setCurrentRow(1)
    second = window.add_geometry(box(60, 60, 80, 80))
    window.tabs.setCurrentIndex(1)
    assert window.tabs.tabText(1) == "Train"
    assert window.training_counts.n_rows == 8 and window.train_btn.isEnabled()
    revision = window.store.revision
    table = window.training_counts
    row = table.get_row_id(4, first_class)
    index = table.proxy.mapFromSource(table.model().index(row, 0))
    assert table.proxy.setData(index, 0, Qt.ItemDataRole.CheckStateRole)
    assert not window.train_btn.isEnabled() and window.finetune_btn.isEnabled()
    assert window.store.revision == revision
    assert window.store.get(first).training and window.store.get(second).training
    jobs = []
    monkeypatch.setattr(window.jobs, "submit", jobs.append)
    window.train()
    assert first_class not in jobs[0]["settings"]["class_ids"]
    row = table.get_row_id(4, first_class)
    assert table.proxy.setData(
        table.proxy.mapFromSource(table.model().index(row, 0)), 2, Qt.ItemDataRole.CheckStateRole
    )
    assert window.train_btn.isEnabled()
    ignore = next(item for item in window.store.classes() if item.is_ignore)
    assert not table.model().index(table.get_row_id(4, ignore.id), 0).flags() & Qt.ItemFlag.ItemIsUserCheckable


def test_finetune_dialog_requirements_settings_and_prediction_reuse(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window = editor
    monkeypatch.setattr("image2image.qt.dialog_annotate.backend_available", lambda _backend: True)
    monkeypatch.setattr("image2image.qt._annotation_finetune.backend_available", lambda _backend: True)
    window._catalogue_project["cellpose"] = str(window.store.path)
    window._catalogues["cellpose"] = [
        {"name": "cpsam", "cached": True, "reason": ""},
        {
            "name": "cpdino",
            "cached": False,
            "reason": "Requires DINOv3 architecture. Downloading weights cannot supply it.",
        },
    ]
    dialog = FineTuneDialog(window, (window.class_id,))
    qtbot.addWidget(dialog)
    dialog.show()
    assert dialog.start.isEnabled()
    assert "Model cached in" in dialog.hint.text() and "href=" in dialog.hint.text()
    assert not dialog.model.model().item(1).isEnabled()
    dialog.model.setCurrentIndex(1)
    assert not dialog.start.isEnabled() and not dialog.download.isEnabled()
    assert "DINOv3" in dialog.hint.text()
    local = tmp_path / "local.pt"
    local.write_bytes(b"weights")
    dialog.checkpoint.setText(str(local))
    assert not dialog.start.isEnabled()  # Weights cannot supply a missing architecture.
    dialog.checkpoint.clear()
    dialog.model.setCurrentIndex(0)
    dialog.epochs.setValue(12)
    dialog.learning_rate.setText("0.00002")
    settings = dialog.settings()
    assert settings.epochs == 12 and settings.learning_rate == 2e-5
    assert settings.class_ids == (window.class_id,)
    checkpoint = tmp_path / "custom.pt"
    checkpoint.write_bytes(b"weights")
    result = {
        "kind": "finetuned",
        "name": "Custom cells",
        "settings": settings.model_copy(update={"downsample": 2}).model_dump(),
        "checkpoint": str(checkpoint),
        "manifest": str(tmp_path / "metadata.json"),
    }
    dialog.close()
    window._job_done({"operation": "finetune"}, result)
    assert window.backend.currentText() == "cellpose"
    assert window._model_value() == str(checkpoint)
    assert window.checkpoint.text() == str(checkpoint)
    request = window._request()
    assert request.model == "cpsam" and request.checkpoint == str(checkpoint)
    assert request.downsample == 2 and request.channels == settings.channels
    assert "cpdino" in window.model_requirements.text() and "DINOv3" in window.model_requirements.text()


def test_review_canvas_picks_proposals_across_filters_pages_and_actions(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = editor
    window.add_geometry(box(0, 0, 128, 96))  # Accepted geometry must not intercept Review picks.
    small = AnnotationObject(window.set_id, "small", box(10, 10, 20, 20), window.class_id, review="proposed")
    distant = [
        AnnotationObject(
            window.set_id,
            f"outside-{index}",
            box(200 + index * 2, 0, 201 + index * 2, 1),
            window.class_id,
            review="proposed",
        )
        for index in range(100)
    ]
    target = AnnotationObject(window.set_id, "target", box(0, 0, 128, 96), window.class_id, review="proposed")
    keys = window.store.apply(EditCommand("Proposals", [(None, obj) for obj in (small, *distant, target)]))
    window.refresh_tables()
    window.tabs.setCurrentIndex(3)
    window.tool.setCurrentText("Interact")
    window.show()
    qtbot.waitExposed(window)
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    window.proposals.filter_edit.setText("no matches")
    canvas = window.view.widget.canvas.native
    qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton, pos=QPoint(canvas.width() // 2, canvas.height() // 2))
    assert window._review_selection == [keys[-1]]
    assert window.proposals.selected_keys() == [keys[-1]]
    assert not window.proposals.filter_edit.text()
    assert len(window._pages["proposed"]) == 2 and len(window.draw_layer.data) == 1
    event = SimpleNamespace(button=1, modifiers=["Shift"], position=(15.0, 15.0))
    list(window._mouse_drag(window.view.viewer, event))
    assert set(window._review_selection) == {keys[0], keys[-1]}
    assert set(window.selected) == {keys[0], keys[-1]} and len(window.draw_layer.data) == 2
    assert window.merge_proposals_btn.isEnabled()
    jobs = []
    monkeypatch.setattr(window.jobs, "submit", jobs.append)
    window.review(True, False)
    assert set(jobs[0]["keys"]) == {keys[0], keys[-1]}


def test_model_controls_are_shared_and_refresh_after_download(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = editor
    for module in ("dialog_annotate", "_annotation_finetune"):
        monkeypatch.setattr(f"image2image.qt.{module}.backend_available", lambda _backend: True)
    for backend, names in (("cellpose", ["cpsam"]), ("micro-sam", ["vit_b_lm", "vit_l_lm"])):
        window._catalogue_project[backend] = str(window.store.path)
        window._catalogues[backend] = [{"name": name, "cached": False, "reason": ""} for name in names]
    window.backend.setCurrentText("cellpose")
    dialog = FineTuneDialog(window, (window.class_id,))
    qtbot.addWidget(dialog)
    assert isinstance(window.model_selector, ModelSelector)
    assert isinstance(dialog.model_selector, ModelSelector)
    assert window.model.currentText() == dialog.model.currentText() == "cpsam (not downloaded)"
    assert window.model_help.toolTip() == dialog.help.toolTip()
    assert window.model_status.text() == dialog.hint.text()
    requests = []
    monkeypatch.setattr(window.jobs, "submit", requests.append)
    dialog.download.click()
    assert requests == [{"operation": "download_model", "backend": "cellpose", "model": "cpsam"}]
    result = {"name": "cpsam", "cached": True, "reason": ""}
    window.jobs.completed.emit({"operation": "download_model", "backend": "cellpose"}, result)
    assert window.model.currentText() == dialog.model.currentText() == "cpsam (cached)"
    assert window.model_status.text() == dialog.hint.text()
    assert "Model cached in" in dialog.hint.text()
    assert not dialog.download.isEnabled() and dialog.start.isEnabled()
    dialog.backend.setCurrentText("micro-sam")
    dialog.model.setCurrentIndex(1)
    dialog.backend.setCurrentText("cellpose")
    dialog.backend.setCurrentText("micro-sam")
    assert dialog.model.currentData() == "vit_l_lm"
    window.jobs.current = {"operation": "download_model", "backend": "micro-sam", "model": "vit_l_lm", "id": "test"}
    window.jobs.queue_changed.emit()
    assert "Cancel download" in window.download_btn.toolTip()
    assert "Cancel download" in dialog.download.toolTip()
    cancelled = []
    monkeypatch.setattr(window.jobs, "cancel", lambda: cancelled.append(True))
    dialog.download.click()
    window.download_btn.click()
    assert cancelled == [True, True]
    window.jobs.cancelling = True
    window.jobs.queue_changed.emit()
    assert not dialog.download.isEnabled() and not window.download_btn.isEnabled()
    window.jobs.current, window.jobs.cancelling = None, False
    monkeypatch.setattr("image2image.qt._annotation_finetune.backend_available", lambda _backend: False)
    dialog._update()
    assert not dialog.start.isEnabled() and not dialog.download.isEnabled()
    assert "Backend unavailable" in dialog.hint.text()


def test_mode_shortcuts_and_contextual_r(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    editor.classes.setFocus()
    for key, mode in (
        (Qt.Key.Key_Plus, "Prompt +"),
        (Qt.Key.Key_Minus, "Prompt -"),
        (Qt.Key.Key_P, "Polygon"),
        (Qt.Key.Key_I, "Interact"),
        (Qt.Key.Key_V, "Interact"),
        (Qt.Key.Key_B, "Brush +"),
        (Qt.Key.Key_F, "Freehand"),
    ):
        qtbot.keyClick(editor.classes, key)
        assert editor.tool.currentText() == mode
    for tab in range(3):
        editor.tabs.setCurrentIndex(tab)
        editor.classes.setFocus()
        qtbot.keyClick(editor.classes, Qt.Key.Key_R)
        assert editor.tool.currentText() == "Prompt box"
    key = editor.store.apply(
        EditCommand("Proposal", [(None, AnnotationObject(editor.set_id, "p", box(5, 5, 20, 20), review="proposed"))])
    )[0]
    editor.tabs.setCurrentIndex(3)
    editor._select_from_canvas([key])
    requests = []
    monkeypatch.setattr(editor.jobs, "submit", requests.append)
    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_R)
    assert requests[-1]["operation"] == "review" and requests[-1]["accept"] is False
    assert editor.tool.currentText() == "Prompt box"
    editor.proposals.filter_edit.setFocus()
    qtbot.keyClicks(editor.proposals.filter_edit, "PIFBR+-")
    assert editor.proposals.filter_edit.text() == "PIFBR+-"
    assert len(requests) == 1
    editor.brush_radius.setFocus()
    qtbot.keyClick(editor.brush_radius, Qt.Key.Key_I)
    assert editor.tool.currentText() == "Prompt box"


def test_canvas_apply_and_clear_shortcuts_do_not_delete_objects(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    editor.add_geometry(box(5, 5, 20, 20))
    editor.tool.setCurrentText("Polygon")
    editor.draw_layer.add(np.array([[30, 30], [30, 50], [50, 50]]), shape_type="polygon")
    editor.channels.setFocus()
    qtbot.keyClick(editor.channels, Qt.Key.Key_Return)
    assert editor.store.count(editor.set_id) == 1
    canvas.setFocus()
    qtbot.keyClick(canvas, Qt.Key.Key_Return)
    assert editor.store.count(editor.set_id) == 2
    qtbot.keyClick(canvas, Qt.Key.Key_Enter)
    assert editor.store.count(editor.set_id) == 2
    for key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
        editor.tool.setCurrentText("Interact")
        editor._show_selected()
        editor.prompt_layer.add(np.array([10, 10]))
        editor._prompt_labels.append(1)
        editor.channels.setFocus()
        qtbot.keyClick(editor.channels, key)
        assert len(editor.draw_layer.data) and len(editor.prompt_layer.data)
        canvas.setFocus()
        qtbot.keyClick(canvas, key)
        assert not len(editor.draw_layer.data) and not len(editor.prompt_layer.data)
        assert editor.store.count(editor.set_id) == 2


def test_shortcuts_leave_active_table_editor_alone(editor: ImageAnnotationWindow, qtbot: QtBot) -> None:
    _annotation_canvas(editor, qtbot)
    editor.add_geometry(box(5, 5, 20, 20))
    table = editor.objects
    table.edit(table.proxy.index(0, 2))
    combo = table.findChild(QComboBox)
    assert combo is not None
    qtbot.waitUntil(combo.view().isVisible)
    combo.hidePopup()
    combo.setFocus()
    qtbot.keyClick(combo, Qt.Key.Key_B)
    assert editor.tool.currentText() == "Interact"


@pytest.mark.parametrize("reviewing", [False, True])
def test_rectangle_selection_finishes_on_release(editor: ImageAnnotationWindow, qtbot: QtBot, reviewing: bool) -> None:
    accepted = editor.add_geometry(box(0, 0, 128, 96))
    proposed = editor.store.apply(
        EditCommand("Proposal", [(None, AnnotationObject(editor.set_id, "p", box(0, 0, 128, 96), review="proposed"))])
    )[0]
    editor.refresh_tables()
    canvas = _annotation_canvas(editor, qtbot)
    editor.tabs.setCurrentIndex(3 if reviewing else 0)
    editor.tool.setCurrentText("Select by rectangle")
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center - QPoint(30, 30))
    qtbot.mouseMove(canvas, center + QPoint(30, 30))
    qtbot.wait(10)
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=center + QPoint(30, 30))
    qtbot.waitUntil(lambda: "selected by area" in editor.status.text())
    expected = proposed if reviewing else accepted
    assert editor.selected == [expected]
    assert (editor.proposals if reviewing else editor.objects).selected_keys() == [expected]
    assert editor.store.count(editor.set_id, review=None) == 2
    assert len(editor.draw_layer.data) == 1


def test_interact_selects_annotate_rows_across_filters_and_pages(editor: ImageAnnotationWindow) -> None:
    small = AnnotationObject(editor.set_id, "small", box(10, 10, 20, 20))
    distant = [AnnotationObject(editor.set_id, f"outside-{i}", box(200 + i * 2, 0, 201 + i * 2, 1)) for i in range(100)]
    target = AnnotationObject(editor.set_id, "target", box(0, 0, 128, 96))
    keys = editor.store.apply(EditCommand("Examples", [(None, obj) for obj in (small, *distant, target)]))
    editor.refresh_tables()
    editor.objects.sortByColumn(2, Qt.SortOrder.DescendingOrder)
    editor.objects.filter_edit.setText("no match")
    list(editor._mouse_drag(editor.view.viewer, SimpleNamespace(button=1, modifiers=[], position=(48, 64))))
    assert editor.selected == [keys[-1]]
    assert editor.objects.selected_keys() == [keys[-1]]
    assert len(editor._pages["accepted"]) == 2 and not editor.objects.filter_edit.text()
    list(editor._mouse_drag(editor.view.viewer, SimpleNamespace(button=1, modifiers=["Shift"], position=(15, 15))))
    assert set(editor.selected) == {keys[0], keys[-1]}
    assert editor.objects.selected_keys() == [keys[0]]
    list(editor._mouse_drag(editor.view.viewer, SimpleNamespace(button=1, modifiers=[], position=(5000, 5000))))
    assert not editor.selected and not editor.objects.selected_keys()


def test_drawing_colors_radius_and_freehand_contrast(editor: ImageAnnotationWindow) -> None:
    assert editor.display_popup.isAncestorOf(editor.brush_radius)
    editor.tool.setCurrentText("Freehand")
    event = SimpleNamespace(button=1, modifiers=[], position=(20, 20), type="mouse_press")
    gesture = editor._mouse_drag(editor.view.viewer, event)
    next(gesture)
    event.type, event.position = "mouse_move", (20, 40)
    next(gesture)
    np.testing.assert_allclose(editor.draw_layer.edge_color, [[1, 0, 1, 1]])
    assert editor.draw_layer.edge_width == [3] and editor.draw_layer.opacity == 1
    editor.preview_colors["freehand_color"].set_color("#3478f6")
    editor.preview_colors["brush_add_color"].set_color("#123456")
    editor.preview_colors["brush_subtract_color"].set_color("#654321")
    editor.brush_radius.setValue(42)
    event.type, event.position = "mouse_release", (40, 40)
    list(gesture)
    assert editor.store.count(editor.set_id) == 1
    saved = editor.store.get(editor.selected[0])
    assert saved.class_id == editor.class_id
    editor.tool.setCurrentText("Brush +")
    editor._preview_stroke([(20, 20), (30, 30)], 2, False, force=True)
    np.testing.assert_allclose(editor.brush_preview.edge_color, [[18 / 255, 52 / 255, 86 / 255, 1]])
    editor._preview_stroke([(20, 20), (30, 30)], 2, True, force=True)
    np.testing.assert_allclose(editor.brush_preview.edge_color, [[101 / 255, 67 / 255, 33 / 255, 1]])
    editor.tool.setCurrentText("Interact")
    editor._show_selected()
    np.testing.assert_allclose(editor.draw_layer.edge_color, [[1, 1, 0, 1]])
    reloaded = AnnotateConfig(_auto_load=True)
    assert reloaded.brush_radius == 42
    assert reloaded.freehand_color == "#3478f6"
    assert reloaded.brush_add_color == "#123456" and reloaded.brush_subtract_color == "#654321"


def test_annotation_class_uses_swatch(editor: ImageAnnotationWindow, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = editor.class_id
    original = next(item for item in editor.store.classes() if item.id == selected)

    def accept(dialog: QDialog) -> int:
        swatch = dialog.findChild(QtColorSwatch)
        assert swatch is not None and swatch.hex_color_no_alpha == original.color
        assert len(dialog.findChildren(QLineEdit)) == 1
        swatch.set_color("#123456")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", accept)
    editor.edit_class()
    assert next(item for item in editor.store.classes() if item.id == selected).color == "#123456"


def _drop_event(paths: list[Path]) -> tuple[QDropEvent, QMimeData]:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return QDropEvent(
        QPointF(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    ), mime


@pytest.mark.parametrize("canvas_target", [False, True])
def test_image_drop_routes_to_project_instead_of_napari(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, canvas_target: bool
) -> None:
    canvas = _annotation_canvas(editor, qtbot)
    image = Path(editor.store.slide(editor.slide_id).path)
    requests = []
    monkeypatch.setattr(editor, "add_slides", requests.append)
    event, mime = _drop_event([image])
    target = canvas if canvas_target else editor
    enter = QDragEnterEvent(
        QPoint(10, 10), Qt.DropAction.CopyAction, mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    )
    QApplication.sendEvent(target, enter)
    assert enter.isAccepted()
    layers = len(editor.view.viewer.layers)
    QApplication.sendEvent(target, event)
    assert event.isAccepted() and requests == [[str(image)]]
    assert len(editor.view.viewer.layers) == layers


def test_project_drop_confirmation_failures_and_busy_state(
    editor: ImageAnnotationWindow, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    original = editor.store.path
    target = tmp_path / "other.i2annotate"
    with AnnotationStore(target, create=True):
        pass
    qtbot.waitUntil(lambda: not editor._threads)
    editor.render_timer.stop()
    monkeypatch.setattr("image2image.qt.dialog_annotate.hp.confirm", lambda *_args: False)
    event, _mime = _drop_event([target])
    editor.dropEvent(event)
    assert not event.isAccepted() and editor.store.path == original
    monkeypatch.setattr("image2image.qt.dialog_annotate.hp.confirm", lambda *_args: True)
    editor.dropEvent(event)
    assert event.isAccepted() and editor.store.path == target
    qtbot.waitUntil(lambda: not editor._threads)
    editor.render_timer.stop()
    invalid = tmp_path / "invalid.i2annotate"
    invalid.mkdir()
    event, _mime = _drop_event([invalid])
    editor.dropEvent(event)
    assert not event.isAccepted() and editor.store.path == target
    lock = QLockFile(str(original / ".editor.lock"))
    assert lock.tryLock(0)
    event, _mime = _drop_event([original])
    editor.dropEvent(event)
    assert not event.isAccepted() and editor.store.path == target
    lock.unlock()
    editor.jobs.current = {"operation": "predict", "id": "test"}
    editor.dropEvent(event)
    assert not event.isAccepted() and editor.store.path == target
    assert "Wait for background work" in editor.status.text()
    editor.jobs.current = None
    event, _mime = _drop_event([target, Path(editor.CONFIG.USER_CONFIG_DIR)])
    editor.dropEvent(event)
    assert "separately" in editor.status.text()
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("not an image")
    event, _mime = _drop_event([unsupported])
    editor.dropEvent(event)
    assert not event.isAccepted() and "supported images" in editor.status.text()


def test_image_drop_creates_project_or_cancels(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    window = ImageAnnotationWindow(run_check_version=False)

    def before_close(widget: ImageAnnotationWindow) -> None:
        widget._closing = True
        widget.render_timer.stop()
        widget.jobs.close()
        qtbot.waitUntil(lambda: not widget._threads, timeout=30000)

    qtbot.addWidget(window, before_close_func=before_close)
    path = tmp_path / "drop.ome.tiff"
    tifffile.imwrite(path, np.zeros((32, 32), dtype=np.uint8), metadata={"axes": "YX"})
    event, _mime = _drop_event([path])
    monkeypatch.setattr(window, "choose_new_project", lambda: None)
    window.dropEvent(event)
    assert not event.isAccepted() and window.store is None
    monkeypatch.setattr(window, "choose_new_project", lambda: window.new_project(tmp_path / "dropped.i2annotate"))
    window.dropEvent(event)
    assert event.isAccepted()
    qtbot.waitUntil(lambda: not window._threads and window.slides.count() == 1, timeout=30000)
    assert Path(window.store.slides()[0].path) == path
