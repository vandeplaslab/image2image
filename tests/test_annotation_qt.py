"""Editor integration, real drawing commands, project persistence, and worker lifecycle."""

from __future__ import annotations

from pathlib import Path
from time import sleep
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from pytestqt.qtbot import QtBot
from qtextra.widgets.qt_table_view_check import QtCheckableTableView
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QApplication, QComboBox, QHeaderView, QStyle, QStyleOptionViewItem
from shapely.geometry import box

from image2image.cli import cli
from image2image.models.annotation import (
    AnnotationClass,
    AnnotationObject,
    EditCommand,
    FeatureSettings,
)
from image2image.qt._annotation_finetune import FineTuneDialog
from image2image.qt.dialog_annotate import ImageAnnotationWindow
from image2image.utils.annotation.ml import backend_available, train_classifier
from image2image.utils.annotation.model_cache import read_classifier_metadata
from image2image.utils.annotation.store import AnnotationStore


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
    assert left_layout.indexOf(window.overview_section) < left_layout.indexOf(window.sets)
    window.overview_section.collapse(animate=False)
    assert not window.overview_section.isExpanded()
    qtbot.waitUntil(lambda: window.overview.widget.visibleRegion().isEmpty())
    window.overview_section.expand(animate=False)
    assert window.overview_section.isExpanded()
    qtbot.waitUntil(lambda: not window.overview.widget.visibleRegion().isEmpty())
    monkeypatch.setattr("image2image.qt.dialog_annotate.backend_available", lambda name: name == "forest")
    window.tabs.setCurrentIndex(2)
    window.backend.setCurrentText("micro-sam")
    assert "Backend unavailable" in window.backend_hint.text()
    assert "annotate-sam" in window.backend_hint.text()
    assert window.backend_hint.isVisible()
    assert window.backend_hint.y() < window.model.y()
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


@pytest.mark.parametrize("mode", ["Freehand", "Brush +", "Brush -", "Alt brush"])
def test_canvas_drags_commit_without_panning(editor: ImageAnnotationWindow, qtbot: QtBot, mode: str) -> None:
    window = editor
    window.show()
    qtbot.waitExposed(window)
    window.render_timer.stop()
    qtbot.waitUntil(lambda: not window._threads, timeout=30000)
    original = box(5, 5, 120, 90)
    if mode in ("Brush -", "Alt brush"):
        key = window.add_geometry(original)
    window.tool.setCurrentText("Brush +" if mode == "Alt brush" else mode)
    canvas = window.view.widget.canvas.native
    center = QPoint(canvas.width() // 2, canvas.height() // 2)
    points = [center + QPoint(-25, -20), center + QPoint(25, -20), center + QPoint(25, 20), center + QPoint(-25, 20)]
    if mode == "Brush +":
        points.append(points[0])
    camera_center = tuple(window.view.viewer.camera.center)
    assert not window.view.viewer.camera.mouse_pan
    modifier = Qt.KeyboardModifier.AltModifier if mode == "Alt brush" else Qt.KeyboardModifier.NoModifier
    qtbot.mousePress(canvas, Qt.MouseButton.LeftButton, modifier, pos=points[0])
    if mode != "Freehand":
        assert len(window.brush_preview.data) > 0
        assert window.draw_layer.cursor_size == window.brush_radius.value() * 2
    for point in points[1:]:
        qtbot.mouseMove(canvas, point)
    assert window.store.count(window.set_id) == (1 if mode in ("Brush -", "Alt brush") else 0)
    qtbot.mouseRelease(canvas, Qt.MouseButton.LeftButton, modifier, pos=points[-1])
    assert not len(window.brush_preview.data)
    assert tuple(window.view.viewer.camera.center) == camera_center
    assert window.store.count(window.set_id) == 1, window.status.text()
    obj = window.store.get(window.selected[0])
    assert obj.geometry.is_valid
    if mode in ("Brush -", "Alt brush"):
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
    window.tool.setCurrentText("Select")
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
    assert "Slide 2/3" in editor.job_label.text()
    assert "ETA ~2 s" in editor.job_label.text()
    jobs._report_progress({"phase": "Publishing", "completed": 0, "total": None, "message": "Publishing"})
    assert details[-1]["eta"] is None
    assert details[-1]["elapsed"] == 4
    assert editor.job_bar.maximum() == 0
    jobs._poll()
    assert details[-1]["elapsed"] == 5
    assert "Elapsed 5 s" in editor.job_label.text()
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
    window.tool.setCurrentText("Select")
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
