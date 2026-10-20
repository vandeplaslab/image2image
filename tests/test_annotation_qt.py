"""Editor integration, real drawing commands, project persistence, and worker lifecycle."""

from __future__ import annotations

from pathlib import Path
from time import sleep

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from pytestqt.qtbot import QtBot
from qtpy.QtCore import QPoint, Qt
from shapely.geometry import box

from image2image.cli import cli
from image2image.models.annotation import AnnotationClass, AnnotationObject, EditCommand, FeatureSettings
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
    assert window.proposals.item(0, 1).text() in {"Tissue", "Stroma"}
    assert window.proposals.item(0, 3).text() == "Needs review"
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
    cell = window.objects.item(0, 3)
    assert cell.checkState() == Qt.CheckState.Checked
    cell.setCheckState(Qt.CheckState.Unchecked)
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
    cell = window.objects.item(1, 3)
    assert not cell.flags() & Qt.ItemFlag.ItemIsEnabled
    assert "Ignore mask" in cell.toolTip()


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
    assert window.proposals.rowCount() == 30
    window.next_page("accepted")
    assert window.objects.rowCount() == 25
    assert previous.isEnabled() and not next_.isEnabled()
    window.next_page("accepted")
    assert window.objects.rowCount() == 25
    window.previous_page("accepted")
    assert window.objects.rowCount() == 100
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


def test_download_cancel_cleans_own_files_clears_queue_and_restarts_worker(
    editor: ImageAnnotationWindow, qtbot: QtBot, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window, jobs = editor, editor.jobs
    monkeypatch.setattr("image2image.utils.annotation.model_cache.USER_CACHE_DIR", tmp_path)
    monkeypatch.setattr("image2image.qt.dialog_annotate.backend_available", lambda _backend: True)
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
    assert window.task_table.rowCount() == 2
    assert "Download model" in window.task_table.item(0, 1).text()
    assert "Queued" in window.task_table.item(1, 0).text()
    assert "Cancel download" in window.download_btn.toolTip()
    qtbot.mouseClick(window.download_btn, Qt.MouseButton.LeftButton)
    assert jobs.cancelling
    assert window.task_table.item(0, 0).text() == "Cancelling"
    qtbot.waitUntil(lambda: not jobs.busy, timeout=5000)
    assert not worker.is_alive()
    assert not own.exists() and other.read_bytes() == b"other job"
    assert not jobs.pending and not window._catalogue_pending
    assert window.task_table.rowCount() == 0
    assert "idle" in window.queue_summary.text()
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
