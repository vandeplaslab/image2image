"""Tests for the registration runner dialog helpers."""

from __future__ import annotations

import typing as ty
from types import SimpleNamespace

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage

from image2image.qt import dialog_runner
from image2image.qt._runner._card import QtRunnerProjectCard
from image2image.qt._runner.utilities import (
    discover_overlap_images,
    has_registration_images,
    project_matches_filters,
    read_review_state,
    write_review_state,
)
from image2image.qt.dialog_runner import (
    ImageRunnerWindow,
    RunnerProject,
)


def test_has_registration_images(tmp_path) -> None:
    """Detect image output files in the project Images directory."""
    assert not has_registration_images(tmp_path)

    image_dir = tmp_path / "Images"
    image_dir.mkdir()
    assert not has_registration_images(tmp_path)

    (image_dir / "nested").mkdir()
    assert not has_registration_images(tmp_path)

    (image_dir / "registered.ome.tiff").write_text("image")
    assert has_registration_images(tmp_path)


def test_review_state_roundtrip_and_invalid(tmp_path) -> None:
    """Load and save runner review sidecar state."""
    assert read_review_state(tmp_path) == "unknown"

    write_review_state(tmp_path, "bad")
    assert read_review_state(tmp_path) == "bad"

    (tmp_path / ".image2image-runner-review.json").write_text("{")
    assert read_review_state(tmp_path) == "unknown"

    (tmp_path / ".image2image-runner-review.json").write_text('{"review_state": "maybe"}')
    assert read_review_state(tmp_path) == "unknown"


def test_project_matches_filters() -> None:
    """Match projects by name, run state, and review state."""
    assert project_matches_filters("sample", "Finished", "good", "sam", "Finished", "Good")
    assert project_matches_filters("sample", "In progress", "unknown", "", "Running", "Unknown")
    assert project_matches_filters("sample", "Already queued", "bad", "", "Queued", "Bad")
    assert project_matches_filters("sample", "Invalid", "bad", "", "Failed", "All")
    assert not project_matches_filters("sample", "Finished", "good", "other", "All", "All")
    assert not project_matches_filters("sample", "Finished", "good", "", "Running", "All")
    assert not project_matches_filters("sample", "Finished", "good", "", "All", "Bad")


def test_discover_overlap_images(tmp_path) -> None:
    """Return sorted overlap PNG files."""
    overlap_dir = tmp_path / "Overlap"
    overlap_dir.mkdir()
    (overlap_dir / "b.png").write_text("b")
    (overlap_dir / "a.png").write_text("a")
    (overlap_dir / "c.txt").write_text("c")

    assert [path.name for path in discover_overlap_images(tmp_path)] == ["a.png", "b.png"]


def _make_runner_project(tmp_path) -> RunnerProject:
    """Create a minimal runner project for card tests."""
    return RunnerProject(
        "elastix",
        tmp_path,
        ty.cast(
            ty.Any,
            SimpleNamespace(name="demo", modalities={}, output_dir=tmp_path, project_dir=tmp_path),
        ),
    )


def test_project_card_overlap_preview_navigation(qtbot, tmp_path) -> None:
    """Display sorted overlap previews and navigate between them by chevron."""
    overlap_dir = tmp_path / "Overlap"
    overlap_dir.mkdir()
    for index, name in enumerate(("b.png", "a.png")):
        path = overlap_dir / name
        image = QImage(12, 12, QImage.Format.Format_RGB32)
        image.fill(index)
        assert image.save(str(path))

    card = QtRunnerProjectCard(_make_runner_project(tmp_path))
    qtbot.addWidget(card)

    assert card.current_overlap_path == overlap_dir / "a.png"
    assert card.prev_overlap_btn.isEnabled() is False
    assert card.next_overlap_btn.isEnabled() is True
    qtbot.mouseClick(card.next_overlap_btn, Qt.MouseButton.LeftButton)
    assert card.current_overlap_path == overlap_dir / "b.png"
    assert card.prev_overlap_btn.isEnabled() is True
    assert card.next_overlap_btn.isEnabled() is False


def test_project_card_refreshes_new_overlap_previews(qtbot, tmp_path) -> None:
    """Show overlap previews created after a card was loaded."""
    card = QtRunnerProjectCard(_make_runner_project(tmp_path))
    qtbot.addWidget(card)

    assert card.current_overlap_path is None
    assert card.overlap_unavailable_label.isHidden() is False

    overlap_dir = tmp_path / "Overlap"
    overlap_dir.mkdir()
    image = QImage(12, 12, QImage.Format.Format_RGB32)
    image.fill(0)
    preview_path = overlap_dir / "preview.png"
    assert image.save(str(preview_path))

    card.set_status("Finished")

    assert card.current_overlap_path == preview_path
    assert card.overlap_image_label.isHidden() is False


def test_remove_project_unloads_card_without_deleting_files(qtbot, tmp_path) -> None:
    """Unload a finished card while preserving its registration directory."""
    window = ImageRunnerWindow(None, run_check_version=False)
    qtbot.addWidget(window)
    project = _make_runner_project(tmp_path)
    window.projects[tmp_path] = project
    window._add_project_card(project)
    window.cards[tmp_path].set_status("Finished")

    window.on_remove_project(tmp_path)

    assert tmp_path.exists()
    assert tmp_path not in window.projects
    assert tmp_path not in window.cards


def test_remove_queued_project_removes_pending_task(qtbot, tmp_path, monkeypatch) -> None:
    """Remove a queued task before unloading its project card."""
    window = ImageRunnerWindow(None, run_check_version=False)
    qtbot.addWidget(window)
    project = _make_runner_project(tmp_path)
    window.projects[tmp_path] = project
    window._add_project_card(project)
    window.cards[tmp_path].set_status("Queued")
    window.task_to_project["task-1"] = tmp_path

    removed: list[str] = []
    queue = SimpleNamespace(pending_queue=["task-1"], running_queue=[], remove_task=removed.append)
    monkeypatch.setattr(dialog_runner, "QUEUE", queue)

    window.on_remove_project(tmp_path)

    assert removed == ["task-1"]
    assert "task-1" not in window.task_to_project
    assert tmp_path not in window.cards


def test_running_project_cannot_be_removed(qtbot, tmp_path) -> None:
    """Keep a running project card loaded and disable its removal action."""
    window = ImageRunnerWindow(None, run_check_version=False)
    qtbot.addWidget(window)
    project = _make_runner_project(tmp_path)
    window.projects[tmp_path] = project
    window._add_project_card(project)
    card = window.cards[tmp_path]
    card.set_status("Running")

    assert card.remove_btn.isEnabled() is False
    window.on_remove_project(tmp_path)

    assert tmp_path in window.projects
    assert tmp_path in window.cards


def test_status_bar_shows_project_state_counts(qtbot, tmp_path) -> None:
    """Show aggregate project states in the runner status bar."""
    window = ImageRunnerWindow(None, run_check_version=False)
    qtbot.addWidget(window)

    assert window.status_counts_label.text() == "Queued: 0, Running: 0, Finished: 0, Failed: 0"

    statuses = ["Queued", "Already queued", "Running", "In progress", "Finished", "Failed", "Invalid"]
    for index, status in enumerate(statuses):
        project_dir = tmp_path / f"project-{index}"
        project = _make_runner_project(project_dir)
        window.projects[project_dir] = project
        window._add_project_card(project)
        window.cards[project_dir].set_status(status)

    window._refresh_progress_report()

    assert window.status_counts_label.text() == "Queued: 2, Running: 2, Finished: 1, Failed: 2"
