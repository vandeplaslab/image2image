"""Tests for the registration runner dialog helpers."""

from __future__ import annotations

import typing as ty
from types import SimpleNamespace

from qtpy.QtCore import Qt
from qtpy.QtGui import QImage

from image2image.qt.dialog_runner import (
    OverlapPreviewDialog,
    RunnerProject,
    discover_overlap_images,
    has_registration_images,
    project_matches_filters,
    read_review_state,
    write_review_state,
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


def test_overlap_preview_dialog_arrow_navigation(qtbot, tmp_path) -> None:
    """Move between overlap preview images with arrow keys."""
    image_paths = []
    for index in range(2):
        path = tmp_path / f"preview_{index}.png"
        image = QImage(12, 12, QImage.Format.Format_RGB32)
        image.fill(index)
        assert image.save(str(path))
        image_paths.append(path)

    project = RunnerProject(
        "elastix",
        tmp_path,
        ty.cast(ty.Any, SimpleNamespace(name="demo")),
    )
    dialog = OverlapPreviewDialog(project, image_paths, "unknown")
    qtbot.addWidget(dialog)

    assert dialog.image_list.currentRow() == 0
    qtbot.keyClick(dialog, Qt.Key.Key_Right)
    assert dialog.image_list.currentRow() == 1
    qtbot.keyClick(dialog, Qt.Key.Key_Left)
    assert dialog.image_list.currentRow() == 0
