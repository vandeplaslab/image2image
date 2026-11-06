"""Tests for the extract peaklist dialog."""

from pathlib import Path

import pytest
from pytestqt.qtbot import QtBot

from image2image.qt._dialogs._extract import ExtractChannelsDialog


def test_extract_dialog_reads_mz_peaklist(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Load m/z values from a CSV peaklist."""
    peaklist = tmp_path / "peaklist.csv"
    peaklist.write_text("mz\n100.1\n200.2\n", encoding="utf-8")

    monkeypatch.setattr("image2image.qt._dialogs._extract.hp.get_filename", lambda *args, **kwargs: peaklist)

    widget = ExtractChannelsDialog(None, key_to_extract="test")
    qtbot.addWidget(widget)

    widget.on_open_peaklist()

    assert widget.table.get_col_data(widget.TABLE_CONFIG.mz) == [100.1, 200.2]


def test_extract_dialog_accepts_current_values(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Extract the current peaklist and ppm after loading and editing them."""
    peaklist = tmp_path / "peaklist.csv"
    peaklist.write_text("m/z\n100.1\n200.2\n", encoding="utf-8")
    monkeypatch.setattr("image2image.qt._dialogs._extract.hp.get_filename", lambda *args, **kwargs: peaklist)
    monkeypatch.setattr("image2image.qt._dialogs._extract.hp.confirm", lambda *args, **kwargs: True)

    widget = ExtractChannelsDialog(None, key_to_extract="test")
    qtbot.addWidget(widget)
    widget.on_open_peaklist()
    widget.table.selectRow(0)
    widget.on_delete_row()
    widget.ppm_edit.setValue(25.0)
    widget.on_accept()

    assert widget.mzs == [200.2]
    assert widget.ppm == 25.0


def test_extract_dialog_does_not_accept_empty_peaklist(qtbot: QtBot, monkeypatch: pytest.MonkeyPatch) -> None:
    """Removing all peaks prevents extraction."""
    warnings = []
    monkeypatch.setattr("image2image.qt._dialogs._extract.hp.warn_pretty", lambda *args: warnings.append(args))
    widget = ExtractChannelsDialog(None, key_to_extract="test")
    qtbot.addWidget(widget)
    widget.mz_edit.setValue(100.0)
    widget.on_add()
    widget.table.selectRow(0)
    widget.on_delete_row()
    widget.on_accept()

    assert warnings
    assert widget.mzs == []
