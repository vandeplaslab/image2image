"""Tests for the extract peaklist dialog."""

from image2image.qt._dialogs._extract import ExtractChannelsDialog


def test_extract_dialog_reads_mz_peaklist(qtbot, monkeypatch, tmp_path) -> None:
    """Load m/z values from a CSV peaklist."""
    peaklist = tmp_path / "peaklist.csv"
    peaklist.write_text("mz\n100.1\n200.2\n", encoding="utf-8")

    monkeypatch.setattr("image2image.qt._dialogs._extract.hp.get_filename", lambda *args, **kwargs: peaklist)

    widget = ExtractChannelsDialog(None, key_to_extract="test")
    qtbot.addWidget(widget)

    widget.on_open_peaklist()

    assert widget.table.get_col_data(widget.TABLE_CONFIG.mz) == [100.1, 200.2]
