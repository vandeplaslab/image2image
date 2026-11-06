"""Tests for extracting ion images from a dataset item."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from napari.layers import Image
from pytestqt.qtbot import QtBot
from qtpy.QtWidgets import QDialog, QWidget

from image2image.qt._dialogs._list import TABLE_CONFIG, QtDatasetItem


class FakeReader:
    """Reader that produces predictable ion images in a worker."""

    key = "sample"
    name = "sample"
    clean_key = "sample"
    path = Path("sample.d")
    reader_type = "image"
    allow_extraction = True
    resolution = 1.0
    shape = (2, 2)
    n_in_pyramid_quick = 1
    dtype = np.dtype("float32")

    def __init__(self) -> None:
        self.data = {"tic": np.ones((2, 2))}
        self.calls: list[tuple[list[float], float]] = []
        self.release = threading.Event()
        self.fail = False

    @property
    def channel_names(self) -> list[str]:
        """Return available channels."""
        return list(self.data)

    def extract(self, mzs: list[float], ppm: float) -> tuple[Path, list[str]]:
        """Create one image for each requested m/z."""
        self.calls.append((mzs, ppm))
        if not self.release.wait(timeout=5):
            raise TimeoutError("Extraction was not released by the test")
        if self.fail:
            raise ValueError("Reader failed")
        labels = []
        for mz in mzs:
            name = f"{mz:.1f}"
            self.data[name] = np.full((2, 2), ppm)
            labels.append(f"{name} | {self.name}")
        return self.path, labels

    def get_channel_pyramid(self, index: int) -> list[np.ndarray]:
        """Return the selected channel."""
        return [self.data[self.channel_names[index]]]

    def get_channel(self, index: int) -> np.ndarray:
        """Return the single-level image for a channel."""
        return self.get_channel_pyramid(index)[0]


@pytest.fixture
def extraction_item(qtbot: QtBot) -> tuple[QtDatasetItem, FakeReader, SimpleNamespace]:
    """Create a dataset item with a fake reader and view."""
    reader = FakeReader()
    parent = QWidget()
    parent.data_model = SimpleNamespace(get_reader_for_key=lambda key: reader)
    parent.view = SimpleNamespace(layers={})
    parent.config = None
    parent.transform_model = None
    qtbot.addWidget(parent)
    item = QtDatasetItem(reader.key, parent)
    qtbot.addWidget(item)
    return item, reader, parent.view


def fake_dialog(monkeypatch: pytest.MonkeyPatch, accepted: bool = True) -> None:
    """Replace the modal dialog with a predictable selection."""

    class Dialog:
        def __init__(self, parent: QWidget, key: str) -> None:
            self.mzs = [100.1, 200.2]
            self.ppm = 25.0

        def exec_(self) -> int:
            return QDialog.DialogCode.Accepted if accepted else QDialog.DialogCode.Rejected

    monkeypatch.setattr("image2image.qt._dialogs.ExtractChannelsDialog", Dialog)


def test_extraction_adds_and_refreshes_channels(
    extraction_item: tuple[QtDatasetItem, FakeReader, SimpleNamespace], qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Extracted channels are shown, and a repeated extraction refreshes the layer."""
    item, reader, view = extraction_item
    fake_dialog(monkeypatch)
    events = []

    def show_channel(state: bool, name: str) -> None:
        events.append((state, name))
        if name not in view.layers:
            channel = name.rsplit(" | ", 1)[0]
            view.layers[name] = Image(reader.get_channel_pyramid(reader.channel_names.index(channel)))

    item.evt_channel.connect(show_channel)
    item.on_extract()
    assert not item.extract_btn.isEnabled()
    qtbot.waitUntil(lambda: bool(reader.calls))
    assert reader.calls == [([100.1, 200.2], 25.0)]
    reader.release.set()
    qtbot.waitUntil(item.extract_btn.isEnabled)

    assert events == [(True, "100.1 | sample"), (True, "200.2 | sample")]
    assert item.table.get_col_data(TABLE_CONFIG.channel_name) == ["tic", "100.1", "200.2"]
    np.testing.assert_array_equal(view.layers["100.1 | sample"].data, np.full((2, 2), 25.0))

    view.layers["100.1 | sample"].data = np.zeros((2, 2))
    item.on_extract()
    qtbot.waitUntil(item.extract_btn.isEnabled)
    assert len(reader.calls) == 2
    assert item.table.get_col_data(TABLE_CONFIG.channel_name) == ["tic", "100.1", "200.2"]
    np.testing.assert_array_equal(view.layers["100.1 | sample"].data, np.full((2, 2), 25.0))


def test_extraction_cancellation_does_not_start_worker(
    extraction_item: tuple[QtDatasetItem, FakeReader, SimpleNamespace], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancelling the modal dialog leaves the reader untouched."""
    item, reader, _ = extraction_item
    fake_dialog(monkeypatch, accepted=False)
    item.on_extract()
    assert not reader.calls
    assert item.extract_btn.isEnabled()


def test_extraction_failure_restores_button(
    extraction_item: tuple[QtDatasetItem, FakeReader, SimpleNamespace], qtbot: QtBot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Worker failures are reported and do not add channels."""
    item, reader, _ = extraction_item
    fake_dialog(monkeypatch)
    messages = []
    errors = []
    monkeypatch.setattr("image2image.qt._dialogs._list.hp.get_main_window", lambda: None)
    monkeypatch.setattr("image2image.qt._dialogs._list.hp.toast", lambda *args: messages.append(args))
    monkeypatch.setattr("image2image.qt._dialogs._list.logger.exception", lambda error: errors.append(error))
    reader.fail = True
    reader.release.set()
    item.on_extract()
    qtbot.waitUntil(item.extract_btn.isEnabled)
    assert messages
    assert len(errors) == 1 and str(errors[0]) == "Reader failed"
    assert item.table.get_col_data(TABLE_CONFIG.channel_name) == ["tic"]
