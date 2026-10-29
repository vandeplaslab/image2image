"""Tests for the mass spectrum dialog."""

from __future__ import annotations

import typing as ty
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from qtpy.QtWidgets import QWidget

from image2image.qt._dialogs._spectrum import (
    AVERAGE_SPECTRUM_CACHE,
    SpectrumDialog,
    find_peak_near,
    to_centroid_sticks,
)

N_PIXELS = 6
SHAPE = (2, 3)


class FakeImzyReader:
    """Minimal imzy reader."""

    def __init__(self, is_centroid: bool) -> None:
        self.is_centroid = is_centroid
        self.n_pixels = N_PIXELS
        self.pixels = np.arange(N_PIXELS)
        self.x_coordinates = np.tile(np.arange(SHAPE[1]), SHAPE[0])
        self.y_coordinates = np.repeat(np.arange(SHAPE[0]), SHAPE[1])
        self.ion_image_calls: list[tuple[float, float]] = []

    def get_spectrum(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return spectrum with peak position that depends on the pixel index."""
        if self.is_centroid:
            mz = np.asarray([100.0, 200.0 + index, 300.0])
            return mz, np.asarray([1.0, 10.0, 5.0])
        mz = np.linspace(100, 300, 2001)
        y = np.exp(-((mz - (200.0 + index)) ** 2) / 0.01)
        return mz, y

    def get_summed_spectrum(self, indices: ty.Iterable[int], silent: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return summed spectrum."""
        mz = np.linspace(100, 300, 2001)
        return mz, np.full(len(mz), float(N_PIXELS))

    def get_ion_image(self, mz: float, ppm: float) -> np.ndarray:
        """Return ion image."""
        self.ion_image_calls.append((mz, ppm))
        return np.full(SHAPE, mz, dtype=np.float32)


class FakeReader:
    """Minimal `CoordinateImageReader`."""

    def __init__(self, is_centroid: bool) -> None:
        self.reader = FakeImzyReader(is_centroid)
        self.data = {"tic": np.ones(SHAPE)}
        self.path = Path(f"/fake/{'centroid' if is_centroid else 'profile'}.imzML")
        self.clean_key = "fake"

    @property
    def channel_names(self) -> list[str]:
        """Channel names."""
        return list(self.data.keys())

    @lru_cache(maxsize=1)  # noqa: B019
    def get_image(self) -> np.ndarray:
        """Return image."""
        return np.dstack(list(self.data.values()))


class FakeItem(QWidget):
    """Minimal `QtDatasetItem`."""

    def __init__(self, reader: FakeReader) -> None:
        super().__init__()
        self.key = "fake"
        self.reader = reader
        self.rows: list[str] = []

    def get_model(self) -> FakeReader:
        """Return reader."""
        return self.reader

    def add_channel_row(self, channel_name: str) -> None:
        """Record added channel."""
        self.rows.append(channel_name)


@pytest.fixture
def make_dialog(qtbot):
    """Make dialog with recorded signals."""

    def _make(is_centroid: bool = True) -> tuple[SpectrumDialog, FakeItem, dict[str, list]]:
        item = FakeItem(FakeReader(is_centroid))
        qtbot.addWidget(item)
        dlg = SpectrumDialog(item)
        qtbot.addWidget(dlg)
        events: dict[str, list] = {"next": [], "add": [], "remove": [], "channel": []}
        dlg.evt_iter_next.connect(lambda *args: events["next"].append(args))
        dlg.evt_iter_add.connect(lambda *args: events["add"].append(args))
        dlg.evt_iter_remove.connect(lambda *args: events["remove"].append(args))
        dlg.evt_channel.connect(lambda *args: events["channel"].append(args))
        dlg.on_update_spectrum()
        return dlg, item, events

    return _make


def _main_line_x(dlg: SpectrumDialog) -> np.ndarray:
    return np.asarray(dlg.view.figure.ax.get_lines()[0].get_xdata())


def test_to_centroid_sticks() -> None:
    """Each centroid is converted to a vertical stick."""
    x, y = to_centroid_sticks(np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]))
    np.testing.assert_array_equal(x, [1, 1, 1, 2, 2, 2])
    np.testing.assert_array_equal(y, [0, 3, 0, 0, 4, 0])


def test_find_peak_near() -> None:
    """Most intense peak within the window is selected."""
    mz = np.asarray([100.0, 100.5, 101.0, 200.0])
    y = np.asarray([1.0, 5.0, 2.0, 10.0])
    assert find_peak_near(mz, y, 100.2, 1.0) == 100.5
    # nothing within the window, so the clicked value is returned
    assert find_peak_near(mz, y, 150.0, 1.0) == 150.0


def test_spectrum_dialog_changes_pixel(make_dialog) -> None:
    """Changing pixel index shows that pixel's spectrum."""
    dlg, _, _ = make_dialog(is_centroid=True)
    assert 200.0 in _main_line_x(dlg)
    dlg.pixel_spin.setValue(3)
    dlg.on_update_spectrum()
    assert 203.0 in _main_line_x(dlg)
    assert dlg.pixel_label.text() == "x=0, y=1"


def test_spectrum_dialog_average(make_dialog, qtbot) -> None:
    """Average spectrum is computed in a thread and divided by the number of pixels."""
    AVERAGE_SPECTRUM_CACHE.clear()
    dlg, item, _ = make_dialog(is_centroid=False)
    dlg.average_check.setChecked(True)
    assert not dlg.pixel_spin.isEnabled()
    qtbot.waitUntil(lambda: str(item.reader.path) in AVERAGE_SPECTRUM_CACHE, timeout=5000)
    np.testing.assert_allclose(dlg._y, 1.0)
    dlg.average_check.setChecked(False)
    assert dlg.pixel_spin.isEnabled()


def test_spectrum_dialog_click_preview_and_add(make_dialog, qtbot) -> None:
    """Clicking near a peak previews its ion image and it can be added to the dataset."""
    dlg, item, events = make_dialog(is_centroid=True)
    reader = item.reader

    dlg.on_select_mz(200.002)
    assert dlg._selected_mz == 200.0
    qtbot.waitUntil(lambda: "m/z 200.000" in reader.data, timeout=5000)
    assert reader.reader.ion_image_calls == [(200.0, 5.0)]
    assert events["next"] == [("fake", 1)]
    assert dlg.add_btn.isEnabled()

    dlg.on_add()
    assert events["add"] == [("fake", 1)]
    assert events["channel"] == [(True, "m/z 200.000 | fake")]
    assert item.rows == ["m/z 200.000"]
    assert not dlg.add_btn.isEnabled()

    # closing keeps the added channel
    dlg.on_close()
    assert "m/z 200.000" in reader.data


def test_spectrum_dialog_replaces_and_drops_pending(make_dialog, qtbot) -> None:
    """New preview replaces the previous one and closing removes previews that were not added."""
    dlg, item, events = make_dialog(is_centroid=False)
    reader = item.reader

    dlg.on_select_mz(200.01)
    qtbot.waitUntil(lambda: len(reader.data) == 2, timeout=5000)
    first = dlg._pending
    assert first is not None
    assert abs(dlg._selected_mz - 200.0) < 0.05

    # changing the window re-extracts the same peak
    dlg.ppm_spin.setValue(10)
    qtbot.waitUntil(lambda: len(reader.reader.ion_image_calls) == 2, timeout=5000)
    assert reader.reader.ion_image_calls[-1][1] == 10
    assert list(reader.data) == ["tic", first]

    dlg.on_select_mz(299.0)
    qtbot.waitUntil(lambda: dlg._pending != first and dlg._pending is not None, timeout=5000)
    assert first not in reader.data
    assert len(reader.data) == 2

    dlg.on_close()
    assert list(reader.data) == ["tic"]
    assert events["remove"] == [("fake", 0)]
