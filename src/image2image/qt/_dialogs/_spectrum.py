"""Mass spectrum dialog."""

from __future__ import annotations

import contextlib
import typing as ty

import numpy as np
from image2image_io.utils.utilities import format_mz
from ims_utils.spectrum import get_window_for_ppm
from loguru import logger
from qtextra import helpers as hp
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtpy.QtCore import Qt, QTimer, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QApplication, QFormLayout
from superqt.utils import create_worker, qdebounced

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader

    from image2image.qt._dialogs._list import QtDatasetItem

# cache of average spectra, keyed by the path of the dataset
AVERAGE_SPECTRUM_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}
# number of screen pixels around the click position that are searched for a peak
CLICK_TOLERANCE_PX = 5
# maximum distance (in screen pixels) between press and release to treat the event as a click
CLICK_MAX_MOVE_PX = 3
BAND_COLOR = "#00a6ff"


def to_centroid_sticks(mz: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert centroid data to line representation so that each peak is drawn as a vertical stick."""
    x_sticks = np.repeat(mz, 3)
    y_sticks = np.zeros(len(x_sticks), dtype=np.float32)
    y_sticks[1::3] = y
    return x_sticks, y_sticks


def find_peak_near(mz: np.ndarray, y: np.ndarray, mz_click: float, tol: float) -> float:
    """Find the most intense peak within the tolerance window around the clicked m/z."""
    if len(mz) == 0:
        return mz_click
    mask = (mz >= mz_click - tol) & (mz <= mz_click + tol)
    if not np.any(mask) or np.nanmax(y[mask]) <= 0:
        return mz_click
    indices = np.flatnonzero(mask)
    return float(mz[indices[np.nanargmax(y[mask])]])


class SpectrumDialog(QtFramelessTool):
    """Dialog to show mass spectrum and extract ion images by clicking on peaks."""

    evt_iter_remove = Signal(str, int)
    evt_iter_next = Signal(str, int)
    evt_iter_add = Signal(str, int)
    evt_channel = Signal(bool, str)  # channel | dataset

    _mz: np.ndarray | None = None
    _y: np.ndarray | None = None
    _selected_mz: float | None = None
    _pending: str | None = None
    _queued_mz: float | None = None
    _is_extracting: bool = False
    _is_closed: bool = False
    _press_xy: tuple[float, float] | None = None
    _click_mz: float | None = None

    def __init__(self, parent: QtDatasetItem):
        self.key = parent.key
        self._item = parent
        self.reader: BaseReader = parent.get_model()
        # channels that existed before the dialog opened or were added by the user - these are never removed
        self._kept: set[str] = set(self.reader.channel_names)
        self._band: list = []
        super().__init__(parent)
        self.setMinimumSize(700, 450)
        self.evt_close.connect(self.on_close)

        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._on_delayed_click)
        canvas = self.view.figure.canvas
        canvas.mpl_connect("button_press_event", self._on_mpl_press)
        canvas.mpl_connect("button_release_event", self._on_mpl_release)
        hp.call_later(self, self.on_update_spectrum, 50)

    @property
    def imzy_reader(self) -> ty.Any:
        """Return the underlying imzy reader."""
        return self.reader.reader  # type: ignore[attr-defined]

    @property
    def is_centroid(self) -> bool:
        """Return whether the dataset is centroided."""
        try:
            return bool(self.imzy_reader.is_centroid)
        except Exception:  # noqa: BLE001
            return False

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Setup UI."""
        _, header_layout = self._make_close_handle(title=f"Mass spectrum - {self.reader.clean_key}")

        n_pixels = int(self.imzy_reader.n_pixels)
        self.pixel_spin = hp.make_int_spin_box(
            self,
            minimum=0,
            maximum=max(n_pixels - 1, 0),
            value=0,
            tooltip="Index of the pixel (frame) whose mass spectrum should be shown.",
            func=self.on_change_pixel,
        )
        self.pixel_label = hp.make_label(self, "", tooltip="Coordinates of the pixel.")
        self.average_check = hp.make_checkbox(
            self,
            "Average spectrum",
            tooltip="Show the average mass spectrum of all pixels instead of a single pixel.<br>"
            "This might take a while for large datasets.",
            func=self.on_update_spectrum,
        )
        self.ppm_spin = hp.make_double_spin_box(
            self,
            minimum=0.5,
            maximum=500,
            value=5,
            step_size=1,
            n_decimals=1,
            suffix=" ppm",
            tooltip="Integration window (in ppm) used when extracting the ion image.",
            func=self.on_change_ppm,
        )
        from qtextraplot._mpl.views import ViewMplLine

        self.view = ViewMplLine(self, x_label="m/z", y_label="Intensity")
        self.mz_label = hp.make_label(self, "No peak selected.", tooltip="Currently selected peak.")
        self.add_btn = hp.make_btn(
            self, "Add to viewer", tooltip="Add the previewed ion image to the dataset.", func=self.on_add
        )
        self.add_btn.setEnabled(False)

        layout = hp.make_form_layout(margin=6)
        layout.addRow(header_layout)
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "Pixel"),
                self.pixel_spin,
                self.pixel_label,
                hp.make_v_line(),
                self.average_check,
                hp.make_v_line(),
                hp.make_label(self, "Window"),
                self.ppm_spin,
                stretch_id=(2,),
            )
        )
        layout.addRow(self.view.figure)
        layout.addRow(hp.make_h_layout(self.mz_label, self.add_btn, stretch_id=(0,)))
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Click on a peak to preview its ion image. Drag to zoom, double-click to reset.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
            )
        )
        return layout

    @qdebounced(timeout=150, leading=False)
    def on_change_pixel(self, _: ty.Any = None) -> None:
        """Change pixel."""
        self.on_update_spectrum()

    def on_update_spectrum(self, _: ty.Any = None) -> None:
        """Update the displayed mass spectrum."""
        is_average = self.average_check.isChecked()
        self.pixel_spin.setEnabled(not is_average)
        if is_average:
            self._show_average_spectrum()
        else:
            self._show_pixel_spectrum(self.pixel_spin.value())

    def _show_pixel_spectrum(self, index: int) -> None:
        reader = self.imzy_reader
        x, y = reader.x_coordinates[index], reader.y_coordinates[index]
        self.pixel_label.setText(f"x={x}, y={y}")
        mz, intensity = reader.get_spectrum(index)
        self._show_spectrum(mz, intensity, as_centroid=self.is_centroid)

    def _show_average_spectrum(self) -> None:
        cache_key = str(self.reader.path)
        if cache_key in AVERAGE_SPECTRUM_CACHE:
            self._show_spectrum(*AVERAGE_SPECTRUM_CACHE[cache_key], as_centroid=False)
            return

        self.average_check.setEnabled(False)
        self.pixel_label.setText("Computing average...")
        create_worker(
            self._compute_average_spectrum,
            _start_thread=True,
            _connect={"returned": self._on_average_spectrum, "errored": self._on_average_failed},
        )

    def _compute_average_spectrum(self) -> tuple[str, np.ndarray, np.ndarray]:
        reader = self.imzy_reader
        mz, intensity = reader.get_summed_spectrum(reader.pixels, silent=True)
        return str(self.reader.path), mz, intensity / max(reader.n_pixels, 1)

    def _on_average_spectrum(self, result: tuple[str, np.ndarray, np.ndarray]) -> None:
        cache_key, mz, intensity = result
        AVERAGE_SPECTRUM_CACHE[cache_key] = (mz, intensity)
        self.average_check.setEnabled(True)
        self.pixel_label.setText("")
        if self.average_check.isChecked():
            self._show_spectrum(mz, intensity, as_centroid=False)

    def _on_average_failed(self, exc: Exception) -> None:
        logger.exception(exc)
        self.average_check.setEnabled(True)
        self.pixel_label.setText("")
        hp.toast(hp.get_main_window(), "Failed to compute average", "Failed to compute average mass spectrum.")

    def _show_spectrum(self, mz: np.ndarray, intensity: np.ndarray, as_centroid: bool) -> None:
        self._mz, self._y = np.asarray(mz), np.asarray(intensity)
        x, y = to_centroid_sticks(self._mz, self._y) if as_centroid else (self._mz, self._y)
        self._remove_band()
        self.view.plot(x, y)
        self._draw_band()

    def _remove_band(self) -> None:
        for artist in self._band:
            with contextlib.suppress(ValueError, NotImplementedError):
                artist.remove()
        self._band.clear()

    def _draw_band(self) -> None:
        """Draw integration window around the selected peak."""
        self._remove_band()
        if self._selected_mz is not None:
            mz = self._selected_mz
            window = get_window_for_ppm(mz, self.ppm_spin.value())
            ax = self.view.figure.ax
            self._band.append(ax.axvspan(mz - window, mz + window, color=BAND_COLOR, alpha=0.3, zorder=1))
            self._band.append(ax.axvline(mz, color=BAND_COLOR, alpha=0.6, lw=1, zorder=2))
        self.view.figure.repaint()

    def _click_tolerance(self) -> float:
        """Return tolerance (in m/z) that corresponds to few screen pixels at the current zoom level."""
        ax = self.view.figure.ax
        xmin, xmax = ax.get_xlim()
        width = max(ax.bbox.width, 1)
        return abs(xmax - xmin) / width * CLICK_TOLERANCE_PX

    def _on_mpl_press(self, evt: ty.Any) -> None:
        if evt.dblclick:
            # double-click resets zoom, so cancel any pending click
            self._click_timer.stop()
            self._press_xy = None
            return
        self._press_xy = (evt.x, evt.y) if evt.button == 1 else None

    def _on_mpl_release(self, evt: ty.Any) -> None:
        if self._press_xy is None or evt.button != 1 or evt.xdata is None:
            return
        px, py = self._press_xy
        self._press_xy = None
        if abs(evt.x - px) > CLICK_MAX_MOVE_PX or abs(evt.y - py) > CLICK_MAX_MOVE_PX:
            return
        self._click_mz = float(evt.xdata)
        self._click_timer.start(QApplication.doubleClickInterval())

    def _on_delayed_click(self) -> None:
        if self._click_mz is not None:
            self.on_select_mz(self._click_mz)
            self._click_mz = None

    def on_select_mz(self, mz_click: float) -> None:
        """Select peak closest to the clicked m/z and preview its ion image."""
        if self._mz is None or self._y is None:
            return
        window = get_window_for_ppm(mz_click, self.ppm_spin.value())
        tol = max(3 * window, self._click_tolerance())
        self._selected_mz = find_peak_near(self._mz, self._y, mz_click, tol)
        self._update_mz_label()
        self._draw_band()
        self._on_preview(self._selected_mz)

    def on_change_ppm(self, _: ty.Any = None) -> None:
        """Update integration window."""
        if self._selected_mz is None:
            return
        self._update_mz_label()
        self._draw_band()
        self._on_preview(self._selected_mz)

    def _update_mz_label(self) -> None:
        if self._selected_mz is None:
            self.mz_label.setText("No peak selected.")
            return
        mz = self._selected_mz
        window = get_window_for_ppm(mz, self.ppm_spin.value())
        self.mz_label.setText(f"<b>m/z {mz:.4f}</b> ± {window:.4f} Da ({self.ppm_spin.value():.1f} ppm)")

    def _drop_pending(self) -> None:
        """Remove previewed ion image that was not added by the user."""
        if self._pending and self._pending not in self._kept:
            self.reader.data.pop(self._pending, None)  # type: ignore[attr-defined]
            self.reader.get_image.cache_clear()  # type: ignore[attr-defined]
            logger.trace(f"Removed preview '{self._pending}' from '{self.key}'.")
        self._pending = None
        self.add_btn.setEnabled(False)

    def _on_preview(self, mz: float) -> None:
        """Extract ion image in a separate thread and show it as temporary layer."""
        if self._is_closed:
            return
        if self._is_extracting:
            self._queued_mz = mz
            return
        name = format_mz(mz)
        if name in self._kept:
            # image already exists - show it instead of extracting it again
            self._drop_pending()
            self._show_preview(name)
            return

        self._is_extracting = True
        self.add_btn.setEnabled(False)
        create_worker(
            self._extract_ion_image,
            mz=mz,
            ppm=self.ppm_spin.value(),
            _start_thread=True,
            _connect={"returned": self._on_extracted, "errored": self._on_extract_failed},
        )

    def _extract_ion_image(self, mz: float, ppm: float) -> tuple[str, np.ndarray]:
        return format_mz(mz), self.imzy_reader.get_ion_image(mz, ppm=ppm)

    def _on_extracted(self, result: tuple[str, np.ndarray]) -> None:
        name, image = result
        self._is_extracting = False
        if self._is_closed:
            return
        if name not in self._kept:
            if self._pending != name:
                self._drop_pending()
            self.reader.data[name] = image  # type: ignore[attr-defined]
            self.reader.get_image.cache_clear()  # type: ignore[attr-defined]
            self._pending = name
        self._show_preview(name)
        self._run_queued()

    def _on_extract_failed(self, exc: Exception) -> None:
        logger.exception(exc)
        self._is_extracting = False
        hp.toast(hp.get_main_window(), "Failed to extract data.", "Failed to extract ion image.")
        self._run_queued()

    def _run_queued(self) -> None:
        if self._queued_mz is not None:
            mz, self._queued_mz = self._queued_mz, None
            self._on_preview(mz)

    def _show_preview(self, name: str) -> None:
        index = self.reader.channel_names.index(name)
        self.add_btn.setEnabled(name not in self._kept)
        self.evt_iter_next.emit(self.key, index)
        logger.trace(f"Previewing '{name}' for '{self.key}'.")

    def on_add(self) -> None:
        """Add previewed ion image to the dataset."""
        name = self._pending
        if not name:
            return
        index = self.reader.channel_names.index(name)
        self._kept.add(name)
        self._pending = None
        self.add_btn.setEnabled(False)
        self.evt_iter_add.emit(self.key, index)
        self._item.add_channel_row(name)
        self.evt_channel.emit(True, f"{name} | {self.key}")
        logger.trace(f"Added '{name}' to '{self.key}'.")

    def on_close(self) -> None:
        """Close event."""
        self._is_closed = True
        self._queued_mz = None
        self._click_timer.stop()
        self.evt_iter_remove.emit(self.key, 0)
        self._drop_pending()
