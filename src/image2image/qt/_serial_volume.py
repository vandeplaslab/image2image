"""Linked orthogonal views and a bounded 3D preview of serial volumes."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import dask.array as da
import numpy as np
import qtextra.helpers as hp
from image2image_io.volume import Volume, open_volume
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import QTimer
from qtpy.QtGui import QShowEvent
from qtpy.QtWidgets import QGridLayout, QVBoxLayout, QWidget
from superqt.utils import create_worker

PreviewResult = tuple[np.ndarray, tuple[float, ...], tuple[float, float]]


class SerialVolumeWidget(QWidget):
    """Explore calibrated slices without materializing the full volume."""

    PREVIEW_BYTES = 64 * 1024 * 1024

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.volume: Volume | None = None
        self._syncing = False
        self._generation = 0
        self._workers: list[Any] = []
        self._pending_preview: tuple[Callable[[], PreviewResult], int, bool] | None = None
        layout = QVBoxLayout(self)
        self.channel = hp.make_combobox(self, func_index=self.show_channel)
        self.coordinates = hp.make_label(self, "Open a reconstructed volume to inspect its physical coordinates.")
        self.preview_label = hp.make_label(self, "Downsampled 3D preview (maximum 64 MiB)")
        layout.addLayout(hp.make_h_layout(hp.make_btn(self, "Open volume...", func=self.choose_volume), self.channel))
        self.contrast = [
            hp.make_double_spin_box(self, minimum=-1e30, maximum=1e30, n_decimals=4, func=self.on_contrast_changed)
            for _ in range(2)
        ]
        layout.addLayout(hp.make_h_layout(hp.make_label(self, "Display contrast min / max"), *self.contrast))
        layout.addWidget(self.coordinates)
        grid = QGridLayout()
        self.views = []
        for index, (name, order) in enumerate((("XY", (0, 1, 2)), ("XZ", (1, 0, 2)), ("YZ", (2, 0, 1)))):
            panel = QWidget(self)
            box = QVBoxLayout(panel)
            box.addWidget(hp.make_label(panel, name))
            view = NapariImageView(panel, add_toolbars=False, disable_new_layers=True, allow_crosshair=False)
            box.addWidget(view.widget)
            view.viewer.dims.events.current_step.connect(partial(self.sync_position, index))
            view.viewer.mouse_drag_callbacks.append(partial(self._click_position, index))
            self.views.append((view, order))
            grid.addWidget(panel, index // 2, index % 2)
        preview_panel = QWidget(self)
        box = QVBoxLayout(preview_panel)
        box.addWidget(self.preview_label)
        self.preview = NapariImageView(
            preview_panel, add_toolbars=False, disable_new_layers=True, allow_crosshair=False
        )
        box.addWidget(self.preview.widget)
        grid.addWidget(preview_panel, 1, 1)
        layout.addLayout(grid)

    def showEvent(self, event: QShowEvent) -> None:
        """Fit slice views once their visible canvas dimensions are known."""
        super().showEvent(event)
        QTimer.singleShot(0, self._fit_slices)

    def _fit_slices(self) -> None:
        if self.volume is not None:
            for view, _order in self.views:
                view.viewer.reset_view()

    def choose_volume(self) -> None:
        """Select an existing OME-Zarr store."""
        path = hp.get_directory(self, "Open OME-Zarr volume")
        if path:
            try:
                self.open(path)
            except (OSError, ValueError, KeyError) as exc:
                hp.warn_pretty(self, str(exc))

    def open(self, path: str | Path) -> None:
        """Open a completed volume and display its first channel."""
        self.volume = open_volume(path)
        with hp.qt_signals_blocked(self.channel):
            self.channel.clear()
            self.channel.addItems(self.volume.channel_names)
            if self.volume.provenance.get("rgb"):
                self.channel.addItem("RGB composite")
                self.channel.setCurrentIndex(len(self.volume.channel_names))
        self.show_channel(self.channel.currentIndex())

    def show_channel(self, index: int) -> None:
        """Share lazy image arrays between orthogonal views and refresh the preview."""
        if self.volume is None or index < 0:
            return
        self._generation += 1
        generation = self._generation
        volume = self.volume
        rgb = index == len(volume.channel_names)
        arrays = [da.moveaxis(level, 0, -1) if rgb else level[index] for level in volume.levels]
        dtype = arrays[0].dtype
        limits = (float(np.iinfo(dtype).min), float(np.iinfo(dtype).max)) if dtype.kind in "ui" else (0.0, 1.0)
        self._syncing = True
        try:
            for view, order in self.views:
                view.viewer.layers.clear()
                layer = view.viewer.add_image(
                    arrays if len(arrays) > 1 else arrays[0],
                    rgb=rgb,
                    multiscale=len(arrays) > 1,
                    scale=volume.spacing,
                    translate=volume.origin,
                    units=("um", "um", "um"),
                    name=self.channel.currentText(),
                    contrast_limits=limits,
                )
                layer.events.contrast_limits.connect(partial(self._sync_contrast, layer))
                view.viewer.dims.order = order
                view.viewer.dims.axis_labels = ("Z", "Y", "X")
                view.viewer.dims.set_current_step((0, 1, 2), tuple(size // 2 for size in arrays[0].shape[:3]))
        finally:
            self._syncing = False
        self.sync_position(0)
        self.preview.viewer.layers.clear()
        self.preview_label.setText("Loading downsampled 3D preview (maximum 64 MiB)...")

        def load_preview() -> PreviewResult:
            level = len(arrays) - 1
            data = arrays[level]
            strides = [1, 1, 1]
            while int(np.prod(data.shape)) * data.dtype.itemsize > self.PREVIEW_BYTES:
                axis = int(np.argmax(data.shape[:3]))
                slices = [slice(None)] * data.ndim
                slices[axis] = slice(None, None, 2)
                data = data[tuple(slices)]
                strides[axis] *= 2
            pixels = data.compute()
            # XY levels preserve Z spacing; additional striding is preview-only.
            scale = tuple(
                s * stride * (1 if axis == 0 else 2**level)
                for axis, (s, stride) in enumerate(zip(volume.spacing, strides, strict=False))
            )
            sample = pixels.reshape(-1)[:: max(1, pixels.size // 100_000)]
            finite = sample[np.isfinite(sample)]
            low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
            return pixels, scale, (float(low), float(high) if high > low else float(low) + 1.0)

        self._pending_preview = (load_preview, generation, rgb)
        self._start_preview()

    def _start_preview(self) -> None:
        if self._workers or self._pending_preview is None:
            return
        load_preview, generation, rgb = self._pending_preview
        self._pending_preview = None
        worker = create_worker(load_preview)
        worker.returned.connect(partial(self._show_preview, generation, rgb))
        worker.errored.connect(
            lambda exc: self.preview_label.setText(f"Preview failed: {exc}") if generation == self._generation else None
        )
        worker.finished.connect(partial(self._preview_finished, worker))
        self._workers.append(worker)
        worker.start()

    def _preview_finished(self, worker: Any) -> None:
        self._workers.remove(worker)
        self._start_preview()

    def _show_preview(self, generation: int, rgb: bool, result: PreviewResult) -> None:
        if generation != self._generation or self.volume is None:
            return
        pixels, scale, limits = result
        layer = self.preview.viewer.add_image(
            pixels,
            rgb=rgb,
            scale=scale,
            translate=self.volume.origin,
            contrast_limits=limits,
            name="Downsampled preview",
            units=("um", "um", "um"),
        )
        self.preview.viewer.dims.ndisplay = 3
        self.preview.viewer.reset_view()
        self.preview.viewer.camera.angles = (25, 35, 30)
        self.preview_label.setText(f"Downsampled 3D preview: {pixels.shape[:3]} ZYX, {pixels.nbytes / 1024**2:.1f} MiB")
        self._sync_contrast(layer)

    def _sync_contrast(self, layer: Any, _event: Any = None) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            for control, value in zip(self.contrast, layer.contrast_limits, strict=True):
                with hp.qt_signals_blocked(control):
                    control.setValue(value)
            for view in [*(item[0] for item in self.views), self.preview]:
                for other in view.viewer.layers:
                    other.contrast_limits = layer.contrast_limits
        finally:
            self._syncing = False

    def on_contrast_changed(self, _value: float) -> None:
        """Apply shared display limits without changing stored intensities."""
        if self._syncing or self.volume is None:
            return
        limits = tuple(control.value() for control in self.contrast)
        if limits[0] < limits[1] and self.views[0][0].viewer.layers:
            self.views[0][0].viewer.layers[0].contrast_limits = limits

    def _click_position(self, index: int, viewer: Any, event: Any) -> None:
        if self.volume is None or event.type != "mouse_press":
            return
        coordinates = tuple(
            round((p - o) / s)
            for p, o, s in zip(event.position[-3:], self.volume.origin, self.volume.spacing, strict=False)
        )
        viewer.dims.set_current_step((0, 1, 2), coordinates)
        self.sync_position(index)

    def sync_position(self, index: int, _event: Any = None) -> None:
        """Synchronize physical slice positions across XY, XZ and YZ views."""
        if self._syncing or self.volume is None:
            return
        self._syncing = True
        try:
            step = self.views[index][0].viewer.dims.current_step
            for view, _order in self.views:
                view.viewer.dims.set_current_step((0, 1, 2), step)
            physical = [o + s * i for o, s, i in zip(self.volume.origin, self.volume.spacing, step, strict=False)]
            present = self.volume.present[min(max(step[0], 0), len(self.volume.present) - 1)]
            self.coordinates.setText(
                f"Z {physical[0]:.3f} um | Y {physical[1]:.3f} um | X {physical[2]:.3f} um"
                + (" | Missing/excluded section" if not present else "")
            )
        finally:
            self._syncing = False

    def dispose(self) -> None:
        """Discard pending previews and release the embedded viewers."""
        self._generation += 1
        self._syncing = True
        self.volume = None
        self._pending_preview = None
        for worker in self._workers:
            worker.quit()
        for view in [*(item[0] for item in self.views), self.preview]:
            view.viewer.layers.clear()
            view.widget.close()
