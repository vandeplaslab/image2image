"""Serial workflow controls added to the existing section-preparation window."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import qtextra.helpers as hp
from qtextra.queue.queue_widget import QUEUE
from qtextra.queue.task import Task
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from image2image.models.serial import SerialSettings
from image2image.qt._serial_volume import SerialVolumeWidget
from image2image.utils.serial import (
    inspect_sections,
    load_serial_project,
    reference_canvas,
    registration_fingerprint,
    save_serial_project,
)
from image2image.utils.utilities import get_cli_path

if TYPE_CHECKING:
    from image2image.qt.dialog_elastix3d import ImageElastix3dWindow


class SerialPanel:
    """Connect preparation controls, subprocess jobs, and volume inspection."""

    def __init__(self, window: ImageElastix3dWindow) -> None:
        self.window = window
        self._loading = True
        self.task: Task | None = None
        self.snapshot: Path | None = None
        self.operation = ""
        sections = window.takeCentralWidget()
        self.tabs = QTabWidget(window)
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(sections, "Sections")
        self.tabs.addTab(self._make_registration_tab(), "Registration")
        self.tabs.addTab(self._make_volume_tab(), "Volume")
        window.setCentralWidget(self.tabs)
        window.setWindowTitle("image2image: Serial Sections")
        window.table.model().setHeaderData(window.TABLE_CONFIG.image_order, Qt.Orientation.Horizontal, "section")
        window.table.setToolTip(
            "Double-click the section column to set physical order. Uncheck keep to preserve an empty section."
        )
        for attr in (
            "group_by_btn",
            "groups_choice",
            "group_mode_btn",
            "slide_mode_btn",
            "auto_ref_btn",
            "iwsireg_btn",
            "project_mode",
            "mask_btn",
            "check_group_btn",
            "uncheck_group_btn",
            "scroll_group_btn",
        ):
            getattr(window, attr).hide()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        for signal, callback in (
            (QUEUE.evt_progress, self.on_progress),
            (QUEUE.evt_finished, self.on_finished),
            (QUEUE.evt_errored, self.on_failed),
            (QUEUE.evt_cancelled, self.on_cancelled),
        ):
            signal.connect(callback)
        self._loading = False
        self.refresh()

    def on_tab_changed(self, index: int) -> None:
        """Show preparation controls only while editing source sections."""
        self.window.statusBar().setVisible(index == 0)
        self.refresh()

    @property
    def settings(self) -> SerialSettings:
        """Return settings belonging to the currently loaded preparation project."""
        if self.window.registration.serial is None:
            self.window.registration.serial = SerialSettings()
        return self.window.registration.serial

    def _make_registration_tab(self) -> QWidget:
        widget = QWidget(self.window)
        form = hp.make_form_layout(parent=widget)
        form.addRow(
            hp.make_label(
                widget, "Set physical section numbers and inclusion in Sections, then choose alignment settings here."
            )
        )
        self.spacing = hp.make_double_spin_box(
            widget, minimum=0.001, maximum=1e6, n_decimals=3, suffix=" um", func=self.on_settings_changed
        )
        self.reference = hp.make_combobox(widget, func=self.on_settings_changed)
        self.preset = hp.make_combobox(
            widget, ["Rigid", "Rigid + affine", "Rigid + affine + deformable"], func=self.on_settings_changed
        )
        self.channel = hp.make_combobox(widget, func=self.on_settings_changed)
        self.downsample = hp.make_int_spin_box(widget, minimum=1, maximum=1024, value=1, func=self.on_settings_changed)
        self.pixel_cap = hp.make_int_spin_box(
            widget, minimum=1000, maximum=100_000_000, value=4_000_000, func=self.on_settings_changed
        )
        for label, control in (
            ("Section spacing", self.spacing),
            ("Reference section", self.reference),
            ("Alignment", self.preset),
            ("Registration channel", self.channel),
            ("Registration downsample", self.downsample),
            ("Maximum registration pixels", self.pixel_cap),
        ):
            form.addRow(label, control)
        self.paths = hp.make_label(widget, "")
        self.paths.setWordWrap(True)
        form.addRow(self.paths)
        self.status = hp.make_label(widget, "Add at least two sections.")
        self.status.setWordWrap(True)
        form.addRow(self.status)
        self.register_btn = hp.make_btn(widget, "Register sections...", func=self.start_registration)
        self.cancel_btn = hp.make_btn(widget, "Cancel processing", func=self.cancel)
        form.addRow(hp.make_h_layout(self.register_btn, self.cancel_btn))
        return widget

    def _make_volume_tab(self) -> QWidget:
        widget = QWidget(self.window)
        layout = QVBoxLayout(widget)
        controls = QWidget(widget)
        form = hp.make_form_layout(parent=controls)
        self.output_spacing = hp.make_double_spin_box(
            controls, minimum=0.0001, maximum=1e6, n_decimals=4, suffix=" um", func=self.on_geometry_changed
        )
        self.canvas = [
            hp.make_double_spin_box(
                controls,
                minimum=-1e9 if index < 2 else 0.0001,
                maximum=1e9,
                n_decimals=4,
                suffix=" um",
                func=self.on_geometry_changed,
            )
            for index in range(4)
        ]
        form.addRow("Output XY spacing", self.output_spacing)
        form.addRow("Canvas X / Y / width / height", hp.make_h_layout(*self.canvas))
        self.reset_canvas_btn = hp.make_btn(controls, "Use reference canvas", func=self.reset_canvas)
        self.reconstruct_btn = hp.make_btn(controls, "Reconstruct and export...", func=self.start_reconstruction)
        form.addRow(hp.make_h_layout(self.reset_canvas_btn, self.reconstruct_btn))
        self.volume_status = hp.make_label(controls, "All source channels are exported; gaps remain empty.")
        self.volume_status.setWordWrap(True)
        form.addRow(self.volume_status)
        layout.addWidget(controls)
        self.volume = SerialVolumeWidget(widget)
        layout.addWidget(self.volume)
        return widget

    def on_settings_changed(self, _value: Any = None) -> None:
        """Save edits immediately, without tying physical order to table sorting."""
        if self._loading:
            return
        settings = self.settings
        settings.spacing_um = self.spacing.value()
        settings.reference = self.reference.currentData()
        settings.preset = ("rigid", "affine", "deformable")[self.preset.currentIndex()]
        settings.registration_channel = max(0, self.channel.currentIndex())
        settings.registration_downsample = self.downsample.value()
        settings.max_registration_pixels = self.pixel_cap.value()
        self.refresh()

    def on_geometry_changed(self, _value: Any = None) -> None:
        """Keep explicit output bounds separate from automatic reference defaults."""
        if not self._loading:
            self.settings.output_spacing_um = self.output_spacing.value()
            self.settings.canvas_xywh = tuple(control.value() for control in self.canvas)
            self.refresh()

    def set_section_number(self, key: str, number: int) -> None:
        """Update a section's physical number, rejecting collisions."""
        self.settings.section_numbers = {**self.settings.section_numbers, key: number}
        self.refresh()

    def reset_canvas(self) -> None:
        """Reset volume geometry to the reference image's physical extent."""
        self.settings.canvas_xywh = None
        self.settings.output_spacing_um = None
        self.refresh()

    def refresh(self) -> None:
        """Refresh validation and result availability after preparation edits."""
        if self._loading:
            return
        self._loading = True
        valid = current = False
        try:
            settings = self.settings
            settings.synchronize(self.window.registration)
            self.spacing.setValue(settings.spacing_um)
            self.preset.setCurrentIndex(("rigid", "affine", "deformable").index(settings.preset))
            self.downsample.setValue(settings.registration_downsample)
            self.pixel_cap.setValue(settings.max_registration_pixels)
            self.reference.clear()
            for key in settings.ordered_keys(self.window.registration, included_only=False):
                self.reference.addItem(f"{settings.section_numbers[key]}: {key}", key)
            self.reference.setCurrentIndex(self.reference.findData(settings.reference))
            for row in range(self.window.table.n_rows):
                key = self.window.table.get_value(self.window.TABLE_CONFIG.key, row)
                if key in settings.section_numbers:
                    self.window.table.set_value(
                        self.window.TABLE_CONFIG.image_order, row, settings.section_numbers[key]
                    )
                    self.window.table.set_value(
                        self.window.TABLE_CONFIG.reference, row, "true" if key == settings.reference else "false"
                    )
            try:
                infos = inspect_sections(self.window.registration)
                info = infos[settings.reference]
                self.channel.clear()
                self.channel.addItems(["RGB brightfield"] if info.rgb else info.channels)
                self.channel.setCurrentIndex(0 if info.rgb else settings.registration_channel)
                self.channel.setEnabled(not info.rgb)
                output_spacing = settings.output_spacing_um or max(item.spacing for item in infos.values())
                canvas = settings.canvas_xywh or reference_canvas(self.window.registration, info)
                self.output_spacing.setValue(output_spacing)
                for control, value in zip(self.canvas, canvas, strict=False):
                    control.setValue(value)
                edges = settings.registration_paths(self.window.registration)
                self.paths.setText(
                    " | ".join(
                        f"{settings.section_numbers[source]} -> {settings.section_numbers[through or target]}"
                        for source, target, through in edges
                    )
                )
                valid = True
                current = settings.registered_fingerprint == registration_fingerprint(self.window.registration, infos)
                current = current and bool(
                    settings.result_dir and (Path(settings.result_dir) / "transforms.json").exists()
                )
                if self.task is None:
                    self.status.setText(
                        "Registration is current. Reconstruct in the Volume tab."
                        if current
                        else "Ready to register. Previous registration is missing or stale."
                    )
                first, last = min(settings.section_numbers.values()), max(settings.section_numbers.values())
                self.volume_status.setText(
                    f"{last - first + 1} Z positions, {last - first + 1 - len(infos)} gaps/exclusions. "
                    "All channels retain native intensities. Expand the canvas to include tissue beyond the reference."
                )
            except (ValueError, OSError, KeyError) as exc:
                self.status.setText(str(exc))
                self.paths.setText("")
        finally:
            busy = self.task is not None
            self.register_btn.setEnabled(valid and not busy)
            self.reconstruct_btn.setEnabled(current and not busy)
            self.cancel_btn.setEnabled(busy)
            self._loading = False

    def _snapshot(self, directory: Path) -> Path:
        snapshot = directory / f"serial-{uuid4().hex}.i2wsiprep.json"
        save_serial_project(snapshot, self.window.registration)
        return snapshot

    def _queue(self, operation: str, directory: Path, output: Path | None = None) -> None:
        inspect_sections(self.window.registration)
        self.snapshot = self._snapshot(directory)
        if getattr(sys, "frozen", False):
            command = [get_cli_path("image2image")]
        else:
            command = [sys.executable, "-m", "image2image"]
        command += ["serial", operation, "--project", str(self.snapshot)]
        if output is not None:
            command += ["--output", str(output)]
        if sys.platform == "win32":
            command[1:] = [subprocess.list2cmdline([argument]) for argument in command[1:]]
        self.operation = operation
        self.task = Task(uuid4().hex, [command], task_name=f"Serial sections: {operation}")
        self.status.setText(f"Queued {operation}...")
        self.refresh()
        QUEUE.add_task(self.task)
        QUEUE.run_queued()

    def start_registration(self) -> None:
        """Snapshot and queue registration in a user-selected result directory."""
        directory = hp.get_directory(self.window, "Serial registration results", base_dir=self.window.CONFIG.output_dir)
        if directory:
            try:
                self.window.CONFIG.output_dir = str(directory)
                self._queue("register", Path(directory))
            except (ValueError, OSError) as exc:
                hp.warn_pretty(self.window, str(exc))

    def start_reconstruction(self) -> None:
        """Queue reconstruction into a new OME-Zarr directory."""
        output = hp.get_save_filename(
            self.window,
            "Export serial volume",
            base_dir=self.window.CONFIG.output_dir,
            base_filename="serial.ome.zarr",
            file_filter="OME-Zarr (*.ome.zarr)",
        )
        if output:
            try:
                path = Path(output)
                if path.exists():
                    raise FileExistsError("Choose a new output name; existing volumes are preserved.")
                self._queue("reconstruct", path.parent, path)
            except (ValueError, OSError) as exc:
                hp.warn_pretty(self.window, str(exc))

    def cancel(self) -> None:
        """Cancel only this window's active subprocess job."""
        if self.task is not None:
            QUEUE.cancel(self.task)

    def on_progress(self, task: Task) -> None:
        """Display the active job's recent output."""
        if self.task is task:
            self.status.setText(task.current_std_out[-1500:])
            self.volume_status.setText(task.current_std_out[-500:])

    def on_finished(self, task: Task) -> None:
        """Adopt results only if the live preparation still matches the job."""
        if self.task is not task or self.snapshot is None:
            return
        self.task = None
        try:
            result = load_serial_project(self.snapshot)
            same_registration = registration_fingerprint(result) == registration_fingerprint(self.window.registration)
            same_geometry = all(
                getattr(result.serial, attr) == getattr(self.settings, attr)
                for attr in ("spacing_um", "section_numbers", "canvas_xywh", "output_spacing_um")
            )
            if same_registration and (self.operation == "register" or same_geometry):
                self.settings.result_dir = result.serial.result_dir
                self.settings.registered_fingerprint = result.serial.registered_fingerprint
                self.settings.volume_path = result.serial.volume_path
                if self.operation == "reconstruct" and self.settings.volume_path:
                    self.volume.open(self.settings.volume_path)
                    self.tabs.setCurrentIndex(2)
            else:
                hp.warn_pretty(self.window, f"Sections changed during processing. Results remain in {self.snapshot}.")
        except (ValueError, OSError, KeyError) as exc:
            hp.warn_pretty(self.window, str(exc))
        self.refresh()

    def on_failed(self, task: Task, error: Any = None) -> None:
        """Keep failures visible and permit a new run."""
        if self.task is task:
            self.task = None
            self.refresh()
            self.status.setText(f"Processing failed: {error or task.current_std_out}")
            self.volume_status.setText(self.status.text())

    def on_cancelled(self, task: Task) -> None:
        """Report cancellation without treating partial outputs as completed."""
        if self.task is task:
            self.task = None
            self.refresh()
            self.status.setText("Processing cancelled. Completed volumes were preserved.")

    def dispose(self) -> None:
        """Stop owned processing and disconnect queue callbacks on window close."""
        self.cancel()
        for signal, callback in (
            (QUEUE.evt_progress, self.on_progress),
            (QUEUE.evt_finished, self.on_finished),
            (QUEUE.evt_errored, self.on_failed),
            (QUEUE.evt_cancelled, self.on_cancelled),
        ):
            signal.disconnect(callback)
        self.volume.dispose()
