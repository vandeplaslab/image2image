"""Standalone OME-Zarr volume assembly with explicit slide and channel selection."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from threading import Event
from typing import Any, cast

import numpy as np
import qtextra.helpers as hp
from qtpy.QtCore import QMimeData, Qt, QTimer, Signal
from qtpy.QtGui import QCloseEvent, QColor, QDrag, QDragEnterEvent, QDragMoveEvent, QDropEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenuBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from image2image.config import SingleAppConfig
from image2image.enums import ALLOWED_IMAGE_FORMATS_MICROSCOPY_ONLY
from image2image.models.volume import MissingChannelPolicy, VolumeChannel, VolumeExportPlan, VolumeSlide
from image2image.qt._dialog_base import Window
from image2image.utils.volume import (
    build_volume_plan,
    detect_slide_number,
    export_volume,
    inspect_volume_file,
    reconcile_channels,
    reorder_slides,
    resolve_volume_selection,
    source_channel,
)


class VolumeConfig(SingleAppConfig):
    """Window preferences independent of the OME-TIFF converter."""

    USER_CONFIG_FILENAME = "config.volume.json"


class VolumeTable(QTableWidget):
    """Request semantic row moves without losing cell widgets or source identities."""

    row_moved = Signal(int, int)
    files_dropped = Signal(list)

    def __init__(self, parent: QWidget, columns: list[str]) -> None:
        super().__init__(0, len(columns), parent)
        self.setHorizontalHeaderLabels(columns)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setSortingEnabled(False)
        header = cast(QHeaderView, self.horizontalHeader())
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

    def startDrag(self, supported_actions: Qt.DropAction) -> None:
        """Own the drag so Qt does not clear source cells after a semantic row move."""
        if self.currentRow() < 0:
            return
        mime = QMimeData()
        mime.setData("application/x-image2image-volume-row", str(self.currentRow()).encode("ascii"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec_(Qt.DropAction.MoveAction)  # type: ignore[attr-defined]
        drag.deleteLater()

    def dragEnterEvent(self, event: QDragEnterEvent | None) -> None:
        """Accept local files and row moves originating in this table."""
        if event is None:
            return
        mime = event.mimeData()
        if (mime is not None and mime.hasUrls()) or event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent | None) -> None:
        """Retain the drop target while dragging over editable cells."""
        if event is None:
            return
        mime = event.mimeData()
        if (mime is not None and mime.hasUrls()) or event.source() is self:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Send file imports or a row move to the owning model."""
        if event is None:
            return
        mime = event.mimeData()
        if mime is not None and mime.hasUrls():
            self.files_dropped.emit([url.toLocalFile() for url in mime.urls() if url.isLocalFile()])
            event.acceptProposedAction()
        elif event.source() is self:
            point = event.position().toPoint() if hasattr(event, "position") else event.pos()  # type: ignore[attr-defined]
            target = self.indexAt(point).row()
            source = int(mime.data("application/x-image2image-volume-row").data()) if mime else self.currentRow()
            self.row_moved.emit(source, target if target >= 0 else self.rowCount() - 1)
            event.acceptProposedAction()
        else:
            event.ignore()


class ImageVolumeWindow(Window):
    """Prepare already aligned slides and export a calibrated OME-Zarr volume."""

    APP_NAME = "volume"
    progress_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None, run_check_version: bool = True, **_kwargs: Any) -> None:
        self.CONFIG = VolumeConfig(_auto_load=True)  # type: ignore[call-arg]
        self.slides: list[VolumeSlide] = []
        self.channels: list[VolumeChannel] = []
        self.worker: Any = None
        self._cancel = Event()
        self._closing = False
        self._refresh_pending = False
        self._updating = False
        self._plan: VolumeExportPlan | None = None
        super().__init__(parent, "image2image: OME-Zarr Volume Builder", run_check_version=run_check_version)
        self.refresh()

    @staticmethod
    def _setup_config() -> None:
        """Readers are opened independently with lazy pyramid settings."""

    def setup_events(self, state: bool = True) -> None:
        """Connect progress signals delivered from the background worker."""
        if state:
            self.progress_changed.connect(self.status.setText)

    def _setup_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.addWidget(
            hp.make_label(
                self,
                "Build an OME-Zarr volume from already aligned slides. Drop images here or use Add images.",
                wrap=True,
            )
        )
        self.tabs = QTabWidget(self)
        layout.addWidget(self.tabs, 1)
        self._make_slides_tab()
        self._make_channels_tab()
        self._make_output_tab()
        self.warning = hp.make_text_edit(self, read_only=True, max_height=130)
        self.warning.setObjectName("warning_label")
        layout.addWidget(self.warning)
        self.status = hp.make_label(self, "Ready. Drop slides to begin.", wrap=True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.export_btn = hp.make_btn(self, "Write OME-Zarr volume", func=self.start_export)
        self.cancel_btn = hp.make_btn(self, "Cancel", func=self.cancel)
        self.cancel_btn.setEnabled(False)
        layout.addLayout(hp.make_h_layout(self.export_btn, self.cancel_btn))
        menu = hp.make_menu(self, "File")
        self.add_action = hp.make_menu_item(self, "Add images...", "Ctrl+I", menu=menu, func=self.choose_images)
        hp.make_menu_item(self, "Quit", menu=menu, func=self.close)
        menubar = cast(QMenuBar, self.menuBar())
        menubar.addMenu(menu)
        menubar.addMenu(self._make_apps_menu())
        menubar.addMenu(self._make_tools_menu())
        self._make_icon()
        self._make_statusbar()

    def _make_slides_tab(self) -> None:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(panel, "Add images...", func=self.choose_images),
                hp.make_btn(panel, "Add OME-Zarr image...", func=self.choose_zarr),
                hp.make_btn(panel, "Remove selected", func=self.remove_slide),
            )
        )
        self.pattern = hp.make_line_edit(
            panel,
            placeholder="Automatic: slide_001, section12, sec-003",
            tooltip="Custom regex with a named 'section' group or one capture group. Applied only when requested.",
        )
        layout.addLayout(hp.make_h_layout(self.pattern, hp.make_btn(panel, "Detect numbers", func=self.detect_numbers)))
        self.slide_table = VolumeTable(
            panel,
            [
                "Include",
                "File",
                "Scene",
                "Detected",
                "Z position",
                "Dimensions (Y X)",
                "Pixel size (um)",
                "Status",
            ],
        )
        self.slide_table.itemChanged.connect(self._slide_edited)
        self.slide_table.row_moved.connect(self.move_slide)
        self.slide_table.files_dropped.connect(self.add_paths)
        layout.addWidget(self.slide_table, 1)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(
                    panel,
                    "Move up",
                    func=lambda: self.move_slide(self.slide_table.currentRow(), self.slide_table.currentRow() - 1),
                ),
                hp.make_btn(
                    panel,
                    "Move down",
                    func=lambda: self.move_slide(self.slide_table.currentRow(), self.slide_table.currentRow() + 1),
                ),
            )
        )
        layout.addWidget(
            hp.make_label(
                panel,
                "Z positions must be unique. Dragging moves images between existing positions; gaps remain blank. "
                "Edit a position to correct numbering. Excluded slides retain their blank Z positions.",
                wrap=True,
            )
        )
        self.tabs.addTab(panel, "Slides")

    def _make_channels_tab(self) -> None:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        self.policy = hp.make_combobox(panel, ["Exclude incomplete channels", "Exclude incomplete slides"])
        self.policy.currentIndexChanged.connect(self._schedule_refresh)
        layout.addWidget(self.policy)
        self.channel_table = VolumeTable(
            panel, ["Include", "Order", "Matched name", "Final name", "Coverage", "Status"]
        )
        self.channel_table.itemChanged.connect(self._channel_edited)
        self.channel_table.row_moved.connect(self.move_channel)
        self.channel_table.files_dropped.connect(self.add_paths)
        layout.addWidget(self.channel_table, 1)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(
                    panel,
                    "Move up",
                    func=lambda: self.move_channel(
                        self.channel_table.currentRow(), self.channel_table.currentRow() - 1
                    ),
                ),
                hp.make_btn(
                    panel,
                    "Move down",
                    func=lambda: self.move_channel(
                        self.channel_table.currentRow(), self.channel_table.currentRow() + 1
                    ),
                ),
            )
        )
        layout.addWidget(
            hp.make_label(
                panel,
                "Source mappings for slide (automatic matching ignores case and surrounding whitespace):",
                wrap=True,
            )
        )
        self.mapping_slide = hp.make_combobox(panel)
        self.mapping_slide.currentIndexChanged.connect(self._refresh_mappings)
        layout.addWidget(self.mapping_slide)
        self.mapping_table = QTableWidget(0, 2, panel)
        self.mapping_table.setHorizontalHeaderLabels(["Output channel", "Source channel"])
        cast(QHeaderView, self.mapping_table.horizontalHeader()).setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.mapping_table, 1)
        self.tabs.addTab(panel, "Channels")

    def _make_output_tab(self) -> None:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        form = hp.make_form_layout()
        self.output_directory = hp.make_line_edit(
            panel, text=self.CONFIG.output_dir, placeholder="Select output directory"
        )
        self.output_directory.textChanged.connect(self._schedule_refresh)
        form.addRow(
            "Output directory",
            hp.make_h_layout(
                self.output_directory,
                hp.make_btn(panel, "Browse...", func=self.choose_output),
            ),
        )
        self.output_name = hp.make_line_edit(panel, text="volume.ome.zarr")
        self.output_name.textChanged.connect(self._schedule_refresh)
        form.addRow("Output name", self.output_name)
        self.z_spacing = hp.make_double_spin_box(panel, minimum=0, maximum=1e6, n_decimals=6, value=0)
        self.z_spacing.setSpecialValueText("Set section spacing")
        self.z_spacing.valueChanged.connect(self._schedule_refresh)
        form.addRow("Z spacing (um)", self.z_spacing)
        self.xy_spacing = hp.make_double_spin_box(panel, minimum=0, maximum=1e6, n_decimals=6, value=0)
        self.xy_spacing.setSpecialValueText("Automatic (coarsest source)")
        self.xy_spacing.valueChanged.connect(self._schedule_refresh)
        form.addRow("XY pixel size (um)", self.xy_spacing)
        self.tile_size = hp.make_combobox(panel, ["256", "512", "1024", "2048"], value="512")
        self.tile_size.currentIndexChanged.connect(self._schedule_refresh)
        form.addRow("XY tile size", self.tile_size)
        form.addRow(
            hp.make_label(
                panel,
                "OME-Zarr 0.5 / Zarr 3. Automatic XY pyramids and default compression. "
                "Native dtype is preserved. Slides are resampled from a common top-left origin and padded with zeros. "
                "Existing outputs are never overwritten.",
                wrap=True,
            )
        )
        self.summary = hp.make_label(panel, "", wrap=True)
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        form.addRow(self.summary)
        layout.addLayout(form)
        layout.addStretch(1)
        self.tabs.addTab(panel, "Output")

    def choose_images(self) -> None:
        """Select microscopy files using the same filter as the Convert app."""
        if self.worker is not None:
            return
        paths = hp.get_filename(
            self, "Add slides", self.CONFIG.last_dir, ALLOWED_IMAGE_FORMATS_MICROSCOPY_ONLY, multiple=True
        )
        if paths:
            self.add_paths(cast(list[str | Path], paths))

    def choose_zarr(self) -> None:
        """Select a directory-backed 2D OME-Zarr source."""
        path = hp.get_directory(self, "Add OME-Zarr image")
        if path:
            self.add_paths([path])

    def choose_output(self) -> None:
        """Select the parent directory for the new OME-Zarr store."""
        path = hp.get_directory(self, "Output directory")
        if path:
            self.output_directory.setText(str(path))

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Import local files dropped onto the window."""
        if event is None:
            return
        mime = event.mimeData()
        if self.worker is None and mime is not None and mime.hasUrls():
            self.add_paths([url.toLocalFile() for url in mime.urls() if url.isLocalFile()])
            event.acceptProposedAction()
        else:
            event.ignore()

    def add_paths(self, paths: list[str | Path]) -> None:
        """Inspect new files in a worker without replacing existing row edits."""
        if self.worker is not None:
            return
        known = {slide.path for slide in self.slides}
        pending = list(dict.fromkeys(Path(path).resolve() for path in paths if path))
        pending = [path for path in pending if path not in known]
        if not pending:
            return
        pattern = self.pattern.text().strip()
        try:
            detect_slide_number("", pattern)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self._cancel.clear()

        def inspect() -> tuple[list[VolumeSlide], list[str]]:
            slides, errors = [], []
            for path in pending:
                if self._cancel.is_set():
                    break
                self.progress_changed.emit(f"Inspecting {path.name}...")
                try:
                    slides.append(inspect_volume_file(path, pattern))
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{path.name}: {exc}")
            return slides, errors

        self.worker = create_worker(
            inspect,
            _start_thread=False,
            _connect={
                "returned": self._accept_inputs,
                "errored": self._worker_error,
                "finished": self._worker_finished,
            },
        )
        self._set_busy(True)
        self.worker.start()

    def _accept_inputs(self, result: tuple[list[VolumeSlide], list[str]]) -> None:
        slides, errors = result
        self.slides.extend(slides)
        self.slides.sort(key=lambda slide: (slide.position is None, slide.position or 0, slide.path.name))
        if slides:
            self.CONFIG.last_dir = str(slides[-1].path.parent)
        self.status.setText("\n".join(errors) if errors else f"Loaded {len(slides)} slide(s).")

    def detect_numbers(self) -> None:
        """Apply filename detection while preserving manually assigned positions."""
        try:
            detected = [detect_slide_number(slide.path.name, self.pattern.text().strip()) for slide in self.slides]
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        for slide, (number, error) in zip(self.slides, detected, strict=True):
            slide.detected_number, slide.detection_error = number, error
            if not slide.manual_position:
                slide.position = number
        self.slides.sort(key=lambda slide: (slide.position is None, slide.position or 0))
        self.refresh()

    def remove_slide(self) -> None:
        """Remove one file and its channel overrides from the assembly."""
        row = self.slide_table.currentRow()
        if 0 <= row < len(self.slides):
            slide = self.slides.pop(row)
            for channel in self.channels:
                channel.overrides.pop(slide.key, None)
            self.refresh()

    def move_slide(self, source: int, target: int) -> None:
        """Reassign images across the existing numbered positions."""
        if not (0 <= source < len(self.slides) and 0 <= target < len(self.slides)) or source == target:
            return
        keys = [slide.key for slide in self.slides]
        keys.insert(target, keys.pop(source))
        try:
            self.slides = reorder_slides(self.slides, keys)
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.refresh()
        self.slide_table.selectRow(target)

    def move_channel(self, source: int, target: int) -> None:
        """Change the output C-axis order without changing channel mappings."""
        if 0 <= source < len(self.channels) and 0 <= target < len(self.channels):
            self.channels.insert(target, self.channels.pop(source))
            self.refresh()
            self.channel_table.selectRow(target)

    def _slide_edited(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        slide = self.slides[item.row()]
        if item.column() == 0:
            slide.included = item.checkState() == Qt.CheckState.Checked
        elif item.column() == 4:
            try:
                value = int(item.text())
                slide.position = value if value >= 0 else None
            except ValueError:
                slide.position = None
            slide.manual_position = True
            self.slides.sort(key=lambda entry: (entry.position is None, entry.position or 0))
        self._schedule_refresh()

    def _channel_edited(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        channel = self.channels[item.row()]
        if item.column() == 0:
            channel.selected = item.checkState() == Qt.CheckState.Checked
        elif item.column() == 3:
            channel.name = item.text()
        self._schedule_refresh()

    def _select_scene(self, key: str, scene: int | None) -> None:
        slide = next(slide for slide in self.slides if slide.key == key)
        slide.scene_index = scene
        for channel in self.channels:
            channel.overrides.pop(key, None)
        self._schedule_refresh()

    def _schedule_refresh(self, *_args: Any) -> None:
        if not self._updating and not self._refresh_pending:
            self._refresh_pending = True
            QTimer.singleShot(0, self.refresh)

    @staticmethod
    def _item(text: str, *, editable: bool = False, checked: bool | None = None) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        if checked is not None:
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        return item

    def refresh(self) -> None:
        """Recompute inclusion, visible warnings, and the validated output snapshot."""
        self._refresh_pending = False
        self._updating = True
        try:
            self.channels = reconcile_channels(self.slides, self.channels)
            policy: MissingChannelPolicy = "channels" if self.policy.currentIndex() == 0 else "slides"
            selection = resolve_volume_selection(self.slides, self.channels, policy)
            channel_names = {channel.key: channel.name for channel in self.channels}
            positions = [slide.position for slide in self.slides]
            self.slide_table.setRowCount(len(self.slides))
            for row, slide in enumerate(self.slides):
                source = slide.source
                reasons = []
                if slide.position is None:
                    reasons.append(slide.detection_error or "Set a valid Z position")
                elif positions.count(slide.position) > 1:
                    reasons.append("Duplicate Z position")
                if source is None:
                    reasons.append("Select one scene")
                missing_keys = tuple(
                    channel.key for channel in self.channels if source_channel(slide, channel)[0] is None
                )
                if missing_keys:
                    reasons.append("Missing/ambiguous: " + ", ".join(channel_names[key] for key in missing_keys))
                if slide.key in selection.excluded_slides:
                    reasons.append("Excluded - blank Z position")
                texts = [
                    "",
                    slide.path.name,
                    "",
                    str(slide.detected_number) if slide.detected_number is not None else "?",
                    str(slide.position) if slide.position is not None else "",
                    " x ".join(map(str, source.shape)) if source else "",
                    f"{source.pixel_size:g}" if source else "",
                    "; ".join(reasons) or "Included",
                ]
                for column, text in enumerate(texts):
                    item = self._item(text, editable=column == 4, checked=slide.included if column == 0 else None)
                    item.setToolTip(str(slide.path) if column == 1 else text)
                    if reasons:
                        item.setBackground(QColor(220, 145, 30, 55))
                    self.slide_table.setItem(row, column, item)
                combo = hp.make_combobox(self.slide_table)
                combo.addItem("Select a scene...", None)
                for candidate in slide.sources:
                    combo.addItem(
                        f"Scene {candidate.scene_index}: {' x '.join(map(str, candidate.shape))}", candidate.scene_index
                    )
                combo.setCurrentIndex(combo.findData(slide.scene_index))
                combo.currentIndexChanged.connect(
                    lambda _index, widget=combo, key=slide.key: self._select_scene(key, widget.currentData())
                )
                self.slide_table.setCellWidget(row, 2, combo)
            self.channel_table.setRowCount(len(self.channels))
            included = [slide for slide in self.slides if slide.included]
            for row, channel in enumerate(self.channels):
                absent = [slide.path.name for slide in included if source_channel(slide, channel)[0] is None]
                notes = []
                if absent:
                    notes.append("Missing/ambiguous in: " + ", ".join(absent))
                if channel.key in selection.excluded_channels:
                    notes.append("Excluded" + (" by policy" if channel.selected else " (not selected)"))
                texts = [
                    "",
                    str(row + 1),
                    channel.key,
                    channel.name,
                    f"{len(included) - len(absent)}/{len(included)}",
                    "; ".join(notes) or "Included",
                ]
                for column, text in enumerate(texts):
                    item = self._item(text, editable=column == 3, checked=channel.selected if column == 0 else None)
                    item.setToolTip(text)
                    if absent or channel.key in selection.excluded_channels:
                        item.setBackground(QColor(220, 145, 30, 55))
                    self.channel_table.setItem(row, column, item)
            selected_key = self.mapping_slide.currentData()
            with hp.qt_signals_blocked(self.mapping_slide):
                self.mapping_slide.clear()
                for slide in self.slides:
                    self.mapping_slide.addItem(slide.path.name, slide.key)
                self.mapping_slide.setCurrentIndex(max(0, self.mapping_slide.findData(selected_key)))
            self._refresh_mappings()
            self._plan = None
            try:
                self._plan = build_volume_plan(
                    self.slides,
                    self.channels,
                    policy=policy,
                    z_spacing=self.z_spacing.value(),
                    xy_spacing=self.xy_spacing.value() or None,
                    tile_size=int(self.tile_size.currentText()),
                )
                plan = self._plan
                level_shapes = [plan.shape]
                while max(level_shapes[-1][2:]) > plan.tile_size:
                    previous = level_shapes[-1]
                    level_shapes.append((*previous[:2], (previous[2] + 1) // 2, (previous[3] + 1) // 2))
                size = sum(math.prod(shape) for shape in level_shapes) * np.dtype(plan.dtype).itemsize
                extent = tuple(size * spacing for size, spacing in zip(plan.shape[1:], plan.spacing, strict=True))
                self.summary.setText(
                    f"CZYX: {plan.shape}\nSpacing ZYX (um): {plan.spacing}\n"
                    f"Extent ZYX (um): {extent}\nOrigin ZYX (um): (0, 0, 0)\nDtype: {plan.dtype}\n"
                    f"Pyramid levels: {len(level_shapes)}\nEstimated uncompressed size: {size / 1024**2:,.2f} MiB"
                )
                self._output_path()
                valid = True
            except ValueError as exc:
                self.summary.setText(str(exc) if self._plan is None else self.summary.text() + f"\n{exc}")
                valid = False
            excluded_channels = [
                channel.name for channel in self.channels if channel.key in selection.excluded_channels
            ]
            excluded_slides = [slide.path.name for slide in self.slides if slide.key in selection.excluded_slides]
            issues = [
                f"Excluded channels ({len(excluded_channels)}): {', '.join(excluded_channels) or 'none'}",
                f"Excluded slides ({len(excluded_slides)}): {', '.join(excluded_slides) or 'none'}",
            ]
            valid_positions = [position for position in positions if position is not None]
            if valid_positions:
                active = {
                    slide.position
                    for slide in self.slides
                    if slide.key in selection.slide_keys and slide.position is not None
                }
                first, last = min(valid_positions), max(valid_positions)
                # Report ranges without allocating every missing position.
                boundaries = [first - 1, *sorted(active), last + 1]
                gaps = [(left + 1, right - 1) for left, right in itertools.pairwise(boundaries) if right - left > 1]
                ranges = [str(left) if left == right else f"{left}-{right}" for left, right in gaps]
                issues.append(f"Blank Z positions ({last - first + 1 - len(active)}): {', '.join(ranges) or 'none'}")
            affected = []
            for slide in included:
                absent = [channel.name for channel in self.channels if source_channel(slide, channel)[0] is None]
                if absent:
                    affected.append(f"{slide.path.name} ({', '.join(absent)})")
            if affected:
                issues.append(f"Slides with missing/ambiguous channels ({len(affected)}): " + "; ".join(affected))
            issues.extend(selection.errors)
            self.warning.setPlainText("\n".join(issues))
            self.export_btn.setEnabled(valid and self.worker is None)
        finally:
            self._updating = False

    def _refresh_mappings(self, *_args: Any) -> None:
        key = self.mapping_slide.currentData()
        slide = next((slide for slide in self.slides if slide.key == key), None)
        self.mapping_table.setRowCount(len(self.channels) if slide and slide.source else 0)
        if slide is None or slide.source is None:
            return
        for row, channel in enumerate(self.channels):
            self.mapping_table.setItem(row, 0, self._item(channel.name))
            combo = hp.make_combobox(self.mapping_table)
            index, error = source_channel(slide, VolumeChannel(channel.key, channel.name))
            label = (
                f"Automatic: {slide.source.channel_names[index]}"
                if index is not None
                else f"Automatic: {error or 'missing'}"
            )
            combo.addItem(label, "auto")
            combo.addItem("Missing / exclude this mapping", None)
            for index, name in enumerate(slide.source.channel_names):
                combo.addItem(f"{index}: {name}", index)
            combo.setCurrentIndex(combo.findData(channel.overrides.get(slide.key, "auto")))
            combo.currentIndexChanged.connect(
                lambda _index, widget=combo, entry=channel, slide_key=slide.key: self._set_mapping(
                    entry, slide_key, widget.currentData()
                )
            )
            self.mapping_table.setCellWidget(row, 1, combo)

    def _set_mapping(self, channel: VolumeChannel, key: str, value: str | int | None) -> None:
        if value == "auto":
            channel.overrides.pop(key, None)
        elif value is None or isinstance(value, int):
            channel.overrides[key] = value
        self._schedule_refresh()

    def _output_path(self) -> Path:
        directory = self.output_directory.text().strip()
        name = self.output_name.text().strip()
        if not directory:
            raise ValueError("Choose an output directory.")
        if not name.endswith(".ome.zarr") or name == ".ome.zarr" or Path(name).name != name or "\\" in name:
            raise ValueError("Use a filename ending in .ome.zarr, without directory components.")
        output = Path(directory).expanduser() / name
        if output.exists():
            raise ValueError(f"Output already exists: {output}")
        if output.parent.exists() and not output.parent.is_dir():
            raise ValueError("Output directory is not a directory.")
        return output

    def start_export(self) -> None:
        """Validate once more and export a frozen snapshot in an independent worker."""
        if self.worker is not None:
            return
        self.refresh()
        if self._plan is None or not self.export_btn.isEnabled():
            return
        output = self._output_path()
        self._cancel.clear()
        self.worker = create_worker(
            export_volume,
            self._plan,
            output,
            progress=self.progress_changed.emit,
            cancelled=self._cancel.is_set,
            _start_thread=False,
            _connect={
                "returned": self._export_returned,
                "errored": self._worker_error,
                "finished": self._worker_finished,
            },
        )
        self.status.setText("Writing OME-Zarr volume...")
        self._set_busy(True)
        self.worker.start()

    def _export_returned(self, path: Path) -> None:
        self.status.setText(f"Exported: {path}")

    def _worker_error(self, exc: Exception) -> None:
        self.status.setText(
            "Cancelled; partial output removed." if isinstance(exc, InterruptedError) else f"Failed: {exc}"
        )

    def _worker_finished(self) -> None:
        self.worker = None
        self._set_busy(False)
        self.refresh()
        if self._closing:
            QTimer.singleShot(0, self.close)

    def _set_busy(self, busy: bool) -> None:
        self.tabs.setEnabled(not busy)
        self.add_action.setEnabled(not busy)
        self.export_btn.setEnabled(not busy and self._plan is not None)
        self.cancel_btn.setEnabled(busy)
        self.allow_drop = not busy
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(busy)

    def cancel(self) -> None:
        """Request cancellation while keeping the worker alive until cleanup completes."""
        if self.worker is not None:
            self._cancel.set()
            self.cancel_btn.setEnabled(False)
            self.status.setText("Cancelling; waiting for cleanup...")

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Keep the window alive until its background reader and writer have stopped."""
        if event is None:
            return
        if self.worker is not None:
            self._closing = True
            self.cancel()
            event.ignore()
            return
        self.CONFIG.output_dir = self.output_directory.text().strip()
        self.CONFIG.save()
        if self._console:
            self._console.close()
        event.accept()
