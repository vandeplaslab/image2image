"""Interactive brightfield stain separation and reference normalization."""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from pathlib import Path
from threading import Event
from typing import Any, cast

import numpy as np
import qtextra.helpers as hp
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QCloseEvent, QDropEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHeaderView,
    QMenuBar,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from image2image.config import SingleAppConfig
from image2image.models.stains import (
    StainEstimation,
    StainOutcome,
    StainOverride,
    StainProfile,
    StainRecipe,
    StainSource,
    stain_preset,
)
from image2image.qt._dialog_base import Window
from image2image.utils.stain_io import (
    FitCache,
    StainPreview,
    export_stain_batch,
    inspect_stain_file,
    load_stain_overview,
    load_stain_recipe,
    preview_stains,
    sample_rgb,
    save_stain_recipe,
)
from image2image.utils.stains import estimate_stain_profile, sample_background, sample_stain_vector


class StainsConfig(SingleAppConfig):
    """Store stain window preferences independently from other mini apps."""

    USER_CONFIG_FILENAME = "config.stains.json"


class ImageStainsWindow(Window):
    """Inspect separated channels, tune vectors, and export reproducible brightfield batches."""

    APP_NAME = "stains"
    progress_changed = Signal(str)
    outcome_ready = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        run_check_version: bool = True,
        image_path: tuple[str, ...] | None = None,
        image_dir: str | None = None,
        **_kwargs: Any,
    ) -> None:
        self.CONFIG = StainsConfig(_auto_load=True)  # type: ignore[call-arg]
        self.inputs: list[tuple[StainSource, ...]] = []
        self.profile = stain_preset()
        self.estimation = StainEstimation()
        self.overrides: dict[str, StainOverride] = {}
        self.reference_profile: StainProfile | None = None
        self.worker: Any = None
        self._operation = ""
        self._cancel = Event()
        self._closing = self._updating = self._syncing = False
        self._generation = 0
        self._preview_pending = False
        self._preview: StainPreview | None = None
        self._preview_key: tuple[Any, ...] | None = None
        self._cache: FitCache = {}
        self._regions: dict[str, tuple[int, int, int, int]] = {}
        self.results: list[StainOutcome] = []
        self._job_buttons: list[Any] = []
        super().__init__(parent, "image2image: Stain Separation and Normalization", run_check_version=run_check_version)
        self.resize(1550, 950)
        self._update_editor()
        paths = list(image_path or ())
        if image_dir:
            paths.extend(
                str(path) for path in sorted(Path(image_dir).iterdir()) if path.is_file() or path.name.endswith(".zarr")
            )
        if paths:
            QTimer.singleShot(0, partial(self.add_paths, paths))

    @staticmethod
    def _setup_config() -> None:
        """Use independent readers without registration preprocessing."""

    def setup_events(self, state: bool = True) -> None:
        """Connect background progress to the Qt thread."""
        if state:
            self.progress_changed.connect(self.status.setText)
            self.outcome_ready.connect(self._record_outcome)

    def _button(self, parent: QWidget, text: str, callback: Any) -> Any:
        button = hp.make_btn(parent, text, func=callback)
        self._job_buttons.append(button)
        return button

    def _setup_ui(self) -> None:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        self.tabs = QTabWidget(panel)
        self.tabs.setMinimumWidth(500)
        self.tabs.addTab(self._make_inputs(), "1. Images")
        scroll = QScrollArea(panel)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._make_stains())
        self.tabs.addTab(scroll, "2. Stains and Reference")
        self.tabs.addTab(self._make_export(), "3. Export")
        splitter = QSplitter(Qt.Orientation.Horizontal, panel)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self._make_preview())
        splitter.setSizes([520, 1030])
        layout.addWidget(splitter, 1)
        self.status = hp.make_label(panel, "Add brightfield images to begin.", wrap=True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.export_btn = hp.make_btn(panel, "Export checked images", func=self.start_export)
        self.export_btn.setEnabled(False)
        self.cancel_btn = hp.make_btn(panel, "Cancel", func=self.cancel)
        self.cancel_btn.setEnabled(False)
        layout.addLayout(hp.make_h_layout(self.export_btn, self.cancel_btn))
        self.setCentralWidget(panel)
        menu = hp.make_menu(self, "File")
        self.add_action = hp.make_menu_item(self, "Add images...", "Ctrl+I", menu=menu, func=self.choose_images)
        self.load_action = hp.make_menu_item(self, "Load recipe...", menu=menu, func=self.choose_recipe)
        self.save_action = hp.make_menu_item(self, "Save recipe...", menu=menu, func=self.choose_save_recipe)
        hp.make_menu_item(self, "Quit", menu=menu, func=self.close)
        menubar = QMenuBar(self)
        for item in (menu, self._make_apps_menu(), self._make_tools_menu(), self._make_help_menu()):
            menubar.addMenu(item)
        self.setMenuBar(menubar)
        self._make_icon()
        self._make_statusbar()
        self.tutorial_btn.hide()
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(250)
        self.preview_timer.timeout.connect(self._start_preview)

    def _make_inputs(self) -> QWidget:
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        box.addLayout(
            hp.make_h_layout(
                self._button(panel, "Add images...", self.choose_images),
                self._button(panel, "Add OME-Zarr...", partial(self.choose_images, True)),
            )
        )
        self.table = QTableWidget(0, 4, panel)
        self.table.setHorizontalHeaderLabels(["Use", "Image", "Scene", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        cast(QHeaderView, self.table.horizontalHeader()).setStretchLastSection(True)
        for index, width in enumerate((38, 140, 75, 130)):
            self.table.setColumnWidth(index, width)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemChanged.connect(self.invalidate)
        box.addWidget(self.table, 1)
        box.addWidget(self._button(panel, "Remove selected", self.remove_selected))
        box.addWidget(self._button(panel, "Use selected image as reference", self.use_selected_reference))
        box.addWidget(
            hp.make_label(
                panel,
                "Choose a scene for multi-scene images. Inputs must be 2D uint8/uint16 brightfield RGB.",
                wrap=True,
            )
        )
        return panel

    def _make_stains(self) -> QWidget:
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        form = hp.make_form_layout()
        self.preset = hp.make_combobox(panel, ["H&E", "H-DAB", "Custom (2 stains)", "Custom (3 stains)"])
        self.preset.currentTextChanged.connect(self._preset_changed)
        self.vector_mode = hp.make_combobox(panel, ["Auto per slide", "Shared vectors"], func=self.invalidate)
        form.addRow("Stain type", self.preset)
        form.addRow("Batch vectors", self.vector_mode)
        box.addLayout(form)
        box.addWidget(hp.make_label(panel, "Reference slide (optional for separation)", bold=True))
        self.reference_scene = hp.make_combobox(panel)
        self.reference_scene.addItem("No reference", None)
        self.reference_scene.currentIndexChanged.connect(self._reference_changed)
        box.addWidget(self.reference_scene)
        box.addLayout(
            hp.make_h_layout(
                self._button(panel, "Load reference...", self.choose_reference),
                self._button(panel, "Zarr...", partial(self.choose_reference, True)),
                self._button(panel, "Clear", self.clear_reference),
            )
        )
        self.normalization = hp.make_combobox(
            panel, ["Match colors only", "Match colors and intensity"], func=self.invalidate
        )
        box.addWidget(self.normalization)
        box.addWidget(
            hp.make_label(
                panel,
                "Intensity matching changes normalized RGB only. Raw stain channels retain signed OD coefficients.",
                wrap=True,
            )
        )
        self.editor_target = hp.make_combobox(
            panel, ["Selected source image", "Reference image"], func=self._update_editor
        )
        box.addWidget(self.editor_target)
        self.editor_label = hp.make_label(panel, "", wrap=True)
        self.editor_label.setTextFormat(Qt.TextFormat.PlainText)
        box.addWidget(self.editor_label)
        self.mapping: list[QComboBox] = []
        mapping_row = hp.make_h_layout()
        for color in ("R", "G", "B"):
            combo = hp.make_combobox(panel)
            combo.setToolTip(f"Source component used as {color}; select explicitly for unlabeled three-channel images.")
            combo.currentIndexChanged.connect(self._mapping_changed)
            mapping_row.addWidget(hp.make_label(panel, color))
            mapping_row.addWidget(combo)
            self.mapping.append(combo)
        box.addLayout(mapping_row)
        self.vector_table = QTableWidget(0, 5, panel)
        self.vector_table.setHorizontalHeaderLabels(["Stain", "Red OD", "Green OD", "Blue OD", "Color"])
        self.vector_table.verticalHeader().hide()
        header = cast(QHeaderView, self.vector_table.horizontalHeader())
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.vector_table.setColumnWidth(column, 100)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.vector_table.setColumnWidth(4, 45)
        self.vector_table.setMaximumHeight(155)
        box.addWidget(self.vector_table)
        white_row = hp.make_h_layout()
        white_row.addWidget(hp.make_label(panel, "Background RGB (0-1)"))
        self.white = [
            hp.make_double_spin_box(panel, minimum=0.0001, maximum=1, default=1, n_decimals=4) for _ in range(3)
        ]
        for spin in self.white:
            spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            white_row.addWidget(spin)
        box.addLayout(white_row)
        box.addLayout(
            hp.make_h_layout(
                self._button(panel, "Apply vector edits", self.apply_vector_edits),
                self._button(panel, "Reset image vectors", self.reset_vectors),
            )
        )
        box.addWidget(self._button(panel, "Use these source vectors for the batch", self.share_vectors))
        box.addWidget(self._button(panel, "Draw sampling rectangle", self.draw_region))
        self.roi_label = hp.make_label(panel, "No rectangle: estimation samples across the slide.", wrap=True)
        box.addWidget(self.roi_label)
        box.addLayout(
            hp.make_h_layout(
                self._button(panel, "Estimate two stains", partial(self.sample_profile, "estimate")),
                self._button(panel, "Clear rectangle", self.clear_region),
            )
        )
        self.sample_channel = hp.make_combobox(panel)
        box.addLayout(
            hp.make_h_layout(
                self.sample_channel, self._button(panel, "Sample this stain", partial(self.sample_profile, "stain"))
            )
        )
        box.addWidget(self._button(panel, "Sample blank background", partial(self.sample_profile, "background")))
        advanced = hp.make_advanced_collapsible(panel, allow_checkbox=False)
        self.min_od = hp.make_double_spin_box(
            panel, minimum=0.001, maximum=2.999, default=0.15, n_decimals=3, func=self.invalidate
        )
        self.max_od = hp.make_double_spin_box(
            panel, minimum=0.01, maximum=12, default=3, n_decimals=2, func=self.invalidate
        )
        self.angle_percentile = hp.make_double_spin_box(
            panel, minimum=0.01, maximum=24, default=1, n_decimals=2, func=self.invalidate
        )
        self.intensity_percentile = hp.make_double_spin_box(
            panel, minimum=90, maximum=99.99, default=99, n_decimals=2, func=self.invalidate
        )
        advanced.addRow("Minimum tissue OD", self.min_od)
        advanced.addRow("Maximum component OD", self.max_od)
        advanced.addRow("Angular percentile", self.angle_percentile)
        advanced.addRow("Intensity percentile", self.intensity_percentile)
        box.addWidget(advanced)
        box.addWidget(
            hp.make_label(
                panel,
                "Residual shows colors outside the two-stain model. Coefficients describe the optical model, not absolute stain quantities.",
                wrap=True,
            )
        )
        box.addStretch(1)
        return panel

    def _make_export(self) -> QWidget:
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        box.addWidget(hp.make_label(panel, "Original-slide stain channels", bold=True))
        self.channel_checks = [
            hp.make_checkbox(panel, name, default=index < 2, func=self.invalidate)
            for index, name in enumerate(self.profile.channel_names)
        ]
        for check in self.channel_checks:
            box.addWidget(check)
        self.channel_dtype = hp.make_combobox(
            panel, ["Float32 - signed OD", "Uint16 - scaled for registration"], func=self.invalidate
        )
        box.addWidget(self.channel_dtype)
        box.addWidget(
            hp.make_label(
                panel,
                "Uint16 clips negative values and uses one recorded percentile scale per slide, never per tile.",
                wrap=True,
            )
        )
        self.normalized_output = hp.make_checkbox(panel, "Also export normalized RGB", func=self.invalidate)
        box.addWidget(self.normalized_output)
        self.output_format = hp.make_combobox(panel, ["OME-TIFF", "OME-Zarr"])
        box.addWidget(self.output_format)
        self.output_directory = hp.make_line_edit(
            panel, text=str(self.CONFIG.output_dir or ""), placeholder="Output directory"
        )
        box.addLayout(hp.make_h_layout(self.output_directory, self._button(panel, "Browse...", self.choose_output)))
        box.addWidget(
            hp.make_label(
                panel,
                "Creates new pyramidal images and JSON records. TIFF export uses temporary disk space for a chunked pyramid.",
                wrap=True,
            )
        )
        self.batch_report = hp.make_text_edit(panel)
        self.batch_report.setReadOnly(True)
        box.addWidget(self.batch_report, 1)
        return panel

    def _make_preview(self) -> QWidget:
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        self.display_mode = hp.make_combobox(
            panel, ["Separated channels", "Reference normalization"], func=self._render_preview
        )
        self.pseudocolor = hp.make_checkbox(panel, "Stain colors", default=True, func=self._render_preview)
        self.display_limit = hp.make_double_spin_box(
            panel, minimum=0.01, maximum=20, default=2, n_decimals=2, func=self._render_preview
        )
        box.addLayout(
            hp.make_h_layout(
                self.display_mode, self.pseudocolor, hp.make_label(panel, "Display OD max"), self.display_limit
            )
        )
        self.native_region = hp.make_checkbox(panel, "Full-resolution crop", func=self.invalidate)
        self.region_y = hp.make_int_spin_box(
            panel, minimum=0, maximum=2**31 - 1, default=0, prefix="Y ", func=self.invalidate
        )
        self.region_x = hp.make_int_spin_box(
            panel, minimum=0, maximum=2**31 - 1, default=0, prefix="X ", func=self.invalidate
        )
        self.region_size = hp.make_int_spin_box(
            panel, minimum=32, maximum=1024, default=512, prefix="Size ", func=self.invalidate
        )
        box.addLayout(hp.make_h_layout(self.native_region, self.region_x, self.region_y, self.region_size))
        grid = QGridLayout()
        self.views: list[NapariImageView] = []
        self.view_labels = []
        for index in range(4):
            part = QWidget(panel)
            layout = QVBoxLayout(part)
            label = hp.make_label(part, ("Original", "Stain 1", "Stain 2", "Residual")[index])
            label.setTextFormat(Qt.TextFormat.PlainText)
            layout.addWidget(label)
            self.view_labels.append(label)
            view = NapariImageView(
                part, add_toolbars=False, disable_new_layers=True, allow_crosshair=False, disable_controls=True
            )
            layout.addWidget(view.widget)
            view.viewer.camera.events.zoom.connect(partial(self._sync_camera, index))
            view.viewer.camera.events.center.connect(partial(self._sync_camera, index))
            self.views.append(view)
            grid.addWidget(part, index // 2, index % 2)
        self.views[0].viewer.mouse_double_click_callbacks.append(self._pick_region)
        box.addLayout(grid, 1)
        self.statistics = hp.make_label(
            panel, "Select an image to preview. Double-click Original to inspect a native crop.", wrap=True
        )
        self.statistics.setTextFormat(Qt.TextFormat.PlainText)
        box.addWidget(self.statistics)
        return panel

    @staticmethod
    def _item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def selected_source(self) -> StainSource | None:
        """Return the explicitly chosen scene in the selected image row."""
        row = self.table.currentRow()
        return cast(QComboBox, self.table.cellWidget(row, 2)).currentData() if row >= 0 else None

    def _editor_source(self) -> StainSource | None:
        return self.reference_scene.currentData() if self.editor_target.currentIndex() else self.selected_source()

    def current_recipe(self) -> StainRecipe:
        """Freeze validated settings, profiles, and reference identity for a worker."""
        return StainRecipe(
            profile=self.profile,
            vector_mode="shared" if self.vector_mode.currentIndex() else "auto",
            estimation=StainEstimation(
                **{
                    **self.estimation.model_dump(),
                    "min_od": self.min_od.value(),
                    "max_od": self.max_od.value(),
                    "angular_percentile": self.angle_percentile.value(),
                    "concentration_percentile": self.intensity_percentile.value(),
                }
            ),
            overrides=tuple(self.overrides.values()),
            reference=self.reference_scene.currentData(),
            reference_profile=self.reference_profile,
            normalization="colors_intensity" if self.normalization.currentIndex() else "colors",
            export_channels=tuple(index for index, check in enumerate(self.channel_checks) if check.isChecked()),
            channel_dtype="uint16" if self.channel_dtype.currentIndex() else "float32",
            export_normalized=self.normalized_output.isChecked(),
        )

    def choose_images(self, zarr: bool = False) -> None:
        """Choose local images or an OME-Zarr image group."""
        paths = (
            [QFileDialog.getExistingDirectory(self, "Add OME-Zarr")]
            if zarr
            else QFileDialog.getOpenFileNames(self, "Add brightfield images")[0]
        )
        self.add_paths([path for path in paths if path])

    def add_paths(self, paths: Sequence[str]) -> None:
        """Inspect images in a background worker without loading full-resolution pixels."""
        if not paths or self.worker:
            return

        def inspect() -> tuple[list[tuple[StainSource, ...]], list[str]]:
            sources, errors = [], []
            for path in paths:
                if self._cancel.is_set():
                    raise InterruptedError("Image loading cancelled.")
                try:
                    sources.append(inspect_stain_file(path))
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append(f"{Path(path).name}: {exc}")
            return sources, errors

        self._start_worker("load", inspect, self._images_loaded)

    def _images_loaded(self, result: tuple[list[tuple[StainSource, ...]], list[str]]) -> None:
        self._updating = True
        for scenes in result[0]:
            if any(existing[0].path == scenes[0].path for existing in self.inputs):
                continue
            row = self.table.rowCount()
            self.inputs.append(scenes)
            self.table.insertRow(row)
            check = self._item("")
            check.setFlags(check.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, check)
            self.table.setItem(row, 1, self._item(scenes[0].path.name))
            self.table.setItem(row, 3, self._item("Pending preview"))
            combo = hp.make_combobox(self.table)
            if len(scenes) > 1:
                combo.addItem("Choose...", None)
            for source in scenes:
                combo.addItem(str(source.scene), source)
            combo.currentIndexChanged.connect(self._selection_changed)
            self.table.setCellWidget(row, 2, combo)
        if self.table.currentRow() < 0 and self.table.rowCount():
            self.table.selectRow(0)
        self._updating = False
        if result[1]:
            self.batch_report.append("\n".join(result[1]))
            self.status.setText("Some images could not be loaded. See Export for details.")
        self._selection_changed()

    def remove_selected(self) -> None:
        """Remove an input and its source override without changing files."""
        row = self.table.currentRow()
        if row < 0 or self.worker:
            return
        for source in self.inputs[row]:
            self.overrides.pop(source.key, None)
        self._updating = True
        self.inputs.pop(row)
        self.table.removeRow(row)
        self._updating = False
        self._selection_changed()

    def _selection_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self._preview = None
        self._preview_key = None
        for view in self.views:
            view.viewer.layers.clear()
        self._update_editor()
        self.invalidate()

    def _preset_changed(self, name: str) -> None:
        if self._updating:
            return
        self.profile = stain_preset(name)
        self.overrides.clear()
        self.reference_profile = None
        self._preview = None
        if len(self.profile.names) == 3:
            self.vector_mode.setCurrentIndex(1)
        self.channel_checks[2].setChecked(len(self.profile.names) == 3)
        self._update_channel_names()
        self._update_editor()
        self.invalidate()

    def _update_channel_names(self) -> None:
        self.vector_mode.model().item(0).setEnabled(len(self.profile.names) == 2)
        for check, name in zip(self.channel_checks, self.profile.channel_names, strict=True):
            check.setText(name)

    def _effective_profile(self) -> StainProfile:
        source = self._editor_source()
        if self.editor_target.currentIndex():
            if self.reference_profile:
                return self.reference_profile
            if self._preview and self._preview.reference_fit:
                return self._preview.reference_fit.profile
        elif source is not None:
            if source.key in self.overrides:
                return self.overrides[source.key].profile
            if self._preview:
                return self._preview.source_fit.profile
        return self.profile

    def _update_editor(self, *_args: Any) -> None:
        if self._updating or not hasattr(self, "white"):
            return
        self._updating = True
        source = self._editor_source()
        profile = self._effective_profile()
        target = "Reference" if self.editor_target.currentIndex() else "Source"
        self.editor_label.setText(f"{target}: {source.path.name if source else 'select an image'}")
        for index, combo in enumerate(self.mapping):
            combo.clear()
            combo.addItem("Select...", -1)
            if source:
                for channel, name in enumerate(source.channels):
                    combo.addItem(f"{channel}: {name}", channel)
                combo.setCurrentIndex(combo.findData(source.rgb_channels[index]) if source.rgb_channels else 0)
        self.vector_table.setRowCount(len(profile.names))
        self.vector_table.setFixedHeight(42 + 34 * len(profile.names))
        self.sample_channel.clear()
        for row, (name, vector) in enumerate(zip(profile.names, profile.vectors, strict=True)):
            label = hp.make_line_edit(self.vector_table, text=name)
            label.setClearButtonEnabled(False)
            label.setCursorPosition(0)
            label.setReadOnly(not self.profile.name.startswith("Custom") or bool(self.editor_target.currentIndex()))
            label.setToolTip("Custom stain names are shared across source and reference profiles.")
            self.vector_table.setCellWidget(row, 0, label)
            for column, value in enumerate(vector, 1):
                spin = hp.make_double_spin_box(
                    self.vector_table, minimum=0, maximum=1, default=value, n_decimals=6, step_size=0.01
                )
                spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
                self.vector_table.setCellWidget(row, column, spin)
            self.vector_table.setRowHeight(row, 34)
            color = np.rint(np.exp(-2 * np.asarray(vector)) * 255).astype(int)
            swatch = hp.make_label(self.vector_table, " ")
            swatch.setStyleSheet(f"background-color: rgb({color[0]}, {color[1]}, {color[2]});")
            self.vector_table.setCellWidget(row, 4, swatch)
            self.sample_channel.addItem(name, row)
        for spin, value in zip(self.white, profile.white, strict=True):
            spin.setValue(value)
        region = self._regions.get(source.key) if source else None
        self.roi_label.setText(
            f"Native rectangle (Y, X, height, width): {region}"
            if region
            else "No rectangle: estimation samples across the slide."
        )
        self._updating = False

    def _mapping_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        source = self._editor_source()
        if source is None:
            return
        mapping = tuple(combo.currentData() for combo in self.mapping)
        if sorted(mapping) != [0, 1, 2]:
            self.status.setText("Choose three distinct RGB components before previewing.")
            return
        updated = source.model_copy(update={"rgb_channels": mapping})
        if self.editor_target.currentIndex():
            self.reference_scene.setItemData(self.reference_scene.currentIndex(), updated)
            self.reference_profile = None
        else:
            row = self.table.currentRow()
            combo = cast(QComboBox, self.table.cellWidget(row, 2))
            combo.setItemData(combo.currentIndex(), updated)
            self.inputs[row] = tuple(updated if item.scene == source.scene else item for item in self.inputs[row])
            self.overrides.pop(source.key, None)
        self.invalidate()

    def _read_editor(self) -> StainProfile:
        names = tuple(
            self.vector_table.cellWidget(row, 0).text().strip() for row in range(self.vector_table.rowCount())
        )
        vectors = tuple(
            tuple(self.vector_table.cellWidget(row, col).value() for col in range(1, 4)) for row in range(len(names))
        )
        return StainProfile(
            name=self.profile.name, names=names, vectors=vectors, white=tuple(spin.value() for spin in self.white)
        )

    def _store_profile(self, source: StainSource, profile: StainProfile, reference: bool) -> None:
        if profile.names != self.profile.names:
            self.profile = StainProfile(**{**self.profile.model_dump(), "names": profile.names})
            self.overrides = {
                key: StainOverride(
                    source=item.source, profile=StainProfile(**{**item.profile.model_dump(), "names": profile.names})
                )
                for key, item in self.overrides.items()
            }
            if self.reference_profile:
                self.reference_profile = StainProfile(**{**self.reference_profile.model_dump(), "names": profile.names})
            self._update_channel_names()
        if reference:
            self.reference_profile = profile
        else:
            self.overrides[source.key] = StainOverride(source=source, profile=profile)
        self._preview = None
        self._update_editor()
        self.invalidate()

    def apply_vector_edits(self) -> None:
        """Validate editor values and store a source or reference override."""
        source = self._editor_source()
        if source is None:
            self.status.setText("Select the image whose vectors you want to edit.")
            return
        try:
            self._store_profile(source, self._read_editor(), bool(self.editor_target.currentIndex()))
        except ValueError as exc:
            self.status.setText(str(exc))

    def reset_vectors(self) -> None:
        """Restore automatic/shared settings for the selected source or reference."""
        source = self._editor_source()
        if self.editor_target.currentIndex():
            self.reference_profile = None
        elif source:
            self.overrides.pop(source.key, None)
        self._preview = None
        self._update_editor()
        self.invalidate()

    def share_vectors(self) -> None:
        """Explicitly replace source overrides with one batch-wide source profile."""
        if self.editor_target.currentIndex():
            self.status.setText("Switch the editor to Selected source image before sharing source vectors.")
            return
        try:
            profile = self._read_editor()
            source = self.selected_source()
            if source:
                self._store_profile(source, profile, False)
            self.profile = profile
            self.overrides.clear()
            self.vector_mode.setCurrentIndex(1)
            self._update_editor()
            self.invalidate()
        except ValueError as exc:
            self.status.setText(str(exc))

    def choose_reference(self, zarr: bool = False) -> None:
        """Inspect a selected reference asynchronously, retaining explicit scene selection."""
        if self.worker:
            return
        path = (
            QFileDialog.getExistingDirectory(self, "Reference OME-Zarr")
            if zarr
            else QFileDialog.getOpenFileName(self, "Reference slide")[0]
        )
        if path:
            self._start_worker("reference", partial(inspect_stain_file, path), self._reference_loaded)

    def _reference_loaded(self, sources: tuple[StainSource, ...]) -> None:
        self._updating = True
        self.reference_scene.clear()
        self.reference_scene.addItem("Choose reference scene...", None)
        for source in sources:
            self.reference_scene.addItem(f"{source.path.name} - scene {source.scene}", source)
        if len(sources) == 1:
            self.reference_scene.setCurrentIndex(1)
        self._updating = False
        self._reference_changed()

    def _reference_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self.reference_profile = None
        self._preview = None
        self._preview_key = None
        self.normalized_output.setChecked(self.reference_scene.currentData() is not None)
        self._update_editor()
        self.invalidate()

    def use_selected_reference(self) -> None:
        """Use the selected source scene as the normalization reference."""
        source = self.selected_source()
        if source:
            profile = self._preview.source_fit.profile if self._preview else None
            self._reference_loaded((source,))
            if profile is not None:
                self.reference_profile = profile
                self._update_editor()
                self.invalidate()

    def clear_reference(self) -> None:
        """Remove the reference and disable normalized RGB export."""
        self._updating = True
        self.reference_scene.clear()
        self.reference_scene.addItem("No reference", None)
        self._updating = False
        self._reference_changed()

    def draw_region(self) -> None:
        """Draw a native-coordinate rectangle on the source or reference image."""
        source = self._editor_source()
        if source is None:
            return
        reference = bool(self.editor_target.currentIndex())
        if reference:
            self.display_mode.setCurrentIndex(1)
        viewer = self.views[2 if reference else 0].viewer
        if "Pixels" not in viewer.layers:
            self.status.setText("Wait for the image preview before drawing a rectangle.")
            return
        if "Sampling rectangle" in viewer.layers:
            viewer.layers.remove("Sampling rectangle")
        layer = viewer.add_shapes(
            ndim=2, name="Sampling rectangle", face_color="transparent", edge_color="yellow", edge_width=2
        )
        layer.mode = "add_rectangle"

        def changed(_event: Any) -> None:
            if not len(layer.data):
                return
            vertices = np.asarray(layer.data[-1])
            low = np.maximum(np.floor(vertices.min(axis=0)).astype(int), 0)
            high = np.minimum(np.ceil(vertices.max(axis=0)).astype(int), source.shape)
            if np.any(high <= low):
                return
            self._regions[source.key] = (int(low[0]), int(low[1]), int(high[0] - low[0]), int(high[1] - low[1]))
            self.roi_label.setText(f"Native rectangle (Y, X, height, width): {self._regions[source.key]}")

        layer.events.data.connect(changed)

    def clear_region(self) -> None:
        """Remove the sampling region and return estimation to whole-slide sampling."""
        source = self._editor_source()
        if source:
            self._regions.pop(source.key, None)
        for view in self.views:
            if "Sampling rectangle" in view.viewer.layers:
                view.viewer.layers.remove("Sampling rectangle")
        self._update_editor()

    def sample_profile(self, kind: str) -> None:
        """Estimate vectors or sample one stain/background from the chosen native region."""
        source = self._editor_source()
        if source is None or self.worker:
            return
        region = self._regions.get(source.key)
        if kind != "estimate" and region is None:
            self.status.setText("Draw a rectangle around a pure stain or blank background first.")
            return
        try:
            recipe, profile = self.current_recipe(), self._read_editor()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        reference, index = bool(self.editor_target.currentIndex()), self.sample_channel.currentIndex()

        def compute() -> StainProfile:
            samples = sample_rgb(source, recipe, region, self._cancel.is_set)
            if kind == "estimate":
                return estimate_stain_profile(samples, profile, recipe.estimation)
            if kind == "background":
                return sample_background(samples, profile)
            return sample_stain_vector(samples, profile, index)

        self._start_worker("sample", compute, lambda result: self._store_profile(source, result, reference))

    def invalidate(self, *_args: Any) -> None:
        """Debounce computation and invalidate older worker generations."""
        if self._updating or not hasattr(self, "preview_timer"):
            return
        self._generation += 1
        self._preview_pending = True
        if self.worker and self._operation == "preview":
            self._cancel.set()
        self.preview_timer.start()
        self._validate_controls()

    def _validate_controls(self) -> None:
        try:
            self.current_recipe()
            valid = any(
                self.table.item(row, 0).checkState() == Qt.CheckState.Checked
                and cast(QComboBox, self.table.cellWidget(row, 2)).currentData() is not None
                for row in range(self.table.rowCount())
            )
        except ValueError:
            valid = False
        self.export_btn.setEnabled(valid and self.worker is None)

    def _start_preview(self) -> None:
        if self.worker or self._closing or not self._preview_pending:
            return
        self._preview_pending = False
        source = self.selected_source()
        if source is None:
            for view in self.views:
                view.viewer.layers.clear()
            return
        try:
            recipe = self.current_recipe()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        generation = self._generation
        region = None
        if self.native_region.isChecked():
            y, x, size = self.region_y.value(), self.region_x.value(), self.region_size.value()
            region = (y, x, min(size, source.shape[0] - y), min(size, source.shape[1] - x))

        def compute() -> Any:
            try:
                return preview_stains(source, recipe, region=region, cache=self._cache, cancelled=self._cancel.is_set)
            except ValueError as exc:
                raw = load_stain_overview(source, self._cancel.is_set)
                reference_raw = (
                    load_stain_overview(recipe.reference, self._cancel.is_set)
                    if recipe.reference and recipe.reference.rgb_channels
                    else None
                )
                return str(exc), raw, reference_raw

        def returned(result: Any) -> None:
            if generation != self._generation or self._closing:
                return
            if not isinstance(result, StainPreview):
                self._preview = None
                message, raw, reference_raw = result
                for view in self.views:
                    view.viewer.layers.clear()
                self._show_pixels(0, raw[0], "Original - adjust vectors or sample a region", raw[1], raw[2])
                if reference_raw:
                    self._show_pixels(2, reference_raw[0], "Reference", reference_raw[1], reference_raw[2])
                self.statistics.setText(message)
                self.status.setText(message)
                self._set_row_status(source, message)
                for view in self.views:
                    view.viewer.reset_view()
                return
            key = (source.key, region, result.original.shape)
            reset = key != self._preview_key
            self._preview_key, self._preview = key, result
            self._update_editor()
            self._render_preview()
            self._set_row_status(source, "Ready - manual" if source.key in self.overrides else "Ready")
            self.statistics.setText(
                f"{'Native crop' if region else 'Pyramid overview / center crop'}; origin YX {result.origin}; "
                f"{result.source_fit.sample_count:,} fitting tissue pixels. "
                f"Negative coefficients: {np.count_nonzero(result.channels < 0):,}. Display changes never affect export."
            )
            self.status.setText("Preview ready. Review separated channels and residual before exporting.")
            if reset:
                for view in self.views:
                    view.viewer.reset_view()
                self._sync_camera(0)

        self._start_worker("preview", compute, returned)

    def _set_row_status(self, source: StainSource, message: str) -> None:
        self._updating = True
        for row in range(self.table.rowCount()):
            item = cast(QComboBox, self.table.cellWidget(row, 2)).currentData()
            if item is not None and item.key == source.key:
                self.table.item(row, 3).setText(message)
                self.table.item(row, 3).setToolTip(message)
        self._updating = False

    def _show_pixels(
        self,
        index: int,
        pixels: np.ndarray,
        title: str,
        scale: tuple[float, float] = (1.0, 1.0),
        origin: tuple[int, int] = (0, 0),
        colormap: Any = "gray",
        limits: tuple[float, float] | None = None,
    ) -> None:
        viewer = self.views[index].viewer
        rgb = pixels.ndim == 3
        if "Pixels" in viewer.layers and viewer.layers["Pixels"].rgb != rgb:
            viewer.layers.remove("Pixels")
        if "Pixels" not in viewer.layers:
            viewer.add_image(
                pixels, name="Pixels", rgb=rgb, scale=scale, translate=origin, colormap=colormap, contrast_limits=limits
            )
        else:
            layer = viewer.layers["Pixels"]
            layer.data, layer.scale, layer.translate = pixels, scale, origin
            if not rgb:
                layer.colormap = colormap
                if limits is not None:
                    layer.contrast_limits = limits
        self.view_labels[index].setText(title)

    def _render_preview(self, *_args: Any) -> None:
        if self._preview is None or not hasattr(self, "views"):
            return
        from napari.utils.colormaps import Colormap

        result = self._preview
        self._show_pixels(0, result.original, "Original", result.scale, result.origin)
        if self.display_mode.currentIndex():
            if result.normalized is None or result.reference_image is None:
                for index in (1, 2):
                    self.views[index].viewer.layers.clear()
                    self.view_labels[index].setText("Select a reference slide")
            else:
                self._show_pixels(1, result.normalized, "Normalized RGB", result.scale, result.origin)
                self._show_pixels(
                    2,
                    result.reference_image,
                    "Reference (independent view)",
                    result.reference_scale,
                    result.reference_origin,
                )
            channels = [(3, 2)]
        else:
            if "Sampling rectangle" in self.views[2].viewer.layers:
                self.views[2].viewer.layers.remove("Sampling rectangle")
            channels = [(1, 0), (2, 1), (3, 2)]
        for view_index, channel in channels:
            name = result.source_fit.profile.channel_names[channel]
            cmap: Any = "gray"
            limits = (0.0, self.display_limit.value())
            if name == "Residual":
                cmap, limits = "coolwarm", (-self.display_limit.value(), self.display_limit.value())
            elif self.pseudocolor.isChecked():
                color = np.exp(-2 * np.asarray(result.source_fit.profile.vectors[channel]))
                cmap = Colormap([[1, 1, 1, 1], [*color, 1]])
            self._show_pixels(
                view_index, result.channels[..., channel], name, result.scale, result.origin, cmap, limits
            )
        self._sync_camera(0)

    def _sync_camera(self, index: int, *_args: Any) -> None:
        if self._syncing or not hasattr(self, "views"):
            return
        linked = (0, 1, 3) if self.display_mode.currentIndex() else (0, 1, 2, 3)
        if index not in linked:
            return
        self._syncing = True
        try:
            camera = self.views[index].viewer.camera
            for other in linked:
                if other != index:
                    target = self.views[other].viewer.camera
                    target.center, target.zoom = camera.center, camera.zoom
        finally:
            self._syncing = False

    def _pick_region(self, _viewer: Any, event: Any) -> None:
        if self.worker and self._operation != "preview":
            return
        y, x = event.position[-2:]
        self._updating = True
        self.region_y.setValue(max(0, int(y) - self.region_size.value() // 2))
        self.region_x.setValue(max(0, int(x) - self.region_size.value() // 2))
        self.native_region.setChecked(True)
        self._updating = False
        self.invalidate()

    def choose_output(self) -> None:
        """Choose a directory for new image products and parameter records."""
        directory = QFileDialog.getExistingDirectory(self, "Output directory", self.output_directory.text())
        if directory:
            self.output_directory.setText(directory)

    def start_export(self) -> None:
        """Freeze settings and export checked scenes; incomplete rows are reported explicitly."""
        if self.worker:
            return
        try:
            recipe = self.current_recipe()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        if not self.output_directory.text().strip():
            self.status.setText("Choose an output directory first.")
            return
        sources = []
        self.batch_report.clear()
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).checkState() != Qt.CheckState.Checked:
                continue
            source = cast(QComboBox, self.table.cellWidget(row, 2)).currentData()
            if source is None:
                self.batch_report.append(f"{self.inputs[row][0].path.name}: skipped; choose a scene")
            else:
                sources.append(source)
        if not sources:
            return
        self.preview_timer.stop()
        self._preview_pending = False
        self.results = []
        run = partial(
            export_stain_batch,
            tuple(sources),
            recipe,
            self.output_directory.text().strip(),
            output_format="ome.zarr" if self.output_format.currentIndex() else "ome.tiff",
            cache=self._cache,
            cancelled=self._cancel.is_set,
            progress=self.progress_changed.emit,
            completed=self.outcome_ready.emit,
        )
        self._start_worker("export", run, self._export_finished)

    def _record_outcome(self, outcome: StainOutcome) -> None:
        self.results.append(outcome)
        self.batch_report.append(f"{outcome.source.path.name}: {outcome.error or ', '.join(map(str, outcome.outputs))}")
        self._set_row_status(outcome.source, outcome.error or "Exported")

    def _export_finished(self, results: list[StainOutcome]) -> None:
        self.results = results
        self.status.setText(
            f"Exported {sum(bool(item.outputs) for item in results)}/{len(results)} images. See Export for results."
        )

    def apply_recipe(self, recipe: StainRecipe) -> None:
        """Restore a validated recipe without recomputing saved overrides or dropping settings."""
        self._updating = True
        self.profile = recipe.profile
        self.estimation = recipe.estimation
        self.overrides = {item.source.key: item for item in recipe.overrides}
        self.reference_profile = recipe.reference_profile
        self.preset.setCurrentText(recipe.profile.name)
        self.vector_mode.setCurrentIndex(recipe.vector_mode == "shared")
        self.reference_scene.clear()
        self.reference_scene.addItem(
            recipe.reference.path.name if recipe.reference else "No reference", recipe.reference
        )
        self.normalization.setCurrentIndex(recipe.normalization == "colors_intensity")
        self.min_od.setValue(recipe.estimation.min_od)
        self.max_od.setValue(recipe.estimation.max_od)
        self.angle_percentile.setValue(recipe.estimation.angular_percentile)
        self.intensity_percentile.setValue(recipe.estimation.concentration_percentile)
        for index, check in enumerate(self.channel_checks):
            check.setChecked(index in recipe.export_channels)
        self.channel_dtype.setCurrentIndex(recipe.channel_dtype == "uint16")
        self.normalized_output.setChecked(recipe.export_normalized)
        self._preview = None
        self._updating = False
        self._update_channel_names()
        self._update_editor()
        self.invalidate()

    def choose_recipe(self) -> None:
        """Load a reusable JSON stain recipe."""
        path = QFileDialog.getOpenFileName(self, "Load stain recipe", filter="JSON (*.json)")[0]
        if path:
            try:
                self.apply_recipe(load_stain_recipe(path))
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))

    def choose_save_recipe(self) -> None:
        """Save the current validated profiles and batch settings."""
        path = QFileDialog.getSaveFileName(self, "Save stain recipe", "stains.json", "JSON (*.json)")[0]
        if path:
            try:
                save_stain_recipe(path, self.current_recipe())
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))

    def _start_worker(self, operation: str, function: Any, returned: Any) -> None:
        self._cancel.clear()
        self._operation = operation
        self.worker = create_worker(
            function,
            _start_thread=False,
            _connect={"returned": returned, "errored": self._worker_error, "finished": self._worker_finished},
        )
        self._set_busy(True)
        self.status.setText(f"{operation.capitalize()} in progress...")
        self.worker.start()

    def _set_busy(self, busy: bool) -> None:
        self.tabs.setEnabled(not busy or self._operation == "preview")
        for button in self._job_buttons:
            button.setEnabled(not busy)
        for action in (self.add_action, self.load_action, self.save_action):
            action.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)
        self.export_btn.setEnabled(not busy)
        self.allow_drop = not busy
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(busy)

    def _worker_error(self, error: Exception) -> None:
        if self._operation == "preview" and isinstance(error, InterruptedError) and self._preview_pending:
            return
        self.status.setText(str(error))
        if self._operation == "export":
            self.batch_report.append(str(error))

    def _worker_finished(self) -> None:
        self.worker = None
        self._operation = ""
        self._set_busy(False)
        self._validate_controls()
        if self._closing:
            QTimer.singleShot(0, self.close)
        elif self._preview_pending:
            self.preview_timer.start()

    def cancel(self) -> None:
        """Request cancellation, keeping readers and viewers alive until the worker exits."""
        self._generation += 1
        self._preview_pending = False
        self.preview_timer.stop()
        self._cancel.set()
        self.cancel_btn.setEnabled(False)
        self.status.setText("Cancelling; waiting for the current tile and cleanup...")

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Accept local image files and OME-Zarr directories while idle."""
        if event is not None and self.allow_drop and event.mimeData() is not None:
            self.add_paths([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Cancel work before closing viewers or saving window preferences."""
        if event is None:
            return
        self._closing = True
        self.preview_timer.stop()
        if self.worker:
            self.cancel()
            event.ignore()
            return
        self.CONFIG.output_dir = self.output_directory.text().strip()
        self.CONFIG.save()
        for view in self.views:
            view.widget.close()
        if self._console:
            self._console.close()
        event.accept()
