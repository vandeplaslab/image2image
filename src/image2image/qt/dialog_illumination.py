"""Interactive illumination correction with native-region previews and a shared batch recipe."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from threading import Event
from typing import Any, cast

import qtextra.helpers as hp
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QCloseEvent, QDropEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHeaderView,
    QMenuBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from image2image.config import SingleAppConfig
from image2image.models.illumination import CorrectionOutcome, CorrectionRecipe, IlluminationSource
from image2image.qt._dialog_base import Window
from image2image.utils.illumination import (
    CorrectionPreview,
    convert_intensity,
    export_correction_batch,
    inspect_illumination_file,
    load_correction_recipe,
    preview_correction,
    save_correction_recipe,
    validate_recipe,
)


class IlluminationConfig(SingleAppConfig):
    """Store window preferences independently from other mini apps."""

    USER_CONFIG_FILENAME = "config.illumination.json"


class ImageIlluminationWindow(Window):
    """Tune a correction recipe, compare pixels, and export compatible 2D scenes."""

    APP_NAME = "illumination"
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
        self.CONFIG = IlluminationConfig(_auto_load=True)  # type: ignore[call-arg]
        self.inputs: list[tuple[IlluminationSource, ...]] = []
        self.recipe_channels: tuple[str, ...] = ()
        self.worker: Any = None
        self._operation = ""
        self._cancel = Event()
        self._closing = False
        self._updating = False
        self._syncing = False
        self._generation = 0
        self._preview_pending = False
        self._preview: CorrectionPreview | None = None
        self._last_preview_key: tuple[Any, ...] | None = None
        self._calibration_cache: dict[str, tuple[float, ...]] = {}
        self.results: list[CorrectionOutcome] = []
        self._mapping_references: dict[str, IlluminationSource | None] = {}
        self._mapped_channels: tuple[str, ...] = ()
        self._export_skipped: list[str] = []
        super().__init__(
            parent, "image2image: Illumination and Background Correction", run_check_version=run_check_version
        )
        self.resize(1550, 950)
        paths = list(image_path or ())
        if image_dir:
            paths.extend(str(p) for p in sorted(Path(image_dir).iterdir()) if p.is_file() or p.name.endswith(".zarr"))
        if paths:
            QTimer.singleShot(0, partial(self.add_paths, paths))

    @staticmethod
    def _setup_config() -> None:
        """Use independent readers without registration preprocessing."""

    def setup_events(self, state: bool = True) -> None:
        """Connect worker messages to the UI thread."""
        if state:
            self.progress_changed.connect(self.status.setText)
            self.outcome_ready.connect(self._record_outcome)

    def _setup_ui(self) -> None:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.tabs = QTabWidget(widget)
        self.tabs.setMinimumWidth(390)
        self.tabs.addTab(self._make_inputs(), "Images")
        self.tabs.addTab(self._make_correction(), "Correction")
        self.tabs.addTab(self._make_output(), "Export")
        preview = self._make_preview()
        splitter = QSplitter(Qt.Orientation.Horizontal, widget)
        splitter.addWidget(self.tabs)
        splitter.addWidget(preview)
        splitter.setSizes([450, 1100])
        layout.addWidget(splitter, 1)
        self.status = hp.make_label(
            widget, "Add images to begin. Display contrast never changes exported values.", wrap=True
        )
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.status)
        self.cancel_btn = hp.make_btn(widget, "Cancel", func=self.cancel)
        self.cancel_btn.setEnabled(False)
        self.export_btn = hp.make_btn(widget, "Export checked images", func=self.start_export)
        self.export_btn.setEnabled(False)
        layout.addLayout(hp.make_h_layout(self.export_btn, self.cancel_btn))
        self.setCentralWidget(widget)
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
        self.add_btn = hp.make_btn(panel, "Add images...", func=self.choose_images)
        self.zarr_btn = hp.make_btn(panel, "Add OME-Zarr...", func=self.choose_zarr)
        self.remove_btn = hp.make_btn(panel, "Remove selected", func=self.remove_selected)
        box.addLayout(hp.make_h_layout(self.add_btn, self.zarr_btn))
        self.table = QTableWidget(0, 4, panel)
        self.table.setHorizontalHeaderLabels(["Use", "Image", "Scene", "Status"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        cast(QHeaderView, self.table.horizontalHeader()).setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        for column, width in enumerate((35, 140, 85, 130)):
            self.table.setColumnWidth(column, width)
        cast(QHeaderView, self.table.horizontalHeader()).setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemChanged.connect(self.invalidate)
        box.addWidget(self.table, 1)
        box.addWidget(self.remove_btn)
        box.addWidget(hp.make_btn(panel, "Use selected image's channel layout", func=self.use_selected_channels))
        box.addWidget(
            hp.make_label(
                panel,
                "Select a row to preview it. All channels are exported in source order. Multi-scene files require a scene choice.",
                wrap=True,
            )
        )
        return panel

    def _make_correction(self) -> QWidget:
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        self.reference_enabled: dict[str, Any] = {}
        self.reference_scene: dict[str, Any] = {}
        self.reference_buttons: list[Any] = []
        for kind in ("dark", "flat"):
            enabled = hp.make_checkbox(panel, f"Apply {kind} frame", func=self.invalidate)
            combo = hp.make_combobox(panel)
            combo.setMinimumContentsLength(12)
            combo.currentIndexChanged.connect(self._references_changed)
            button = hp.make_btn(panel, "Load...", func=partial(self.choose_reference, kind))
            zarr = hp.make_btn(panel, "Zarr...", func=partial(self.choose_reference, kind, True))
            self.reference_enabled[kind], self.reference_scene[kind] = enabled, combo
            self.reference_buttons.extend([button, zarr])
            box.addLayout(hp.make_h_layout(enabled, button, zarr))
            box.addWidget(combo)
        self.flat_corrected = hp.make_checkbox(panel, "Flat already dark-corrected", func=self.invalidate)
        box.addWidget(self.flat_corrected)
        self.mapping = QTableWidget(0, 3, panel)
        self.mapping.setHorizontalHeaderLabels(["Source channel", "Dark channel", "Flat channel"])
        cast(QHeaderView, self.mapping.horizontalHeader()).setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.mapping.setMaximumHeight(210)
        box.addWidget(self.mapping)
        form = hp.make_form_layout()
        self.method = hp.make_combobox(panel)
        for label, value in (
            ("None", "none"),
            ("Gaussian", "gaussian"),
            ("Morphology", "morphology"),
            ("Rolling ball", "rolling_ball"),
        ):
            self.method.addItem(label, value)
        self.method.currentIndexChanged.connect(self.invalidate)
        self.polarity = hp.make_combobox(panel)
        self.polarity.addItem("Bright features", "bright")
        self.polarity.addItem("Dark features", "dark")
        self.polarity.currentIndexChanged.connect(self.invalidate)
        self.sigma = hp.make_double_spin_box(
            panel, minimum=0.1, maximum=4096, default=50, suffix=" px", func=self.invalidate
        )
        self.radius = hp.make_int_spin_box(
            panel, minimum=1, maximum=4096, default=50, suffix=" px", func=self.invalidate
        )
        self.ball_height = hp.make_double_spin_box(
            panel, minimum=0.000001, maximum=1e15, default=50, n_decimals=6, func=self.invalidate
        )
        self.minimum_response = hp.make_double_spin_box(
            panel, minimum=1e-9, maximum=0.99, default=1e-6, n_decimals=9, step_size=1e-6, func=self.invalidate
        )
        form.addRow("Background estimator", self.method)
        form.addRow("Feature polarity", self.polarity)
        form.addRow("Gaussian sigma", self.sigma)
        form.addRow("Morphology / ball radius", self.radius)
        form.addRow("Ball height (intensity units)", self.ball_height)
        form.addRow("Minimum normalized flat", self.minimum_response)
        box.addLayout(form)
        box.addWidget(
            hp.make_label(
                panel,
                "Order: subtract dark, divide by normalized flat, subtract estimated background. Dark features retain negative residuals. Morphology uses a square footprint. Invalid calibration blocks export.",
                wrap=True,
            )
        )
        box.addStretch(1)
        return panel

    def _make_output(self) -> QWidget:
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        form = hp.make_form_layout()
        self.output_directory = hp.make_line_edit(panel, self.CONFIG.output_dir)
        self.format = hp.make_combobox(panel)
        self.format.addItem("OME-TIFF", "ome.tiff")
        self.format.addItem("OME-Zarr", "ome.zarr")
        self.dtype = hp.make_combobox(panel)
        for label, value in (
            ("Float32 (retain negatives)", "float32"),
            ("Source dtype", "source"),
            ("Uint8 (clip)", "uint8"),
            ("Uint16 (clip)", "uint16"),
        ):
            self.dtype.addItem(label, value)
        self.dtype.currentIndexChanged.connect(self.invalidate)
        form.addRow("Directory", self.output_directory)
        form.addRow(hp.make_btn(panel, "Choose directory...", func=self.choose_output))
        form.addRow("Format", self.format)
        form.addRow("Output dtype", self.dtype)
        box.addLayout(form)
        box.addWidget(
            hp.make_label(
                panel,
                "Float32 retains negatives. Integer output clips to its type's limits and rounds to nearest even. No automatic intensity rescaling. Every image includes a JSON parameter record; existing outputs are never overwritten.",
                wrap=True,
            )
        )
        box.addWidget(
            hp.make_label(
                panel,
                "OME-TIFF and OME-Zarr use lossless compression and pyramids. TIFF export uses temporary disk space for a chunked working image. RGB is stored as separate R/G/B channels.",
                wrap=True,
            )
        )
        self.batch_report = hp.make_text_edit(panel, read_only=True)
        box.addWidget(self.batch_report, 1)
        return panel

    def _make_preview(self) -> QWidget:
        panel = QWidget(self)
        box = QVBoxLayout(panel)
        self.channel = hp.make_combobox(panel, func=self.invalidate)
        self.preview_mode = hp.make_combobox(
            panel, ["Overview (approximate)", "Full-resolution region"], func=self.invalidate
        )
        self.display_mode = hp.make_combobox(panel, ["Corrected", "Estimated background"], func=self._render_preview)
        box.addLayout(hp.make_h_layout(self.channel, self.preview_mode, self.display_mode))
        self.region_x = hp.make_int_spin_box(
            panel, minimum=0, maximum=2**31 - 1, default=0, prefix="X ", func=self.invalidate
        )
        self.region_y = hp.make_int_spin_box(
            panel, minimum=0, maximum=2**31 - 1, default=0, prefix="Y ", func=self.invalidate
        )
        self.region_size = hp.make_int_spin_box(
            panel, minimum=16, maximum=1024, default=512, prefix="Size ", func=self.invalidate
        )
        box.addLayout(hp.make_h_layout(self.region_x, self.region_y, self.region_size))
        box.addWidget(
            hp.make_label(
                panel,
                "Double-click the original to center a full-resolution region. Coordinates are native pixels.",
                wrap=True,
            )
        )
        row = hp.make_h_layout()
        self.views: list[NapariImageView] = []
        for index, title in enumerate(("Original", "Corrected / background")):
            part = QWidget(panel)
            layout = QVBoxLayout(part)
            layout.addWidget(hp.make_label(part, title))
            view = NapariImageView(
                part, add_toolbars=False, disable_new_layers=True, allow_crosshair=False, disable_controls=True
            )
            layout.addWidget(view.widget)
            view.viewer.camera.events.zoom.connect(partial(self._sync_camera, index))
            view.viewer.camera.events.center.connect(partial(self._sync_camera, index))
            self.views.append(view)
            row.addWidget(part)
        self.views[0].viewer.mouse_double_click_callbacks.append(self._pick_region)
        box.addLayout(row, 1)
        self.shared_limits = hp.make_checkbox(panel, "Shared display limits", default=True, func=self._render_preview)
        self.display_min = hp.make_double_spin_box(
            panel, minimum=-1e100, maximum=1e100, default=0, n_decimals=4, func=self._render_preview
        )
        self.display_max = hp.make_double_spin_box(
            panel, minimum=-1e100, maximum=1e100, default=65535, n_decimals=4, func=self._render_preview
        )
        box.addLayout(
            hp.make_h_layout(
                self.shared_limits,
                self.display_min,
                self.display_max,
                hp.make_btn(panel, "Fit both", func=self.fit_display),
            )
        )
        self.statistics = hp.make_label(panel, "No preview", wrap=True)
        self.statistics.setTextFormat(Qt.TextFormat.PlainText)
        box.addWidget(self.statistics)
        return panel

    @staticmethod
    def _item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def selected_source(self) -> IlluminationSource | None:
        """Return the explicitly selected scene in the highlighted row."""
        row = self.table.currentRow()
        return self._scene_at(row) if 0 <= row < len(self.inputs) else None

    def current_recipe(self) -> CorrectionRecipe:
        """Freeze widget state and explicit channel mappings into an immutable recipe."""
        references, mappings = {}, {}
        for column, kind in enumerate(("dark", "flat"), 1):
            enabled = self.reference_enabled[kind].isChecked()
            reference = self.reference_scene[kind].currentData() if enabled else None
            if enabled and reference is None:
                raise ValueError(f"Select the {kind} reference scene.")
            references[kind] = reference
            mappings[f"{kind}_channels"] = (
                tuple(self._mapped_channel(i, column) for i in range(self.mapping.rowCount())) if enabled else ()
            )
        return CorrectionRecipe.model_validate(
            {
                "channels": self.recipe_channels,
                **references,
                **mappings,
                "flat_dark_corrected": self.flat_corrected.isChecked(),
                "minimum_response": self.minimum_response.value(),
                "background": self.method.currentData(),
                "polarity": self.polarity.currentData(),
                "sigma": self.sigma.value(),
                "radius": self.radius.value(),
                "height": self.ball_height.value(),
                "output_dtype": self.dtype.currentData(),
            }
        )

    def _scene_at(self, row: int) -> IlluminationSource | None:
        return cast(IlluminationSource | None, cast(QComboBox, self.table.cellWidget(row, 2)).currentData())

    def _mapped_channel(self, row: int, column: int) -> int:
        return int(cast(QComboBox, self.mapping.cellWidget(row, column)).currentData())

    def _rebuild_mappings(self, previous: CorrectionRecipe | None = None) -> None:
        old = {
            kind: tuple(self._mapped_channel(row, col) for row in range(self.mapping.rowCount()))
            for col, kind in enumerate(("dark", "flat"), 1)
        }
        self.mapping.setRowCount(len(self.recipe_channels))
        for row, name in enumerate(self.recipe_channels):
            self.mapping.setItem(row, 0, self._item(f"{row}: {name}"))
            for column, kind in enumerate(("dark", "flat"), 1):
                combo = hp.make_combobox(self.mapping)
                combo.addItem("Select explicitly...", -1)
                reference = self.reference_scene[kind].currentData()
                if reference:
                    for index, channel in enumerate(reference.channels):
                        combo.addItem(f"{index}: {channel}", index)
                    matches = [i for i, channel in enumerate(reference.channels) if channel == name]
                    unchanged = (
                        self._mapped_channels == self.recipe_channels
                        and self._mapping_references.get(kind) == reference
                    )
                    mapping = getattr(previous, f"{kind}_channels") if previous else old[kind] if unchanged else ()
                    selected = mapping[row] if row < len(mapping) else matches[0] if len(matches) == 1 else -1
                    combo.setCurrentIndex(max(0, combo.findData(selected)))
                combo.currentIndexChanged.connect(self.invalidate)
                self.mapping.setCellWidget(row, column, combo)
        self._mapped_channels = self.recipe_channels
        self._mapping_references = {kind: combo.currentData() for kind, combo in self.reference_scene.items()}

    def _references_changed(self, *_args: Any) -> None:
        if self._updating:
            return
        self._rebuild_mappings()
        self._calibration_cache.clear()
        self.invalidate()

    def _selection_changed(self) -> None:
        if self._updating:
            return
        source = self.selected_source()
        self._updating = True
        self.channel.clear()
        if source:
            self.channel.addItems([f"{i}: {name}" for i, name in enumerate(source.channels)])
            self.region_x.setMaximum(source.shape[1] - 1)
            self.region_y.setMaximum(source.shape[0] - 1)
            if not self.recipe_channels:
                self.recipe_channels = source.channels
                self._rebuild_mappings()
        self._updating = False
        self.invalidate()

    def use_selected_channels(self) -> None:
        """Explicitly adopt a new source layout for the shared batch recipe."""
        source = self.selected_source()
        if source:
            self.recipe_channels = source.channels
            self._rebuild_mappings()
            self.invalidate()

    def invalidate(self, *_args: Any) -> None:
        """Debounce processing and invalidate any older in-flight preview."""
        if self._updating or not hasattr(self, "preview_timer"):
            return
        self._generation += 1
        self._preview_pending = True
        self.statistics.setText("Preview out of date; updating...")
        self.sigma.setEnabled(self.method.currentData() == "gaussian")
        self.radius.setEnabled(self.method.currentData() in ("morphology", "rolling_ball"))
        self.ball_height.setEnabled(self.method.currentData() == "rolling_ball")
        self.polarity.setEnabled(self.method.currentData() in ("morphology", "rolling_ball"))
        self._validate_rows()
        if self._operation == "preview":
            self._cancel.set()
        self.preview_timer.start()

    def _validate_rows(self) -> list[IlluminationSource]:
        valid = []
        self._updating = True
        try:
            try:
                recipe, error = self.current_recipe(), ""
            except ValueError as exc:
                recipe, error = None, str(exc)
            for row in range(len(self.inputs)):
                source = self._scene_at(row)
                message = error
                if source is None:
                    message = "Select a scene"
                elif recipe is not None:
                    try:
                        validate_recipe(source, recipe)
                    except ValueError as exc:
                        message = str(exc)
                cast(QTableWidgetItem, self.table.item(row, 3)).setText(
                    message or "Compatible; pixels checked during export"
                )
                if (
                    source
                    and not message
                    and cast(QTableWidgetItem, self.table.item(row, 0)).checkState() == Qt.CheckState.Checked
                ):
                    valid.append(source)
            self.export_btn.setEnabled(bool(valid) and self.worker is None)
        finally:
            self._updating = False
        return valid

    def choose_images(self) -> None:
        """Select microscopy files for the batch queue."""
        paths, _ = QFileDialog.getOpenFileNames(self, "Add 2D microscopy images")
        self.add_paths(paths)

    def choose_zarr(self) -> None:
        """Select an OME-Zarr image directory."""
        path = QFileDialog.getExistingDirectory(self, "Select OME-Zarr image")
        if path:
            self.add_paths([path])

    def add_paths(self, paths: list[str]) -> None:
        """Inspect new files in a worker, leaving multi-scene choices explicit."""
        if self.worker or not paths:
            return
        known = {str(scenes[0].path) for scenes in self.inputs}
        paths = list(dict.fromkeys(str(Path(p).resolve()) for p in paths if str(Path(p).resolve()) not in known))
        if not paths:
            return

        def inspect() -> tuple[list[tuple[IlluminationSource, ...]], list[str]]:
            found, errors = [], []
            for path in paths:
                if self._cancel.is_set():
                    raise InterruptedError("Import cancelled.")
                try:
                    found.append(inspect_illumination_file(path))
                except (OSError, ValueError, RuntimeError) as exc:
                    errors.append(f"{Path(path).name}: {exc}")
            return found, errors

        self._start_worker("import", inspect, self._images_loaded)

    def _images_loaded(self, result: tuple[list[tuple[IlluminationSource, ...]], list[str]]) -> None:
        found, errors = result
        self._updating = True
        for scenes in found:
            row = self.table.rowCount()
            self.inputs.append(scenes)
            self.table.insertRow(row)
            check = self._item("")
            check.setFlags(check.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Checked)
            self.table.setItem(row, 0, check)
            item = self._item(scenes[0].path.name)
            item.setToolTip(str(scenes[0].path))
            self.table.setItem(row, 1, item)
            combo = hp.make_combobox(self.table)
            if len(scenes) > 1:
                combo.addItem("Select scene...", None)
            for source in scenes:
                combo.addItem(f"Scene {source.scene}", source)
            combo.currentIndexChanged.connect(self._selection_changed)
            self.table.setCellWidget(row, 2, combo)
            self.table.setItem(row, 3, self._item(""))
        if self.table.currentRow() < 0 and self.inputs:
            self.table.selectRow(0)
        self._updating = False
        self._selection_changed()
        if errors:
            self.batch_report.setPlainText("\n".join(errors))
        self.status.setText(f"Added {len(found)} image(s). " + "\n".join(errors))

    def remove_selected(self) -> None:
        """Remove a queued input without changing its file."""
        row = self.table.currentRow()
        if row < 0 or self.worker:
            return
        self._updating = True
        self.inputs.pop(row)
        self.table.removeRow(row)
        self._updating = False
        self._selection_changed()

    def choose_reference(self, kind: str, directory: bool = False) -> None:
        """Load one prepared master calibration frame and its scene choices."""
        if self.worker:
            return
        path = (
            QFileDialog.getExistingDirectory(self, f"Select {kind} OME-Zarr")
            if directory
            else QFileDialog.getOpenFileName(self, f"Select master {kind} frame")[0]
        )
        if path:
            self._start_worker(
                "reference", partial(inspect_illumination_file, path), partial(self._reference_loaded, kind)
            )

    def _reference_loaded(self, kind: str, scenes: tuple[IlluminationSource, ...]) -> None:
        self._updating = True
        combo = self.reference_scene[kind]
        combo.clear()
        if len(scenes) > 1:
            combo.addItem("Select scene...", None)
        for source in scenes:
            combo.addItem(f"{source.path.name} / scene {source.scene}", source)
        self.reference_enabled[kind].setChecked(True)
        self._updating = False
        self._references_changed()

    def choose_recipe(self) -> None:
        """Load a saved recipe while retaining the current image queue."""
        path, _ = QFileDialog.getOpenFileName(self, "Load correction recipe", filter="JSON (*.json)")
        if path:
            try:
                self.apply_recipe(load_correction_recipe(path))
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))

    def apply_recipe(self, recipe: CorrectionRecipe) -> None:
        """Restore all numerical and reference choices from a validated recipe."""
        if self.worker:
            return
        self._updating = True
        self.recipe_channels = recipe.channels
        for kind in ("dark", "flat"):
            reference = getattr(recipe, kind)
            combo = self.reference_scene[kind]
            combo.clear()
            if reference:
                combo.addItem(f"{reference.path.name} / scene {reference.scene}", reference)
            self.reference_enabled[kind].setChecked(reference is not None)
        self._rebuild_mappings(recipe)
        self.flat_corrected.setChecked(recipe.flat_dark_corrected)
        self.minimum_response.setValue(recipe.minimum_response)
        self.method.setCurrentIndex(self.method.findData(recipe.background))
        self.polarity.setCurrentIndex(self.polarity.findData(recipe.polarity))
        self.sigma.setValue(recipe.sigma)
        self.radius.setValue(recipe.radius)
        self.ball_height.setValue(recipe.height)
        self.dtype.setCurrentIndex(self.dtype.findData(recipe.output_dtype))
        self._calibration_cache.clear()
        self._updating = False
        self.invalidate()

    def choose_save_recipe(self) -> None:
        """Save the shared correction settings as a reusable JSON recipe."""
        try:
            recipe = self.current_recipe()
            path, _ = QFileDialog.getSaveFileName(self, "Save correction recipe", "correction.json", "JSON (*.json)")
            if path:
                save_correction_recipe(path, recipe)
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))

    def choose_output(self) -> None:
        """Choose the directory for new corrected images and parameter records."""
        path = QFileDialog.getExistingDirectory(self, "Output directory", self.output_directory.text())
        if path:
            self.output_directory.setText(path)

    def _start_preview(self) -> None:
        if self.worker or self._closing or not self._preview_pending:
            return
        self._preview_pending = False
        source = self.selected_source()
        if source is None:
            self._preview = None
            self.statistics.setText("Select an image and scene to preview.")
            for view in self.views:
                view.viewer.layers.clear()
            return
        try:
            recipe = self.current_recipe()
            validate_recipe(source, recipe)
        except ValueError as exc:
            self.statistics.setText(str(exc))
            return
        generation = self._generation
        region = None
        if self.preview_mode.currentIndex() == 1:
            y, x = self.region_y.value(), self.region_x.value()
            size = self.region_size.value()
            region = (y, x, min(size, source.shape[0] - y), min(size, source.shape[1] - x))
        key = (source.path, source.scene, self.channel.currentIndex(), region)
        compute = partial(
            preview_correction,
            source,
            recipe,
            max(0, self.channel.currentIndex()),
            region=region,
            cancelled=self._cancel.is_set,
            calibration_cache=self._calibration_cache,
        )

        def returned(result: CorrectionPreview) -> None:
            if generation != self._generation or self._closing:
                return
            reset = key != self._last_preview_key
            self._last_preview_key = key
            self._preview = result
            if reset:
                with hp.qt_signals_blocked(self.display_min, self.display_max):
                    self.display_min.setValue(min(0, float(result.original.min())))
                    self.display_max.setValue(max(float(result.original.max()), self.display_min.value() + 1))
            self._render_preview()
            self.status.setText("Preview ready. Display limits do not affect correction or export.")
            if reset:
                self.views[0].viewer.reset_view()
                self._sync_camera(0)
            stats = result.statistics
            self.statistics.setText(
                f"{'Approximate overview' if region is None else 'Full-resolution region'}; "
                f"original {stats['original_minimum']:.6g} to {stats['original_maximum']:.6g}; "
                f"corrected before conversion {stats['minimum']:.6g} to {stats['maximum']:.6g}.\n"
                f"Preview uses export dtype {stats['output_dtype']}. "
                f"Negative pixels: {stats['negative']:,}/{stats['pixels']:,}; "
                f"integer clipping: low {stats['clipped_low']:,}, high {stats['clipped_high']:,}. "
                "Preview statistics only; export measures the complete image."
            )

        self._start_worker("preview", compute, returned)

    def _render_preview(self, *_args: Any) -> None:
        if self._preview is None or self._updating or not hasattr(self, "shared_limits"):
            return
        result = self._preview
        right = (
            convert_intensity(result.corrected, result.statistics["output_dtype"])
            if self.display_mode.currentIndex() == 0
            else result.background
        )
        for index, (view, data) in enumerate(zip(self.views, (result.original, right), strict=True)):
            limits = (self.display_min.value(), self.display_max.value())
            if index == 1 and not self.shared_limits.isChecked():
                limits = (float(data.min()), float(data.max()))
            if limits[1] <= limits[0]:
                limits = (limits[0], limits[0] + 1)
            if len(view.viewer.layers):
                layer = view.viewer.layers[0]
                layer.data = data
                layer.scale = (result.scale,) * 2
                layer.translate = result.origin
                layer.contrast_limits = limits
            else:
                view.viewer.add_image(
                    data,
                    name="Original" if index == 0 else "Result",
                    scale=(result.scale,) * 2,
                    translate=result.origin,
                    contrast_limits=limits,
                    colormap="gray",
                )

    def fit_display(self) -> None:
        """Fit shared display limits to both arrays without changing processing values."""
        if self._preview:
            with hp.qt_signals_blocked(self.display_min, self.display_max):
                self.display_min.setValue(
                    min(float(self._preview.original.min()), float(self._preview.corrected.min()))
                )
                self.display_max.setValue(
                    max(float(self._preview.original.max()), float(self._preview.corrected.max()))
                )
            self._render_preview()

    def _sync_camera(self, index: int, *_args: Any) -> None:
        if self._syncing or len(self.views) != 2:
            return
        self._syncing = True
        try:
            source, target = self.views[index].viewer.camera, self.views[1 - index].viewer.camera
            target.center, target.zoom = source.center, source.zoom
        finally:
            self._syncing = False

    def _pick_region(self, _viewer: Any, event: Any) -> None:
        if self.worker and self._operation != "preview":
            return
        y, x = event.position[-2:]
        self._updating = True
        self.region_y.setValue(max(0, int(y) - self.region_size.value() // 2))
        self.region_x.setValue(max(0, int(x) - self.region_size.value() // 2))
        self.preview_mode.setCurrentIndex(1)
        self._updating = False
        self.invalidate()

    def start_export(self) -> None:
        """Freeze a shared recipe and export compatible checked scenes sequentially."""
        if self.worker:
            return
        if not self._validate_rows():
            return
        sources = []
        self._export_skipped = []
        for row, scenes in enumerate(self.inputs):
            if cast(QTableWidgetItem, self.table.item(row, 0)).checkState() != Qt.CheckState.Checked:
                continue
            source = self._scene_at(row)
            if source is None:
                self._export_skipped.append(f"{scenes[0].path.name}: skipped; select a scene")
            else:
                sources.append(source)
        if not self.output_directory.text().strip():
            self.status.setText("Choose an output directory before exporting.")
            return
        self.preview_timer.stop()
        self._preview_pending = False
        recipe = self.current_recipe()
        self.results = []
        self.batch_report.clear()
        run = partial(
            export_correction_batch,
            tuple(sources),
            recipe,
            self.output_directory.text().strip(),
            output_format=self.format.currentData(),
            cancelled=self._cancel.is_set,
            progress=self.progress_changed.emit,
            completed=self.outcome_ready.emit,
        )
        self._start_worker("export", run, self._export_returned)

    def _record_outcome(self, outcome: CorrectionOutcome) -> None:
        self.results.append(outcome)
        self.batch_report.append(f"{outcome.source.path.name}: {outcome.error or str(outcome.output)}")

    def _export_returned(self, results: list[CorrectionOutcome]) -> None:
        self.results = results
        self.batch_report.setPlainText(
            "\n".join([*self._export_skipped, *(f"{r.source.path.name}: {r.error or str(r.output)}" for r in results)])
        )
        count = sum(result.output is not None for result in results)
        self.status.setText(
            f"Exported {count}/{len(results) + len(self._export_skipped)} checked image(s). See Export for results."
        )

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
        for button in (self.add_btn, self.zarr_btn, self.remove_btn, *self.reference_buttons):
            button.setEnabled(not busy)
        for action in (self.add_action, self.load_action, self.save_action):
            action.setEnabled(not busy)
        self.export_btn.setEnabled(not busy and bool(self.inputs))
        self.cancel_btn.setEnabled(busy)
        self.allow_drop = not busy
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(busy)

    def _worker_error(self, exc: Exception) -> None:
        if isinstance(exc, InterruptedError) and self._operation == "preview" and self._preview_pending:
            return
        self.status.setText(str(exc))
        if self._operation == "preview":
            self.statistics.setText(str(exc))
        elif self._operation == "export":
            self.batch_report.append(str(exc))

    def _worker_finished(self) -> None:
        self.worker = None
        self._operation = ""
        self._set_busy(False)
        self._validate_rows()
        if self._closing:
            QTimer.singleShot(0, self.close)
        elif self._preview_pending:
            self.preview_timer.start()

    def cancel(self) -> None:
        """Request cancellation and retain the worker until its readers and outputs are closed."""
        self._preview_pending = False
        self.preview_timer.stop()
        self._cancel.set()
        self.cancel_btn.setEnabled(False)
        self.status.setText("Cancelling; waiting for cleanup...")

    def dropEvent(self, event: QDropEvent | None) -> None:
        """Accept local microscopy paths when no worker owns the input state."""
        if event is not None and self.allow_drop:
            mime = event.mimeData()
            if mime is None:
                return
            self.add_paths([url.toLocalFile() for url in mime.urls() if url.isLocalFile()])
            event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Wait for cancellation before releasing Qt viewers or saving preferences."""
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
