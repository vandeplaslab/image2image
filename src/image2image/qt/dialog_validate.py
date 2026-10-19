"""Interactive independent registration validation and deformation analysis."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any

import numpy as np
import qtextra.helpers as hp
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QListWidget,
    QMenuBar,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker

from image2image.config import SingleAppConfig
from image2image.models.validation import (
    ImageGeometry,
    MaskCorrespondence,
    ValidationAnalysis,
    ValidationImage,
    ValidationLandmark,
    ValidationSession,
)
from image2image.qt._dialog_base import Window
from image2image.utils.validation import (
    AnalysisCancelled,
    analysis_grid,
    analyze_validation_session,
    discover_validation_candidates,
    export_validation_landmarks,
    import_validation_landmarks,
    inspect_validation_image,
    load_validation_session,
    open_validation_reader,
    require_same_pair,
    sample_image,
    save_validation_session,
)
from image2image.utils.validation_report import comparison_rows, export_validation_report, normalize_image


class ValidateConfig(SingleAppConfig):
    """Window preferences kept separate from registration app configuration."""

    USER_CONFIG_FILENAME = "config.validate.json"


class ImageValidationWindow(Window):
    """Compare frozen registrations using a separate set of validation annotations."""

    APP_NAME = "validate"
    progress_changed = Signal(int, str)

    def __init__(
        self,
        parent: QWidget | None = None,
        run_check_version: bool = True,
        project_dir: str | Path | None = None,
        **_kwargs: Any,
    ) -> None:
        self.CONFIG = ValidateConfig(_auto_load=True)
        self.session = ValidationSession()
        self.analysis: ValidationAnalysis | None = None
        self._generation = 0
        self._updating = False
        self._syncing = False
        self._dirty = False
        self._closing = False
        self._pending_pair: tuple[float, float] | None = None
        self._adding_pair = False
        self._workers: list[Any] = []
        self._cancel = Event()
        self._result_directory: TemporaryDirectory | None = None
        self._session_path: Path | None = None
        super().__init__(
            parent, "image2image: Registration Validation and Deformation Analysis", run_check_version=run_check_version
        )
        self.resize(1550, 950)
        if project_dir:
            self._try(partial(self.open_path, project_dir))

    @staticmethod
    def _setup_config() -> None:
        """Use reader-local configuration for validation inputs."""

    def setup_events(self, state: bool = True) -> None:
        """Connect worker progress to the Qt thread."""
        if state:
            self.progress_changed.connect(self._show_progress)

    def _try(self, function: Any) -> Any:
        try:
            return function()
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            self.status.setText(str(exc))
            hp.warn_pretty(self, str(exc), "Validation input")
            return None

    @staticmethod
    def _table(parent: QWidget, headings: list[str], editable: bool = True) -> QTableWidget:
        table = QTableWidget(0, len(headings), parent)
        table.setHorizontalHeaderLabels(headings)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setStretchLastSection(True)
        if not editable:
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        return table

    def _setup_ui(self) -> None:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.run_btn = hp.make_btn(widget, "Analyze", func=self.start_analysis)
        self.cancel_btn = hp.make_btn(widget, "Cancel", func=self.cancel_analysis)
        self.cancel_btn.setEnabled(False)
        self.export_btn = hp.make_btn(widget, "Export report...", func=self.choose_export)
        self.export_btn.setEnabled(False)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(widget, "New session", func=self.new_session),
                hp.make_btn(widget, "Open session...", func=self.choose_session),
                hp.make_btn(widget, "Save session...", func=self.choose_save),
                self.run_btn,
                self.cancel_btn,
                self.export_btn,
            )
        )
        self.tabs = QTabWidget(widget)
        self.tabs.addTab(self._make_inputs(), "Inputs")
        self.tabs.addTab(self._make_landmarks(), "Landmarks")
        self.tabs.addTab(self._make_masks(), "Masks")
        self.tabs.addTab(self._make_results(), "Results")
        self.tabs.setMinimumWidth(520)
        self.view_stack = QStackedWidget(widget)
        raw = QWidget(widget)
        raw_layout = hp.make_h_layout(parent=raw)
        self.views: list[NapariImageView] = []
        self.point_layers: list[Any] = []
        for index, name in enumerate(("Fixed", "Moving")):
            panel = QWidget(raw)
            box = QVBoxLayout(panel)
            box.addWidget(hp.make_label(panel, f"{name} image - native physical coordinates (um)"))
            view = NapariImageView(panel, add_toolbars=False, disable_new_layers=True, allow_crosshair=False)
            box.addWidget(view.widget)
            view.viewer.mouse_drag_callbacks.append(partial(self._pair_click, index))
            view.viewer.camera.events.zoom.connect(partial(self._sync_zoom, index))
            self.views.append(view)
            raw_layout.addWidget(panel)
        self.view_stack.addWidget(raw)
        self.result_view = NapariImageView(widget, add_toolbars=True, disable_new_layers=True, allow_crosshair=False)
        self.view_stack.addWidget(self.result_view.widget)
        splitter = QSplitter(Qt.Orientation.Horizontal, widget)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.view_stack)
        splitter.setSizes([550, 1000])
        layout.addWidget(splitter, 1)
        self.status = hp.make_label(widget, "Add registrations of one fixed/moving image pair.", wrap=True)
        layout.addWidget(self.status)
        self.setCentralWidget(widget)
        self._make_icon()
        self._make_statusbar()
        self.tutorial_btn.hide()
        menubar = QMenuBar(self)
        for menu in (self._make_tools_menu(), self._make_apps_menu(), self._make_help_menu()):
            menubar.addAction(menu.menuAction())
        self.setMenuBar(menubar)
        self.tabs.currentChanged.connect(self._tab_changed)
        self._refresh_inputs()

    def _make_inputs(self) -> QWidget:
        widget = QWidget(self)
        form = hp.make_form_layout(parent=widget)
        self.image_labels = []
        self.channels = []
        for index, name in enumerate(("Fixed", "Moving")):
            label = hp.make_label(widget, "No image", wrap=True)
            channel = hp.make_combobox(widget, func_index=partial(self._channel_changed, index))
            self.image_labels.append(label)
            self.channels.append(channel)
            form.addRow(hp.make_btn(widget, f"{name} image...", func=partial(self.choose_image, index)), label)
            form.addRow(f"{name} channel", channel)
        form.addRow(
            hp.make_h_layout(
                hp.make_btn(widget, "Add Register file...", func=self.choose_register),
                hp.make_btn(widget, "Add Elastix project...", func=self.choose_elastix),
            )
        )
        self.candidate_list = QListWidget(widget)
        form.addRow(self.candidate_list)
        form.addRow(hp.make_btn(widget, "Remove selected candidate", func=self.remove_candidate))
        self.downsample = hp.make_int_spin_box(widget, minimum=1, maximum=10000, value=1, func=self._resolution_changed)
        form.addRow("Requested analysis downsample", self.downsample)
        self.grid_label = hp.make_label(widget, "The common grid is capped at 4 million pixels.", wrap=True)
        form.addRow(self.grid_label)
        form.addRow(
            hp.make_label(
                widget,
                "Image paths and calibration must match across candidates. "
                "Analysis never fits or edits a registration.",
                wrap=True,
            )
        )
        return widget

    def _make_landmarks(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.units = hp.make_combobox(widget, ["px", "um"], func=self._units_changed)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(widget, "CSV units"),
                self.units,
                hp.make_btn(widget, "Import CSV...", func=self.choose_landmarks),
                hp.make_btn(widget, "Export CSV...", func=self.choose_landmark_export),
            )
        )
        layout.addWidget(
            hp.make_label(
                widget,
                "Table and views use um. Add a pair by clicking fixed, then moving. "
                "Drag existing points or edit coordinates below.",
                wrap=True,
            )
        )
        self.landmark_table = self._table(widget, ["Use", "ID", "Fixed X", "Fixed Y", "Moving X", "Moving Y", "Note"])
        self.landmark_table.itemChanged.connect(self._landmark_edited)
        self.landmark_table.itemSelectionChanged.connect(self.focus_landmark)
        layout.addWidget(self.landmark_table)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(widget, "Add pair", func=self.begin_pair),
                hp.make_btn(widget, "Delete pair", func=self.delete_pair),
            )
        )
        layout.addWidget(hp.make_label(widget, self.session.independence_note, wrap=True))
        return widget

    def _make_masks(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.mask_labels = []
        for index, name in enumerate(("Fixed", "Moving")):
            label = hp.make_label(widget, "No mask", wrap=True)
            self.mask_labels.append(label)
            layout.addLayout(
                hp.make_h_layout(hp.make_btn(widget, f"{name} mask...", func=partial(self.choose_mask, index)), label)
            )
        layout.addWidget(
            hp.make_label(
                widget,
                "Pair matching anatomical labels. Background 0 is excluded. "
                "Masks use nearest-neighbor sampling on the common fixed grid.",
                wrap=True,
            )
        )
        self.mask_table = self._table(widget, ["Region", "Fixed label", "Moving label"])
        self.mask_table.itemChanged.connect(self._mask_edited)
        layout.addWidget(self.mask_table)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(widget, "Add label pair", func=self.add_mask_pair),
                hp.make_btn(widget, "Delete label pair", func=self.delete_mask_pair),
                hp.make_btn(widget, "Clear masks", func=self.clear_masks),
            )
        )
        return widget

    def _make_results(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        self.summary_table = self._table(widget, ["Candidate", "Count", "Mean", "RMS", "P95", "Max", "Coverage"], False)
        layout.addWidget(
            hp.make_label(widget, "Validation errors: um in moving frame. No universal pass/fail threshold.", wrap=True)
        )
        layout.addWidget(self.summary_table)
        self.result_candidate = hp.make_combobox(widget, func_index=self.show_result)
        self.summary_table.itemSelectionChanged.connect(
            lambda: self.result_candidate.setCurrentIndex(self.summary_table.currentRow())
        )
        self.result_mode = hp.make_combobox(
            widget,
            [
                "Image overlap",
                "Validation errors",
                "Mask disagreement",
                "Area change (log2)",
                "Signed Jacobian",
                "Nonpositive Jacobian",
            ],
            func_index=lambda _: self.show_result(),
        )
        layout.addWidget(self.result_candidate)
        layout.addWidget(self.result_mode)
        self.result_detail = hp.make_label(widget, "Analyze to see results.", wrap=True)
        layout.addWidget(self.result_detail)
        self.error_table = self._table(widget, ["ID", "dX (um)", "dY (um)", "Error (um)", "Exclusion"], False)
        layout.addWidget(self.error_table)
        return widget

    def invalidate(self) -> None:
        """Cancel obsolete work and prevent stale measurements from being shown or exported."""
        if self._updating:
            return
        self._generation += 1
        self._cancel.set()
        self._dirty = True
        self.analysis = None
        self.export_btn.setEnabled(False)
        self.summary_table.setRowCount(0)
        self.error_table.setRowCount(0)
        self.result_view.viewer.layers.clear()
        with hp.qt_signals_blocked(self.result_candidate):
            self.result_candidate.clear()
        self.result_detail.setText("Inputs changed. Analyze again to update results.")
        if self._result_directory:
            self._result_directory.cleanup()
            self._result_directory = None

    def _refresh_inputs(self) -> None:
        self._updating = True
        try:
            for index, image in enumerate((self.session.fixed, self.session.moving)):
                self.image_labels[index].setText(
                    "No image"
                    if image is None
                    else f"{image.path}\nShape YX {image.geometry.shape}; spacing XY {image.geometry.spacing} um"
                )
                self.channels[index].clear()
                if image:
                    self.channels[index].addItems(image.channel_names or ["0"])
                    self.channels[index].setCurrentIndex(image.channel)
            self.candidate_list.clear()
            self.candidate_list.addItems([candidate.name for candidate in self.session.candidates])
            self.downsample.setValue(self.session.downsample)
            self.units.setCurrentText(self.session.landmark_units)
            if self.session.fixed:
                grid = analysis_grid(self.session)
                self.grid_label.setText(
                    f"Analysis grid YX {grid.shape}; spacing XY {grid.spacing} um. "
                    "Landmark precision is independent of downsampling."
                )
            for index, mask in enumerate((self.session.fixed_mask, self.session.moving_mask)):
                self.mask_labels[index].setText(
                    "No mask"
                    if mask is None
                    else f"{mask.path}\nSpacing XY {mask.geometry.spacing}; origin XY {mask.geometry.origin} um"
                )
        finally:
            self._updating = False
        self._refresh_landmarks()
        self._refresh_masks()

    def open_path(self, path: str | Path) -> None:
        """Open a saved validation session or add a native registration candidate."""
        if str(path).endswith(".i2validate.json"):
            session, changes = load_validation_session(path)
            self.invalidate()
            self.session = session
            self._session_path = Path(path)
            self._dirty = False
            self._refresh_inputs()
            self._load_views()
            self.status.setText(
                "\n".join(changes) if changes else "Session loaded. Analyze to compute current results."
            )
            return
        choices = discover_validation_candidates(path)
        if len(choices) > 1:
            name, ok = QInputDialog.getItem(
                self, "Select registered pair", "Registration path", [item[0].name for item in choices], editable=False
            )
            if not ok:
                return
            selected = next(item for item in choices if item[0].name == name)
        else:
            selected = choices[0]
        candidate, fixed, moving = selected
        selected_pair = self.session.model_copy(
            update={"fixed": self.session.fixed or fixed, "moving": self.session.moving or moving}
        )
        require_same_pair(selected_pair, fixed, moving)
        if any(c.path == candidate.path and c.source == candidate.source for c in self.session.candidates):
            raise ValueError("This candidate is already loaded.")
        self.invalidate()
        self.session.fixed, self.session.moving = selected_pair.fixed, selected_pair.moving
        self.session.candidates = [*self.session.candidates, candidate]
        self._refresh_inputs()
        self._load_views()
        self.status.setText("Candidate added. Supply independent landmarks or corresponding masks, then analyze.")

    def choose_register(self) -> None:
        """Choose a saved landmark registration."""
        path, _ = QFileDialog.getOpenFileName(self, "Add Register candidate", "", "Register (*.i2r.json *.i2r.toml)")
        if path:
            self._try(partial(self.open_path, path))

    def choose_elastix(self) -> None:
        """Choose a completed Elastix project directory."""
        path = QFileDialog.getExistingDirectory(self, "Add completed Elastix project")
        if path:
            self._try(partial(self.open_path, path))

    def choose_session(self) -> None:
        """Replace the current session after preserving unsaved annotations."""
        if not self._may_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open validation session", "", "Validation (*.i2validate.json)")
        if path:
            self._try(partial(self.open_path, path))

    def choose_save(self) -> bool:
        """Save editable evidence and transform references, without embedding image data."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save validation session",
            str(self._session_path or "comparison.i2validate.json"),
            "Validation (*.i2validate.json)",
        )
        if not path:
            return False
        saved = self._try(partial(save_validation_session, path, self.session))
        if saved:
            self._session_path, self._dirty = saved, False
            self.status.setText(f"Saved {saved}")
        return bool(saved)

    def _may_discard(self) -> bool:
        if not self._dirty:
            return True
        choice = QMessageBox.question(
            self,
            "Unsaved validation annotations",
            "Save this session before closing it?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        return choice == QMessageBox.StandardButton.Discard or (
            choice == QMessageBox.StandardButton.Save and self.choose_save()
        )

    def new_session(self) -> None:
        """Start a fresh comparison without silently discarding annotations."""
        if self._may_discard():
            self.invalidate()
            self.session = ValidationSession()
            self._session_path = None
            self._dirty = False
            self._refresh_inputs()
            self._load_views()

    def choose_image(self, index: int) -> None:
        """Select a native image before adding registrations or relink the same input."""
        path, _ = QFileDialog.getOpenFileName(self, "Choose native image")
        if path:
            image = self._try(partial(inspect_validation_image, path))
            if image:
                self.invalidate()
                setattr(self.session, "fixed" if index == 0 else "moving", image)
                self._refresh_inputs()
                self._load_views()

    def remove_candidate(self) -> None:
        """Remove the selected registration without touching its source files."""
        row = self.candidate_list.currentRow()
        if row >= 0:
            self.invalidate()
            self.session.candidates = [c for i, c in enumerate(self.session.candidates) if i != row]
            self._refresh_inputs()

    def _channel_changed(self, index: int, channel: int) -> None:
        if self._updating or channel < 0:
            return
        image = self.session.fixed if index == 0 else self.session.moving
        if image:
            self.invalidate()
            image.channel = channel
            self._load_views()

    def _resolution_changed(self, value: int) -> None:
        if not self._updating:
            self.session.downsample = value
            self.invalidate()
            self._refresh_inputs()

    def _units_changed(self, value: str) -> None:
        if not self._updating:
            self.session.landmark_units = value
            self._dirty = True

    def _load_views(self) -> None:
        self._updating = True
        try:
            for view in self.views:
                view.viewer.layers.clear()
            self.point_layers.clear()
            for index, image in enumerate((self.session.fixed, self.session.moving)):
                view = self.views[index]
                if image and Path(image.path).exists():
                    grid = analysis_grid(ValidationSession(fixed=image))
                    array, _ = sample_image(image, grid)
                    view.viewer.add_image(
                        array,
                        name=f"Image preview ({grid.spacing[0]:.4g}, {grid.spacing[1]:.4g} um/px)",
                        scale=grid.spacing[::-1],
                        translate=grid.origin[::-1],
                        units=("um", "um"),
                    )
                layer = view.viewer.add_points(
                    np.empty((0, 2)),
                    name="Validation landmarks",
                    size=6 * max(analysis_grid(ValidationSession(fixed=image)).spacing) if image else 6,
                    units=("um", "um"),
                )
                layer.mode = "select"
                layer.events.data.connect(partial(self._points_moved, index))
                layer.selected_data.events.active.connect(partial(self._point_selected, index))
                self.point_layers.append(layer)
                view.viewer.reset_view()
        finally:
            self._updating = False
        self._refresh_landmarks()
        QTimer.singleShot(0, self._reset_native_views)

    def _reset_native_views(self) -> None:
        if self._closing:
            return
        self._syncing = True
        try:
            for view in self.views:
                view.viewer.reset_view()
            zoom = min(view.viewer.camera.zoom for view in self.views)
            for view in self.views:
                view.viewer.camera.zoom = zoom
        finally:
            self._syncing = False

    def _refresh_landmarks(self, sync_layers: bool = True) -> None:
        self._updating = True
        try:
            self.landmark_table.setRowCount(len(self.session.landmarks))
            for row, point in enumerate(self.session.landmarks):
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
                use.setCheckState(Qt.CheckState.Checked if point.enabled else Qt.CheckState.Unchecked)
                self.landmark_table.setItem(row, 0, use)
                for col, value in enumerate((point.id, *point.fixed, *point.moving, point.note), 1):
                    self.landmark_table.setItem(row, col, QTableWidgetItem(str(value)))
            for index, layer in enumerate(self.point_layers if sync_layers else []):
                layer.data = np.asarray(
                    [p.fixed[::-1] if index == 0 else p.moving[::-1] for p in self.session.landmarks]
                ).reshape(-1, 2)
                layer.face_color = ["lime" if p.enabled else "gray" for p in self.session.landmarks] or "lime"
                layer.text = {"string": [p.id for p in self.session.landmarks], "color": "white", "size": 10}
        finally:
            self._updating = False

    def _landmark_edited(self, _item: QTableWidgetItem) -> None:
        if self._updating:
            return
        try:
            pairs = []
            for row in range(self.landmark_table.rowCount()):
                values = [self.landmark_table.item(row, col).text() for col in range(1, 7)]
                pairs.append(
                    ValidationLandmark(
                        id=values[0],
                        fixed=(float(values[1]), float(values[2])),
                        moving=(float(values[3]), float(values[4])),
                        note=values[5],
                        enabled=self.landmark_table.item(row, 0).checkState() == Qt.CheckState.Checked,
                    )
                )
            self.session.landmarks = pairs
            self.invalidate()
        except ValueError as exc:
            self.status.setText(f"Invalid landmark edit: {exc}")
        self._refresh_landmarks()

    def _points_moved(self, index: int, _event: Any = None) -> None:
        if self._updating:
            return
        if getattr(_event, "action", None) in ("changing", "adding", "removing"):
            return
        data = self.point_layers[index].data
        if len(data) != len(self.session.landmarks):
            self._refresh_landmarks()
            return
        pairs = [p.model_copy() for p in self.session.landmarks]
        for point, yx in zip(pairs, data, strict=True):
            setattr(point, "fixed" if index == 0 else "moving", tuple(yx[::-1]))
        self.session.landmarks = pairs
        self.invalidate()
        self._refresh_landmarks(sync_layers=False)

    def _point_selected(self, index: int, _event: Any = None) -> None:
        if not self._updating and self.point_layers[index].selected_data:
            self.landmark_table.selectRow(min(self.point_layers[index].selected_data))

    def focus_landmark(self) -> None:
        """Focus both native views on the same independent anatomical landmark."""
        if self._updating:
            return
        row = self.landmark_table.currentRow()
        if 0 <= row < len(self.session.landmarks):
            self._updating = True
            try:
                point = self.session.landmarks[row]
                for view, layer, xy in zip(self.views, self.point_layers, (point.fixed, point.moving), strict=True):
                    view.viewer.camera.center = (0, xy[1], xy[0])
                    layer.selected_data = {row}
            finally:
                self._updating = False

    def _sync_zoom(self, index: int, _event: Any = None) -> None:
        if self._syncing or self._updating or len(self.views) != 2:
            return
        self._syncing = True
        try:
            self.views[1 - index].viewer.camera.zoom = self.views[index].viewer.camera.zoom
        finally:
            self._syncing = False

    def begin_pair(self) -> None:
        """Begin a manual pair; no transformed location is inserted as validation evidence."""
        if not self.session.fixed or not self.session.moving:
            self.status.setText("Load both images before annotating.")
            return
        self._adding_pair, self._pending_pair = True, None
        self.tabs.setCurrentIndex(1)
        self.status.setText("Click the independent landmark in the fixed image, then its counterpart in moving.")

    def _pair_click(self, index: int, _viewer: Any, event: Any) -> None:
        if not self._adding_pair or event.type != "mouse_press" or event.button != 1:
            return
        xy = tuple(float(v) for v in event.position[-2:][::-1])
        if index == 0:
            self._pending_pair = xy
            self.status.setText("Now click the corresponding landmark in the moving image.")
        elif self._pending_pair is not None:
            self.session.landmarks = [*self.session.landmarks, ValidationLandmark(fixed=self._pending_pair, moving=xy)]
            self._adding_pair, self._pending_pair = False, None
            self.invalidate()
            self._refresh_landmarks()
            self.landmark_table.selectRow(len(self.session.landmarks) - 1)
            self.status.setText("Validation pair added.")

    def delete_pair(self) -> None:
        """Remove a paired annotation from both native views."""
        row = self.landmark_table.currentRow()
        if row >= 0:
            self.session.landmarks = [p for i, p in enumerate(self.session.landmarks) if i != row]
            self.invalidate()
            self._refresh_landmarks()

    def choose_landmarks(self) -> None:
        """Import paired CSV annotations, retaining existing pairs with distinct IDs."""
        path, _ = QFileDialog.getOpenFileName(self, "Import independent landmarks", "", "Landmarks (*.csv)")
        if path:

            def load() -> None:
                points = import_validation_landmarks(path, self.session, self.session.landmark_units)
                self.session.landmarks = [*self.session.landmarks, *points]
                self.invalidate()
                self._refresh_landmarks()

            self._try(load)

    def choose_landmark_export(self) -> None:
        """Export all paired coordinates using the CSV unit selector."""
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export landmarks ({self.session.landmark_units})", "landmarks.csv", "CSV (*.csv)"
        )
        if path:
            self._try(partial(export_validation_landmarks, path, self.session))

    def choose_mask(self, index: int) -> None:
        """Load a label mask and explicitly confirm its physical geometry."""
        path, _ = QFileDialog.getOpenFileName(self, "Choose integer-label mask")
        if not path:
            return
        mask = self._try(partial(inspect_validation_image, path))
        if mask is None:
            return
        image = self.session.fixed if index == 0 else self.session.moving
        if image and mask.geometry.shape == image.geometry.shape:
            mask.geometry = image.geometry.model_copy()
        dlg = QDialog(self)
        dlg.setWindowTitle("Mask calibration")
        form = hp.make_form_layout(parent=dlg)
        controls = []
        for name, value in zip(
            ("Spacing X (um)", "Spacing Y (um)", "Origin X (um)", "Origin Y (um)"),
            (*mask.geometry.spacing, *mask.geometry.origin),
            strict=True,
        ):
            control = hp.make_double_spin_box(
                dlg, minimum=0.000001 if name.startswith("Spacing") else -1e9, maximum=1e9, value=value, n_decimals=6
            )
            controls.append(control)
            form.addRow(name, control)
        form.addRow(
            hp.make_h_layout(
                hp.make_btn(dlg, "Use geometry", func=dlg.accept), hp.make_btn(dlg, "Cancel", func=dlg.reject)
            )
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        values = [c.value() for c in controls]
        mask.geometry = ImageGeometry(shape=mask.geometry.shape, spacing=tuple(values[:2]), origin=tuple(values[2:]))
        self._try(partial(self.set_mask, index, mask))

    def set_mask(self, index: int, mask: ValidationImage) -> None:
        """Attach a calibrated mask and initialize matching labels from the analysis preview."""
        reader = open_validation_reader(mask)
        try:
            array = reader.get_channel(mask.channel, split_rgb=True)
            if array.ndim != 2 or array.dtype.kind not in "bui":
                raise ValueError("Masks must be 2D nonnegative integer labels.")
        finally:
            reader.close()
        self.invalidate()
        setattr(self.session, "fixed_mask" if index == 0 else "moving_mask", mask)
        if self.session.fixed_mask and self.session.moving_mask:
            labels = []
            for image in (self.session.fixed_mask, self.session.moving_mask):
                grid = analysis_grid(ValidationSession(fixed=image, max_pixels=250_000))
                pixels, _ = sample_image(image, grid, labels=True)
                labels.append({int(v) for v in np.unique(pixels) if v > 0})
            self.session.mask_pairs = [
                MaskCorrespondence(name=str(v), fixed_label=v, moving_label=v) for v in sorted(labels[0] & labels[1])
            ]
        self._refresh_inputs()
        self.status.setText(
            "Mask loaded. Default label pairs use sampled labels; add small or missing regions explicitly."
        )

    def _refresh_masks(self) -> None:
        self._updating = True
        try:
            self.mask_table.setRowCount(len(self.session.mask_pairs))
            for row, pair in enumerate(self.session.mask_pairs):
                for col, value in enumerate((pair.name, pair.fixed_label, pair.moving_label)):
                    self.mask_table.setItem(row, col, QTableWidgetItem(str(value)))
        finally:
            self._updating = False

    def _mask_edited(self, _item: QTableWidgetItem) -> None:
        if self._updating:
            return
        try:
            self.session.mask_pairs = [
                MaskCorrespondence(
                    name=self.mask_table.item(row, 0).text(),
                    fixed_label=int(self.mask_table.item(row, 1).text()),
                    moving_label=int(self.mask_table.item(row, 2).text()),
                )
                for row in range(self.mask_table.rowCount())
            ]
            self.invalidate()
        except ValueError as exc:
            self.status.setText(f"Invalid label pair: {exc}")
        self._refresh_masks()

    def add_mask_pair(self) -> None:
        """Add an editable positive label correspondence."""
        label = max([p.fixed_label for p in self.session.mask_pairs] or [0]) + 1
        self.session.mask_pairs = [
            *self.session.mask_pairs,
            MaskCorrespondence(name=str(label), fixed_label=label, moving_label=label),
        ]
        self.invalidate()
        self._refresh_masks()

    def delete_mask_pair(self) -> None:
        """Remove the selected anatomical correspondence."""
        row = self.mask_table.currentRow()
        if row >= 0:
            self.session.mask_pairs = [p for i, p in enumerate(self.session.mask_pairs) if i != row]
            self.invalidate()
            self._refresh_masks()

    def clear_masks(self) -> None:
        """Clear mask evidence without deleting input files."""
        self.session.fixed_mask = self.session.moving_mask = None
        self.session.mask_pairs = []
        self.invalidate()
        self._refresh_inputs()

    def start_analysis(self) -> None:
        """Analyze a frozen input snapshot outside the Qt thread."""
        if self._workers:
            return
        self.invalidate()
        generation = self._generation
        self._cancel = Event()
        directory = TemporaryDirectory(prefix="i2validate-")
        snapshot = self.session.model_copy(deep=True)
        worker = create_worker(
            analyze_validation_session,
            snapshot,
            directory.name,
            self._cancel,
            lambda text: self.progress_changed.emit(generation, text),
            _start_thread=False,
            _ignore_errors=True,
        )
        worker.returned.connect(partial(self.accept_analysis, generation, directory))
        worker.errored.connect(partial(self._analysis_error, generation))
        worker.finished.connect(partial(self._analysis_finished, worker, directory))
        self._workers.append(worker)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.status.setText("Loading registration definitions...")
        worker.start()

    def accept_analysis(self, generation: int, directory: TemporaryDirectory, result: ValidationAnalysis) -> None:
        """Publish only results belonging to the current input generation."""
        if generation != self._generation or self._cancel.is_set() or self._closing:
            return
        self.analysis = result
        self._result_directory = directory
        self.export_btn.setEnabled(True)
        rows = comparison_rows(result)
        self.summary_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(
                (
                    values["candidate"],
                    f"{values['validation_count']}/{values['validation_eligible']}",
                    values["validation_mean_um"],
                    values["validation_rms_um"],
                    values["validation_p95_um"],
                    values["validation_maximum_um"],
                    values["coverage"],
                )
            ):
                self.summary_table.setItem(row, col, QTableWidgetItem(self._format(value)))
        with hp.qt_signals_blocked(self.result_candidate):
            self.result_candidate.clear()
            self.result_candidate.addItems([c.name for c in result.candidates])
        self.tabs.setCurrentIndex(3)
        self.show_result()
        self.status.setText("Analysis complete. Results describe the supplied evidence and sampled spatial resolution.")

    def _analysis_error(self, generation: int, error: Exception) -> None:
        if generation == self._generation and not self._closing:
            self.status.setText(
                "Analysis cancelled." if isinstance(error, AnalysisCancelled) else f"Analysis failed: {error}"
            )

    def _analysis_finished(self, worker: Any, directory: TemporaryDirectory) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        if directory is not self._result_directory:
            directory.cleanup()
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        if self._cancel.is_set() and not self._closing:
            self.status.setText("Analysis cancelled. Inputs are ready for a new run.")
        if self._closing:
            self.close()

    def cancel_analysis(self) -> None:
        """Request cancellation at the next bounded processing step."""
        self._cancel.set()
        self._generation += 1
        self.status.setText("Cancelling analysis...")

    def _show_progress(self, generation: int, text: str) -> None:
        if generation == self._generation and not self._closing:
            self.status.setText(text)

    @staticmethod
    def _format(value: Any) -> str:
        return "Unavailable" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)

    def _tab_changed(self, index: int) -> None:
        self.view_stack.setCurrentIndex(1 if index == 3 else 0)

    def show_result(self, _index: int = 0) -> None:
        """Display calibrated diagnostics for the selected candidate without interpolating landmark errors."""
        if self.analysis is None or self.result_candidate.currentIndex() < 0:
            return
        result = self.analysis.candidates[self.result_candidate.currentIndex()]
        with hp.qt_signals_blocked(self.summary_table):
            self.summary_table.selectRow(self.result_candidate.currentIndex())
        if result.map_path is None:
            return
        viewer = self.result_view.viewer
        viewer.layers.clear()
        grid = self.analysis.grid
        kwargs = {"scale": grid.spacing[::-1], "translate": grid.origin[::-1], "units": ("um", "um")}
        with np.load(result.map_path) as arrays:
            mode = self.result_mode.currentIndex()
            legend = ""
            if mode == 0:
                fixed, moving = normalize_image(arrays["fixed_image"]), normalize_image(arrays["registered_image"])
                viewer.add_image(np.stack((fixed, moving, fixed), axis=-1), rgb=True, name="Image overlap", **kwargs)
                legend = "Magenta: fixed; green: registered moving; gray: agreement."
            elif mode == 1:
                points = [p for p in result.landmarks if p["error_um"] is not None]
                if points:
                    data = np.array([[p["moving_y"], p["moving_x"]] for p in points])
                    errors = np.array([p["error_um"] for p in points])
                    maximum = max(
                        [
                            p["error_um"]
                            for c in self.analysis.candidates
                            for p in c.landmarks
                            if p["error_um"] is not None
                        ]
                        or [1]
                    )
                    viewer.add_points(
                        data,
                        name="Validation error (um)",
                        features={"error_um": errors},
                        face_color="error_um",
                        face_colormap="magma",
                        face_contrast_limits=(0, max(maximum, 1e-9)),
                        size=6 * max(grid.spacing),
                        units=("um", "um"),
                    )
                    vectors = np.stack((data, np.array([[p["dy_um"], p["dx_um"]] for p in points])), axis=1)
                    viewer.add_vectors(vectors, name="Residual G(fixed)-moving", edge_color="cyan", units=("um", "um"))
                    legend = f"Moving frame, um. Dark to light: 0 to {maximum:.4g} um; cyan arrows are residuals."
                else:
                    legend = "No independent validation landmarks available."
            elif mode == 2:
                if "fixed_mask" in arrays and self.session.mask_pairs:
                    disagreement = np.zeros((*grid.shape, 3), dtype=np.float32)
                    for pair in self.session.mask_pairs:
                        a, b = arrays["fixed_mask"] == pair.fixed_label, arrays["moving_mask"] == pair.moving_label
                        disagreement[a & b] = (0.3, 0.8, 0.3)
                        disagreement[a & ~b] = (1, 0, 1)
                        disagreement[b & ~a] = (0, 0.7, 1)
                    viewer.add_image(disagreement, rgb=True, name="Mask correspondence", **kwargs)
                    legend = "Green: agreement; magenta: fixed only; cyan: moving only.\n" + "\n".join(
                        f"{p['name']}: Dice {self._format(p['dice'])}, IoU {self._format(p['iou'])}; {p['reason']}"
                        for p in result.masks
                    )
                else:
                    legend = "No corresponding mask labels available."
            else:
                key = ("log2_area_ratio", "jacobian", "nonpositive")[mode - 3]
                data = arrays[key]
                finite = data[np.isfinite(data)]
                limit = max(float(np.max(np.abs(finite))), 0.1) if finite.size else 1
                limits = (
                    (-limit, limit)
                    if mode == 3
                    else (float(finite.min()), float(finite.max()) + 1e-9)
                    if finite.size
                    else (0, 1)
                )
                viewer.add_image(
                    np.nan_to_num(data).astype(np.float32),
                    name=key,
                    colormap="coolwarm" if mode == 3 else "viridis",
                    contrast_limits=limits,
                    **kwargs,
                )
                invalid = ~np.isfinite(data) if mode < 5 else ~arrays["valid"]
                overlay = np.zeros((*grid.shape, 4), dtype=np.float32)
                overlay[invalid] = (0.5, 0.5, 0.5, 1)
                if mode == 3:
                    overlay[arrays["nonpositive"]] = (0.5, 0, 0.5, 1)
                viewer.add_image(overlay, rgb=True, name="Unavailable / nonpositive", **kwargs)
                legend = (
                    f"Display range {limits[0]:.4g} to {limits[1]:.4g}. Gray: unavailable. "
                    "log2 area ratio: positive expansion, negative compression, moving to fixed. "
                    "Signed Jacobian: fixed to moving. Nonpositive values can include intended reflections. "
                    "All maps include global affine scaling."
                )
        viewer.reset_view()
        self.result_detail.setText(
            f"{legend}\nFitting RMS: {self._format(result.fitting.get('rms'))} um "
            f"({result.fitting.get('count', 0)} points). Source coverage: {result.deformation['coverage']:.1%}. "
            f"Nonpositive fraction: {self._format(result.deformation['nonpositive_fraction'])}."
        )
        self.error_table.setRowCount(len(result.landmarks))
        for row, point in enumerate(result.landmarks):
            for col, key in enumerate(("id", "dx_um", "dy_um", "error_um", "excluded")):
                self.error_table.setItem(row, col, QTableWidgetItem(self._format(point[key])))

    def choose_export(self) -> None:
        """Write a complete visual report into a newly named directory."""
        if self.analysis is None:
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent directory for report")
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "Report directory", "New directory name", text="validation-report")
        if not ok or not name or Path(name).name != name:
            return
        path = self._try(partial(export_validation_report, self.analysis, Path(parent) / name))
        if path:
            self.status.setText(f"Exported {path}")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cancel running work before disposing readers, views, and temporary maps."""
        if not self._closing and not self._may_discard():
            event.ignore()
            return
        self._closing = True
        if self._workers:
            self.cancel_analysis()
            event.ignore()
            return
        self._updating = True
        for view in [*self.views, self.result_view]:
            view.close()
        if self._result_directory:
            self._result_directory.cleanup()
        event.accept()
