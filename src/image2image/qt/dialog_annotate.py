"""Annotation and Label Editor with a persistent object store and local ML assistance."""

from __future__ import annotations

import contextlib
import json
import math
import shutil
import sqlite3
from collections.abc import Generator
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from functools import partial
from importlib.metadata import version
from pathlib import Path
from queue import Empty, Queue
from threading import Event
from time import monotonic
from typing import Any, Literal, cast
from uuid import uuid4
from zipfile import BadZipFile

import numpy as np
import qtextra.helpers as hp
from pydantic import Field
from qtextra.utils.table_config import TableConfig
from qtextra.widgets.qt_button_segmented import QtSegmentedButton
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import QEvent, QLockFile, QObject, QSignalBlocker, Qt, QTimer
from qtpy.QtGui import (  # type: ignore[attr-defined]
    QCloseEvent,
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QShortcut,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenuBar,
    QPlainTextEdit,
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QTextEdit,
    QWidget,
)
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from superqt.utils import create_worker

from image2image.config import SingleAppConfig
from image2image.models.annotation import (
    AnnotationClass,
    AnnotationObject,
    AnnotationSlide,
    CoordinateSpec,
    EditCommand,
    FeatureSettings,
    PredictionRequest,
    SlideGeometry,
)
from image2image.qt._annotation_finetune import FineTuneDialog
from image2image.qt._annotation_jobs import AnnotationJobs
from image2image.qt._annotation_model import ModelSelector
from image2image.qt._annotation_table import AnnotationTable
from image2image.qt._dialog_base import Window
from image2image.utils.annotation.geometry import (
    brush_object,
    brush_stroke,
    edit_geometry,
    merge_objects,
    split_object,
    split_preview,
)
from image2image.utils.annotation.images import PatchReader, inspect_slide
from image2image.utils.annotation.ml import backend_available, load_forest
from image2image.utils.annotation.model_cache import read_classifier_metadata
from image2image.utils.annotation.render import AnnotationRenderer
from image2image.utils.annotation.store import AnnotationStore
from image2image.utils.annotation.wand import WandLayer, WandSource


@dataclass(eq=False)
class _PaintStroke:
    """Pending ordered paint operations, owned by one canvas gesture."""

    mode: str
    context: tuple[Any, ...]
    start: tuple[float, float]
    point: tuple[float, float]
    subtract: bool
    radius: float
    zoom: float
    class_id: str | None
    fallback: list[int]
    target: AnnotationObject | None = None
    geometry: BaseGeometry = dataclass_field(default_factory=Polygon)
    base: BaseGeometry = dataclass_field(default_factory=Polygon)
    points: list[tuple[float, float]] = dataclass_field(default_factory=list)
    queue: Queue[tuple[tuple[float, float], bool] | None] = dataclass_field(default_factory=Queue)
    cancelled: Event = dataclass_field(default_factory=Event)
    revision: int = 0
    layers: tuple[WandLayer, ...] = ()
    initial_subtract: bool = False
    background: tuple[float, float, float] = (0, 0, 0)


def _table_key(table: AnnotationTable, row: int) -> int:
    return int(table.key_at(row))


def _selected_keys(table: AnnotationTable) -> list[int]:
    return list(map(int, table.selected_keys()))


class AnnotateConfig(SingleAppConfig):
    """Keep editor preferences separate from registration configuration."""

    USER_CONFIG_FILENAME = "config.annotate.json"

    auto_add_polygons: bool = True
    opacity_annotations: int = Field(default=100, ge=0, le=100)
    opacity_predictions: int = Field(default=100, ge=0, le=100)
    opacity_drawings: int = Field(default=100, ge=0, le=100)
    opacity_prompts: int = Field(default=100, ge=0, le=100)
    brush_radius: int = Field(default=15, ge=1, le=500)
    wand_radius: int = Field(default=75, ge=1, le=500)
    wand_sensitivity: int = Field(default=15, ge=1, le=100)
    freehand_color: str = Field(default="#ff00ff", pattern=r"^#[0-9a-fA-F]{6}$")
    brush_add_color: str = Field(default="#00ff00", pattern=r"^#[0-9a-fA-F]{6}$")
    brush_subtract_color: str = Field(default="#ff6347", pattern=r"^#[0-9a-fA-F]{6}$")


class CoordinateDialog(QDialog):
    """Explicit exchange frame and raster ID mapping options."""

    def __init__(self, parent: QWidget, *, raster: bool, exporting: bool, geometry: SlideGeometry) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotation coordinates")
        layout = hp.make_form_layout(parent=self)
        layout.addRow(
            hp.make_label(
                self, "XY coordinates refer to pixel edges; (0, 0) is the full-resolution upper-left corner.", wrap=True
            )
        )
        self.units = hp.make_combobox(self, ["px", "um"] if not raster else ["px"])
        layout.addRow("Coordinates", self.units)
        self.offset = hp.make_line_edit(self, text="0, 0")
        self.downsample = hp.make_line_edit(self, text="1, 1")
        layout.addRow("Patch offset X, Y (full-resolution px)", self.offset)
        layout.addRow("Downsample X, Y", self.downsample)
        self.mapped = hp.make_checkbox(self, "Map IDs to integers and write a reversible ID sidecar")
        self.mapped.setVisible(raster and exporting)
        layout.addRow(self.mapped)
        layout.addRow(
            hp.make_label(
                self,
                (
                    f"Pixel size X, Y: {geometry.spacing} um. Origin X, Y: {geometry.origin} um."
                    if geometry.spacing
                    else "Calibration unknown. Pixel coordinates are available; calibrate the slide for physical exchange."
                ),
                wrap=True,
            )
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def coordinates(self) -> CoordinateSpec:
        """Read and validate the explicit coordinate frame."""
        offset = tuple(float(value.strip()) for value in self.offset.text().split(","))
        downsample = tuple(float(value.strip()) for value in self.downsample.text().split(","))
        return CoordinateSpec.model_validate(
            {"units": self.units.currentText(), "offset": offset, "downsample": downsample}
        )


class ImageAnnotationWindow(Window):
    """Edit complete objects while rendering only the current image viewport."""

    APP_NAME = "annotate"
    view: NapariImageView

    def __init__(
        self,
        parent: QWidget | None = None,
        run_check_version: bool = True,
        project_dir: str | Path | None = None,
        image_path: list[str] | None = None,
        **_kwargs: Any,
    ) -> None:
        self.CONFIG = AnnotateConfig(_auto_load=True)  # type: ignore[call-arg]
        self.store: AnnotationStore | None = None
        self._project_lock: QLockFile | None = None
        self.reader: PatchReader | None = None
        self.renderer: AnnotationRenderer | None = None
        self.slide_id: str | None = None
        self.selected: list[int] = []
        self._review_selection: list[int] = []
        self._excluded_training_classes: set[str] = set()
        self._threads: list[Any] = []
        self._forest_scans: set[Any] = set()
        self._closing = False
        self._syncing = False
        self._rendering = False
        self._generation = 0
        self._pages: dict[str, list[int]] = {"accepted": [0], "proposed": [0]}
        self._page_last: dict[str, int] = {"accepted": 0, "proposed": 0}
        self._page_totals: dict[str, int] = {"accepted": 0, "proposed": 0}
        self._class_labels: dict[str, str | None] = {}
        self._pagination: dict[str, tuple[Any, Any, Any]] = {}
        self._mode = "Interact"
        self._ring_layout: list[int] = []
        self._image_layers: list[Any] = []
        self._overlay = None
        self._proposal_overlay = None
        self._edit_revision = 0
        self._prompt_labels: list[int] = []
        self._backend_state: dict[str, dict[str, Any]] = {}
        self._active_backend = ""
        self._changing_backend = False
        self._forest_path = ""
        self._forest_metadata: dict[str, Any] | None = None
        self._forest_catalogue: dict[str, dict[str, Any]] = {}
        self._forest_scan_pending: Path | None = None
        self._forest_scan_preferred = ""
        self._forest_scan_again = False
        self._catalogues: dict[str, list[dict[str, Any]]] = {}
        self._catalogue_pending: set[str] = set()
        self._catalogue_errors: dict[str, str] = {}
        self._catalogue_project: dict[str, str | None] = {}
        self._selected_custom_checkpoint = ""
        self._stroke_revision = 0
        self._preview_updated = 0.0
        self._paint: _PaintStroke | None = None
        self._wand_strokes: list[_PaintStroke] = []
        self._wand_worker: Any = None
        self._paint_committing = False
        self.jobs = AnnotationJobs()
        super().__init__(parent, "image2image: Annotation and Label Editor", run_check_version=run_check_version)
        self.jobs.setParent(self)
        self.resize(1650, 1000)
        if project_dir:
            self._try(partial(self.open_project, project_dir))
        if image_path:
            self.status.setText("Create or open a project, then add the source images.")

    @staticmethod
    def _setup_config() -> None:
        """Use reader-local settings and retain native pixel coordinates."""

    def setup_events(self, state: bool = True) -> None:
        """Connect background job progress and viewport changes."""
        if state:
            self.jobs.progress.connect(self._job_message)
            self.jobs.details.connect(self._job_progress)
            self.jobs.completed.connect(self._job_done)
            self.jobs.failed.connect(self._job_failed)
            self.jobs.busy_changed.connect(self._job_busy)
            self.jobs.queue_changed.connect(self._refresh_queue)
            self.view.viewer.camera.events.zoom.connect(self.schedule_render)
            self.view.viewer.camera.events.zoom.connect(self._cancel_paint)
            self.tabs.currentChanged.connect(self._cancel_paint)
            self.view.viewer.camera.events.center.connect(self.schedule_render)
            self.view.viewer.camera.events.center.connect(self._cancel_paint)

    def _try(self, function: Any) -> Any:
        try:
            return function()
        except (
            ValueError,
            TypeError,
            KeyError,
            OSError,
            RuntimeError,
            ImportError,
            EOFError,
            BadZipFile,
            sqlite3.Error,
        ) as exc:
            self._error(str(exc))
            return None

    def _error(self, message: str) -> None:
        self._set_status(message)
        hp.warn_pretty(self, message, "Annotation editor")

    def _button(self, parent: QWidget, text: str, function: Any) -> Any:
        return hp.make_btn(parent, text, func=lambda: self._try(function))

    def _qta_button(self, parent: QWidget, icon: str, tooltip: str, function: Any) -> Any:
        return hp.make_qta_btn(parent, icon, tooltip=tooltip, func=lambda: self._try(function), size_preset="small")

    def _setup_ui(self) -> None:
        main = QWidget(self)
        layout = hp.make_v_layout(parent=main)
        self._make_queue_dialog()
        layout.addLayout(
            hp.make_h_layout(
                self._button(main, "New project...", self.choose_new_project),
                self._button(main, "Open project...", self.choose_project),
                self._button(main, "Add slides...", self.choose_slides),
                self._button(main, "Import...", self.choose_import),
                self._button(main, "Export...", self.choose_export),
                self._button(main, "Undo", self.undo),
                self._button(main, "Redo", self.redo),
            )
        )
        splitter = QSplitter(Qt.Orientation.Horizontal, main)
        left = QWidget(splitter)
        left_box = hp.make_v_layout(parent=left, margin=2, spacing=2)
        left_box.addWidget(hp.make_h_line_with_text(left, "Slides"))
        # Preview
        self.overview = NapariImageView(left, add_toolbars=False, disable_new_layers=True, allow_crosshair=False)
        self.overview.widget.setFixedHeight(160)
        left_box.addWidget(self.overview.widget)
        # Slides
        self.slides = QListWidget(left)
        self.slides.currentRowChanged.connect(self._slide_selected)
        left_box.addWidget(self.slides, 1)
        left_box.addLayout(
            hp.make_h_layout(
                self._button(left, "Relocate...", self.relocate_slide),
                self._button(left, "Calibration...", self.calibrate_slide),
            )
        )
        self.overview.viewer.mouse_drag_callbacks.append(self._overview_click)
        self.sets = hp.make_combobox(left, func_index=self._set_changed)
        left_box.addWidget(hp.make_h_line_with_text(left, "Annotation set"))
        left_box.addLayout(
            hp.make_h_layout(
                self.sets,
                self._qta_button(left, "add", "Create a new annotation set", self.new_set),
                self._qta_button(left, "remove", "Remove the selected annotation set", self.remove_set),
                self._qta_button(left, "combine", "Combine annotation sets", self.choose_combine_sets),
                stretch_id=(0,),
                spacing=1,
                margin=1,
            ),
        )
        left_box.addWidget(hp.make_h_line_with_text(left, "Shared classes"))
        self.classes = QListWidget(left)
        left_box.addLayout(
            hp.make_h_layout(
                self._qta_button(left, "add", "add a new class", self.add_class),
                self._qta_button(left, "edit", "Remove the selected class", self.edit_class),
                self._qta_button(left, "visible", "Show/hide classes", self.toggle_class_visible),
                self._qta_button(left, "locked", "Lock/unlock classes", self.toggle_class_locked),
                stretch_after=True,
                spacing=1,
                margin=1,
            ),
        )
        left_box.addWidget(self.classes, 1)
        center = QWidget(splitter)
        center_box = hp.make_v_layout(parent=center)
        self.tool = hp.make_combobox(
            center,
            [
                "Interact",
                "Polygon",
                "Freehand",
                "Rectangle",
                "Brush +",
                "Brush -",
                "Wand",
                "Vertices",
                "Cut",
                "Select by polygon",
                "Select by rectangle",
                "Prompt +",
                "Prompt -",
                "Prompt box",
            ],
            func=self.set_mode,
        )
        self.show_predictions = hp.make_checkbox(center, "Show predictions", default=True, func=self.schedule_render)
        self.show_predictions.setToolTip("Show predictions in the active annotation set awaiting acceptance in Review.")
        center_box.addLayout(
            hp.make_h_layout(
                self.tool,
                self._button(center, "Apply drawing", self.apply_drawing),
                self._button(center, "Clear drawing", self.clear_drawing),
            )
        )
        self.auto_add_polygons = hp.make_checkbox(
            center, "Auto-add polygons", default=self.CONFIG.auto_add_polygons, func=self._auto_add_changed
        )
        self.auto_add_polygons.setToolTip("Save polygons on double-click. Uncheck to use Apply drawing instead.")
        self.display_btn = self._button(center, "Display...", self.show_display_controls)
        center_box.addLayout(
            hp.make_h_layout(
                self.auto_add_polygons,
                self.show_predictions,
                self._button(center, "Clear prompts", self.clear_prompts),
                self.display_btn,
                stretch_after=True,
            )
        )
        self.view = NapariImageView(center, add_toolbars=False, disable_new_layers=True, allow_crosshair=False)
        center_box.addWidget(self.view.widget, 1)
        self.draw_layer = self.view.viewer.add_shapes(
            name="Drawing / selected rings", ndim=2, edge_color="yellow", face_color="transparent", edge_width=2
        )
        self.brush_preview = self.view.viewer.add_shapes(
            name="Brush preview", ndim=2, edge_color="lime", face_color="transparent", edge_width=3, opacity=1
        )
        self.prompt_layer = self.view.viewer.add_points(name="AI prompts", ndim=2, size=8, face_color="lime")
        self.prompt_box_layer = self.view.viewer.add_shapes(
            name="AI prompt boxes", ndim=2, edge_color="cyan", face_color="transparent", edge_width=2
        )
        self.view.viewer.mouse_drag_callbacks.append(self._mouse_drag)
        self.draw_layer.mouse_double_click_callbacks.append(self._drawing_finished)
        self._make_display_controls()
        self.tabs = QTabWidget(splitter)
        self.tabs.addTab(self._annotation_panel(), "Annotate")
        self.tabs.addTab(self._train_panel(), "Train")
        self.tabs.addTab(self._predict_panel(), "Predict")
        self.tabs.addTab(self._review_panel(), "Review")
        self.tabs.currentChanged.connect(lambda index: self._refresh_training_summary() if index == 1 else None)
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([250, 1000, 370])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(main)
        self._make_statusbar()
        self._make_icon()
        menubar = QMenuBar(self)
        for menu in (self._make_apps_menu(), self._make_help_menu()):
            menubar.addAction(menu.menuAction())
        self.setMenuBar(menubar)
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(100)
        self.render_timer.timeout.connect(self._render)
        self._shortcuts = []
        for key, function in (
            ("Ctrl+Z", self.undo),
            ("Ctrl+Shift+Z", self.redo),
            ("Escape", self.clear_drawing),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(partial(self._try, function))
            self._shortcuts.append(shortcut)
        self.set_mode(self.tool.currentText())
        QApplication.instance().installEventFilter(self)

    def _make_statusbar(self) -> None:
        super()._make_statusbar()
        self.status = hp.make_label(self, "Create or open an .i2annotate project. Edits are saved automatically.")
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.tool_hint = hp.make_label(self, "", wrap=False)
        self.tool_hint.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.queue_btn = hp.make_qta_btn(
            self, "queue", tooltip="Show task queue", func=self.show_task_queue, size_preset="small"
        )
        self.cancel_btn = hp.make_btn(
            self, "Cancel job", tooltip="Stop the active job and clear queued tasks.", func=self.jobs.cancel
        )
        self.cancel_btn.setEnabled(False)
        self.statusbar.insertPermanentWidget(1, self.queue_btn)
        self.statusbar.insertPermanentWidget(2, self.cancel_btn)
        self.statusbar.addWidget(self.status, 1)
        self.statusbar.addWidget(self.tool_hint, 1)

    def _auto_add_changed(self, enabled: bool) -> None:
        self.CONFIG.update(auto_add_polygons=bool(enabled))

    def _make_display_controls(self) -> None:
        self.display_popup = QDialog(self, Qt.WindowType.Popup)
        self.display_popup.setWindowTitle("Display")
        self.display_popup.setMinimumWidth(300)
        layout = hp.make_form_layout(parent=self.display_popup)
        self.brush_radius = hp.make_int_spin_box(
            self.display_popup,
            minimum=1,
            maximum=500,
            value=self.CONFIG.brush_radius,
            func=self._brush_radius_changed,
        )
        layout.addRow("Brush radius (screen px)", self.brush_radius)
        self.wand_radius = hp.make_int_spin_box(
            self.display_popup,
            minimum=1,
            maximum=500,
            value=self.CONFIG.wand_radius,
            func=partial(self._wand_setting_changed, "wand_radius"),
        )
        self.wand_sensitivity = hp.make_int_spin_box(
            self.display_popup,
            minimum=1,
            maximum=100,
            value=self.CONFIG.wand_sensitivity,
            func=partial(self._wand_setting_changed, "wand_sensitivity"),
        )
        self.wand_radius.setToolTip("Maximum local search radius in screen pixels. Zoom out to fill larger regions.")
        self.wand_sensitivity.setToolTip(
            "Allowed displayed color difference (%). Higher values fill more varied regions."
        )
        layout.addRow("Wand radius (screen px)", self.wand_radius)
        layout.addRow("Wand sensitivity (%)", self.wand_sensitivity)
        self.preview_colors: dict[str, Any] = {}
        for field, label in (
            ("freehand_color", "Freehand preview"),
            ("brush_add_color", "Brush + preview"),
            ("brush_subtract_color", "Brush - preview"),
        ):
            swatch = hp.make_swatch(
                self.display_popup, getattr(self.CONFIG, field), func=partial(self._preview_color_changed, field)
            )
            self.preview_colors[field] = swatch
            layout.addRow(label, swatch)
        self.opacity_sliders: dict[str, Any] = {}
        for group, label in (
            ("annotations", "Annotations (%)"),
            ("predictions", "Predictions (%)"),
            ("drawings", "Drawings (%)"),
            ("prompts", "Prompts (%)"),
        ):
            slider = hp.make_slider_with_text(
                self.display_popup,
                value=getattr(self.CONFIG, f"opacity_{group}"),
                func=partial(self._opacity_changed, group),
            )
            self.opacity_sliders[group] = slider
            layout.addRow(label, slider)
        self._update_opacity()

    def _brush_radius_changed(self, radius: int) -> None:
        self._cancel_paint()
        self.CONFIG.update(brush_radius=radius)
        self.set_mode(self._mode)

    def _wand_setting_changed(self, name: str, value: int) -> None:
        self._cancel_paint()
        self.CONFIG.update(**{name: value})
        self.set_mode(self._mode)

    def _preview_color_changed(self, field: str, _color: np.ndarray) -> None:
        self.CONFIG.update(**{field: self.preview_colors[field].hex_color_no_alpha})
        if field == "freehand_color" and self._mode == "Freehand":
            self.draw_layer.edge_color = self.CONFIG.freehand_color
        elif field.startswith("brush_"):
            self.brush_preview.edge_color = (
                self.CONFIG.brush_subtract_color if self._mode == "Brush -" else self.CONFIG.brush_add_color
            )

    def show_display_controls(self) -> None:
        """Show overlay opacity and drawing preview controls."""
        hp.show_below_widget(self.display_popup, self.display_btn)

    def _opacity_changed(self, group: str, value: int) -> None:
        self.CONFIG.update(**{f"opacity_{group}": value})
        self._update_opacity()

    def _update_opacity(self) -> None:
        for group, layers in (
            ("annotations", (self._overlay,)),
            ("predictions", (self._proposal_overlay,)),
            ("drawings", (self.draw_layer, self.brush_preview)),
            ("prompts", (self.prompt_layer, self.prompt_box_layer)),
        ):
            for layer in layers:
                if layer is not None:
                    layer.opacity = getattr(self.CONFIG, f"opacity_{group}") / 100

    def _annotation_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = hp.make_v_layout(parent=panel)
        self.channels = hp.make_line_edit(panel, text="0")
        self.display_mode = hp.make_combobox(panel, ["Channels", "RGB"])
        layout.addWidget(hp.make_label(panel, "Channel indices (comma separated)"))
        layout.addWidget(self.channels)
        layout.addWidget(self.display_mode)
        layout.addWidget(self._button(panel, "Update image channels", self.refresh_image))
        self.search = hp.make_line_edit(panel)
        self.search.setPlaceholderText("Object ID (or int:12 / str:12)")
        self.search.returnPressed.connect(lambda: self._try(self.find_object))
        layout.addLayout(hp.make_h_layout(self.search, self._button(panel, "Find", self.find_object)))
        self.objects = AnnotationTable(
            panel,
            TableConfig()
            .add("Training", "training", dtype="bool", checkable=True, tooltip="Include in classifier training")
            .add("ID", "id", width=60, resizable=True, tooltip="Original object ID (read-only)")
            .add("Class", "class", width=100, resizable=True, tooltip="Double-click to choose a class")
            .add("Area (px2)", "area", dtype="float", tooltip="Object area in full-resolution pixels (read-only)"),
            paged=True,
            editable={"class", "training"},
            on_edit=self._table_edited,
        )
        self.objects.proxy.evt_filtered.connect(partial(self._page_status, "accepted"))
        self.objects.selectionModel().selectionChanged.connect(lambda *_: self._table_selection())
        self.objects.doubleClicked.connect(lambda index: self.zoom_selected() if index.column() != 2 else None)
        layout.addWidget(self.objects.filter_widget)
        layout.addWidget(self.objects, 1)
        layout.addLayout(self._pagination_controls(panel, "accepted"))
        self.selection_label = hp.make_label(panel, "No selection", wrap=True)
        layout.addWidget(self.selection_label)
        layout.addWidget(self._button(panel, "Object properties...", self.edit_properties))
        layout.addWidget(self._button(panel, "Assign selected class", self.assign_class))
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Include in training", partial(self.set_training, True)),
                self._button(panel, "Exclude from training", partial(self.set_training, False)),
            )
        )
        layout.addWidget(self._button(panel, "Lock/unlock selected objects", self.toggle_object_lock))
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Merge...", self.merge_selected),
                self._button(panel, "Delete", self.delete_selected),
            )
        )
        return panel

    def _train_panel(self) -> QWidget:
        panel, scroll = hp.make_scroll_area(self)
        outer = hp.make_v_layout(parent=panel)
        content = QWidget(panel)
        form = hp.make_form_layout(parent=content)
        outer.addWidget(content)
        outer.addStretch(1)
        form.addRow(
            hp.make_label(
                panel,
                "Train a CPU random forest to classify pixels using multiscale intensity, edge, and texture features. "
                "Include accepted examples for at least two classes, usually including Background. "
                "Choose classes below for this training run. Fine-tune Cellpose or SAM with the separate action below.",
                wrap=True,
            )
        )
        self.classifier_name = hp.make_line_edit(panel, text="Tissue classifier")
        self.train_downsample = hp.make_int_spin_box(panel, minimum=1, maximum=256, value=4)
        self.feature_sigma = hp.make_int_spin_box(panel, minimum=1, maximum=32, value=8)
        self.sample_count = hp.make_int_spin_box(panel, minimum=100, maximum=100000, value=10000)
        self.random_seed = hp.make_int_spin_box(panel, minimum=0, maximum=2**31 - 1, value=42)
        self.max_depth = hp.make_int_spin_box(panel, minimum=1, maximum=1000, value=24)
        form.addRow("Classifier name", self.classifier_name)
        form.addRow("Training downsample", self.train_downsample)
        form.addRow("Largest feature scale", self.feature_sigma)
        form.addRow("Samples per class", self.sample_count)
        form.addRow("Random seed", self.random_seed)
        form.addRow("Maximum tree depth", self.max_depth)
        self.random_seed.setToolTip("Controls sampling and forest randomness. Reuse the seed to reproduce training.")
        self.max_depth.setToolTip(
            "Maximum depth of each tree. Larger values can fit finer detail and increase overfitting."
        )
        self.train_downsample.setToolTip("Working resolution: 4 uses one pixel per 4 x 4 full-resolution pixels.")
        self.feature_sigma.setToolTip("Largest Gaussian feature scale in working-resolution pixels; minimum is 1.")
        self.sample_count.setToolTip("Upper limit per class. Classes use equal counts, limited by the smallest class.")
        self.training_summary = hp.make_label(panel, "No training examples.", wrap=True)
        form.addRow(self.training_summary)
        self.training_counts = AnnotationTable(
            panel,
            TableConfig()
            .add(
                "Use", "use", dtype="bool", checkable=True, tooltip="Enable this class for the next model training run"
            )
            .add("Class", "class", width=100, resizable=True)
            .add("Included", "included", sizing="contents")
            .add("Total", "total", dtype="int", sizing="contents"),
            editable={"use"},
            on_edit=self._training_class_edited,
        )
        self.training_counts.setMaximumHeight(190)
        form.addRow(self.training_counts.filter_widget)
        form.addRow(self.training_counts)
        self.train_btn = self._button(panel, "Train from included annotations", self.train)
        self.train_btn.setEnabled(False)
        form.addRow(self.train_btn)
        self.finetune_btn = self._button(panel, "Fine-tune Cellpose / SAM...", self.choose_finetuning)
        form.addRow(self.finetune_btn)
        form.addRow(
            hp.make_label(
                panel,
                "New manual drawings and imports are included automatically. Change the Training checkboxes in Annotate "
                "to exclude examples or include reviewed predictions. Existing saved choices are preserved. "
                "Ignore regions exclude overlapping pixels in the same annotation set. Counts show candidate objects "
                "before masking and sampling. Training starts only when you click Train.\n\n"
                "Class checkboxes choose this run's classes without changing individual object flags. "
                "Channels come from Annotate. The forest uses 100 trees and the seed/depth selected above. "
                "Examples, Ignore masks, and settings are saved with the classifier.",
                wrap=True,
            )
        )
        return scroll

    def on_pressed_link(self, link: str) -> None:
        """Go to the requested task."""
        from koyo.path import open_directory_universal

        if "goto:" in link:
            link = link.split("goto:")[-1]
        open_directory_universal(link)

    def _predict_panel(self) -> QWidget:
        panel, scroll = hp.make_scroll_area(self)
        form = hp.make_form_layout(parent=panel)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._predict_form = form
        self.backend = hp.make_combobox(panel, ["forest", "micro-sam", "cellpose"])
        self.model_selector = ModelSelector(panel, self.jobs, self.on_pressed_link)
        self.model = self.model_selector.model
        self.checkpoint = hp.make_line_edit(panel)
        self.device = hp.make_combobox(
            panel, ["auto", "cpu", "cuda", "mps"], tooltip="Device: auto chooses CUDA, then Apple MPS, then CPU."
        )
        self.predict_downsample = hp.make_int_spin_box(panel, minimum=1, maximum=256, value=1)
        self.tile_size = hp.make_combobox(panel, ["256", "512", "1024"])
        self.tile_size.setCurrentText("512")
        form.addRow("Backend", self.backend)
        self.backend_hint = hp.make_label(panel, "", wrap=True)
        form.addRow(self.backend_hint)

        self.download_btn = self.model_selector.download
        self.download_btn.clicked.connect(lambda: self._try(self.download_selected_model))
        self.model_help = self.model_selector.help
        form.addRow(hp.make_label(panel, "Model"))
        form.addRow(self.model_selector)
        self.model_summary = hp.make_label(panel, "", wrap=True)
        form.addRow(self.model_summary)
        self.model_status = self.model_selector.status
        self.model_requirements = self.model_selector.requirements
        self.choose_model_btn = self._button(panel, "Load classifier...", self.choose_model)
        form.addRow(self.choose_model_btn)
        form.addRow("Optional local checkpoint", self.checkpoint)
        self.choose_checkpoint_btn = self._button(panel, "Choose checkpoint...", self.choose_checkpoint)
        form.addRow(self.choose_checkpoint_btn)
        form.addRow(hp.make_h_line(panel))
        form.addRow("Device", self.device)
        form.addRow("Prediction downsample", self.predict_downsample)
        form.addRow("Patch size", self.tile_size)
        self.view_btn = self._button(panel, "Preview current view", self.predict_view)
        form.addRow(self.view_btn)
        self.roi_segment = QtSegmentedButton("Predict drawn rectangle", panel, func=lambda: self._try(self.predict_roi))
        self.roi_segment.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.draw_roi_btn, _ = self.roi_segment.add_action(
            "mdi6.rectangle-outline",
            "Draw rectangle",
            lambda: self.tool.setCurrentText("Rectangle"),
            size_preset="small",
        )
        form.addRow(self.roi_segment)
        self.prompt_segment = QtSegmentedButton(
            "Run point/box assistance", panel, func=lambda: self._try(self.predict_prompts)
        )
        self.prompt_segment.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self.prompt_btn = self.prompt_segment.button
        self.prompt_tool_buttons = [
            self.prompt_segment.add_action(icon, mode, partial(self.tool.setCurrentText, mode), size_preset="small")[0]
            for icon, mode in (
                ("mdi6.plus", "Prompt +"),
                ("mdi6.minus", "Prompt -"),
                ("mdi6.rectangle-outline", "Prompt box"),
            )
        ]
        self.prompt_segment.setToolTip(
            "Each positive point or box produces a separate instance. Negative points are shared exclusion hints."
        )
        form.addRow(self.prompt_segment)
        self.slide_separator = hp.make_h_line(panel)
        form.addRow(self.slide_separator)
        self.slide_btn = self._button(panel, "Predict current slide", self.predict_slide)
        self.all_slides_btn = self._button(panel, "Queue all slides", self.predict_all)
        form.addRow(self.slide_btn)
        form.addRow(self.all_slides_btn)
        form.addRow(hp.make_label(panel, "Predictions stay separate until accepted in Review.", wrap=True))
        self.predict_buttons = [self.view_btn, self.roi_segment.button, self.slide_btn, self.all_slides_btn]
        self.backend.currentTextChanged.connect(self._backend_changed)
        self.model.currentIndexChanged.connect(self._model_selected)
        self.checkpoint.textChanged.connect(self._update_prediction_controls)
        self._backend_changed()
        return scroll

    def _review_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = hp.make_v_layout(parent=panel)
        layout.addWidget(
            hp.make_label(
                panel, "Choose a prediction set on the left. Proposed objects appear in this paged list.", wrap=True
            )
        )
        self.proposals = AnnotationTable(
            panel,
            TableConfig()
            .add("ID", "id", width=60, resizable=True, tooltip="Original object ID (read-only)")
            .add("Class", "class", width=100, resizable=True, tooltip="Double-click to choose a class")
            .add("Area (px2)", "area", dtype="float", width=80, resizable=True)
            .add("Review note", "review"),
            paged=True,
            editable={"class"},
            on_edit=self._table_edited,
        )
        self.proposals.proxy.evt_filtered.connect(partial(self._page_status, "proposed"))
        self.proposals.doubleClicked.connect(
            lambda index: self._zoom_proposal(index.row(), index.column()) if index.column() != 1 else None
        )
        self.proposals.selectionModel().selectionChanged.connect(lambda *_: self._table_selection(self.proposals))
        layout.addWidget(self.proposals.filter_widget)
        layout.addWidget(self.proposals, 1)
        self.proposal_details = hp.make_label(panel, "Select a proposal to see its source model.", wrap=True)
        layout.addWidget(self.proposal_details)
        layout.addLayout(self._pagination_controls(panel, "proposed"))
        self.accept_selected_btn = self._button(panel, "Accept selected (A)", partial(self.review, True, False))
        self.reject_selected_btn = self._button(panel, "Reject selected (R)", partial(self.review, False, False))
        layout.addLayout(hp.make_h_layout(self.accept_selected_btn, self.reject_selected_btn))
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Accept batch", partial(self.review, True, True)),
                self._button(panel, "Reject batch", partial(self.review, False, True)),
            )
        )
        self.refine_btn = self._button(panel, "Refine selected (E)", self.refine_proposal)
        self.merge_proposals_btn = self._button(panel, "Merge selected (M)", self.merge_proposals)
        layout.addLayout(hp.make_h_layout(self.refine_btn, self.merge_proposals_btn))
        self.accept_selected_btn.setToolTip("Accept selected proposals (A). Undo restores their proposed state.")
        self.reject_selected_btn.setToolTip("Reject selected proposals (R). Undo restores them.")
        self.refine_btn.setToolTip("Edit one selected proposal with vertices or brush tools (E).")
        self.merge_proposals_btn.setToolTip("Merge selected proposals and choose the surviving object ID (M).")
        return panel

    def _make_queue_dialog(self) -> None:
        self.queue_dialog = QDialog(self)
        self.queue_dialog.setWindowTitle("Annotation task queue")
        self.queue_dialog.resize(650, 320)
        layout = hp.make_v_layout(parent=self.queue_dialog)
        layout.addWidget(
            hp.make_label(
                self.queue_dialog,
                "Tasks run in order. Cancel stops the active task and removes queued tasks.",
                wrap=True,
            )
        )
        self.task_table = AnnotationTable(
            self.queue_dialog, TableConfig().add("Status", "status", sizing="contents").add("Task", "task")
        )
        layout.addWidget(self.task_table.filter_widget)
        layout.addWidget(self.task_table, 1)
        layout.addWidget(hp.make_btn(self.queue_dialog, "Cancel active and queued tasks", func=self.jobs.cancel))

    def show_task_queue(self) -> None:
        """Show active and queued worker operations without interrupting the editor."""
        self._refresh_queue()
        self.queue_dialog.show()
        self.queue_dialog.raise_()
        self.queue_dialog.activateWindow()

    def _refresh_queue(self) -> None:
        tasks = ([self.jobs.current] if self.jobs.current else []) + list(self.jobs.pending)
        rows = []
        for row, job in enumerate(tasks):
            active = job is self.jobs.current
            status = (
                ("Cancelling" if self.jobs.cancelling else self.jobs._phase or "Starting")
                if active
                else f"Queued {row if self.jobs.current else row + 1}"
            )
            request = job.get("request", {})
            operation = job["operation"]
            description = {
                "model_catalogue": "Discover models",
                "download_model": "Download model",
                "train": "Train classifier",
                "finetune": "Fine-tune instance model",
                "predict": "Predict",
                "combine_sets": "Combine annotation sets",
            }.get(operation, operation.capitalize())
            details = job.get("model") or job.get("name") or request.get("model") or ""
            if request.get("slide_id") and self.store:
                details = f"{self.store.slide(request['slide_id']).name}: {request['backend']} / {Path(details).name}"
            elif job.get("backend"):
                details = " / ".join(value for value in (job["backend"], details) if value)
            elif job.get("path"):
                details = Path(job["path"]).name
            rows.append([status, f"{description}: {details}".rstrip(": "), job.get("id", row)])
        self.task_table.set_rows(rows)
        self.task_table.resizeRowsToContents()
        self.queue_btn.setToolTip(f"Show task queue ({len(tasks)} tasks)")
        if self.jobs.cancelling:
            self._set_status("Cancelling the active task and clearing the queue...")
        self._update_prediction_controls()

    def _require_project(self) -> AnnotationStore:
        if self.store is None:
            raise ValueError("Create or open a project first.")
        return self.store

    def _require_slide(self) -> AnnotationSlide:
        store = self._require_project()
        if self.slide_id is None:
            raise ValueError("Add and select a slide first.")
        return store.slide(self.slide_id)

    @property
    def set_id(self) -> str | None:
        """Return the selected ID namespace."""
        return cast(str | None, self.sets.currentData())

    @property
    def class_id(self) -> str | None:
        """Return the selected shared class, or no class."""
        item = self.classes.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def choose_new_project(self) -> None:
        """Choose a new persistent project directory."""
        path, _ = QFileDialog.getSaveFileName(self, "New annotation project", "", "Annotation project (*.i2annotate)")
        if path:
            self.new_project(path if path.endswith(".i2annotate") else path + ".i2annotate")

    def new_project(self, path: str | Path) -> None:
        """Create an empty project with common tissue classes."""
        with AnnotationStore(path, create=True) as store:
            for name, color in (
                ("Tissue", "#ef5350"),
                ("Stroma", "#42a5f5"),
                ("Background", "#ffd54f"),
                ("Nucleus", "#ab47bc"),
                ("Cells", "#66bb6a"),
                ("Ignore", "#ff8a65"),
                ("Other", "#26c6da"),
                ("Necrosis", "#8d6e63"),
            ):
                store.save_class(AnnotationClass(name=name, color=color, is_ignore=name == "Ignore"))
        self.open_project(path)

    def choose_project(self) -> None:
        """Open an existing annotation project."""
        path = QFileDialog.getExistingDirectory(self, "Open .i2annotate project")
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        """Load a project and recover abandoned staging sets before starting work."""
        if self.jobs.busy or any(worker not in self._forest_scans for worker in self._threads):
            raise ValueError("Wait for background work or cancel it before switching projects.")
        path = Path(path).expanduser().resolve()
        if self.store and self.store.path == path:
            return
        lock = QLockFile(str(path / ".editor.lock"))
        lock.setStaleLockTime(0)
        if not lock.tryLock(0):
            raise ValueError("This project is already open in another editor, or its directory is not writable.")
        try:
            new_store = AnnotationStore(path)
            new_store.recover()
        except BaseException:
            lock.unlock()
            raise
        new_store.db.execute("PRAGMA busy_timeout=100")
        self._generation += 1
        if self.store:
            self.store.close()
        if self._project_lock:
            self._project_lock.unlock()
        self._project_lock = lock
        self.store = new_store
        self._excluded_training_classes.clear()
        self._backend_state.clear()
        self._selected_custom_checkpoint = ""
        self._catalogues = {
            backend: [entry for entry in entries if not entry.get("custom")]
            for backend, entries in self._catalogues.items()
        }
        self._catalogue_project.clear()
        self._active_backend = ""
        self._forest_path = ""
        self._forest_metadata = None
        self._backend_changed()
        self.renderer = AnnotationRenderer(new_store.path)
        self.slide_id = None
        self.selected.clear()
        self.clear_drawing()
        for layer in self._image_layers:
            if layer in self.view.viewer.layers:
                self.view.viewer.layers.remove(layer)
        self._image_layers.clear()
        if self.reader:
            self.reader.close()
            self.reader = None
        self.overview.viewer.layers.clear()
        for layer in (self._overlay, self._proposal_overlay):
            if layer is not None:
                layer.data = np.zeros((1, 1, 4), dtype=np.uint8)
        self._syncing = True
        self.slides.clear()
        for slide in new_store.slides():
            self.slides.addItem(slide.name)
            cast(QListWidgetItem, self.slides.item(self.slides.count() - 1)).setData(Qt.ItemDataRole.UserRole, slide.id)
        self._syncing = False
        self.refresh_classes()
        self.refresh_sets()
        self.slides.setCurrentRow(0)
        if self.slides.count():
            self._slide_selected(0)
        self.status.setText(f"Opened {new_store.path.name}. Edits are saved automatically.")

    def choose_slides(self) -> None:
        """Add source image files to the current project."""
        self._require_project()
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add 2D slides", "", "Images (*.tif *.tiff *.svs *.ndpi *.czi *.qptiff *.png *.jpg);;All files (*)"
        )
        if paths:
            self.add_slides(paths)
        else:
            path = QFileDialog.getExistingDirectory(self, "Or select an OME-Zarr image (Cancel to stop)")
            if path:
                self.add_slides([path])

    def add_slides(self, paths: list[str]) -> None:
        """Inspect image metadata in a background thread and add valid slide references."""
        store = self._require_project()
        self._require_project()
        self._generation += 1
        generation = self._generation

        def done(slides: list) -> None:
            if generation != self._generation:
                return
            for slide in slides:
                store.save_slide(slide)
                store.create_set(slide.id, "Manual annotations")
                self.slides.addItem(slide.name)
                cast(QListWidgetItem, self.slides.item(self.slides.count() - 1)).setData(
                    Qt.ItemDataRole.UserRole, slide.id
                )
            self.slides.setCurrentRow(self.slides.count() - 1)

        self._background(lambda: [inspect_slide(path) for path in paths], done)

    def _background(self, function: Any, done: Any, *, discard: Any = None) -> Any:
        worker = create_worker(function)
        self._threads.append(worker)
        worker.returned.connect(lambda value: done(value) if not self._closing else discard(value) if discard else None)
        worker.errored.connect(lambda exc: self._error(str(exc)) if not self._closing else None)
        worker.finished.connect(lambda: self._threads.remove(worker) if worker in self._threads else None)
        worker.start()
        return worker

    def _slide_selected(self, row: int) -> None:
        if self._syncing or row < 0 or self.store is None:
            return
        slide_id = cast(QListWidgetItem, self.slides.item(row)).data(Qt.ItemDataRole.UserRole)
        if slide_id == self.slide_id:
            return
        self.slide_id = slide_id
        self._generation += 1
        self.selected.clear()
        self.clear_drawing()
        self.refresh_sets()
        slide = self.store.slide(slide_id)
        self.channels.setText(", ".join(str(index) for index in range(min(3, max(1, len(slide.channel_names))))))
        self.display_mode.setCurrentText("RGB" if len(slide.channel_names) == 3 else "Channels")
        self._try(self.refresh_image)

    def refresh_image(self) -> None:
        """Load lazy image layers and a bounded navigation overview."""
        self._cancel_paint()
        store = self._require_project()
        slide = self._require_slide()
        path = store.resolve_path(slide)
        channels = self._channels()
        self._generation += 1
        generation = self._generation
        rgb = self.display_mode.currentText() == "RGB"

        def load() -> tuple:
            reader = PatchReader(path, series=slide.series)
            try:
                ds = max(1, max(reader.shape) / 256)
                patch = reader.read((0, 0, reader.shape[1], reader.shape[0]), downsample=ds, channels=channels)
            except BaseException:
                reader.close()
                raise
            else:
                return reader, patch

        def done(result: tuple) -> None:
            reader, overview = result
            if generation != self._generation:
                reader.close()
                return
            for layer in self._image_layers:
                if layer in self.view.viewer.layers:
                    self.view.viewer.layers.remove(layer)
            if self.reader:
                self.reader.close()
            self.reader = reader
            self._image_layers.clear()
            rgb_mode = False if rgb and len(channels) not in (3, 4) else rgb
            arrays = []
            for level in reader.display_levels:
                if "c" in reader.axes:
                    selection: list[Any] = [slice(None)] * level.ndim
                    selection[reader.axes.index("c")] = list(channels)
                    array = level[tuple(selection)]
                    array = array.transpose(tuple(reader.axes.index(axis) for axis in "yxc"))
                else:
                    array = level[..., None]
                arrays.append(array)
            for index in range(1 if rgb_mode else len(channels)):
                data = arrays if rgb_mode else [array[..., index] for array in arrays]
                sample = overview.data if rgb_mode else overview.data[..., index]
                low, high = np.percentile(sample, (1, 99))
                layer = self.view.viewer.add_image(
                    data if len(data) > 1 else data[0],
                    multiscale=len(data) > 1,
                    rgb=rgb_mode,
                    name=slide.name if rgb_mode else f"Channel {channels[index]}",
                    contrast_limits=(float(low), float(max(high, low + 1))),
                    blending="additive" if not rgb_mode else "translucent",
                    colormap=("green", "magenta", "cyan")[index % 3] if len(channels) > 1 and not rgb_mode else "gray",
                    translate=(0.5, 0.5),
                )
                layer.metadata["annotation_channels"] = channels if rgb_mode else (channels[index],)
                for setting in ("visible", "contrast_limits", "gamma", "colormap", "opacity", "blending", "data"):
                    getattr(layer.events, setting).connect(self._cancel_paint)
                self._image_layers.append(layer)
            for layer in self._image_layers:
                self.view.viewer.layers.move(self.view.viewer.layers.index(layer), 0)
            self.overview.viewer.layers.clear()
            data = overview.data if rgb_mode else overview.data[..., 0]
            self.overview.viewer.add_image(
                data,
                rgb=rgb_mode,
                scale=overview.downsample[::-1],
                translate=(overview.downsample[1] / 2, overview.downsample[0] / 2),
            )
            self.overview.viewer.reset_view()
            self.view.viewer.reset_view()
            self.set_mode(self.tool.currentText())
            self.schedule_render()

        self._background(load, done, discard=lambda result: result[0].close())

    def _channels(self) -> tuple[int, ...]:
        values = tuple(int(value.strip()) for value in self.channels.text().split(",") if value.strip())
        if not values or min(values) < 0 or len(set(values)) != len(values):
            raise ValueError("Enter distinct, nonnegative channel indices.")
        return values

    def refresh_classes(self) -> None:
        """Refresh class names and controls without changing their IDs."""
        current = self.class_id
        self.classes.clear()
        if self.store:
            for index, item in enumerate(self.store.classes()):
                self.classes.addItem(
                    f"{item.name}{' [Ignore mask]' if item.is_ignore else ''}"
                    f"{' [hidden]' if not item.visible else ''}{' [locked]' if item.locked else ''}"
                )
                cast(QListWidgetItem, self.classes.item(index)).setData(Qt.ItemDataRole.UserRole, item.id)
                cast(QListWidgetItem, self.classes.item(index)).setData(
                    Qt.ItemDataRole.DecorationRole, QColor(item.color)
                )
                if item.id == current:
                    self.classes.setCurrentRow(index)
            if self.classes.currentRow() < 0:
                self.classes.setCurrentRow(0)

    def refresh_sets(self) -> None:
        """Refresh namespaces and preserve the active set where possible."""
        current = self.set_id
        self._syncing = True
        self.sets.clear()
        if self.store and self.slide_id:
            for item in self.store.sets(self.slide_id):
                self.sets.addItem(
                    f"{item['name']} ({item['object_count']:,} {'object' if item['object_count'] == 1 else 'objects'})",
                    item["id"],
                )
            index = self.sets.findData(current)
            self.sets.setCurrentIndex(max(0, index))
        self._syncing = False
        self._set_changed()

    def _refresh_set_counts(self) -> None:
        if self.store and self.slide_id:
            for info in self.store.sets(self.slide_id):
                index = self.sets.findData(info["id"])
                if index >= 0:
                    self.sets.setItemText(
                        index,
                        f"{info['name']} ({info['object_count']:,} {'object' if info['object_count'] == 1 else 'objects'})",
                    )

    def choose_combine_sets(self) -> None:
        """Select same-slide sets and name a combined copy, retaining the source sets."""
        store = self._require_project()
        slide = self._require_slide()
        dialog = QDialog(self)
        dialog.setWindowTitle("Combine annotation sets")
        dialog.resize(580, 400)
        layout = hp.make_v_layout(parent=dialog)
        layout.addWidget(
            hp.make_label(
                dialog,
                "Select at least two sets with Ctrl/Shift. A combined copy keeps the original sets and resolves duplicate object IDs.",
                wrap=True,
            )
        )
        table = AnnotationTable(
            dialog, TableConfig().add("Annotation set", "set").add("Objects", "objects", dtype="int", sizing="contents")
        )
        table.set_rows(
            [[info["name"], info["object_count"], info["id"]] for info in store.sets(slide.id)], selected=[self.set_id]
        )
        layout.addWidget(table.filter_widget)
        layout.addWidget(table, 1)
        name = hp.make_line_edit(dialog, text="Combined annotations")
        layout.addWidget(hp.make_label(dialog, "Combined set name"))
        layout.addWidget(name)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            self.combine_sets(table.selected_keys(), name.text())

    def combine_sets(self, set_ids: list[str], name: str) -> None:
        """Queue a cancellable combined copy of two or more same-slide sets."""
        store = self._require_project()
        if len(set(set_ids)) < 2:
            raise ValueError("Select at least two annotation sets.")
        self.jobs.submit({"operation": "combine_sets", "project": str(store.path), "set_ids": set_ids, "name": name})

    def remove_set(self) -> None:
        """Remove the active set from the editor; Undo restores it with its objects."""
        store = self._require_project()
        if not self.set_id:
            raise ValueError("Select an annotation set first.")
        if self.jobs.busy:
            raise ValueError("Wait for background work or cancel it before removing a set.")
        store.remove_set(self.set_id)
        self.refresh_sets()
        self._changed()

    def _set_changed(self, *_args: Any) -> None:
        if self._syncing:
            return
        self._pages = {"accepted": [0], "proposed": [0]}
        self.clear_drawing()
        self.selected.clear()
        self._review_selection.clear()
        self.refresh_tables()
        self.schedule_render()

    def _pagination_controls(self, panel: QWidget, review: str) -> Any:
        previous = self._button(panel, "Previous", partial(self.previous_page, review))
        next_ = self._button(panel, "Next", partial(self.next_page, review))
        label = hp.make_label(panel, "Page 1 of 1; 0 objects (100 per page)", wrap=True)
        previous.setEnabled(False)
        next_.setEnabled(False)
        self._pagination[review] = previous, next_, label
        return hp.make_h_layout(previous, label, next_)

    def refresh_tables(self) -> None:
        """Fetch a page of objects rather than populating a million-row Qt table."""
        self._syncing = True
        classes = {item.id: item for item in self.store.classes()} if self.store else {}
        names = [item.name for item in classes.values()]
        class_names = {
            key: item.name
            if names.count(item.name) == 1 and item.name != "Unclassified"
            else f"{item.name} ({key[:8]})"
            for key, item in classes.items()
        }
        self._class_labels = {"Unclassified": None, **{name: key for key, name in class_names.items()}}
        for table, review in ((self.objects, "accepted"), (self.proposals, "proposed")):
            selected = self.selected if review == "accepted" else self._review_selection
            model = table.model()
            model.disabled, model.tooltips, model.colors = set(), {}, {}
            table.set_class_choices(list(self._class_labels))
            previous, next_, label = self._pagination[review]
            if not self.store or not self.set_id:
                previous.setEnabled(False)
                next_.setEnabled(False)
                label.setText("Page 1 of 1; 0 objects (100 per page)")
                table.set_rows([])
                self._page_totals[review] = 0
                continue
            pages = self._pages[review]
            objects = list(self.store.iter_objects(set_id=self.set_id, review=review, after=pages[-1], limit=101))
            while not objects and len(pages) > 1:
                pages.pop()
                objects = list(self.store.iter_objects(set_id=self.set_id, review=review, after=pages[-1], limit=101))
            self._page_totals[review] = self.store.count(self.set_id, review=review)
            previous.setEnabled(len(pages) > 1)
            next_.setEnabled(len(objects) > 100)
            self._page_last[review] = objects[min(100, len(objects)) - 1].key if objects else 0
            rows = []
            for obj in objects[:100]:
                annotation_class = classes.get(obj.class_id or "")
                eligible = annotation_class is not None and not annotation_class.is_ignore
                locked = bool(
                    obj.properties.get("locked")
                    or obj.properties.get("isLocked")
                    or (annotation_class and annotation_class.locked)
                )
                if locked:
                    model.disabled.add((obj.key, table.config.find_col_id("class")))
                if annotation_class:
                    model.colors[obj.key] = annotation_class.color
                if review == "accepted":
                    if locked or not eligible:
                        model.disabled.add((obj.key, 0))
                    model.tooltips[obj.key, 0] = (
                        "Ignore mask: excludes training pixels in this annotation set."
                        if annotation_class and annotation_class.is_ignore
                        else "Assign a class before including this annotation."
                        if not eligible
                        else "Unlock this object and its class to change training inclusion."
                        if locked
                        else "Include this accepted annotation in the next classifier training run."
                    )
                row = [str(obj.object_id), class_names.get(obj.class_id, "Unclassified"), round(obj.geometry.area, 1)]
                rows.append(
                    [bool(obj.training and eligible), *row, obj.key]
                    if review == "accepted"
                    else [*row, "Check tile seam" if obj.properties.get("ambiguous_seam") else "Needs review", obj.key]
                )
            table.set_rows(rows, selected=selected)
            self._page_status(review)
        self._syncing = False
        self.selection_label.setText(f"{len(self.selected)} selected.")
        self._refresh_training_summary()
        self._proposal_details()
        self._refresh_set_counts()

    def _page_status(self, review: str) -> None:
        table = self.proposals if review == "proposed" else self.objects
        total = self._page_totals[review]
        count = table.proxy.rowCount()
        self._pagination[review][2].setText(
            f"Page {len(self._pages[review])} of {max(1, math.ceil(total / 100))}\nShowing {count} of {total:,} objects (100 per page)"
            + (f"; {count} of {table.n_rows} loaded rows match" if table.filter_edit.text() else "")
        )

    def _refresh_training_summary(self) -> None:
        # Aggregate the project only when these counts are visible, keeping drawing and paging responsive.
        if self.tabs.currentIndex() != 1:
            return
        rows = self.store.training_summary() if self.store else []
        classes = self.store.classes() if self.store else []
        counts: dict[str, list[int]] = {item.id: [0, 0] for item in classes}
        for row in rows:
            if row["class_id"] in counts:
                counts[row["class_id"]][0] += row["included"]
                counts[row["class_id"]][1] += row["total"]
        model = self.training_counts.model()
        model.disabled = {(item.id, 0) for item in classes if item.is_ignore}
        model.colors = {item.id: item.color for item in classes}
        model.tooltips = {
            (item.id, 0): "Ignore regions mask training pixels and cannot be trained as a class."
            if item.is_ignore
            else "Choose this class for the next training run; individual object flags are preserved."
            for item in classes
        }
        self.training_counts.set_rows(
            [
                [
                    not item.is_ignore and item.id not in self._excluded_training_classes,
                    item.name,
                    "Mask" if item.is_ignore else counts[item.id][0],
                    counts[item.id][1],
                    item.id,
                ]
                for item in classes
            ]
        )
        selected = {
            item.id for item in classes if not item.is_ignore and item.id not in self._excluded_training_classes
        }
        included = sum(counts[key][0] for key in selected)
        available = sum(counts[key][0] > 0 for key in selected)
        self.training_summary.setText(
            f"Across all project slides: {included:,} included annotations in {available} enabled classes. "
            "The forest requires at least two classes."
        )
        self.train_btn.setEnabled(available >= 2 and not self.jobs.busy)
        self.finetune_btn.setEnabled(available >= 1 and not self.jobs.busy)

    def _training_class_edited(self, key: str, _column: str, value: Any) -> bool:
        if value:
            self._excluded_training_classes.discard(key)
        else:
            self._excluded_training_classes.add(key)
        self._refresh_training_summary()
        return True

    def _table_edited(self, key: int, column: str, value: Any) -> bool:
        if self._syncing:
            return False

        def edit() -> bool:
            if column == "training":
                self.set_training(bool(value), keys=[key])
            else:
                store = self._require_project()
                obj = store.get(key)
                store.apply(EditCommand("Assign class", [(key, replace(obj, class_id=self._class_labels[value]))]))
                self._changed(obj.geometry.bounds)
            return True

        return bool(self._try(edit))

    def _proposal_details(self) -> None:
        if self._syncing or not hasattr(self, "proposal_details"):
            return
        keys = self._review_selection
        self.refine_btn.setEnabled(len(keys) == 1 and not self.jobs.busy)
        self.merge_proposals_btn.setEnabled(len(keys) >= 2 and not self.jobs.busy)
        self.accept_selected_btn.setEnabled(bool(keys) and not self.jobs.busy)
        self.reject_selected_btn.setEnabled(bool(keys) and not self.jobs.busy)
        if not keys or not self.store:
            self.proposal_details.setText("Select a proposal to see its source model.")
            return
        obj = self.store.get(keys[0])
        request = obj.provenance.get("request", {})
        model = obj.provenance.get("model", {})
        self.proposal_details.setText(
            f"{len(keys)} selected. Backend: {request.get('backend', model.get('package', 'Unknown'))}\n"
            f"Model: {request.get('model') or request.get('checkpoint') or 'Not recorded'}"
        )

    def next_page(self, review: str | None = None) -> None:
        """Advance through objects using indexed keyset pagination."""
        review = review or ("proposed" if self.tabs.currentIndex() == 3 else "accepted")
        table = self.proposals if review == "proposed" else self.objects
        if table.n_rows and self._pagination[review][1].isEnabled():
            self._pages[review].append(self._page_last[review])
            self.refresh_tables()

    def previous_page(self, review: str | None = None) -> None:
        """Return to the preceding object page."""
        review = review or ("proposed" if self.tabs.currentIndex() == 3 else "accepted")
        if len(self._pages[review]) > 1:
            self._pages[review].pop()
            self.refresh_tables()

    def _table_selection(self, table: AnnotationTable | None = None) -> None:
        if not self._syncing:
            self.selected = _selected_keys(table if table is not None else self.objects)
            if table is self.proposals:
                self._review_selection = list(self.selected)
            self._show_selected()
            self._proposal_details()

    def _select_from_canvas(self, keys: list[int], focus_key: int | None = None) -> None:
        reviewing = self.tabs.currentIndex() == 3
        review = "proposed" if reviewing else "accepted"
        table = self.proposals if reviewing else self.objects
        self.selected = list(keys)
        if reviewing:
            self._review_selection = list(keys)
        focus_key = focus_key if focus_key is not None else keys[0] if keys else None
        self._syncing = True
        try:
            loaded = {row[-1] for row in table.model().get_data()}
            if focus_key is not None and focus_key not in loaded:
                self._pages[review] = self.store.page_cursors(self.set_id, focus_key, review=review)
                self.refresh_tables()
                self._syncing = True
            visible = {table.key_at(row) for row in range(table.proxy.rowCount())}
            if focus_key is not None and focus_key not in visible:
                table.filter_edit.clear()
            table.select_keys(keys, scroll_to=focus_key)
        finally:
            self._syncing = False
        self._show_selected()
        self._proposal_details()

    def _show_selected(self) -> None:
        if not self._paint_committing:
            self._cancel_paint()
        self._stroke_revision += 1
        summary = f"{len(self.selected)} selected."
        if len(self.selected) == 1 and self.store:
            obj = self.store.get(self.selected[0])
            annotation_class = next((item for item in self.store.classes() if item.id == obj.class_id), None)
            if annotation_class is None:
                training = "Assign a class to use for training"
            elif annotation_class.is_ignore:
                training = "Ignore mask: excludes training pixels"
            elif obj.review != "accepted":
                training = "Accept and include to use for training"
            else:
                training = "Included in training" if obj.training else "Excluded from training"
            summary += f" ID: {obj.object_id}\n{obj.review.capitalize()} | {training}"
        self.selection_label.setText(summary)
        self.draw_layer.data = []
        self._ring_layout.clear()
        if self.selected and self.store and self._mode not in ("Polygon", "Rectangle", "Freehand"):
            rings = []
            for key in self.selected[:100]:
                obj = self.store.get(key)
                polygons = list(obj.geometry.geoms) if obj.geometry.geom_type == "MultiPolygon" else [obj.geometry]
                for polygon in polygons:
                    self._ring_layout.append(1 + len(polygon.interiors))
                    for ring in (polygon.exterior, *polygon.interiors):
                        rings.append(np.asarray(ring.coords)[:-1, ::-1])
            if rings:
                self.draw_layer.add(rings, shape_type="polygon", edge_color="yellow", edge_width=2)
            self._edit_revision = self.store.revision
        self.set_mode(self.tool.currentText())

    def set_mode(self, mode: str) -> None:
        """Activate a drawing or selection tool using the editor's pixel coordinate frame."""
        previous_mode, self._mode = self._mode, mode
        if previous_mode != mode:
            self._cancel_paint()
            self._stroke_revision += 1
            if hasattr(self, "brush_preview"):
                self.brush_preview.data = []
        if not hasattr(self, "draw_layer"):
            return
        if previous_mode != mode and mode in (
            "Polygon",
            "Rectangle",
            "Cut",
            "Select by polygon",
            "Select by rectangle",
            "Freehand",
        ):
            self.draw_layer.data = []
        mapping = {
            "Polygon": "add_polygon",
            "Rectangle": "add_rectangle",
            "Vertices": "direct",
            "Cut": "add_path",
            "Select by polygon": "add_polygon",
            "Select by rectangle": "add_rectangle",
        }
        self.draw_layer.mode = mapping.get(mode, "pan_zoom")
        self.prompt_box_layer.mode = "add_rectangle" if mode == "Prompt box" else "pan_zoom"
        self.prompt_box_layer.mouse_pan = False
        self.view.viewer.layers.selection.select_only(
            self.prompt_box_layer if mode == "Prompt box" else self.draw_layer
        )
        self.draw_layer.mouse_pan = mode == "Interact"
        if mode in ("Brush +", "Brush -", "Wand"):
            self.draw_layer.cursor = "circle"
            self.draw_layer.cursor_size = (
                self.wand_radius.value() if mode == "Wand" else self.brush_radius.value()
            ) * 2
        self.prompt_layer.mode = "pan_zoom"
        self.tool_hint.setText(
            {
                "Interact": "Click to select; Shift-click to add. Drag to pan; scroll to zoom.",
                "Polygon": "Click vertices; double-click to finish. Auto-add saves immediately; otherwise use Apply drawing.",
                "Rectangle": "Drag a rectangle, then Apply drawing or Predict drawn rectangle.",
                "Freehand": "Drag a region; saved on release. Escape cancels.",
                "Brush +": "Start on empty space for a new object, or inside an object to edit it. Alt subtracts; Escape cancels.",
                "Brush -": "Start inside an object, or select one first. Drag to subtract; Escape cancels.",
                "Wand": "Click or drag to fill similar visible pixels. Alt subtracts; zoom out for larger areas. Escape cancels.",
                "Vertices": "Select one object and edit its yellow vertices, then Apply drawing.",
                "Cut": "Select one object, draw a cut line, then Apply drawing to choose the surviving ID.",
                "Select by polygon": "Click vertices; double-click to select intersecting objects in the active set.",
                "Select by rectangle": "Drag a rectangle to select intersecting objects in the active set.",
                "Prompt +": "Click to mark foreground. Right-click a point or inside a box to remove it.",
                "Prompt -": "Click to mark an exclusion. Right-click a point or inside a box to remove it.",
                "Prompt box": "Drag a box. Right-click a point or inside a box to remove it. Clear prompts removes all hints.",
            }.get(mode, "")
        )

    def _drawing_finished(self, _layer: Any, event: Any) -> None:
        mode = self._mode
        if (
            event.button != 1
            or not _layer._is_creating
            or not (mode == "Select by polygon" or (mode == "Polygon" and self.auto_add_polygons.isChecked()))
        ):
            return
        self._schedule_drawing_completion()

    def _schedule_drawing_completion(self) -> None:
        mode = self._mode
        self._stroke_revision += 1
        revision = self._stroke_revision
        context = (self._generation, self.slide_id, self.set_id, self.tabs.currentIndex())

        def finish() -> None:
            # Napari finalizes its shape after custom double-click callbacks run.
            if (
                not self._closing
                and revision == self._stroke_revision
                and mode == self._mode
                and context == (self._generation, self.slide_id, self.set_id, self.tabs.currentIndex())
                and len(self.draw_layer.data)
            ):
                self._stroke_revision += 1
                self._try(self.apply_drawing)

        QTimer.singleShot(0, finish)

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Route editor shortcuts and drops before napari handles canvas events."""
        if not isinstance(obj, QWidget) or QWidget.window(obj) is not self:
            return super().eventFilter(obj, event)
        if obj is self and event.type() == QEvent.Type.WindowDeactivate:
            self._cancel_paint()
        if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            key_event = cast(QKeyEvent, event)
            if key_event.key() == Qt.Key.Key_Alt and not key_event.isAutoRepeat() and self._paint is not None:
                stroke = self._paint
                self._paint_point(
                    stroke, stroke.point, stroke.mode == "Brush -" or event.type() == QEvent.Type.KeyPress, force=True
                )
        if event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            key_event = cast(QKeyEvent, event)
            action = self._shortcut_action(key_event)
            if action is not None:
                event.accept()
                if event.type() == QEvent.Type.KeyPress and not key_event.isAutoRepeat():
                    self._try(action)
                return True
        if event.type() in (QEvent.Type.DragEnter, QEvent.Type.DragMove):
            self.dragEnterEvent(cast(QDragEnterEvent, event))
            return True
        if event.type() == QEvent.Type.Drop:
            self.dropEvent(cast(QDropEvent, event))
            return True
        if (
            obj is self.view.widget.canvas.native
            and self._mode == "Select by rectangle"
            and event.type() == QEvent.Type.MouseButtonRelease
            and cast(QMouseEvent, event).button() == Qt.MouseButton.LeftButton
            and self.draw_layer._is_creating
        ):
            self._schedule_drawing_completion()
        if (
            obj is self.view.widget.canvas.native
            and self._mode in ("Prompt +", "Prompt -", "Prompt box")
            and event.type()
            in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick)
        ):
            mouse = cast(QMouseEvent, event)
            if mouse.button() == Qt.MouseButton.RightButton:
                if event.type() == QEvent.Type.MouseButtonPress:
                    point = mouse.position() if hasattr(mouse, "position") else mouse.localPos()
                    canvas = self.view.widget.canvas
                    position = canvas._map_canvas2world((point.x(), point.y()), canvas.view)
                    self._remove_prompt(position[-2:])
                return True
        return super().eventFilter(obj, event)

    def _shortcut_action(self, event: QKeyEvent) -> Any:
        focus = QApplication.focusWidget()
        if (
            QApplication.activePopupWidget() is not None
            or QApplication.activeModalWidget() is not None
            or isinstance(focus, QLineEdit | QTextEdit | QPlainTextEdit | QAbstractSpinBox)
            or (isinstance(focus, QComboBox) and focus.isEditable())
            or event.modifiers()
            & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier | Qt.KeyboardModifier.AltModifier)
        ):
            return None
        parent = focus
        while parent is not None and parent is not self:
            if isinstance(parent, QAbstractItemView) and parent.state() == QAbstractItemView.State.EditingState:
                return None
            parent = parent.parentWidget()
        canvas = self.view.widget.canvas.native
        on_canvas = focus is canvas or (focus is not None and canvas.isAncestorOf(focus))
        key = event.key()
        if on_canvas and key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            return self.apply_drawing
        if on_canvas and key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            return self.clear_drawing
        modes = {
            Qt.Key.Key_Plus: "Prompt +",
            Qt.Key.Key_Minus: "Prompt -",
            Qt.Key.Key_P: "Polygon",
            Qt.Key.Key_I: "Interact",
            Qt.Key.Key_V: "Interact",
            Qt.Key.Key_B: "Brush +",
            Qt.Key.Key_W: "Wand",
            Qt.Key.Key_F: "Freehand",
        }
        mode = "Prompt +" if event.text() == "+" else modes.get(key)
        if key == Qt.Key.Key_R and self.tabs.currentIndex() != 3:
            mode = "Prompt box"
        if mode:
            return partial(self.tool.setCurrentText, mode)
        if key == Qt.Key.Key_M:
            return (
                partial(self._review_action, self.merge_proposals)
                if self.tabs.currentIndex() == 3
                else self.merge_selected
            )
        if self.tabs.currentIndex() == 3:
            actions = {
                Qt.Key.Key_A: partial(self.review, True, False),
                Qt.Key.Key_R: partial(self.review, False, False),
                Qt.Key.Key_E: self.refine_proposal,
            }
            if key in actions:
                return partial(self._review_action, actions[key])
        return None

    def _drop_paths(self, event: QDropEvent) -> tuple[list[Path], bool]:
        urls = event.mimeData().urls()
        if not urls or any(not url.isLocalFile() for url in urls):
            raise ValueError("Drop a local annotation project or supported images.")
        paths = list(dict.fromkeys(Path(url.toLocalFile()).resolve() for url in urls))
        projects = [path for path in paths if path.suffix.lower() == ".i2annotate"]
        if projects:
            if len(paths) != 1 or len(projects) != 1:
                raise ValueError("Drop one annotation project, or a batch of images, separately.")
            if not projects[0].is_dir():
                raise ValueError("The annotation project directory does not exist.")
            return projects, True
        suffixes = (
            ".tif",
            ".tiff",
            ".svs",
            ".ndpi",
            ".czi",
            ".scn",
            ".qptiff",
            ".qptiff.raw",
            ".qptiff.intermediate",
            ".png",
            ".jpg",
            ".jpeg",
            ".zarr",
        )
        if any(not path.exists() or not path.name.lower().endswith(suffixes) for path in paths):
            raise ValueError("Drop supported images, including TIFF, CZI, JPEG, PNG, or OME-Zarr.")
        return paths, False

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Accept supported project and image drops throughout the editor."""
        try:
            self._drop_paths(event)
        except ValueError as exc:
            self._set_status(str(exc))
            event.ignore()
        else:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        """Open a dropped project or add dropped images through the project workflow."""
        try:
            paths, project = self._drop_paths(event)
        except ValueError as exc:
            self._set_status(str(exc))
            event.ignore()
            return
        if self._try(partial(self._open_dropped_paths, paths, project)):
            event.acceptProposedAction()
        else:
            event.ignore()

    def _open_dropped_paths(self, paths: list[Path], project: bool) -> bool:
        if project:
            path = paths[0]
            if self.store and self.store.path == path:
                return True
            if self.jobs.busy or any(worker not in self._forest_scans for worker in self._threads):
                raise ValueError("Wait for background work or cancel it before switching projects.")
            if self.store and not hp.confirm(
                self,
                f"Close {self.store.path.name} and open {path.name}? Edits are saved. Temporary drawings and prompts will be discarded.",
                "Open annotation project",
            ):
                return False
            self.open_project(path)
        else:
            if self.store is None:
                self.choose_new_project()
            if self.store is None:
                return False
            self.add_slides([str(path) for path in paths])
        return True

    def _remove_prompt(self, yx: tuple[float, float]) -> None:
        points = self.prompt_layer.data
        if len(points):
            distances = np.linalg.norm(points - np.asarray(yx), axis=1) * self.view.viewer.camera.zoom
            index = int(np.argmin(distances))
            if distances[index] <= 8:
                self.prompt_layer.selected_data = {index}
                self.prompt_layer.remove_selected()
                self.prompt_layer.selected_data = set()
                self._prompt_labels.pop(index)
                return
        for index in reversed(range(len(self.prompt_box_layer.data))):
            box = Polygon(self.prompt_box_layer.data[index])
            if box.covers(Point(yx)):
                self.prompt_box_layer.selected_data = {index}
                self.prompt_box_layer.remove_selected()
                self.prompt_box_layer.selected_data = set()
                return

    def clear_prompts(self) -> None:
        """Remove all AI prompt points and boxes without changing annotation drawings."""
        self.prompt_layer.selected_data = set()
        self.prompt_layer.data = np.empty((0, 2))
        self.prompt_box_layer.selected_data = set()
        self.prompt_box_layer.data = []
        self._prompt_labels.clear()

    def clear_drawing(self) -> None:
        """Discard temporary drawing and prompts without touching stored annotations."""
        self._cancel_paint()
        self._stroke_revision += 1
        self.draw_layer.data = []
        self.brush_preview.data = []
        self.clear_prompts()
        self._ring_layout.clear()

    def _mouse_drag(self, _viewer: Any, event: Any) -> Any:
        if not self.store or not self.slide_id or event.button != 1:
            return
        xy = (float(event.position[-1]), float(event.position[-2]))
        if self._mode == "Interact":
            reviewing = self.tabs.currentIndex() == 3
            obj = self.store.hit_test(
                self.slide_id, xy, set_id=self.set_id, review="proposed" if reviewing else "accepted"
            )
            extend = "Shift" in event.modifiers
            previous = self._review_selection if reviewing else self.selected
            self.selected = (
                (
                    [*previous, obj.key]
                    if extend and obj and obj.key not in previous
                    else list(previous)
                    if extend and obj
                    else [obj.key]
                )
                if obj
                else []
            )
            self._select_from_canvas(self.selected, obj.key if obj else None)
            return
        if self._mode in ("Prompt +", "Prompt -"):
            # Color only the next prompt, without editing a previously selected point.
            self.prompt_layer.selected_data = set()
            self.prompt_layer.current_face_color = "lime" if self._mode == "Prompt +" else "tomato"
            self.prompt_layer.add(np.asarray(xy[::-1]))
            self._prompt_labels.append(1 if self._mode == "Prompt +" else 0)
            return
        if self._mode in ("Brush +", "Brush -", "Wand"):
            stroke = self._begin_paint(xy, self._mode == "Brush -" or "Alt" in event.modifiers)
            if stroke is None:
                return
            yield
            while event.type == "mouse_move":
                if not self._paint_valid(stroke):
                    self._cancel_paint()
                    return
                point = (float(event.position[-1]), float(event.position[-2]))
                self._paint_point(stroke, point, stroke.mode == "Brush -" or "Alt" in event.modifiers)
                yield
            if self._paint_valid(stroke):
                point = (float(event.position[-1]), float(event.position[-2]))
                self._paint_point(stroke, point, stroke.mode == "Brush -" or "Alt" in event.modifiers, force=True)
                self._paint = None
                if stroke.mode == "Wand":
                    stroke.queue.put(None)
                else:
                    self._try(partial(self._commit_paint, stroke, stroke.geometry))
            else:
                self._cancel_paint()
            return
        if self._mode != "Freehand":
            return
        mode, slide_id = self._mode, self.slide_id
        stroke_revision = self._stroke_revision
        points = [xy]
        yield
        while event.type == "mouse_move":
            if self._mode != mode or self.slide_id != slide_id or stroke_revision != self._stroke_revision:
                self.brush_preview.data = []
                return
            point = (float(event.position[-1]), float(event.position[-2]))
            if np.linalg.norm(np.asarray(point) - points[-1]) > max(0.2, 1 / self.view.viewer.camera.zoom):
                points.append(point)
                self.draw_layer.data = []
                self.draw_layer.add(
                    np.asarray(points)[:, ::-1],
                    shape_type="path",
                    edge_color=self.CONFIG.freehand_color,
                    edge_width=3,
                )
            yield
        self.brush_preview.data = []
        if self._mode != mode or self.slide_id != slide_id or stroke_revision != self._stroke_revision:
            return
        point = (float(event.position[-1]), float(event.position[-2]))
        if point != points[-1]:
            points.append(point)
        if len(points) >= 3:
            self.draw_layer.data = []
            self.draw_layer.add(
                np.asarray(points)[:, ::-1],
                shape_type="polygon",
                edge_color=self.CONFIG.freehand_color,
                edge_width=3,
            )
            self._try(lambda: self.add_geometry(Polygon(points)))

    def _paint_context(self) -> tuple[Any, ...]:
        return (
            self._generation,
            id(self.store),
            self.slide_id,
            self.set_id,
            self.tabs.currentIndex(),
            self._mode,
            float(self.view.viewer.camera.zoom),
            self.view.widget.canvas.bgcolor,
            tuple(id(layer) for layer in self.view.viewer.layers if layer in self._image_layers),
        )

    def _paint_valid(self, stroke: _PaintStroke) -> bool:
        return not self._closing and not stroke.cancelled.is_set() and stroke.context == self._paint_context()

    def _cancel_paint(self, *_args: Any) -> None:
        for stroke in [self._paint, *self._wand_strokes]:
            if stroke is not None:
                stroke.cancelled.set()
                stroke.queue.put(None)
        self._paint = None
        self._wand_strokes.clear()
        if hasattr(self, "brush_preview"):
            self.brush_preview.data = []

    def _wand_layers(self) -> tuple[WandLayer, ...]:
        layers = []
        for layer in self.view.viewer.layers:
            if layer not in self._image_layers or not layer.visible or layer.opacity <= 0:
                continue
            layers.append(
                WandLayer(
                    tuple(layer.metadata["annotation_channels"]),
                    tuple(layer.contrast_limits),
                    float(layer.gamma),
                    None if layer.rgb else layer.colormap.map(np.linspace(0, 1, 1024)).copy(),
                    float(layer.opacity),
                    str(layer.blending),
                )
            )
        return tuple(layers)

    def _resolve_paint_target(self, stroke: _PaintStroke) -> None:
        store = self._require_project()
        review = "proposed" if self.tabs.currentIndex() == 3 else "accepted"
        target = store.hit_test(self.slide_id, stroke.start, set_id=self.set_id, review=review)
        if target is None and stroke.initial_subtract and len(stroke.fallback) == 1:
            selected = store.get(stroke.fallback[0])
            if selected.set_id == self.set_id and selected.review == review:
                target = selected
        if stroke.initial_subtract and target is None:
            raise ValueError(
                "Start on an object or select exactly one object before subtracting with the brush or wand."
            )
        if target is not None:
            store._require_unlocked(target)
        stroke.target = target
        stroke.geometry = stroke.base = target.geometry if target is not None else Polygon()
        stroke.revision = store.revision

    def _begin_paint(self, xy: tuple[float, float], subtract: bool) -> _PaintStroke | None:
        zoom = max(0.0001, float(self.view.viewer.camera.zoom))
        stroke = _PaintStroke(
            self._mode,
            self._paint_context(),
            xy,
            xy,
            subtract,
            self.wand_radius.value() if self._mode == "Wand" else self.brush_radius.value(),
            zoom,
            self.class_id,
            list(self.selected),
            initial_subtract=subtract,
        )
        if stroke.mode == "Wand":
            stroke.background = QColor(self.view.widget.canvas.bgcolor).getRgbF()[:3]
            stroke.layers = self._wand_layers()
            if not stroke.layers:
                self._error("Show at least one image channel before using the wand.")
                return None
        try:
            if stroke.mode == "Wand":
                self._wand_strokes.append(stroke)
                stroke.queue.put((xy, subtract))
                self._start_wand()
            else:
                self._resolve_paint_target(stroke)
                stroke.points = [xy]
                self._paint_point(stroke, xy, subtract, force=True)
        except (ValueError, KeyError) as exc:
            stroke.cancelled.set()
            self._error(str(exc))
            return None
        self._paint = stroke
        if stroke.cancelled.is_set():
            self._paint = None
            return None
        return stroke

    def _paint_point(
        self, stroke: _PaintStroke, xy: tuple[float, float], subtract: bool, *, force: bool = False
    ) -> None:
        if not self._paint_valid(stroke):
            return
        previous, previous_subtract = stroke.point, stroke.subtract
        if stroke.mode == "Wand":
            if xy != previous or subtract != previous_subtract:
                stroke.queue.put((xy, subtract))
            stroke.point, stroke.subtract = xy, subtract
            if subtract != previous_subtract:
                self.brush_preview.edge_color = (
                    self.CONFIG.brush_subtract_color if subtract else self.CONFIG.brush_add_color
                )
        else:
            radius = stroke.radius / stroke.zoom
            if subtract != previous_subtract:
                footprint = brush_stroke(stroke.points, radius, fill=not previous_subtract)
                stroke.base = stroke.base.difference(footprint) if previous_subtract else stroke.base.union(footprint)
                stroke.points = [previous]
                force = True
            if xy != stroke.points[-1]:
                stroke.points.append(xy)
            stroke.point, stroke.subtract = xy, subtract
            if not force and monotonic() - self._preview_updated < 1 / 30:
                return
            footprint = brush_stroke(stroke.points, radius, fill=not subtract)
            stroke.geometry = stroke.base.difference(footprint) if subtract else stroke.base.union(footprint)
            self._preview_geometry(stroke.geometry, subtract)
        self.tool_hint.setText(
            f"{stroke.mode}: {'subtracting' if subtract else 'adding'}. Release to save; Escape cancels."
        )

    def _preview_geometry(self, geometry: BaseGeometry, subtract: bool) -> None:
        self._preview_updated = monotonic()
        polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        rings = [
            np.asarray(ring.coords)[:, ::-1]
            for polygon in polygons
            if not polygon.is_empty
            for ring in (polygon.exterior, *polygon.interiors)
        ]
        color = self.CONFIG.brush_subtract_color if subtract else self.CONFIG.brush_add_color
        if rings and len(rings) == len(self.brush_preview.data):
            self.brush_preview.data = rings
            self.brush_preview.edge_color = color
        elif rings:
            self.brush_preview.data = []
            self.brush_preview.add(
                rings,
                shape_type="path",
                edge_width=3,
                edge_color=color,
            )
        else:
            self.brush_preview.data = []

    def _commit_paint(self, stroke: _PaintStroke, geometry: BaseGeometry) -> None:
        if not self._paint_valid(stroke):
            return
        store = self._require_project()
        target = stroke.target
        self.brush_preview.data = []
        if (target is None and geometry.is_empty) or (target is not None and target.geometry.equals(geometry)):
            self.set_mode(self._mode)
            return
        obj = (
            replace(target, geometry=geometry)
            if target is not None
            else AnnotationObject(
                self.set_id,
                store.next_id(self.set_id),
                geometry,
                stroke.class_id,
                provenance={"source": "manual"},
                training=True,
            )
        )
        self._paint_committing = True
        try:
            keys = store.apply(
                EditCommand(
                    f"{stroke.mode} stroke", [(target.key if target else None, None if geometry.is_empty else obj)]
                ),
                expected_revision=stroke.revision,
            )
            self.selected = keys if not geometry.is_empty else []
            if target is not None and target.review == "proposed":
                self._review_selection = list(self.selected)
            bounds = target.geometry.union(geometry).bounds if target is not None else geometry.bounds
            self._changed(bounds)
        finally:
            self._paint_committing = False

    def _start_wand(self) -> None:
        if self._wand_worker is not None or not self._wand_strokes:
            return
        stroke = self._wand_strokes[0]
        if not self._paint_valid(stroke):
            self._wand_strokes.pop(0)
            self._start_wand()
            return
        try:
            self._resolve_paint_target(stroke)
            slide = self._require_slide()
            path, series = self.store.resolve_path(slide), slide.series
        except (ValueError, KeyError) as exc:
            stroke.cancelled.set()
            self._wand_strokes.pop(0)
            self._error(str(exc))
            self._start_wand()
            return
        sensitivity = self.CONFIG.wand_sensitivity / 100

        def work() -> Generator[tuple[BaseGeometry, bool], None, BaseGeometry | None]:
            geometry, previous, updated, operation = stroke.base, None, 0.0, stroke.initial_subtract
            dirty = False
            with PatchReader(path, series=series) as reader:
                source = WandSource(reader, stroke.layers, background=stroke.background)
                while not stroke.cancelled.is_set():
                    try:
                        timeout = max(0.001, 1 / 30 - (monotonic() - updated)) if dirty else 0.03
                        item = stroke.queue.get(timeout=timeout)
                    except Empty:
                        if dirty:
                            updated, dirty = monotonic(), False
                            yield geometry, operation
                        continue
                    if item is None:
                        return geometry
                    point, operation = item
                    distance = math.dist(previous, point) if previous is not None else 0
                    spacing = max(1, min(4, stroke.radius / 4)) / stroke.zoom
                    count = max(1, math.ceil(distance / spacing))
                    seeds = np.linspace(previous or point, point, count + 1)[1:]
                    pieces = []
                    for seed in seeds:
                        if stroke.cancelled.is_set():
                            return None
                        pieces.append(source.region(tuple(seed), stroke.radius, stroke.zoom, sensitivity))
                    footprint = unary_union(pieces)
                    geometry = geometry.difference(footprint) if operation else geometry.union(footprint)
                    previous = point
                    dirty = True
                    if monotonic() - updated >= 1 / 30:
                        updated, dirty = monotonic(), False
                        yield geometry, operation
            return None

        def preview(result: tuple[BaseGeometry, bool]) -> None:
            if self._paint_valid(stroke):
                stroke.geometry = result[0]
                self._preview_geometry(result[0], stroke.subtract)

        worker = create_worker(work, _ignore_errors=True)
        self._wand_worker = worker
        self._threads.append(worker)
        worker.yielded.connect(preview)
        worker.returned.connect(
            lambda geometry: (
                self._try(partial(self._commit_paint, stroke, geometry))
                if geometry is not None and self._paint_valid(stroke)
                else None
            )
        )
        worker.errored.connect(partial(self._wand_failed, stroke))
        worker.finished.connect(partial(self._finish_wand, worker, stroke))
        worker.start()

    def _wand_failed(self, stroke: _PaintStroke, exc: Exception) -> None:
        if self._paint_valid(stroke):
            self._error(str(exc))
        stroke.cancelled.set()
        if self._paint is stroke:
            self._paint = None
        self.brush_preview.data = []

    def _finish_wand(self, worker: Any, stroke: _PaintStroke) -> None:
        if worker in self._threads:
            self._threads.remove(worker)
        self._wand_worker = None
        if stroke in self._wand_strokes:
            self._wand_strokes.remove(stroke)
        if not self._closing:
            self._start_wand()

    def _preview_stroke(
        self, points: list[tuple[float, float]], radius: float, subtract: bool, *, force: bool = False
    ) -> None:
        now = monotonic()
        if not force and now - self._preview_updated < 1 / 30:
            return
        stroke = brush_stroke(points, radius, fill=not subtract)
        self._preview_geometry(stroke, subtract)

    def _brush(self, stroke: Any, subtract: bool) -> None:
        store = self._require_project()
        if len(self.selected) != 1:
            raise ValueError("Brush editing requires exactly one selected object.")
        old = store.get(self.selected[0]).geometry
        brush_object(store, self.selected[0], stroke, subtract=subtract)
        if subtract and old.difference(stroke).is_empty:
            self._review_selection = [key for key in self._review_selection if key not in self.selected]
            self.selected.clear()
        self._changed(old.union(stroke).bounds)

    def add_geometry(self, geometry: Any) -> int:
        """Create a manually drawn object in the active namespace."""
        store = self._require_project()
        self._require_slide()
        if not self.set_id:
            raise ValueError("Create an annotation set first.")
        obj = AnnotationObject(
            self.set_id,
            store.next_id(self.set_id),
            geometry,
            self.class_id,
            provenance={"source": "manual"},
            training=True,
        )
        self.selected = store.apply(EditCommand("Draw object", [(None, obj)]))
        self._changed(geometry.bounds)
        return self.selected[0]

    def apply_drawing(self) -> None:
        """Commit a drawing, polygon-ring edit, or a split after reviewing its survivor."""
        store = self._require_project()
        slide = self._require_slide()
        if not len(self.draw_layer.data):
            return
        if self._mode == "Vertices":
            if len(self.selected) != 1 or store.revision != self._edit_revision:
                raise ValueError("Select exactly one current object before editing vertices.")
            polygons, index = [], 0
            for ring_count in self._ring_layout:
                rings = [np.asarray(self.draw_layer.data[index + j])[:, ::-1] for j in range(ring_count)]
                polygons.append(Polygon(rings[0], rings[1:]))
                index += ring_count
            if index != len(self.draw_layer.data):
                raise ValueError("Ring count changed. Use brush tools to add/remove holes or components.")
            geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
            old = store.get(self.selected[0]).geometry
            edit_geometry(store, self.selected[0], geometry)
            self._changed(old.union(geometry).bounds)
        elif self._mode == "Cut":
            if len(self.selected) != 1:
                raise ValueError("Select one object before drawing a cut.")
            cut = LineString(np.asarray(self.draw_layer.data[-1])[:, ::-1])
            obj = store.get(self.selected[0])
            pieces = split_preview(obj, cut)
            options = [f"Piece {index + 1}: area {piece.area:.1f} px2" for index, piece in enumerate(pieces)]
            # Show piece numbers on a temporary points layer during survivor selection.
            centers = np.asarray([[piece.representative_point().y, piece.representative_point().x] for piece in pieces])
            preview = self.view.viewer.add_points(
                centers, name="Split preview", size=8, features={"piece": np.arange(1, len(pieces) + 1)}, text="piece"
            )
            try:
                choice, ok = QInputDialog.getItem(
                    self, "Split preview", f"Keep ID {obj.object_id} on:", options, 0, False
                )
                if ok:
                    self.selected = split_object(store, obj.key, cut, options.index(choice))
                    self._changed(obj.geometry.bounds)
            finally:
                self.view.viewer.layers.remove(preview)
        elif self._mode in ("Select by polygon", "Select by rectangle"):
            polygon = Polygon(np.asarray(self.draw_layer.data[-1])[:, ::-1])
            reviewing = self.tabs.currentIndex() == 3
            review = "proposed" if reviewing else "accepted"
            keys = list(store.selection_in(slide.id, polygon, set_id=self.set_id, review=review))
            self._select_from_canvas(keys)
            self.status.setText(f"{len(self.selected)} objects selected by area.")
        elif self._mode in ("Rectangle", "Polygon"):
            self.add_geometry(Polygon(np.asarray(self.draw_layer.data[-1])[:, ::-1]))

    def _changed(self, bounds: tuple | None = None) -> None:
        store = self._require_project()
        if self.renderer:
            self.renderer.invalidate(self.slide_id, bounds, revision=store.revision)
        self.refresh_tables()
        self._show_selected()
        self.schedule_render()
        self.status.setText("Saved. Ctrl+Z undoes the last edit.")

    def assign_class(self) -> None:
        """Assign the selected class to all selected objects in one command."""
        store = self._require_project()
        store.apply(
            EditCommand(
                "Assign class", [(key, replace(store.get(key), class_id=self.class_id)) for key in self.selected]
            )
        )
        self._changed()

    def edit_properties(self) -> None:
        """Inspect provenance and edit the selected object's original JSON properties."""
        store = self._require_project()
        if len(self.selected) != 1:
            raise ValueError("Select one object to inspect its properties.")
        obj = store.get(self.selected[0])
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Object {obj.object_id}")
        dialog.resize(580, 600)
        layout = hp.make_form_layout(parent=dialog)
        properties = hp.make_text_edit(dialog)
        properties.setPlainText(json.dumps(obj.properties, indent=2, ensure_ascii=False))
        provenance = hp.make_text_edit(dialog, read_only=True, max_height=180)
        provenance.setPlainText(json.dumps(obj.provenance, indent=2, ensure_ascii=False))
        layout.addRow("Properties (JSON)", properties)
        layout.addRow("Provenance and lineage", provenance)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel, dialog
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec():
            values = json.loads(properties.toPlainText())
            if not isinstance(values, dict):
                raise ValueError("Object properties must be a JSON object.")
            store.apply(EditCommand("Edit properties", [(obj.key, replace(obj, properties=values))]))
            self._changed()

    def toggle_training(self) -> None:
        """Toggle inclusion for selected accepted training examples."""
        store = self._require_project()
        objects = [store.get(key) for key in self.selected]
        self.set_training(not all(obj.training for obj in objects))

    def set_training(self, enabled: bool, *, keys: list[int] | None = None) -> None:
        """Include or exclude selected accepted examples in one undoable command."""
        store = self._require_project()
        objects = [store.get(key) for key in (self.selected if keys is None else keys)]
        classes = {item.id: item for item in store.classes()}
        changes = []
        for obj in objects:
            annotation_class = classes.get(obj.class_id or "")
            if obj.review != "accepted" or annotation_class is None or annotation_class.is_ignore:
                continue
            if obj.training != enabled:
                changes.append((obj.key, replace(obj, training=enabled)))
        if not changes:
            return
        store.apply(EditCommand("Include training examples" if enabled else "Exclude training examples", changes))
        self._changed()

    def toggle_object_lock(self) -> None:
        """Toggle explicit object locks while respecting class locks."""
        store = self._require_project()
        changes = []
        for key in self.selected:
            obj = store.get(key)
            properties = dict(obj.properties)
            properties["isLocked"] = not bool(properties.get("isLocked"))
            properties["locked"] = properties["isLocked"]
            changes.append((key, replace(obj, properties=properties)))
        store.apply(EditCommand("Toggle object lock", changes), allow_unlock=True)
        self._changed()

    def delete_selected(self) -> None:
        """Delete selected objects with full undo support."""
        self._require_project().apply(EditCommand("Delete objects", [(key, None) for key in self.selected]))
        self.selected.clear()
        self._changed()

    def merge_selected(self) -> None:
        """Choose a survivor and output class before merging selected objects."""
        store = self._require_project()
        if len(self.selected) < 2:
            raise ValueError("Select at least two objects to merge.")
        options = [repr(store.get(key).object_id) for key in self.selected]
        choice, ok = QInputDialog.getItem(self, "Merge objects", "Keep this object ID:", options, 0, False)
        if ok:
            key = merge_objects(store, self.selected, self.selected[options.index(choice)], self.class_id)
            self.selected = [key]
            self._changed()

    def merge_proposals(self) -> None:
        """Union selected Review proposals without accepting them or changing their class silently."""
        store = self._require_project()
        keys = self._review_selection
        if len(keys) < 2:
            raise ValueError("Select at least two proposals to merge.")
        objects = [store.get(key) for key in keys]
        classes = {obj.class_id for obj in objects}
        class_id = objects[0].class_id
        if len(classes) > 1:
            labels = list(self._class_labels)
            label, ok = QInputDialog.getItem(self, "Merge proposals", "Class for merged object:", labels, 0, False)
            if not ok:
                return
            class_id = self._class_labels[label]
            if class_id is None:
                raise ValueError("Choose a shared class when merging proposals with different classes.")
        options = [repr(obj.object_id) for obj in objects]
        choice, ok = QInputDialog.getItem(self, "Merge proposals", "Keep this object ID:", options, 0, False)
        if ok:
            survivor = merge_objects(store, keys, keys[options.index(choice)], class_id)
            self.selected = [survivor]
            self._review_selection = [survivor]
            self._pages["proposed"] = store.page_cursors(self.set_id, survivor, review="proposed")
            self._changed()
            row = self.proposals.get_row_id(4, survivor)
            if row >= 0:
                source = self.proposals.model().index(row, 0)
                visible = self.proposals.proxy.mapFromSource(source)
                if visible.isValid():
                    self.proposals.selectRow(visible.row())

    def undo(self) -> None:
        """Undo a persisted command and refresh the editor."""
        store = self._require_project()
        if store.history_size(undo=True) > 1000:
            self.jobs.submit({"operation": "undo", "project": str(store.path)})
            return
        store.undo()
        self.selected.clear()
        self.refresh_classes()
        self.refresh_sets()
        self._changed()

    def redo(self) -> None:
        """Redo a persisted command and refresh the editor."""
        store = self._require_project()
        if store.history_size(undo=False) > 1000:
            self.jobs.submit({"operation": "redo", "project": str(store.path)})
            return
        store.redo()
        self.selected.clear()
        self.refresh_classes()
        self.refresh_sets()
        self._changed()

    def add_class(self) -> None:
        """Create a shared class with a chosen name and color."""
        self._require_project()
        self._edit_class_dialog(AnnotationClass(name="New class"))

    def edit_class(self) -> None:
        """Edit the selected shared class without changing its ID."""
        store = self._require_project()
        item = next((item for item in store.classes() if item.id == self.class_id), None)
        if item:
            self._edit_class_dialog(item)

    def _edit_class_dialog(self, item: AnnotationClass) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Annotation class")
        form = hp.make_form_layout(parent=dialog)
        name = hp.make_line_edit(dialog, text=item.name)
        color = hp.make_swatch(dialog, item.color, tooltip="Choose the annotation class color.")

        ignore = hp.make_checkbox(dialog, "Use as Ignore mask", default=item.is_ignore)
        ignore.setToolTip("Exclude overlapping training pixels in the same annotation set. This class is not learned.")
        form.addRow("Name", name)
        form.addRow("Color", color)
        form.addRow(ignore)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec():
            updated = item.model_copy(
                update={"name": name.text().strip(), "color": color.hex_color_no_alpha, "is_ignore": ignore.isChecked()}
            )
            self._require_project().save_class(AnnotationClass.model_validate(updated.model_dump()))
            self.refresh_classes()
            self._changed()

    def _toggle_class(self, field: str) -> None:
        store = self._require_project()
        item = next((item for item in store.classes() if item.id == self.class_id), None)
        if item:
            setattr(item, field, not getattr(item, field))
            store.save_class(item)
            self.refresh_classes()
            self._changed()

    def toggle_class_visible(self) -> None:
        """Toggle the selected class overlay visibility."""
        self._toggle_class("visible")

    def toggle_class_locked(self) -> None:
        """Toggle edit protection for every object in the selected class."""
        self._toggle_class("locked")

    def new_set(self) -> None:
        """Create an independent namespace for additional annotation objects."""
        store = self._require_project()
        slide = self._require_slide()
        name, ok = QInputDialog.getText(self, "New annotation set", "Name:")
        if ok and name.strip():
            key = store.create_set(slide.id, name.strip())
            self.refresh_sets()
            self.sets.setCurrentIndex(self.sets.findData(key))

    def find_object(self) -> None:
        """Search both string and integer forms of an exact public ID."""
        store = self._require_project()
        if self.set_id is None:
            raise ValueError("Choose an annotation set first.")
        value = self.search.text()
        if value.startswith("int:"):
            obj = store.find_id(self.set_id, int(value[4:]))
        elif value.startswith("str:"):
            obj = store.find_id(self.set_id, value[4:])
        else:
            obj = store.find_id(self.set_id, value)
            if obj is None:
                with contextlib.suppress(ValueError):
                    obj = store.find_id(self.set_id, int(value))
        if obj is None:
            raise ValueError("Object ID not found in the selected set.")
        self.selected = [obj.key]
        self._show_selected()
        self.zoom_selected()

    def zoom_selected(self) -> None:
        """Center the image canvas on the current object selection."""
        if self.selected and self.store:
            geometry = self.store.get(self.selected[0]).geometry
            x0, y0, x1, y1 = geometry.bounds
            self.view.viewer.camera.center = (0, (y0 + y1) / 2, (x0 + x1) / 2)
            self.view.viewer.camera.zoom = (
                min(self.view.widget.width(), self.view.widget.height()) / max(1, x1 - x0, y1 - y0) * 0.7
            )

    def _overview_click(self, _viewer: Any, event: Any) -> None:
        if event.button == 1:
            self.view.viewer.camera.center = (0, *event.position[-2:])

    def viewport(self) -> tuple[float, float, float, float]:
        """Return the current image viewport in full-resolution pixel XY."""
        cy, cx = self.view.viewer.camera.center[-2:]
        zoom = max(0.0001, self.view.viewer.camera.zoom)
        width, height = self.view.widget.width() / zoom, self.view.widget.height() / zoom
        return cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2

    def schedule_render(self, *_args: Any) -> None:
        """Debounce viewport changes and retain at most one pending render."""
        if hasattr(self, "render_timer") and not self._closing:
            self.render_timer.start()

    def _render(self) -> None:
        if not self.store or not self.slide_id or not self.renderer:
            return
        if self._rendering:
            self.render_timer.start()
            return
        self._rendering = True
        store = self.store
        generation, slide_id, set_id = self._generation, self.slide_id, self.set_id
        bounds, renderer = self.viewport(), self.renderer
        proposals = self.show_predictions.isChecked()

        def work() -> tuple:
            accepted = renderer.render(slide_id, bounds, set_id=set_id)
            proposed = renderer.render(slide_id, bounds, set_id=set_id, review="proposed") if proposals else None
            return accepted, proposed

        def done(result: tuple) -> None:
            self._rendering = False
            if generation != self._generation or set_id != self.set_id or result[0].revision != store.revision:
                self.schedule_render()
                return
            for attr, frame, name in (
                ("_overlay", result[0], "Accepted annotations"),
                ("_proposal_overlay", result[1], "Proposed annotations"),
            ):
                layer = getattr(self, attr)
                if frame is None:
                    if layer is not None:
                        layer.visible = False
                    continue
                translate = (frame.offset[1] + frame.downsample / 2, frame.offset[0] + frame.downsample / 2)
                if layer is None:
                    layer = self.view.viewer.add_image(
                        frame.rgba, rgb=True, name=name, scale=(frame.downsample,) * 2, translate=translate
                    )
                    setattr(self, attr, layer)
                else:
                    layer.data = frame.rgba
                    layer.scale = (frame.downsample,) * 2
                    layer.translate = translate
                    layer.visible = True
                self.view.viewer.layers.move(self.view.viewer.layers.index(layer), -1)
            self._update_opacity()
            if self._proposal_overlay is not None:
                self._proposal_overlay.visible = self.show_predictions.isChecked() and result[1] is not None
            for layer in (self.draw_layer, self.prompt_layer, self.prompt_box_layer, self.brush_preview):
                self.view.viewer.layers.move(self.view.viewer.layers.index(layer), -1)
            self.set_mode(self._mode)

        worker = create_worker(work)
        self._threads.append(worker)
        worker.returned.connect(lambda value: done(value) if not self._closing else None)
        worker.errored.connect(lambda exc: self.status.setText(f"Overlay: {exc}") if not self._closing else None)
        worker.finished.connect(partial(self._finish_render, worker))
        worker.start()

    def _finish_render(self, worker: Any) -> None:
        self._rendering = False
        if worker in self._threads:
            self._threads.remove(worker)

    def relocate_slide(self) -> None:
        """Relink a missing image, retaining annotations only for a matching pixel grid."""
        store = self._require_project()
        slide = self._require_slide()
        path, _ = QFileDialog.getOpenFileName(self, "Relocate source image")
        if path:
            inspected = inspect_slide(path, series=slide.series)
            if inspected.geometry.shape != slide.geometry.shape:
                raise ValueError("Replacement image dimensions differ from the annotation grid.")
            slide.path, slide.fingerprint = inspected.path, inspected.fingerprint
            store.save_slide(slide)
            self.refresh_image()

    def calibrate_slide(self) -> None:
        """Set independent X/Y pixel sizes and origin explicitly."""
        store = self._require_project()
        slide = self._require_slide()
        spacing = slide.geometry.spacing or (1, 1)
        values, ok = QInputDialog.getText(
            self,
            "Slide calibration",
            "Pixel size X, Y (um), then origin X, Y (um):",
            text=", ".join(str(v) for v in (*spacing, *slide.geometry.origin)),
        )
        if ok:
            values = [float(value.strip()) for value in values.split(",")]
            if len(values) != 4:
                raise ValueError("Enter four values: size X, size Y, origin X, origin Y.")
            slide.geometry.spacing, slide.geometry.origin = tuple(values[:2]), tuple(values[2:])
            store.save_slide(slide)
            self._changed()

    def choose_import(self) -> None:
        """Choose vector or raster annotations and an explicit coordinate frame."""
        store = self._require_project()
        slide = self._require_slide()
        path, _ = QFileDialog.getOpenFileName(
            self, "Import annotations", "", "Annotations (*.geojson *.json *.tif *.tiff)"
        )
        if not path:
            path = QFileDialog.getExistingDirectory(self, "Or choose an OME-Zarr label image")
        if not path:
            return
        raster = Path(path).suffix.lower() not in (".json", ".geojson")
        dialog = CoordinateDialog(self, raster=raster, exporting=False, geometry=slide.geometry)
        if dialog.exec():
            coordinates = dialog.coordinates()
            if coordinates.units == "um" and slide.geometry.spacing is None:
                raise ValueError("Set slide calibration before importing physical coordinates.")
            self.jobs.submit(
                {
                    "operation": "import",
                    "project": str(store.path),
                    "slide_id": slide.id,
                    "path": path,
                    "format": "labels" if raster else "geojson",
                    "coordinates": coordinates.model_dump(),
                }
            )

    def choose_export(self) -> None:
        """Export the current set with explicit coordinates and reversible raster IDs."""
        store = self._require_project()
        slide = self._require_slide()
        if not self.set_id:
            raise ValueError("Choose an annotation set first.")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export annotations", "", "GeoJSON (*.geojson);;Labels (*.tif *.ome.tiff *.ome.zarr)"
        )
        if not path:
            return
        raster = Path(path).suffix.lower() not in (".json", ".geojson")
        dialog = CoordinateDialog(self, raster=raster, exporting=True, geometry=slide.geometry)
        if dialog.exec():
            self.jobs.submit(
                {
                    "operation": "export",
                    "project": str(store.path),
                    "set_id": self.set_id,
                    "path": path,
                    "format": "labels" if raster else "geojson",
                    "mapped_ids": dialog.mapped.isChecked(),
                    "coordinates": dialog.coordinates().model_dump(),
                }
            )

    def train(self) -> None:
        """Train using included accepted project examples in a spawned process."""
        store = self._require_project()
        settings = FeatureSettings(
            channels=self._channels(),
            downsample=self.train_downsample.value(),
            sigma_max=self.feature_sigma.value(),
            samples_per_class=self.sample_count.value(),
            seed=self.random_seed.value(),
            max_depth=self.max_depth.value(),
            class_ids=tuple(
                item.id
                for item in store.classes()
                if not item.is_ignore and item.id not in self._excluded_training_classes
            ),
        )
        self.jobs.submit(
            {
                "operation": "train",
                "project": str(store.path),
                "name": self.classifier_name.text(),
                "settings": settings.model_dump(),
            }
        )

    def choose_finetuning(self) -> None:
        """Inspect pretrained fine-tuning settings and queue an explicit instance-training run."""
        store = self._require_project()
        class_ids = tuple(
            item.id for item in store.classes() if not item.is_ignore and item.id not in self._excluded_training_classes
        )
        if not class_ids:
            raise ValueError("Enable at least one instance class in Train.")
        dialog = FineTuneDialog(self, class_ids)
        if dialog.exec():
            self.jobs.submit(
                {
                    "operation": "finetune",
                    "project": str(store.path),
                    "name": dialog.name.text(),
                    "settings": dialog.settings().model_dump(),
                }
            )
        dialog.deleteLater()

    def _ensure_model_catalogue(self, backend: str) -> None:
        project = str(self.store.path) if self.store else None
        if (
            backend != "forest"
            and backend_available(backend)
            and backend not in self._catalogue_pending
            and (backend not in self._catalogues or self._catalogue_project.get(backend) != project)
        ):
            self._catalogue_pending.add(backend)
            self.jobs.submit(
                {"operation": "model_catalogue", "backend": backend, **({"project": project} if project else {})}
            )

    def choose_model(self) -> None:
        """Validate and copy an external classifier into this project's model library."""
        store = self._require_project()
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose classifier", str(self.store.path / "models") if self.store else "", "Classifier (*.npz)"
        )
        if path:
            _forest, metadata = load_forest(path)
            error = self._classifier_error(metadata)
            if error:
                raise ValueError(error)
            source = Path(path).resolve()
            destination = store.path / "models" / source.name
            if source.parent != destination.parent:
                if destination.exists():
                    destination = destination.with_stem(f"{destination.stem}-{uuid4().hex[:8]}")
                temporary = destination.with_name(f".{uuid4().hex}.tmp")
                try:
                    shutil.copyfile(source, temporary)
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                destination = source
            self.backend.setCurrentText("forest")
            self._refresh_model_selector(str(destination))

    def _model_value(self) -> str:
        return str(self.model.currentData() or "")

    def _classifier_error(self, metadata: dict[str, Any]) -> str:
        if metadata["sklearn"].split(".")[:2] != version("scikit-learn").split(".")[:2]:
            return "Classifier scikit-learn version differs. Retrain it in this environment."
        if self.store and not set(metadata["classes"]).issubset({item.id for item in self.store.classes()}):
            return "Classifier classes are missing from this project. Load a classifier trained with this project's classes."
        return ""

    def _refresh_model_selector(self, preferred: str = "") -> None:
        backend = self.backend.currentText()
        preferred = preferred or self._model_value()
        if backend == "forest" and self.store:
            project = self.store.path
            self._forest_scan_preferred = preferred
            if self._forest_scan_pending == project:
                self._forest_scan_again = True
                return
            self._forest_scan_pending = project
            self._forest_scan_again = False

            def scan() -> list[tuple[str, dict[str, Any] | None]]:
                found = []
                paths = sorted((project / "models").glob("*.npz"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
                for path in paths:
                    try:
                        metadata = read_classifier_metadata(path)
                    except (ValueError, KeyError, OSError, TypeError, EOFError, BadZipFile):
                        metadata = None
                    found.append((str(path), metadata))
                return found

            def done(found: list[tuple[str, dict[str, Any] | None]]) -> None:
                if self._forest_scan_pending != project:
                    return
                self._forest_scan_pending = None
                if not self.store or self.store.path != project:
                    return
                if self._forest_scan_again and self.backend.currentText() == "forest":
                    self._refresh_model_selector(self._forest_scan_preferred)
                    return
                self._forest_catalogue = {path: metadata for path, metadata in found if metadata is not None}
                if self.backend.currentText() == "forest":
                    entries = [
                        (
                            f"{metadata['name']} - {Path(path).name}" if metadata else Path(path).name,
                            path,
                            self._classifier_error(metadata) if metadata else "Invalid classifier artifact.",
                        )
                        for path, metadata in found
                    ]
                    self._set_model_entries(entries, self._forest_scan_preferred)

            def failed(_exc: Exception) -> None:
                if self._forest_scan_pending == project:
                    self._forest_scan_pending = None

            worker = self._background(scan, done)
            self._forest_scans.add(worker)
            worker.errored.connect(failed)
            worker.finished.connect(lambda: self._forest_scans.discard(worker))
            return
        self.model_selector.set_catalogue(backend, self._catalogues.get(backend, []), preferred)
        self._model_selected()

    def _set_model_entries(self, entries: list[tuple[str, str, str]], preferred: str) -> None:
        self.model_selector.set_entries(self.backend.currentText(), entries, preferred)
        self._model_selected()

    def _model_selected(self, *_args: Any) -> None:
        if self._changing_backend:
            return
        if self.backend.currentText() == "forest":
            self._try(self._read_classifier)
        else:
            entry = next(
                (
                    item
                    for item in self._catalogues.get(self.backend.currentText(), [])
                    if item["name"] == self._model_value()
                ),
                {},
            )
            custom = entry.get("custom")
            if custom:
                self.checkpoint.setText(entry["path"])
                self.predict_downsample.setValue(custom["settings"]["downsample"])
                self._selected_custom_checkpoint = entry["path"]
            elif self.checkpoint.text() == self._selected_custom_checkpoint:
                self.checkpoint.clear()
                self._selected_custom_checkpoint = ""
            self._update_prediction_controls()

    def _read_classifier(self) -> None:
        if self.backend.currentText() != "forest":
            return
        path = self._model_value()
        if not path:
            self._forest_metadata = None
            self._forest_path = ""
        else:
            metadata = self._forest_catalogue.get(path) or read_classifier_metadata(path)
            error = self._classifier_error(metadata)
            if error:
                raise ValueError(error)
            self._forest_metadata, self._forest_path = metadata, path
            self.predict_downsample.setValue(metadata["settings"]["downsample"])
        self._update_prediction_controls()

    def choose_checkpoint(self) -> None:
        """Choose an existing pretrained model checkpoint."""
        path, _ = QFileDialog.getOpenFileName(self, "Choose local checkpoint")
        if path:
            self.checkpoint.setText(path)

    def _backend_changed(self, *_args: Any) -> None:
        backend = self.backend.currentText()
        if backend == self._active_backend:
            self._update_prediction_controls()
            return
        if self._active_backend:
            self._backend_state[self._active_backend] = {
                "model": self._model_value(),
                "checkpoint": self.checkpoint.text(),
                "device": self.device.currentText(),
                "downsample": self.predict_downsample.value(),
                "tile": self.tile_size.currentText(),
            }
        self._active_backend = backend
        state = self._backend_state.get(backend, {})
        self._changing_backend = True
        with QSignalBlocker(self.model):
            self.model.clear()
        self.checkpoint.setText(state.get("checkpoint", ""))
        self.device.setCurrentText(state.get("device", "cpu" if backend == "forest" else "auto"))
        self.predict_downsample.setValue(state.get("downsample", 1))
        self.tile_size.setCurrentText(state.get("tile", "512"))
        self._refresh_model_selector(
            state.get("model") or {"forest": "", "micro-sam": "vit_b_lm", "cellpose": "cpsam"}[backend]
        )
        self._changing_backend = False
        if backend == "forest":
            self._read_classifier()
        else:
            self._ensure_model_catalogue(backend)
        self._update_prediction_controls()

    def download_selected_model(self) -> None:
        """Download selected weights, or cancel the active model download."""
        if self.jobs.current and self.jobs.current["operation"] == "download_model":
            self.jobs.cancel()
            return
        backend, model = self.backend.currentText(), self._model_value()
        if backend == "forest" or not model:
            raise ValueError("Select an available pretrained model first.")
        self.jobs.submit({"operation": "download_model", "backend": backend, "model": model})

    def _update_prediction_controls(self, *_args: Any) -> None:
        if self._changing_backend:
            return
        backend = self.backend.currentText()
        forest, sam = backend == "forest", backend == "micro-sam"
        available = backend_available(backend)
        metadata = self._forest_metadata if self._model_value() == self._forest_path else None
        entry = next((item for item in self._catalogues.get(backend, []) if item["name"] == self._model_value()), None)
        checkpoint = Path(self.checkpoint.text()).is_file() if self.checkpoint.text() else False
        ready = (
            metadata is not None
            if forest
            else bool(not (entry and entry.get("reason")) and (checkpoint or (entry and entry["cached"])))
        )
        explanations = {
            "forest": "Uses the random forest created in Train or a saved classifier. Runs on CPU.",
            "micro-sam": "Predict actions detect instances automatically. Point/box assistance uses your prompts. Use Train's fine-tuning action to adapt SAM.",
            "cellpose": "Uses pretrained or fine-tuned Cellpose weights for cell instances. Fine-tuning is optional and available in Train.",
        }
        message = explanations[backend]
        if not available:
            extra = "annotate-sam" if sam else "annotate-cellpose"
            message = f"Backend unavailable. Install image2image[{extra}] in the app environment.\n{message}"
        elif forest and metadata is None:
            message = "Train or choose a classifier before predicting.\n" + message
        elif not forest and backend in self._catalogue_errors:
            message = f"Could not load model catalogue: {self._catalogue_errors[backend]}\n{message}"
        elif not forest and backend in self._catalogue_pending:
            message = "Loading the installed model catalogue...\n" + message
        elif not forest and not ready:
            message = "Download the selected model or choose a local checkpoint before predicting.\n" + message
        else:
            message = "Backend available. " + message
        self.backend_hint.setText(message)
        hp.set_object_name(self.backend_hint, object_name="warning_label" if not available else "")
        self.choose_model_btn.setVisible(forest)
        self.model_selector.update_state(
            backend, entry, self._catalogues.get(backend, []), available=available, checkpoint=self.checkpoint.text()
        )
        for widget, visible in (
            (self.checkpoint, not forest),
            (self.choose_checkpoint_btn, not forest),
            (self.device, not forest),
            (self.download_btn, not forest),
        ):
            widget.setVisible(visible)
            label = self._predict_form.labelForField(widget)
            if label:
                label.setVisible(visible)
        self.predict_downsample.setEnabled(not forest)
        self.predict_downsample.setToolTip(
            "Saved with the classifier." if forest else "Working image downsample for pretrained inference."
        )
        for button in self.predict_buttons:
            button.setEnabled(available and ready and not self.jobs.busy)
            button.setToolTip(message)
        self.prompt_btn.setEnabled(available and sam and ready and not self.jobs.busy)
        for button in self.prompt_tool_buttons:
            button.setEnabled(sam)
        if forest and metadata:
            settings = FeatureSettings.model_validate(metadata["settings"])
            self.predict_downsample.setValue(settings.downsample)
            names = metadata.get("class_names", {})
            self.model_summary.setText(
                f"Classifier: {metadata['name']}\nClasses: {', '.join(names.get(key, key) for key in metadata['classes'])}\n"
                f"Channels: {', '.join(map(str, settings.channels))}; downsample: {settings.downsample}\n"
                f"Features: scales {settings.sigma_min:g}-{settings.sigma_max:g}; "
                f"samples/class: {metadata.get('samples_per_class', settings.samples_per_class):,}\n"
                f"Trees: {settings.trees}; seed: {settings.seed}; maximum depth: {settings.max_depth}"
            )
        else:
            self.model_summary.setText(
                "Choose a saved .npz classifier to see its classes and settings."
                if forest
                else "Uses channels from Annotate and the resolution below. Assigns the selected shared class to detections."
            )
            if entry and entry.get("custom"):
                training = entry["custom"]["settings"]
                self.predict_downsample.setValue(training["downsample"])
                self.predict_downsample.setEnabled(False)
                self.model_summary.setText(
                    f"Fine-tuned model: {entry['custom']['name']}\nBase: {training['model']}; epochs: {training['epochs']}; learning rate: {training['learning_rate']:g}\nSaved channels: {training['channels']}; downsample: {training['downsample']}. Assigns the selected shared class to detections."
                )
        if forest:
            self.model_help.setToolTip(self.model_summary.text())

    def _request(
        self, bounds: tuple | None = None, *, prompts: bool = False, slide_id: str | None = None
    ) -> PredictionRequest:
        store = self._require_project()
        slide = self._require_slide()
        backend = self.backend.currentText()
        if not backend_available(backend):
            raise ValueError(self.backend_hint.text())
        downsample = self.predict_downsample.value()
        if backend == "forest":
            if not self._model_value():
                raise ValueError("Train or choose a classifier before predicting.")
            if self._model_value() != self._forest_path or self._forest_metadata is None:
                self._read_classifier()
            if self._forest_metadata is None:
                raise ValueError("Choose a valid classifier before predicting.")
            settings = FeatureSettings.model_validate(self._forest_metadata["settings"])
            channels, downsample = settings.channels, settings.downsample
        else:
            channels = self._channels()
            entry = next(
                (item for item in self._catalogues.get(backend, []) if item["name"] == self._model_value()), None
            )
            if self.checkpoint.text():
                if not Path(self.checkpoint.text()).is_file():
                    raise ValueError("Choose an existing local checkpoint.")
            elif entry is None or entry["reason"] or not entry["cached"]:
                raise ValueError("Download an available model or choose a local checkpoint before predicting.")
            if entry and entry.get("custom"):
                channels = tuple(entry["custom"]["settings"]["channels"])
                downsample = entry["custom"]["settings"]["downsample"]
        points, prompt_boxes = [], []
        if prompts:
            points = [
                (float(point[1]), float(point[0]), label)
                for point, label in zip(self.prompt_layer.data, self._prompt_labels, strict=True)
            ]
            prompt_boxes = [Polygon(np.asarray(ring)[:, ::-1]).bounds for ring in self.prompt_box_layer.data]
        return PredictionRequest(
            project=str(store.path),
            slide_id=slide_id or slide.id,
            revision=store.revision,
            backend=cast(Literal["forest", "micro-sam", "cellpose"], self.backend.currentText()),
            model=entry["custom"]["settings"]["model"]
            if backend != "forest" and entry and entry.get("custom")
            else self._model_value(),
            sam_mode="prompts" if prompts else "automatic",
            checkpoint=self.checkpoint.text(),
            device=cast(
                Literal["auto", "cpu", "cuda", "mps"], "cpu" if backend == "forest" else self.device.currentText()
            ),
            channels=channels,
            downsample=downsample,
            tile_size=int(self.tile_size.currentText()),
            bounds=bounds,
            points=points,
            boxes=prompt_boxes,
            class_id=self.class_id,
        )

    def _submit_prediction(self, request: PredictionRequest, *, queue_index: int = 1, queue_total: int = 1) -> None:
        store = self._require_project()
        self.jobs.submit(
            {
                "operation": "predict",
                "project": str(store.path),
                "request": request.model_dump(),
                "queue_index": queue_index,
                "queue_total": queue_total,
            }
        )

    def predict_view(self) -> None:
        """Preview predictions for the visible image rectangle."""
        self._submit_prediction(self._request(self.viewport()))

    def predict_roi(self) -> None:
        """Predict within the most recently drawn rectangle."""
        if not len(self.draw_layer.data):
            raise ValueError("Draw a rectangle first.")
        self._submit_prediction(self._request(Polygon(np.asarray(self.draw_layer.data[-1])[:, ::-1]).bounds))

    def predict_slide(self) -> None:
        """Queue a complete slide prediction."""
        self._submit_prediction(self._request())

    def predict_all(self) -> None:
        """Queue the project slides with shared immutable model settings."""
        slides = self._require_project().slides()
        for index, slide in enumerate(slides, 1):
            self._submit_prediction(self._request(slide_id=slide.id), queue_index=index, queue_total=len(slides))

    def predict_prompts(self) -> None:
        """Run point/box assistance in the current viewport."""
        if self.backend.currentText() != "micro-sam":
            raise ValueError("Select the micro-sam backend for point/box assistance.")
        self._submit_prediction(self._request(self.viewport(), prompts=True))

    def _job_busy(self, busy: bool) -> None:
        self.cancel_btn.setEnabled(busy)
        self.tool_hint.setVisible(not busy)
        self.progress_bar.setVisible(busy)
        self._update_prediction_controls()
        self._proposal_details()
        self._refresh_training_summary()

    def _job_message(self, message: str) -> None:
        # Measured progress already updates the status with elapsed time and ETA.
        if self.jobs.cancelling or not self.jobs.busy:
            self._set_status(message)

    def _set_status(self, message: str) -> None:
        self.status.setText(message)
        self.status.setToolTip(message)

    def _job_progress(self, details: dict[str, Any]) -> None:
        completed, total = details.get("completed", 0), details.get("total")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 1000 if total else 0)
        if total:
            self.progress_bar.setValue(min(1000, int(completed / total * 1000)))

        def duration(seconds: float) -> str:
            minutes, seconds_ = divmod(max(0, math.ceil(seconds)), 60)
            return f"{minutes} min {seconds_:02d} s" if minutes else f"{seconds_} s"

        elapsed = duration(details["elapsed"])
        eta = details["eta"]
        remaining = f"; ETA ~{duration(eta)}" if eta is not None else ""
        units = f" ({completed:,}/{total:,})" if total else ""
        if self.jobs.current and self.jobs.current.get("operation") == "download_model":
            units = f" ({completed / 2**20:.1f}" + (f"/{total / 2**20:.1f}" if total else "") + " MiB)"
        queue = f"Slide {details['position']}/{details['queue_total']}; " if details["queue_total"] > 1 else ""
        phase, message = details.get("phase", ""), details["message"]
        if phase and phase != message:
            message = f"{phase}: {message}"
        self._set_status(f"{queue}{message}{units}; Elapsed {elapsed}{remaining}")

    def _job_failed(self, job: dict, message: str) -> None:
        if self._closing:
            return
        self._set_status(message)
        if job["operation"] == "model_catalogue":
            backend = job["backend"]
            self._catalogue_pending.discard(backend)
            self._catalogue_errors[backend] = message
            self._update_prediction_controls()
        elif "cancelled" in message.lower():
            self.status.setText(message)
        else:
            self._error(message)

    def _job_done(self, job: dict, result: Any) -> None:
        if self._closing:
            return
        self._set_status(f"Completed {job['operation'].replace('_', ' ')}.")
        if job["operation"] in ("model_catalogue", "download_model"):
            backend = job["backend"]
            if job["operation"] == "model_catalogue":
                self._catalogue_pending.discard(backend)
                self._catalogue_errors.pop(backend, None)
                self._catalogues[backend] = result
                self._catalogue_project[backend] = job.get("project")
            else:
                entries = self._catalogues.setdefault(backend, [])
                entries[:] = [entry for entry in entries if entry["name"] != result["name"]] + [result]
            if self.backend.currentText() == backend:
                preferred = self._model_value() or self._backend_state.get(backend, {}).get("model")
                self._refresh_model_selector(preferred or ("vit_b_lm" if backend == "micro-sam" else "cpsam"))
            self._set_status(f"Completed {job['operation'].replace('_', ' ')}.")
            return
        if job["operation"] in ("undo", "redo", "review"):
            self.selected.clear()
        if job["operation"] == "train":
            self.backend.setCurrentText("forest")
            self._refresh_model_selector(str(result))
        elif job["operation"] == "finetune":
            from image2image.utils.annotation.worker import finetuned_entry

            backend = result["settings"]["backend"]
            self._catalogues.setdefault(backend, []).append(finetuned_entry(result))
            self.backend.setCurrentText(backend)
            self._refresh_model_selector(result["checkpoint"])
            self.tabs.setCurrentIndex(2)
        self.refresh_classes()
        self.refresh_sets()
        if job["operation"] == "predict":
            key = result["set_id"]
            index = self.sets.findData(key)
            if index >= 0:
                self.sets.setCurrentIndex(index)
                self.tabs.setCurrentIndex(3)
        elif job["operation"] in ("import", "combine_sets"):
            self.sets.setCurrentIndex(self.sets.findData(result))
        if self.renderer:
            self.renderer.invalidate()
        self.schedule_render()
        self._set_status(
            f"Completed {job['operation']}."
            + (" Review predictions before accepting them." if job["operation"] == "predict" else "")
        )

    def review(self, accept: bool, batch: bool) -> None:
        """Accept or reject a selected subset or the complete current prediction batch."""
        store = self._require_project()
        keys = None if batch else list(self._review_selection)
        if keys == []:
            raise ValueError("Select proposals to review.")
        self.jobs.submit(
            {"operation": "review", "project": str(store.path), "set_id": self.set_id, "accept": accept, "keys": keys}
        )

    def _review_action(self, function: Any) -> None:
        if self.tabs.currentIndex() == 3 and not self.jobs.busy and self._review_selection:
            self._try(function)

    def _zoom_proposal(self, row: int, _column: int) -> None:
        self.selected = [_table_key(self.proposals, row)]
        self.zoom_selected()

    def refine_proposal(self) -> None:
        """Load one proposed object for manual vertex or brush correction before acceptance."""
        keys = self._review_selection
        if len(keys) != 1:
            raise ValueError("Select one proposal to refine.")
        self.selected = keys
        self._show_selected()
        self.tool.setCurrentText("Vertices")
        self.zoom_selected()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Cancel jobs and release image handles after outstanding reads have finished."""
        if event is None:
            return
        self._closing = True
        self._cancel_paint()
        self._stroke_revision += 1
        self.brush_preview.data = []
        self.render_timer.stop()
        self.jobs.close()
        if self._threads:
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
        QApplication.instance().removeEventFilter(self)
        if self.reader:
            self.reader.close()
            self.reader = None
        if self.store:
            self.store.close()
            self.store = None
        if self._project_lock:
            self._project_lock.unlock()
            self._project_lock = None
        event.accept()
