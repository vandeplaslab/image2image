"""Annotation and Label Editor with a persistent object store and local ML assistance."""

from __future__ import annotations

import contextlib
import json
import sqlite3
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import qtextra.helpers as hp
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import QLockFile, Qt, QTimer
from qtpy.QtGui import QCloseEvent, QKeySequence, QShortcut  # type: ignore[attr-defined]
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenuBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
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
from image2image.qt._annotation_jobs import AnnotationJobs
from image2image.qt._dialog_base import Window
from image2image.utils.annotation.geometry import (
    brush_object,
    edit_geometry,
    merge_objects,
    split_object,
    split_preview,
)
from image2image.utils.annotation.images import PatchReader, inspect_slide
from image2image.utils.annotation.ml import backend_available, load_forest
from image2image.utils.annotation.render import AnnotationRenderer
from image2image.utils.annotation.store import AnnotationStore


def _table_key(table: QTableWidget, row: int) -> int:
    item = table.item(row, 0)
    if item is None:
        raise ValueError("Select an object row first.")
    return int(item.data(Qt.ItemDataRole.UserRole))


def _selected_keys(table: QTableWidget) -> list[int]:
    selection = table.selectionModel()
    return [_table_key(table, index.row()) for index in selection.selectedRows()] if selection else []


class AnnotateConfig(SingleAppConfig):
    """Keep editor preferences separate from registration configuration."""

    USER_CONFIG_FILENAME = "config.annotate.json"


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
        self.offset = QLineEdit("0, 0", self)
        self.downsample = QLineEdit("1, 1", self)
        layout.addRow("Patch offset X, Y (full-resolution px)", self.offset)
        layout.addRow("Downsample X, Y", self.downsample)
        self.mapped = QCheckBox("Map IDs to integers and write a reversible ID sidecar", self)
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
        self._threads: list[Any] = []
        self._closing = False
        self._syncing = False
        self._rendering = False
        self._generation = 0
        self._page_after = 0
        self._page_stack: list[int] = []
        self._mode = "Select"
        self._ring_layout: list[int] = []
        self._image_layers: list[Any] = []
        self._overlay = None
        self._proposal_overlay = None
        self._edit_revision = 0
        self._prompt_labels: list[int] = []
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
            self.jobs.progress.connect(self.status.setText)
            self.jobs.completed.connect(self._job_done)
            self.jobs.failed.connect(lambda _job, message: self._error(message))
            self.jobs.busy_changed.connect(self.cancel_btn.setEnabled)
            self.view.viewer.camera.events.zoom.connect(self.schedule_render)
            self.view.viewer.camera.events.center.connect(self.schedule_render)

    def _try(self, function: Any) -> Any:
        try:
            return function()
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ImportError, sqlite3.Error) as exc:
            self._error(str(exc))
            return None

    def _error(self, message: str) -> None:
        self.status.setText(message)
        hp.warn_pretty(self, message, "Annotation editor")

    def _button(self, parent: QWidget, text: str, function: Any) -> Any:
        return hp.make_btn(parent, text, func=lambda: self._try(function))

    def _setup_ui(self) -> None:
        main = QWidget(self)
        layout = QVBoxLayout(main)
        self.cancel_btn = hp.make_btn(main, "Cancel job", func=self.jobs.cancel)
        self.cancel_btn.setEnabled(False)
        layout.addLayout(
            hp.make_h_layout(
                self._button(main, "New project...", self.choose_new_project),
                self._button(main, "Open project...", self.choose_project),
                self._button(main, "Add slides...", self.choose_slides),
                self._button(main, "Import...", self.choose_import),
                self._button(main, "Export...", self.choose_export),
                self._button(main, "Undo", self.undo),
                self._button(main, "Redo", self.redo),
                self.cancel_btn,
            )
        )
        splitter = QSplitter(Qt.Orientation.Horizontal, main)
        left = QWidget(splitter)
        left_box = QVBoxLayout(left)
        left_box.addWidget(hp.make_label(left, "Slides"))
        self.slides = QListWidget(left)
        self.slides.currentRowChanged.connect(self._slide_selected)
        left_box.addWidget(self.slides, 1)
        left_box.addLayout(
            hp.make_h_layout(
                self._button(left, "Relocate...", self.relocate_slide),
                self._button(left, "Calibration...", self.calibrate_slide),
            )
        )
        self.sets = hp.make_combobox(left, func_index=self._set_changed)
        left_box.addWidget(hp.make_label(left, "Annotation set"))
        left_box.addWidget(self.sets)
        left_box.addWidget(self._button(left, "New annotation set", self.new_set))
        left_box.addWidget(hp.make_label(left, "Shared classes"))
        self.classes = QListWidget(left)
        left_box.addWidget(self.classes, 1)
        left_box.addLayout(
            hp.make_h_layout(
                self._button(left, "Add class", self.add_class), self._button(left, "Edit class", self.edit_class)
            )
        )
        left_box.addLayout(
            hp.make_h_layout(
                self._button(left, "Show/hide", self.toggle_class_visible),
                self._button(left, "Lock/unlock", self.toggle_class_locked),
            )
        )
        left_box.addWidget(hp.make_label(left, "Overview (click to navigate)"))
        self.overview = NapariImageView(left, add_toolbars=False, disable_new_layers=True, allow_crosshair=False)
        self.overview.widget.setMaximumHeight(190)
        left_box.addWidget(self.overview.widget)
        self.overview.viewer.mouse_drag_callbacks.append(self._overview_click)
        center = QWidget(splitter)
        center_box = QVBoxLayout(center)
        self.tool = hp.make_combobox(
            center,
            [
                "Select",
                "Polygon",
                "Freehand",
                "Rectangle",
                "Brush +",
                "Brush -",
                "Vertices",
                "Cut",
                "Select region",
                "Prompt +",
                "Prompt -",
                "Prompt box",
            ],
            func=self.set_mode,
        )
        self.brush_radius = hp.make_int_spin_box(center, minimum=1, maximum=500, value=15)
        self.show_predictions = QCheckBox("Show proposals", center)
        self.show_predictions.setChecked(True)
        self.show_predictions.toggled.connect(self.schedule_render)
        center_box.addLayout(
            hp.make_h_layout(
                self.tool,
                hp.make_label(center, "Brush px"),
                self.brush_radius,
                self._button(center, "Apply drawing", self.apply_drawing),
                self._button(center, "Clear drawing", self.clear_drawing),
                self.show_predictions,
            )
        )
        self.view = NapariImageView(center, add_toolbars=False, disable_new_layers=True, allow_crosshair=False)
        center_box.addWidget(self.view.widget, 1)
        self.draw_layer = self.view.viewer.add_shapes(
            name="Drawing / selected rings", ndim=2, edge_color="yellow", face_color="transparent", edge_width=2
        )
        self.prompt_layer = self.view.viewer.add_points(name="AI prompts", ndim=2, size=8, face_color="lime")
        self.view.viewer.mouse_drag_callbacks.append(self._mouse_drag)
        self.tabs = QTabWidget(splitter)
        self.tabs.addTab(self._annotation_panel(), "Annotate")
        self.tabs.addTab(self._train_panel(), "Train")
        self.tabs.addTab(self._predict_panel(), "Predict")
        self.tabs.addTab(self._review_panel(), "Review")
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(self.tabs)
        splitter.setSizes([250, 1000, 370])
        layout.addWidget(splitter, 1)
        self.status = hp.make_label(
            main, "Create or open an .i2annotate project. Edits are saved automatically.", wrap=True
        )
        layout.addWidget(self.status)
        self.setCentralWidget(main)
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
            ("Delete", self.delete_selected),
            ("Escape", self.clear_drawing),
            ("Return", self.apply_drawing),
            ("M", self.merge_selected),
            ("V", lambda: self.tool.setCurrentText("Select")),
            ("P", lambda: self.tool.setCurrentText("Polygon")),
            ("B", lambda: self.tool.setCurrentText("Brush +")),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(partial(self._try, function))
            self._shortcuts.append(shortcut)

    def _annotation_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        self.channels = QLineEdit("0", panel)
        self.display_mode = hp.make_combobox(panel, ["Channels", "RGB"])
        layout.addWidget(hp.make_label(panel, "Channel indices (comma separated)"))
        layout.addWidget(self.channels)
        layout.addWidget(self.display_mode)
        layout.addWidget(self._button(panel, "Update image channels", self.refresh_image))
        self.search = QLineEdit(panel)
        self.search.setPlaceholderText("Object ID (or int:12 / str:12)")
        self.search.returnPressed.connect(lambda: self._try(self.find_object))
        layout.addLayout(hp.make_h_layout(self.search, self._button(panel, "Find", self.find_object)))
        self.objects = QTableWidget(0, 3, panel)
        self.objects.setHorizontalHeaderLabels(["ID", "Class", "Area (px2)"])
        self.objects.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.objects.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.objects.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.objects.itemSelectionChanged.connect(self._table_selection)
        self.objects.cellDoubleClicked.connect(lambda *_: self.zoom_selected())
        layout.addWidget(self.objects, 1)
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Previous", self.previous_page), self._button(panel, "Next", self.next_page)
            )
        )
        self.selection_label = hp.make_label(panel, "No selection", wrap=True)
        layout.addWidget(self.selection_label)
        layout.addWidget(self._button(panel, "Object properties...", self.edit_properties))
        layout.addWidget(self._button(panel, "Assign selected class", self.assign_class))
        layout.addWidget(self._button(panel, "Mark/unmark training examples", self.toggle_training))
        layout.addWidget(self._button(panel, "Lock/unlock selected objects", self.toggle_object_lock))
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Merge...", self.merge_selected),
                self._button(panel, "Delete", self.delete_selected),
            )
        )
        layout.addWidget(
            hp.make_label(
                panel,
                "Polygon: double-click to finish, then Apply drawing.\nBrush: drag to add; Alt subtracts.\nVertices: edit yellow rings, then Apply.\nCut: draw a line, then choose the survivor.",
                wrap=True,
            )
        )
        return panel

    def _train_panel(self) -> QWidget:
        panel = QWidget(self)
        form = hp.make_form_layout(parent=panel)
        form.addRow(
            hp.make_label(
                panel,
                "Assign classes to examples and mark them for training. Include examples for at least two classes.",
                wrap=True,
            )
        )
        self.classifier_name = QLineEdit("Tissue classifier", panel)
        self.train_downsample = hp.make_int_spin_box(panel, minimum=1, maximum=256, value=4)
        self.feature_sigma = hp.make_int_spin_box(panel, minimum=1, maximum=32, value=8)
        self.sample_count = hp.make_int_spin_box(panel, minimum=100, maximum=100000, value=10000)
        form.addRow("Classifier name", self.classifier_name)
        form.addRow("Training downsample", self.train_downsample)
        form.addRow("Largest feature scale", self.feature_sigma)
        form.addRow("Samples per class", self.sample_count)
        form.addRow(self._button(panel, "Train from project examples", self.train))
        form.addRow(
            hp.make_label(
                panel,
                "Channels come from the Annotate panel. Training examples, feature settings, and random seed are saved with the classifier.",
                wrap=True,
            )
        )
        return panel

    def _predict_panel(self) -> QWidget:
        panel = QWidget(self)
        form = hp.make_form_layout(parent=panel)
        self.backend = hp.make_combobox(
            panel, ["forest", "micro-sam", "cellpose"], func=lambda *_: self._backend_changed()
        )
        self.model = QLineEdit(panel)
        self.checkpoint = QLineEdit(panel)
        self.device = hp.make_combobox(panel, ["auto", "cpu", "cuda", "mps"])
        self.predict_downsample = hp.make_int_spin_box(panel, minimum=1, maximum=256, value=1)
        self.tile_size = hp.make_combobox(panel, ["256", "512", "1024"])
        self.tile_size.setCurrentText("512")
        self.download_model = QCheckBox("Allow model download for this run", panel)
        form.addRow("Backend", self.backend)
        form.addRow("Classifier path / model name", self.model)
        form.addRow(self._button(panel, "Choose classifier...", self.choose_model))
        form.addRow("Optional local checkpoint", self.checkpoint)
        form.addRow(self._button(panel, "Choose checkpoint...", self.choose_checkpoint))
        form.addRow("Device", self.device)
        form.addRow("Prediction downsample", self.predict_downsample)
        form.addRow("Patch size", self.tile_size)
        form.addRow(self.download_model)
        form.addRow(self._button(panel, "Preview current view", self.predict_view))
        form.addRow(self._button(panel, "Predict drawn rectangle", self.predict_roi))
        form.addRow(self._button(panel, "Predict current slide", self.predict_slide))
        form.addRow(self._button(panel, "Queue all slides", self.predict_all))
        form.addRow(self._button(panel, "Run point/box assistance", self.predict_prompts))
        self.backend_hint = hp.make_label(
            panel, "Pretrained models are optional. Predictions remain separate until reviewed.", wrap=True
        )
        form.addRow(self.backend_hint)
        return panel

    def _review_panel(self) -> QWidget:
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        layout.addWidget(
            hp.make_label(
                panel, "Choose a prediction set on the left. Proposed objects appear in this paged list.", wrap=True
            )
        )
        self.proposals = QTableWidget(0, 2, panel)
        self.proposals.setHorizontalHeaderLabels(["ID", "Review note"])
        self.proposals.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.proposals.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.proposals.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.proposals.cellDoubleClicked.connect(self._zoom_proposal)
        layout.addWidget(self.proposals, 1)
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Previous", self.previous_page), self._button(panel, "Next", self.next_page)
            )
        )
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Accept selected", partial(self.review, True, False)),
                self._button(panel, "Reject selected", partial(self.review, False, False)),
            )
        )
        layout.addLayout(
            hp.make_h_layout(
                self._button(panel, "Accept batch", partial(self.review, True, True)),
                self._button(panel, "Reject batch", partial(self.review, False, True)),
            )
        )
        layout.addWidget(self._button(panel, "Refine selected proposal", self.refine_proposal))
        return panel

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
            for name, color in (("Tissue", "#ef5350"), ("Stroma", "#42a5f5"), ("Background", "#bdbdbd")):
                store.save_class(AnnotationClass(name=name, color=color))
        self.open_project(path)

    def choose_project(self) -> None:
        """Open an existing annotation project."""
        path = QFileDialog.getExistingDirectory(self, "Open .i2annotate project")
        if path:
            self.open_project(path)

    def open_project(self, path: str | Path) -> None:
        """Load a project and recover abandoned staging sets before starting work."""
        if self.jobs.busy or self._threads:
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

    def _background(self, function: Any, done: Any, *, discard: Any = None) -> None:
        worker = create_worker(function)
        self._threads.append(worker)
        worker.returned.connect(lambda value: done(value) if not self._closing else discard(value) if discard else None)
        worker.errored.connect(lambda exc: self._error(str(exc)) if not self._closing else None)
        worker.finished.connect(lambda: self._threads.remove(worker) if worker in self._threads else None)
        worker.start()

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
                    f"{item.name}{' [hidden]' if not item.visible else ''}{' [locked]' if item.locked else ''}"
                )
                cast(QListWidgetItem, self.classes.item(index)).setData(Qt.ItemDataRole.UserRole, item.id)
                from qtpy.QtGui import QColor

                cast(QListWidgetItem, self.classes.item(index)).setForeground(QColor(item.color))
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
                self.sets.addItem(item["name"], item["id"])
            index = self.sets.findData(current)
            self.sets.setCurrentIndex(max(0, index))
        self._syncing = False
        self._set_changed()

    def _set_changed(self, *_args: Any) -> None:
        if self._syncing:
            return
        self._page_after = 0
        self._page_stack.clear()
        self.selected.clear()
        self.refresh_tables()
        self.schedule_render()

    def refresh_tables(self) -> None:
        """Fetch a page of objects rather than populating a million-row Qt table."""
        self._syncing = True
        classes = {item.id: item.name for item in self.store.classes()} if self.store else {}
        for table, review in ((self.objects, "accepted"), (self.proposals, "proposed")):
            table.setRowCount(0)
            if not self.store or not self.set_id:
                continue
            for row, obj in enumerate(
                self.store.iter_objects(set_id=self.set_id, review=review, after=self._page_after, limit=100)
            ):
                table.insertRow(row)
                values = (
                    [str(obj.object_id), classes.get(obj.class_id or "", "Unclassified"), f"{obj.geometry.area:.1f}"]
                    if review == "accepted"
                    else [str(obj.object_id), "Check tile seam" if obj.properties.get("ambiguous_seam") else "Ready"]
                )
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(value)
                    cell.setData(Qt.ItemDataRole.UserRole, obj.key)
                    table.setItem(row, column, cell)
                if obj.key in self.selected:
                    table.selectRow(row)
        self._syncing = False
        self.selection_label.setText(f"{len(self.selected)} selected.")

    def next_page(self) -> None:
        """Advance through objects using indexed keyset pagination."""
        table = self.proposals if self.tabs.currentIndex() == 3 else self.objects
        if table.rowCount():
            self._page_stack.append(self._page_after)
            self._page_after = _table_key(table, table.rowCount() - 1)
            self.refresh_tables()

    def previous_page(self) -> None:
        """Return to the preceding object page."""
        if self._page_stack:
            self._page_after = self._page_stack.pop()
            self.refresh_tables()

    def _table_selection(self) -> None:
        if not self._syncing:
            self.selected = _selected_keys(self.objects)
            self._show_selected()

    def _show_selected(self) -> None:
        summary = f"{len(self.selected)} selected."
        if len(self.selected) == 1 and self.store:
            obj = self.store.get(self.selected[0])
            summary += (
                f" ID: {obj.object_id}\n{obj.review.capitalize()} | Training example: {'yes' if obj.training else 'no'}"
            )
        self.selection_label.setText(summary)
        self.draw_layer.data = []
        self._ring_layout.clear()
        if self.selected and self.store:
            rings = []
            for key in self.selected[:100]:
                obj = self.store.get(key)
                polygons = list(obj.geometry.geoms) if obj.geometry.geom_type == "MultiPolygon" else [obj.geometry]
                for polygon in polygons:
                    self._ring_layout.append(1 + len(polygon.interiors))
                    for ring in (polygon.exterior, *polygon.interiors):
                        rings.append(np.asarray(ring.coords)[:-1, ::-1])
            if rings:
                self.draw_layer.add(rings, shape_type="polygon")
            self._edit_revision = self.store.revision
        self.set_mode(self.tool.currentText())

    def set_mode(self, mode: str) -> None:
        """Activate a drawing or selection tool using the editor's pixel coordinate frame."""
        previous_mode, self._mode = self._mode, mode
        if not hasattr(self, "draw_layer"):
            return
        if previous_mode != mode and mode in ("Polygon", "Rectangle", "Cut", "Select region", "Prompt box", "Freehand"):
            self.draw_layer.data = []
        mapping = {
            "Polygon": "add_polygon",
            "Rectangle": "add_rectangle",
            "Vertices": "direct",
            "Cut": "add_path",
            "Select region": "add_polygon",
            "Prompt box": "add_rectangle",
        }
        self.draw_layer.mode = mapping.get(mode, "pan_zoom")
        self.view.viewer.layers.selection.select_only(self.draw_layer)
        self.prompt_layer.mode = "pan_zoom"

    def clear_drawing(self) -> None:
        """Discard temporary drawing and prompts without touching stored annotations."""
        self.draw_layer.data = []
        self.prompt_layer.data = np.empty((0, 2))
        self._prompt_labels: list[int] = []
        self._ring_layout.clear()

    def _mouse_drag(self, _viewer: Any, event: Any) -> Any:
        if not self.store or not self.slide_id or event.button != 1:
            return
        xy = (float(event.position[-1]), float(event.position[-2]))
        if self._mode == "Select":
            obj = self.store.hit_test(self.slide_id, xy, set_id=self.set_id)
            extend = "Shift" in event.modifiers
            self.selected = (
                ([*self.selected, obj.key] if extend and obj and obj.key not in self.selected else [obj.key])
                if obj
                else []
            )
            self._show_selected()
            return
        if self._mode in ("Prompt +", "Prompt -"):
            self.prompt_layer.add(np.asarray(xy[::-1]))
            self._prompt_labels.append(1 if self._mode == "Prompt +" else 0)
            return
        if self._mode not in ("Freehand", "Brush +", "Brush -"):
            return
        points = [xy]
        yield
        while event.type == "mouse_move":
            point = (float(event.position[-1]), float(event.position[-2]))
            if np.linalg.norm(np.asarray(point) - points[-1]) > max(0.2, 1 / self.view.viewer.camera.zoom):
                points.append(point)
                if self._mode == "Freehand":
                    self.draw_layer.data = []
                    self.draw_layer.add(np.asarray(points)[:, ::-1], shape_type="path")
            yield
        if self._mode == "Freehand":
            if len(points) >= 3:
                self.draw_layer.data = []
                self.draw_layer.add(np.asarray(points)[:, ::-1], shape_type="polygon")
                self._try(lambda: self.add_geometry(Polygon(points)))
        else:
            radius = self.brush_radius.value() / max(0.01, self.view.viewer.camera.zoom)
            stroke = (LineString(points) if len(points) > 1 else Point(points[0])).buffer(radius)
            subtract = self._mode == "Brush -" or "Alt" in event.modifiers
            if self.selected:
                self._try(lambda: self._brush(stroke, subtract))
            elif not subtract:
                self._try(lambda: self.add_geometry(stroke))

    def _brush(self, stroke: Any, subtract: bool) -> None:
        store = self._require_project()
        if len(self.selected) != 1:
            raise ValueError("Brush editing requires exactly one selected object.")
        old = store.get(self.selected[0]).geometry
        brush_object(store, self.selected[0], stroke, subtract=subtract)
        self._changed(old.union(stroke).bounds)

    def add_geometry(self, geometry: Any) -> int:
        """Create a manually drawn object in the active namespace."""
        store = self._require_project()
        self._require_slide()
        if not self.set_id:
            raise ValueError("Create an annotation set first.")
        obj = AnnotationObject(
            self.set_id, store.next_id(self.set_id), geometry, self.class_id, provenance={"source": "manual"}
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
        elif self._mode == "Select region":
            polygon = Polygon(np.asarray(self.draw_layer.data[-1])[:, ::-1])
            self.selected = list(store.selection_in(slide.id, polygon, set_id=self.set_id))
            self._show_selected()
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
        """Mark selected objects as explicit examples for tissue training."""
        store = self._require_project()
        objects = [store.get(key) for key in self.selected]
        enabled = not all(obj.training for obj in objects)
        store.apply(
            EditCommand("Mark training examples", [(obj.key, replace(obj, training=enabled)) for obj in objects])
        )
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

    def undo(self) -> None:
        """Undo a persisted command and refresh the editor."""
        store = self._require_project()
        if store.history_size(undo=True) > 1000:
            self.jobs.submit({"operation": "undo", "project": str(store.path)})
            return
        store.undo()
        self.selected.clear()
        self.refresh_classes()
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
        self._changed()

    def add_class(self) -> None:
        """Create a shared class with a chosen name and color."""
        store = self._require_project()
        name, ok = QInputDialog.getText(self, "New class", "Name:")
        if ok and name.strip():
            color = QColorDialog.getColor(parent=self)
            store.save_class(AnnotationClass(name=name.strip(), color=color.name() if color.isValid() else "#ef5350"))
            self.refresh_classes()

    def edit_class(self) -> None:
        """Edit the selected shared class without changing its ID."""
        store = self._require_project()
        item = next((item for item in store.classes() if item.id == self.class_id), None)
        if item:
            name, ok = QInputDialog.getText(self, "Edit class", "Name:", text=item.name)
            if ok and name.strip():
                color = QColorDialog.getColor(parent=self)
                item.name = name.strip()
                if color.isValid():
                    item.color = color.name()
                store.save_class(item)
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
            for layer in (self.draw_layer, self.prompt_layer):
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
        """Train using explicitly marked project examples in a spawned process."""
        store = self._require_project()
        settings = FeatureSettings(
            channels=self._channels(),
            downsample=self.train_downsample.value(),
            sigma_max=self.feature_sigma.value(),
            samples_per_class=self.sample_count.value(),
        )
        self.jobs.submit(
            {
                "operation": "train",
                "project": str(store.path),
                "name": self.classifier_name.text(),
                "settings": settings.model_dump(),
            }
        )

    def choose_model(self) -> None:
        """Select a saved classifier and restore its required inference settings."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose classifier", str(self.store.path / "models") if self.store else "", "Classifier (*.npz)"
        )
        if path:
            _forest, metadata = load_forest(path)
            self.model.setText(path)
            self.channels.setText(", ".join(str(value) for value in metadata["settings"]["channels"]))
            self.predict_downsample.setValue(metadata["settings"]["downsample"])

    def choose_checkpoint(self) -> None:
        """Choose an existing pretrained model checkpoint."""
        path, _ = QFileDialog.getOpenFileName(self, "Choose local checkpoint")
        if path:
            self.checkpoint.setText(path)

    def _backend_changed(self) -> None:
        backend = self.backend.currentText()
        if backend != "forest":
            self.model.setText("vit_b_lm" if backend == "micro-sam" else "cpsam")
        if hasattr(self, "backend_hint"):
            self.backend_hint.setText(
                "Backend available. Predictions require review."
                if backend_available(backend)
                else f"Optional backend missing: install image2image[{'annotate-sam' if backend == 'micro-sam' else 'annotate-cellpose'}]."
            )

    def _request(
        self, bounds: tuple | None = None, *, prompts: bool = False, slide_id: str | None = None
    ) -> PredictionRequest:
        store = self._require_project()
        slide = self._require_slide()
        points, prompt_box = [], None
        if prompts:
            points = [
                (float(point[1]), float(point[0]), label)
                for point, label in zip(self.prompt_layer.data, self._prompt_labels, strict=True)
            ]
            if len(self.draw_layer.data) and self._mode == "Prompt box":
                prompt_box = Polygon(np.asarray(self.draw_layer.data[-1])[:, ::-1]).bounds
        return PredictionRequest(
            project=str(store.path),
            slide_id=slide_id or slide.id,
            revision=store.revision,
            backend=cast(Literal["forest", "micro-sam", "cellpose"], self.backend.currentText()),
            model=self.model.text(),
            checkpoint=self.checkpoint.text(),
            device=cast(Literal["auto", "cpu", "cuda", "mps"], self.device.currentText()),
            channels=self._channels(),
            downsample=self.predict_downsample.value(),
            tile_size=int(self.tile_size.currentText()),
            bounds=bounds,
            points=points,
            box=prompt_box,
            class_id=self.class_id,
            allow_download=self.download_model.isChecked(),
        )

    def _submit_prediction(self, request: PredictionRequest) -> None:
        store = self._require_project()
        self.jobs.submit({"operation": "predict", "project": str(store.path), "request": request.model_dump()})

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
        for slide in self._require_project().slides():
            self._submit_prediction(self._request(slide_id=slide.id))

    def predict_prompts(self) -> None:
        """Run point/box assistance in the current viewport."""
        if self.backend.currentText() != "micro-sam":
            raise ValueError("Select the micro-sam backend for point/box assistance.")
        self._submit_prediction(self._request(self.viewport(), prompts=True))

    def _job_done(self, job: dict, result: Any) -> None:
        if self._closing:
            return
        if job["operation"] in ("undo", "redo", "review"):
            self.selected.clear()
        if job["operation"] == "train":
            self.model.setText(str(result))
            self.backend.setCurrentText("forest")
            self.predict_downsample.setValue(job["settings"]["downsample"])
        self.refresh_classes()
        self.refresh_sets()
        if job["operation"] == "predict":
            key = result["set_id"]
            index = self.sets.findData(key)
            if index >= 0:
                self.sets.setCurrentIndex(index)
                self.tabs.setCurrentIndex(3)
        elif job["operation"] == "import":
            self.sets.setCurrentIndex(self.sets.findData(result))
        if self.renderer:
            self.renderer.invalidate()
        self.schedule_render()
        self.status.setText(
            f"Completed {job['operation']}."
            + (" Review predictions before accepting them." if job["operation"] == "predict" else "")
        )

    def review(self, accept: bool, batch: bool) -> None:
        """Accept or reject a selected subset or the complete current prediction batch."""
        store = self._require_project()
        keys = None if batch else _selected_keys(self.proposals)
        if keys == []:
            raise ValueError("Select proposals to review.")
        self.jobs.submit(
            {"operation": "review", "project": str(store.path), "set_id": self.set_id, "accept": accept, "keys": keys}
        )

    def _zoom_proposal(self, row: int, _column: int) -> None:
        self.selected = [_table_key(self.proposals, row)]
        self.zoom_selected()

    def refine_proposal(self) -> None:
        """Load one proposed object for manual vertex or brush correction before acceptance."""
        keys = _selected_keys(self.proposals)
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
        self.render_timer.stop()
        self.jobs.close()
        if self._threads:
            event.ignore()
            QTimer.singleShot(100, self.close)
            return
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
