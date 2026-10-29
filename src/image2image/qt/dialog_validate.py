"""Interactive independent registration validation and deformation analysis."""

from __future__ import annotations

import json
from functools import partial
from html import escape
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from typing import Any
from uuid import uuid4

import numpy as np
import qtextra.helpers as hp
from qtextra.config import THEMES
from qtextra.utils.table_config import TableConfig
from qtextraplot._napari.image.wrapper import NapariImageView
from qtpy.QtCore import Qt, QTimer, Signal
from qtpy.QtGui import QCloseEvent, QKeySequence, QShortcut, QUndoCommand, QUndoStack
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QFormLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QWidget,
)
from superqt.utils import create_worker

import image2image.constants as constants
from image2image.config import SingleAppConfig
from image2image.models.validation import (
    CandidateAnalysis,
    ImageGeometry,
    MaskCorrespondence,
    PairAnalysis,
    StudyAnalysis,
    ValidationAnalysis,
    ValidationCandidate,
    ValidationImage,
    ValidationLandmark,
    ValidationPair,
    ValidationSession,
    ValidationStudy,
)
from image2image.qt._dialog_base import Window
from image2image.utils.validation import (
    AnalysisCancelled,
    ValidationTransform,
    adopt_validation_calibration,
    analysis_grid,
    analyze_validation_session,
    analyze_validation_study,
    discover_validation_candidates,
    export_validation_landmarks,
    import_validation_landmarks,
    inspect_validation_image,
    landmark_errors,
    landmark_exclusions,
    landmark_warnings,
    load_validation_study,
    load_validation_transform,
    merge_landmarks,
    open_validation_reader,
    require_same_pair,
    sample_image,
    save_validation_study,
    validation_contrast_limits,
    validation_image_pyramid,
)
from image2image.utils.validation_auto import TemplateMatch, match_through_transform
from image2image.utils.validation_report import comparison_rows, export_validation_study_report, normalize_image

IDENTITY = ValidationCandidate(id="aligned-images", name="Already aligned images", kind="aligned", path="")
LANDMARK_COLUMNS = ["Use", "ID", "X (fixed)", "Y (fixed)", "X (moving)", "Y (moving)", "Note", "Source", "Error (um)"]
RESULT_MODES = {
    "overlap": "Image overlap",
    "validation_errors": "Validation errors",
    "mask_disagreement": "Mask disagreement",
    "area_change": "Area change (log2)",
    "signed_jacobian": "Signed Jacobian",
    "nonpositive_jacobian": "Nonpositive Jacobian",
    "residual_shift": "Residual shift map",
    "automatic_landmarks": "Automatic landmarks",
}


class LandmarkCommand(QUndoCommand):
    """Undoable replacement of one pair's landmark list."""

    def __init__(
        self,
        window: ImageValidationWindow,
        pair_id: str,
        before: list[ValidationLandmark],
        after: list[ValidationLandmark],
        text: str,
        sync_layers: bool = True,
    ) -> None:
        super().__init__(text)
        self.window, self.pair_id = window, pair_id
        self.before = [point.model_copy() for point in before]
        self.after = [point.model_copy() for point in after]
        self._sync_layers = sync_layers

    def redo(self) -> None:
        """Apply the new landmarks; only the first application may skip layer synchronization."""
        self.window._apply_landmarks(self.pair_id, self.after, self._sync_layers)
        self._sync_layers = True

    def undo(self) -> None:
        """Restore the previous landmarks."""
        self.window._apply_landmarks(self.pair_id, self.before, True)


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
        self.study = ValidationStudy()
        self._pair_index = 0
        self.session = self.study.pairs[0].session
        self.analysis: ValidationAnalysis | None = None
        self._results: dict[str, ValidationAnalysis] = {}
        self._failures: dict[str, str] = {}
        self._directories: dict[str, TemporaryDirectory] = {}
        self._readers: list[Any | None] = [None, None]
        self._view_keys: list[tuple | None] = [None, None]
        self._display_states: dict[tuple, dict[str, Any]] = {}
        self._camera_states: dict[str, list[tuple[Any, float]]] = {}
        self._selected_candidates: dict[str, str] = {}
        self._generation = 0
        self._updating = False
        self._syncing = False
        self._selecting_layer = False
        self._dirty = False
        self._closing = False
        self._pending_pair: tuple[float, float] | None = None
        self._adding_pair = False
        self._workers: list[Any] = []
        self._cancel = Event()
        self._result_directory: TemporaryDirectory | None = None
        self._session_path: Path | None = None
        self._mask_readers: list[Any | None] = [None, None]
        self._transforms: dict[tuple, ValidationTransform] = {}
        self._transform_errors: dict[tuple, str] = {}
        self._transform_workers: list[Any] = []
        self._assist_workers: list[Any] = []
        self._detect_cancel = Event()
        self._suggestion: TemplateMatch | None = None
        self._assist_generation = 0
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
            THEMES.evt_theme_changed.connect(self._apply_fonts)

    def _apply_fonts(self) -> None:
        """Use one readable theme-based body size for controls, messages, and viewer annotations."""
        size = max(11, THEMES.get_font_size())
        font = self.font()
        font.setPointSize(size)
        self.setFont(font)
        # self.centralWidget().setStyleSheet(
        #     "QLabel, QAbstractButton, QComboBox, QLineEdit, QAbstractSpinBox, QAbstractItemView, "
        #     "QHeaderView, QHeaderView::section, QTabBar, QTabBar::tab, QGroupBox {"
        #     f"font-size: {size}pt;"
        #     "}"
        # )
        # self.menuBar().setStyleSheet(f"font-size: {size}pt;")
        # self.statusBar().setStyleSheet(f"font-size: {size}pt;")
        for tabs in self.findChildren(QTabWidget):
            tabs.tabBar().setElideMode(Qt.TextElideMode.ElideNone)
        for view in (*self.views, self.result_view):
            view.viewer.scale_bar.font_size = size
            view.viewer.text_overlay.font_size = size
        if "Validation error (um)" in self.result_view.viewer.layers:
            layer = self.result_view.viewer.layers["Validation error (um)"]
            layer.text.size = size
            layer.face_colorbar.font_size = size

    def _try(self, function: Any) -> Any:
        try:
            return function()
        except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
            self._set_status(str(exc))
            hp.warn_pretty(self, str(exc), "Validation input")
            return None

    @staticmethod
    def _table(parent: QWidget, headings: list[str], editable: bool = True) -> QTableWidget:
        config = TableConfig()
        for index, heading in enumerate(headings):
            config.add(heading, str(index), sizing="contents")
        table = hp.make_table(parent, config)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.horizontalHeader().setStretchLastSection(True)
        if not editable:
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        return table

    @staticmethod
    def _set_message(label: QLabel, text: str, title: str = "") -> None:
        if not title:
            breaks = [(text.find(separator), separator) for separator in (". ", "\n") if separator in text]
            if breaks:
                index, separator = min(breaks)
                title, text = text[:index], text[index + len(separator) :]
            else:
                title, text = text, ""
        body = escape(text).replace("\n", "<br>")
        label.setText(f"<b>{escape(title)}</b>" + (f"<br>{body}" if body else ""))

    def _message_label(self, parent: QWidget, text: str) -> QLabel:
        label = hp.make_label(
            parent,
            wrap=True,
            selectable=True,
            text_format=Qt.TextFormat.RichText,
        )
        self._set_message(label, text)
        return label

    def _set_status(self, text: str, title: str = "") -> None:
        self._set_message(self.status, text, title)

    def _setup_ui(self) -> None:
        self.undo_stack = QUndoStack(self)
        widget = QWidget(self)
        layout = hp.make_v_layout(parent=widget)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(widget, "New study", func=self.new_session),
                hp.make_btn(widget, "Open study...", func=self.choose_session),
                hp.make_btn(widget, "Save study...", func=self.choose_save),
            )
        )
        self.tabs = QTabWidget(widget)
        self.tabs.addTab(self._make_inputs(), "Inputs")
        self.tabs.addTab(self._make_landmarks(), "Landmarks")
        self.tabs.addTab(self._make_masks(), "Region overlap (optional)")
        self.tabs.addTab(self._make_results(), "Analysis")
        self.tabs.setMinimumWidth(520)
        self.view_stack = QStackedWidget(widget)
        raw = QWidget(widget)
        raw_layout = hp.make_v_layout(parent=raw, spacing=1, margin=1)
        self.link_views = hp.make_checkbox(
            raw,
            "Link views in physical coordinates",
            value=True,
            func=self._link_views_changed,
            tooltip="Show corresponding locations in both images. With a registration selected, views are linked "
            "through it; otherwise through calibrated physical coordinates. Disable for independent navigation.",
        )
        self.native_splitter = QSplitter(Qt.Orientation.Vertical, raw)
        raw_layout.addWidget(self.native_splitter)
        raw_layout.addWidget(self.link_views)
        self.views: list[NapariImageView] = []
        self.coordinate_labels: list[QLabel] = []
        self.point_layers: list[Any] = []
        for index, name in enumerate(("Fixed", "Moving")):
            panel = QWidget(raw)
            box = hp.make_v_layout(parent=panel)
            coordinates = hp.make_label(panel, "X: -- um, Y: -- um")
            self.coordinate_labels.append(coordinates)
            box.addLayout(
                hp.make_h_layout(
                    hp.make_label(panel, f"{name} image - native physical coordinates (um)"),
                    coordinates,
                    stretch_id=0,
                )
            )
            view = NapariImageView(
                panel, add_toolbars=True, disable_new_layers=True, allow_crosshair=False, allow_extraction=False
            )
            view.viewer.scale_bar.visible = True
            box.addWidget(view.widget)
            view.viewer.mouse_drag_callbacks.append(partial(self._pair_click, index))
            view.viewer.camera.events.zoom.connect(partial(self._sync_zoom, index))
            view.viewer.camera.events.center.connect(partial(self._sync_zoom, index))
            view.viewer.layers.selection.events.active.connect(partial(self._select_native_layer, index))
            view.viewer.cursor.events.position.connect(partial(self._cursor_changed, index))
            self.views.append(view)
            self.native_splitter.addWidget(panel)
        self.view_stack.addWidget(raw)
        result_panel = QWidget(widget)
        result_layout = hp.make_v_layout(parent=result_panel)
        self.result_coordinates = hp.make_label(result_panel, "X: -- um, Y: -- um")
        result_layout.addWidget(self.result_coordinates)
        self.result_view = NapariImageView(
            result_panel, add_toolbars=True, disable_new_layers=True, allow_crosshair=False, allow_extraction=False
        )
        self.result_view.viewer.cursor.events.position.connect(partial(self._cursor_changed, None))
        self.result_view.viewer.scale_bar.visible = True
        self.validation_legend = hp.make_group_box(result_panel, "Independent landmark errors (um, moving frame)")
        self.validation_summary = self._message_label(self.validation_legend, "")
        self.error_labels = hp.make_checkbox(
            self.validation_legend, "Show landmark IDs and errors", value=True, func=self._error_labels_changed
        )
        hp.make_v_layout(
            self.validation_summary,
            hp.make_label(
                self.validation_legend,
                "Point color shows positional error in um; use the colorbar at the upper right of the image. "
                "Smaller errors mean closer agreement at those landmarks. "
                "Cyan arrows run from the observed moving point to the position predicted by the registration, "
                "at their actual physical length. The background is the native moving image for context. "
                "Errors are measured only at the landmarks; no accuracy is inferred between them and no "
                "universal pass/fail threshold is applied.",
                wrap=True,
            ),
            self.error_labels,
            parent=self.validation_legend,
            margin=6,
            spacing=3,
        )
        self.validation_legend.hide()
        result_layout.addWidget(self.validation_legend)
        result_layout.addWidget(self.result_view.widget)
        self.view_stack.addWidget(result_panel)
        splitter = QSplitter(Qt.Orientation.Horizontal, widget)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.view_stack)
        splitter.setSizes([550, 1000])
        layout.addWidget(splitter, 1)
        self.status = self._message_label(
            widget, "Load aligned images or import a registration, add evidence, then analyze."
        )
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
        self._make_shortcuts()
        self._apply_fonts()
        self._refresh_inputs()

    def _make_shortcuts(self) -> None:
        for sequence, func in (
            (QKeySequence.StandardKey.Undo, self.undo_stack.undo),
            (QKeySequence.StandardKey.Redo, self.undo_stack.redo),
            (QKeySequence("Ctrl+Shift+Z"), self.undo_stack.redo),
            (QKeySequence("A"), self.begin_pair),
        ):
            QShortcut(sequence, self, activated=func)
        # Enabled only while relevant, so editors keep Esc and Return otherwise.
        self._cancel_shortcut = QShortcut(QKeySequence("Esc"), self, activated=self.cancel_pair)
        self._accept_shortcuts = [
            QShortcut(QKeySequence(key), self, activated=self.accept_suggestion) for key in ("Return", "Enter")
        ]
        self._cancel_shortcut.setEnabled(False)
        for shortcut in self._accept_shortcuts:
            shortcut.setEnabled(False)
        for sequence in (QKeySequence.StandardKey.Delete, QKeySequence("Backspace")):
            QShortcut(
                sequence, self.landmark_table, activated=self.delete_pair, context=Qt.ShortcutContext.WidgetShortcut
            )

    def _make_inputs(self) -> QWidget:
        widget, scroll = hp.make_scroll_area(self, horizontal=Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        form = hp.make_form_layout(parent=widget)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.image_labels = []
        self.channels = []
        self.image_sections: list[QWidget] = []
        self.geometry_buttons: list[QPushButton] = []
        self.remove_image_buttons: list[QPushButton] = []
        for index, name in enumerate(("Fixed", "Moving")):
            group = hp.make_group_box(widget, name, is_flat=False)
            self.image_sections.append(group)
            box = hp.make_v_layout(parent=group)
            geometry_btn = hp.make_qta_btn(
                group,
                "edit",
                tooltip=f"Update {name.lower()} image spacing and origin (um).",
                size_preset="small",
                func=partial(self.edit_image_geometry, index),
            )
            remove_btn = hp.make_qta_btn(
                group,
                "delete",
                tooltip=f"Remove the {name.lower()} image from the app. Source files are preserved.",
                size_preset="small",
                func=self.remove_fixed_image if index == 0 else self.remove_moving_image,
            )
            self.geometry_buttons.append(geometry_btn)
            self.remove_image_buttons.append(remove_btn)
            widgets = [
                hp.make_btn(group, f"Load {name.lower()} image...", func=partial(self.choose_image, index)),
                geometry_btn,
                remove_btn,
            ]
            if index == 1:
                widgets[0].setToolTip(
                    "Load one or more moving images. New images create separate pairs; loaded images are selected."
                )
            box.addLayout(
                hp.make_h_layout(
                    *widgets,
                    stretch_id=0,
                    spacing=1,
                )
            )
            if index == 1:
                self.pair_selector = hp.make_combobox(group, func_index=self._select_pair)
                self.remove_moving_btn = remove_btn
                box.addWidget(self.pair_selector)
            label = hp.make_label(group, "No image", wrap=True, selectable=True, text_format=Qt.TextFormat.PlainText)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
            channel = hp.make_combobox(group, func_index=partial(self._channel_changed, index))
            self.image_labels.append(label)
            self.channels.append(channel)
            box.addWidget(label)
            box.addWidget(channel)
            form.addRow(group)
        form.addRow(
            hp.make_h_layout(
                hp.make_btn(
                    widget,
                    "Add fiducial registration (.i2r)...",
                    func=self.choose_register,
                    tooltip="Add a landmark registration saved by the Register app.",
                ),
                hp.make_btn(
                    widget,
                    "Add image2image-reg project...",
                    func=self.choose_elastix,
                    tooltip="Add every registration path of a completed image2image-reg (Elastix) project.",
                ),
            )
        )
        self.candidate_list = QListWidget(widget)
        self.candidate_list.currentItemChanged.connect(self._candidate_selected)
        form.addRow(self.candidate_list)
        form.addRow(hp.make_btn(widget, "Remove selected registration", func=self.remove_candidate))
        form.addRow(
            hp.make_label(
                widget,
                "Without a registration project, images are compared directly in calibrated physical space. "
                "Validation landmarks measure residuals; they are never used to fit a new registration.",
                wrap=True,
            )
        )
        self.evidence_label = self._message_label(widget, "")
        form.addRow(self.evidence_label)
        return scroll

    def _make_landmarks(self) -> QWidget:
        widget = QWidget(self)
        layout = hp.make_v_layout(parent=widget)
        self.units = hp.make_combobox(widget, ["px", "um"], func=self._units_changed)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(widget, "Import CSV...", func=self.choose_landmarks),
                hp.make_btn(widget, "Export CSV...", func=self.choose_landmark_export),
                hp.make_label(widget, "CSV units"),
                self.units,
            )
        )
        self.edit_landmarks = hp.make_checkbox(
            widget,
            "Edit landmarks in the viewers",
            value=False,
            func=self._landmark_edit_mode,
            tooltip="Select the landmark layers to drag existing points. Image controls are active otherwise.",
        )
        layout.addWidget(
            hp.make_label(
                widget,
                "Table and views use um. Add landmarks by clicking fixed, then moving; adding continues until Esc. "
                "Drag existing points or edit coordinates below. Shortcuts: A add, Esc stop, Del delete, "
                "Ctrl+Z undo.",
                wrap=True,
            )
        )
        layout.addLayout(
            hp.make_h_layout(
                hp.make_btn(widget, "Add landmark", func=self.begin_pair),
                hp.make_btn(widget, "Delete landmark", func=self.delete_pair),
            )
        )
        layout.addWidget(self.edit_landmarks)
        self.landmark_table = self._table(widget, LANDMARK_COLUMNS)
        self.landmark_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.landmark_table.itemChanged.connect(self._landmark_edited)
        self.landmark_table.itemSelectionChanged.connect(self.focus_landmark)
        layout.addWidget(self.landmark_table)
        assistance = hp.make_group_box(widget, "Assistance")
        self.assist = hp.make_checkbox(
            assistance,
            "Suggest the moving point by template matching",
            value=False,
            tooltip="After a fixed click, search near the selected registration's prediction for the same image "
            "content. Press Enter to accept the suggestion (stored as assisted) or click the moving image to place "
            "the point yourself (stored as manual).",
        )
        self.assist_template = hp.make_double_spin_box(assistance, minimum=0, maximum=1e5, value=0, n_decimals=1)
        self.assist_search = hp.make_double_spin_box(assistance, minimum=0, maximum=1e5, value=0, n_decimals=1)
        for control in (self.assist_template, self.assist_search):
            control.setSpecialValueText("Auto")
            control.setSuffix(" um")
        self.detect_btn = hp.make_btn(
            assistance,
            "Detect landmarks",
            func=self.detect_landmarks,
            tooltip="Find matching image features through the selected registration and add them to the table "
            "as automatic landmarks for review.",
        )
        self.detect_count = hp.make_int_spin_box(assistance, minimum=1, maximum=1000, value=50)
        self.detect_count.setPrefix("Max ")
        self.remove_automatic_btn = hp.make_btn(
            assistance, "Remove automatic", func=self.remove_automatic, tooltip="Delete every automatic landmark."
        )
        hp.make_v_layout(
            self.assist,
            hp.make_h_layout(
                hp.make_label(assistance, "Template"),
                self.assist_template,
                hp.make_label(assistance, "Search radius"),
                self.assist_search,
                hp.make_help_label(assistance, constants.ASSIST_HELP),
                stretch_id=4,
            ),
            hp.make_h_layout(self.detect_btn, self.detect_count, self.remove_automatic_btn, stretch_id=0),
            parent=assistance,
            margin=6,
            spacing=3,
        )
        layout.addWidget(assistance)
        layout.addWidget(hp.make_label(widget, self.session.independence_note, wrap=True))
        return widget

    def _make_masks(self) -> QWidget:
        widget = QWidget(self)
        layout = hp.make_v_layout(parent=widget)
        layout.addWidget(
            hp.make_label(
                widget,
                constants.MASKS_HELP,
                wrap=True,
            )
        )
        self.mask_labels = []
        for index, name in enumerate(("Fixed", "Moving")):
            label = hp.make_label(widget, "No mask", wrap=True)
            self.mask_labels.append(label)
            layout.addLayout(
                hp.make_h_layout(hp.make_btn(widget, f"{name} mask...", func=partial(self.choose_mask, index)), label)
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
        layout = hp.make_v_layout(parent=widget)
        self.run_btn = hp.make_btn(
            widget, "Analyze pair", func=self.start_analysis, tooltip="Analyze the selected image pair."
        )
        self.run_all_btn = hp.make_btn(
            widget,
            "Analyze all",
            func=self.start_analysis_all,
            tooltip="Analyze image pairs without current results. Hold Shift to analyze every pair again.",
        )
        self.cancel_btn = hp.make_btn(widget, "Cancel", func=self.cancel_analysis)
        self.cancel_btn.setEnabled(False)
        layout.addLayout(hp.make_h_layout(self.run_btn, self.run_all_btn, self.cancel_btn))
        self.downsample = hp.make_int_spin_box(widget, minimum=1, maximum=10000, value=1, func=self._resolution_changed)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(widget, "Requested analysis downsample"),
                self.downsample,
                hp.make_help_label(widget, constants.DOWNSAMPLE_HELP),
                stretch_id=2,
            )
        )
        self.grid_label = hp.make_label(widget, "The common grid is capped at 4 million pixels.", wrap=True)
        layout.addWidget(self.grid_label)
        automatic = hp.make_group_box(widget, "Automatic evidence (all image pairs)")
        self.auto_residual_map = hp.make_checkbox(
            automatic,
            "Residual shift map",
            value=False,
            func=self._automatic_changed,
            tooltip="Tile-wise phase correlation between the fixed and registered moving images.",
        )
        self.auto_tile = hp.make_double_spin_box(
            automatic, minimum=0, maximum=1e6, value=0, n_decimals=1, func=self._automatic_changed
        )
        self.auto_tile.setSpecialValueText("Auto")
        self.auto_tile.setSuffix(" um")
        hp.make_v_layout(
            hp.make_h_layout(
                self.auto_residual_map,
                hp.make_label(automatic, "Tile"),
                self.auto_tile,
                hp.make_help_label(automatic, constants.AUTOMATIC_HELP),
                stretch_id=0,
            ),
            parent=automatic,
            margin=6,
            spacing=3,
        )
        layout.addWidget(automatic)
        self.summary_table = self._table(
            widget,
            [
                "Candidate",
                "Used/eligible",
                "Mean (um)",
                "Median (um)",
                "RMS (um)",
                "P95 (um)",
                "Max (um)",
                "Auto median (um)",
                "Shift map median (um)",
            ],
            False,
        )
        self.summary_table.setMaximumHeight(220)
        layout.addLayout(
            hp.make_h_layout(
                hp.make_label(
                    widget,
                    "Independent landmark accuracy (um, moving frame)",
                    bold=True,
                ),
                hp.make_help_label(widget, constants.LANDMARK_HELP),
                stretch_id=0,
            )
        )
        layout.addWidget(self.summary_table)
        self.result_candidate = hp.make_combobox(widget, func_index=self.show_result)
        self.summary_table.itemSelectionChanged.connect(
            lambda: self.result_candidate.setCurrentIndex(self.summary_table.currentRow())
        )
        self.result_mode = hp.make_combobox(widget, list(RESULT_MODES.values()), func_index=self._diagnostic_changed)
        self.diagnostic_help = hp.make_help_label(widget, constants.DIAGNOSTIC_HELP["overlap"])
        layout.addWidget(self.result_candidate)
        layout.addLayout(hp.make_h_layout(self.result_mode, self.diagnostic_help, stretch_id=0))
        self.result_detail = self._message_label(widget, "Analyze to see results.")
        layout.addWidget(self.result_detail)
        self.error_table = self._table(
            widget, ["ID", "Source", "dX (um)", "dY (um)", "Error (um)", "Exclusion", "Warning"], False
        )
        details = QTabWidget(widget)
        details.addTab(self._measurement_panel(widget, self.error_table, constants.LANDMARK_HELP), "Landmark residuals")
        self.fitting_table = self._table(widget, ["Statistic", "Value"], False)
        details.addTab(
            self._measurement_panel(widget, self.fitting_table, constants.FITTING_HELP),
            "Fitting residuals",
        )
        self.overlap_table = self._table(
            widget, ["Region", "Dice", "IoU", "Fixed area (um2)", "Moving area (um2)", "Joint coverage", "Note"], False
        )
        details.addTab(
            self._measurement_panel(widget, self.overlap_table, constants.OVERLAP_HELP),
            "Region overlap",
        )
        self.deformation_table = self._table(widget, ["Statistic", "Value"], False)
        details.addTab(
            self._measurement_panel(widget, self.deformation_table, constants.DEFORMATION_HELP),
            "Deformation",
        )
        self.automatic_table = self._table(widget, ["Statistic", "Residual shift map", "Automatic landmarks"], False)
        details.addTab(
            self._measurement_panel(widget, self.automatic_table, constants.AUTOMATIC_HELP),
            "Automatic",
        )
        layout.addWidget(details, 1)
        self.report_include_paths = hp.make_checkbox(
            widget,
            "Include source directory paths in reports",
            value=False,
            tooltip="Enable to retain full source paths. Otherwise HTML, PDF, JSON, CSV and figure labels retain "
            "filenames only. The exported study copy will need its source files located again.",
        )
        layout.addWidget(self.report_include_paths)
        self.export_btn = hp.make_btn(widget, "Export report...", func=self.choose_export)
        self.export_btn.setEnabled(False)
        layout.addWidget(self.export_btn)
        return widget

    @staticmethod
    def _measurement_panel(parent: QWidget, table: QTableWidget, help_text: str) -> QWidget:
        panel = QWidget(parent)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        hp.make_h_layout(
            table,
            hp.make_help_label(panel, help_text),
            parent=panel,
            stretch_id=(0,),
            spacing=1,
            margin=0,
            widget_alignment={1: Qt.AlignmentFlag.AlignTop},
        )
        return panel

    def _diagnostic_changed(self, index: int) -> None:
        self.diagnostic_help.setToolTip(constants.DIAGNOSTIC_HELP[list(RESULT_MODES)[index]])
        self.show_result()

    @property
    def _pair_id(self) -> str:
        return self.study.pairs[self._pair_index].id

    def _clear_result_widgets(self, message: str) -> None:
        self.analysis = None
        self._result_directory = None
        for table in (
            self.summary_table,
            self.error_table,
            self.fitting_table,
            self.overlap_table,
            self.deformation_table,
            self.automatic_table,
        ):
            table.setRowCount(0)
        self.result_view.viewer.layers.clear()
        self.validation_legend.hide()
        self.validation_summary.clear()
        with hp.qt_signals_blocked(self.result_candidate):
            self.result_candidate.clear()
        self._set_message(self.result_detail, message)
        self.export_btn.setEnabled(bool(self._results or self._failures))

    def _drop_result(self, pair_id: str) -> None:
        self._results.pop(pair_id, None)
        self._failures.pop(pair_id, None)
        directory = self._directories.pop(pair_id, None)
        if directory is not None and directory not in self._directories.values():
            if self._result_directory is directory:
                self.result_view.viewer.layers.clear()
            directory.cleanup()

    def _clear_study_results(self) -> None:
        for pair_id in list(self._directories):
            self._drop_result(pair_id)
        self._results.clear()
        self._failures.clear()
        self._clear_result_widgets("Analyze to see results.")

    def invalidate(self) -> None:
        """Cancel obsolete work and prevent stale measurements from being shown or exported."""
        if self._updating:
            return
        self._generation += 1
        self._cancel.set()
        self._dirty = True
        self._drop_result(self._pair_id)
        self._clear_result_widgets("Inputs changed. Use Analyze pair or Analyze all to update results.")

    def _select_pair(self, index: int, *, offer_copy: bool = True) -> None:
        if self._updating or index < 0 or index == self._pair_index:
            return
        previous = self.study.pairs[self._pair_index] if 0 <= self._pair_index < len(self.study.pairs) else None
        self._pair_index = index
        self.session = self.study.pairs[index].session
        self._adding_pair, self._pending_pair = False, None
        self.edit_landmarks.setChecked(False)
        self._clear_result_widgets("Analyze this pair to see available measurements.")
        if offer_copy:
            self._offer_landmark_copy(previous)
        self._refresh_inputs()
        self._load_views()
        self._display_analysis()

    def _offer_landmark_copy(self, previous: ValidationPair | None) -> None:
        if self.session.landmarks or self.session.fixed is None or self.session.moving is None:
            return
        sources = [pair for pair in self.study.pairs if pair.id != self._pair_id and pair.session.landmarks]
        if not sources:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Copy validation landmarks?")
        selector = hp.make_combobox(dlg, [pair.name for pair in sources])
        for index, pair in enumerate(sources):
            selector.setItemData(
                index, pair.session.moving.path if pair.session.moving else "", Qt.ItemDataRole.ToolTipRole
            )
            if previous is not None and pair.id == previous.id:
                selector.setCurrentIndex(index)
        hp.make_v_layout(
            hp.make_label(
                dlg,
                "This image pair has no landmarks. Copy existing pairs from another moving image? "
                "Coordinates stay in um and Use flags are preserved. Check that copied moving points identify "
                "the same tissue locations in this image; known fitting points and out-of-image points remain excluded.",
                wrap=True,
            ),
            selector,
            hp.make_h_layout(
                hp.make_btn(dlg, "Copy", func=dlg.accept),
                hp.make_btn(dlg, "Don't copy", func=dlg.reject),
            ),
            parent=dlg,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        source = sources[selector.currentIndex()]
        self._set_landmarks(
            [point.model_copy(deep=True, update={"id": uuid4().hex[:8]}) for point in source.session.landmarks],
            "Copy landmarks",
        )
        self._set_status(f"Copied {len(self.session.landmarks)} landmark pair(s) from {source.name}.")

    def _candidate_selected(self, item: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        if self._updating or item is None:
            return
        pair_id, candidate_id = item.data(Qt.ItemDataRole.UserRole)
        self._selected_candidates[pair_id] = candidate_id
        index = next(index for index, pair in enumerate(self.study.pairs) if pair.id == pair_id)
        if index != self._pair_index:
            self._select_pair(index)
            return
        self._ensure_transform()
        self._refresh_landmarks(sync_layers=False)
        if self.analysis is not None:
            selected = next((i for i, result in enumerate(self.analysis.candidates) if result.id == candidate_id), -1)
            self.result_candidate.setCurrentIndex(selected)

    def add_moving_image(self) -> None:
        """Load moving images using the same action as the Moving section's Load button."""
        self.choose_moving_images()

    def choose_moving_images(self) -> None:
        """Load one or more moving images, retaining existing pairs and their annotations."""
        if self._workers:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "Load moving images")
        if not paths:
            return
        pairs = list(self.study.pairs)
        selected: list[str] = []
        added: list[str] = []
        issues: list[str] = []
        previous = self.study.pairs[self._pair_index]
        for path in paths:
            try:
                image = inspect_validation_image(path)
            except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
                issues.append(f"{Path(path).name}: {exc}")
                continue
            existing = next(
                (
                    pair
                    for pair in pairs
                    if pair.session.moving is not None
                    and Path(pair.session.moving.path).resolve() == Path(image.path).resolve()
                    and pair.session.moving.reader_kws == image.reader_kws
                ),
                None,
            )
            if existing is not None:
                if existing.session.moving.fingerprint and existing.session.moving.fingerprint != image.fingerprint:
                    issues.append(f"{image.path} changed. Remove its image pair before loading it again.")
                    continue
                selected.append(existing.id)
                continue
            if not self._confirm_calibration(image):
                return  # Cancel aborts the whole batch; unreadable or changed images are skipped instead.
            empty_index = next(
                (i for i, pair in enumerate(pairs) if pair.session.moving is None and not pair.session.candidates), None
            )
            fixed = self.session.fixed.model_copy(deep=True) if self.session.fixed else None
            if empty_index is None:
                pair = ValidationPair(session=ValidationSession(fixed=fixed, moving=image))
                pairs.append(pair)
            else:
                pair = pairs[empty_index].model_copy(deep=True)
                pair.session.moving = image
                pairs[empty_index] = pair
            selected.append(pair.id)
            added.append(pair.id)
        if not selected:
            if issues:
                self._set_status("\n".join(issues), "No moving images loaded")
            return
        if added:
            self.study = ValidationStudy(pairs=pairs)
            self._generation += 1
            self._cancel.set()
            self._dirty = True
            for pair_id in added:
                self._drop_result(pair_id)
        index = next(i for i, pair in enumerate(self.study.pairs) if pair.id == selected[0])
        if index != self._pair_index:
            self._select_pair(index)
        else:
            self.session = self.study.pairs[index].session
            self._offer_landmark_copy(previous)
            self._refresh_inputs()
            self._load_views()
            self._display_analysis()
        reused = len(selected) - len(added)
        self._set_status(
            "\n".join(issues) if issues else "Existing image pairs and annotations are retained.",
            f"Loaded {len(added)} new moving image(s)"
            + (f"; selected {reused} already loaded" if reused else "")
            + (f"; skipped {len(issues)}" if issues else ""),
        )

    def remove_moving_image(self) -> None:
        """Remove the selected pair's annotations and results without changing its source files."""
        if self._workers or self.session.moving is None:
            return
        if not hp.confirm(self, "Remove this image pair and its validation annotations from the study?"):
            return
        self.undo_stack.clear()
        self.invalidate()
        index = self._pair_index
        if len(self.study.pairs) > 1:
            self.study.pairs = [pair for i, pair in enumerate(self.study.pairs) if i != index]
            self._pair_index = -1
            self._select_pair(min(index, len(self.study.pairs) - 1), offer_copy=False)
        else:
            self.session.moving = self.session.moving_mask = None
            self.session.candidates, self.session.landmarks, self.session.mask_pairs = [], [], []
            self._adding_pair, self._pending_pair = False, None
            self.edit_landmarks.setChecked(False)
            self._refresh_inputs()
            self._load_views()

    def remove_fixed_image(self) -> None:
        """Remove the shared fixed image and dependent evidence, keeping moving image files loaded."""
        if self._workers or self.session.fixed is None:
            return
        if not hp.confirm(
            self, "Remove the shared fixed image and its registrations and landmarks from all image pairs?"
        ):
            return
        self.undo_stack.clear()
        self.invalidate()
        self._clear_study_results()
        for pair in self.study.pairs:
            pair.session.fixed = pair.session.fixed_mask = None
            pair.session.candidates, pair.session.landmarks, pair.session.mask_pairs = [], [], []
        self._adding_pair, self._pending_pair = False, None
        self.edit_landmarks.setChecked(False)
        self._refresh_inputs()
        self._load_views()

    def edit_image_geometry(self, index: int) -> None:
        """Edit physical calibration and keep annotations anchored to their native image pixels."""
        image = self.session.fixed if index == 0 else self.session.moving
        if image is None or self._workers:
            return
        updated = image.model_copy(deep=True)
        if not self._confirm_calibration(updated, force=True):
            return
        if updated.geometry == image.geometry:
            return
        old = image.geometry.model_copy()
        pairs = self.study.pairs if index == 0 else [self.study.pairs[self._pair_index]]
        self.undo_stack.clear()
        self.invalidate()
        for pair in pairs:
            self._drop_result(pair.id)
            selected = pair.session.fixed if index == 0 else pair.session.moving
            if selected is None:
                continue
            for point in pair.session.landmarks:
                coordinates = point.fixed if index == 0 else point.moving
                physical = updated.geometry.to_physical(old.to_pixels(np.asarray(coordinates)))
                setattr(point, "fixed" if index == 0 else "moving", tuple(physical))
            mask = pair.session.fixed_mask if index == 0 else pair.session.moving_mask
            if mask is not None and mask.geometry == old:
                mask.geometry = updated.geometry.model_copy()
                mask.calibration_source, mask.calibration_confirmed = "user", True
            selected.geometry = updated.geometry.model_copy()
            selected.calibration_source, selected.calibration_confirmed = "user", True
        self._refresh_inputs()
        self._load_views()
        self._reset_native_views()

    def _refresh_inputs(self) -> None:
        self._updating = True
        try:
            self.pair_selector.clear()
            for pair in self.study.pairs:
                self.pair_selector.addItem(pair.name)
                if pair.session.moving:
                    self.pair_selector.setItemData(
                        self.pair_selector.count() - 1, pair.session.moving.path, Qt.ItemDataRole.ToolTipRole
                    )
            self.pair_selector.setCurrentIndex(self._pair_index)
            for index, image in enumerate((self.session.fixed, self.session.moving)):
                self.geometry_buttons[index].setEnabled(image is not None and not self._workers)
                self.remove_image_buttons[index].setEnabled(image is not None and not self._workers)
                self.image_labels[index].setText(
                    "No image"
                    if image is None
                    else f"{Path(image.path).name}\n"
                    f"Dimensions: {image.geometry.shape[1]} x {image.geometry.shape[0]}\n"
                    f"Extent XY: {image.geometry.shape[1] * image.geometry.spacing[0]:.6g} x "
                    f"{image.geometry.shape[0] * image.geometry.spacing[1]:.6g} um\n"
                    f"Spacing XY: {image.geometry.spacing} um; origin XY: {image.geometry.origin} um\n"
                    f"Calibration: {image.calibration_source}"
                )
                self.image_labels[index].setToolTip(image.path if image else "Choose a calibrated image.")
                self._refresh_channels(index, image)
            self.candidate_list.clear()
            for pair in self.study.pairs:
                for candidate in pair.session.candidates:
                    item = QListWidgetItem(f"{pair.name}: {candidate.name}", self.candidate_list)
                    item.setData(Qt.ItemDataRole.UserRole, (pair.id, candidate.id))
                    item.setToolTip(
                        f"{candidate.path}\nMoving: {pair.session.moving.path if pair.session.moving else ''}"
                    )
                    if pair.id == self._pair_id and candidate.id == self._selected_candidates.get(
                        pair.id, pair.session.candidates[0].id
                    ):
                        self.candidate_list.setCurrentItem(item)
            self.downsample.setValue(self.session.downsample)
            self.units.setCurrentText(self.session.landmark_units)
            self.auto_residual_map.setChecked(self.session.auto_residual_map)
            self.auto_tile.setValue(self.session.auto_tile_um or 0)
            self.grid_label.setText("The common analysis grid is capped at 4 million pixels.")
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
                    else f"{Path(mask.path).name}\nSpacing XY {mask.geometry.spacing}; origin XY {mask.geometry.origin} um"
                )
                self.mask_labels[index].setToolTip(mask.path if mask else "")
            self._set_message(
                self.evidence_label,
                (
                    f"{len(self.session.candidates)} registration(s); "
                    if self.session.candidates
                    else "Already aligned image comparison; "
                )
                + f"{sum(point.enabled for point in self.session.landmarks)} enabled validation landmarks; "
                f"{len(self.session.mask_pairs)} region correspondence(s). "
                "Analyze to calculate available fitting, validation, overlap, and deformation measurements.",
            )
        finally:
            self._updating = False
        self._refresh_landmarks()
        self._refresh_masks()

    def _refresh_channels(self, index: int, image: ValidationImage | None) -> None:
        combo = self.channels[index]
        with hp.qt_signals_blocked(combo):
            combo.clear()
            if image is not None:
                if image.is_rgb:
                    combo.addItem("RGB", "rgb")
                for channel, name in enumerate(image.channel_names or ["0"]):
                    combo.addItem(name, channel)
                display = image.display_channel if image.display_channel is not None else image.channel
                combo.setCurrentIndex(combo.findData(display))
                combo.setToolTip("Native display only. Registered overlap uses the saved scalar analysis channel.")
            combo.setEnabled(image is not None)

    def open_path(self, path: str | Path) -> None:
        """Open a saved study or import every compatible native registration path."""
        if str(path).endswith(".i2validate.json"):
            study, changes = load_validation_study(path)
            for pair in study.pairs:
                for image in (pair.session.fixed, pair.session.moving):
                    if image is not None and not self._confirm_calibration(image):
                        return
            self.invalidate()
            self._clear_study_results()
            self.study, self._pair_index = study, 0
            self.session = study.pairs[0].session
            self._selected_candidates.clear()
            self._session_path = Path(path)
            self._dirty = False
            self._refresh_inputs()
            self._load_views()
            self.undo_stack.clear()
            self._set_status("\n".join(changes) if changes else "Study loaded. Analyze to compute current results.")
            return
        issues: list[str] = []
        choices = discover_validation_candidates(path, errors=issues)
        if not choices:
            self._set_status("\n".join(issues) or "No registration paths were found.", "No registrations imported")
            return
        staged = self.study.model_copy(deep=True)
        reference = next((pair.session.fixed for pair in staged.pairs if pair.session.fixed), None)
        if reference is None:
            references: list[ValidationImage] = []
            for _candidate, fixed, _moving in choices:
                if not any(
                    Path(image.path).resolve() == Path(fixed.path).resolve()
                    and image.geometry == fixed.geometry
                    and image.reader_kws == fixed.reader_kws
                    for image in references
                ):
                    references.append(fixed)
            reference = references[0]
            if len(references) > 1:
                labels = [
                    f"{image.path}; spacing XY {image.geometry.spacing} um; origin XY {image.geometry.origin} um; "
                    f"reader {image.reader_kws}"
                    for image in references
                ]
                name, ok = QInputDialog.getItem(
                    self,
                    "Choose shared fixed reference",
                    "Import all paths using this reference",
                    labels,
                    editable=False,
                )
                if not ok:
                    return
                reference = references[labels.index(name)].model_copy(deep=True)
        if not self._confirm_calibration(reference):
            return
        added: list[tuple[str, str]] = []
        for candidate, fixed, moving in choices:
            adopt_validation_calibration(fixed, reference)
            for pair in staged.pairs:
                adopt_validation_calibration(moving, pair.session.moving)
            try:
                require_same_pair(
                    ValidationSession(fixed=reference, moving=moving), fixed, moving, allow_user_calibration=True
                )
            except ValueError:
                issues.append(f"{candidate.name}: incompatible fixed reference ({fixed.path}).")
                continue
            pair = next(
                (
                    pair
                    for pair in staged.pairs
                    if pair.session.moving is not None
                    and Path(pair.session.moving.path).resolve() == Path(moving.path).resolve()
                    and pair.session.moving.reader_kws == moving.reader_kws
                ),
                None,
            )
            if pair is not None and any(
                c.path == candidate.path and c.source == candidate.source for c in pair.session.candidates
            ):
                issues.append(f"{candidate.name}: already loaded; skipped duplicate.")
                continue
            if not self._confirm_calibration(moving):
                return
            if pair is None:
                pair = next(
                    (pair for pair in staged.pairs if pair.session.moving is None and not pair.session.candidates), None
                )
                if pair is None:
                    pair = ValidationPair()
                    staged.pairs.append(pair)
                pair.session.fixed = reference.model_copy(deep=True)
                pair.session.moving = moving
            elif pair.session.fixed is None:
                pair.session.fixed = reference.model_copy(deep=True)
            try:
                require_same_pair(pair.session, fixed, moving, allow_user_calibration=True)
            except ValueError as exc:
                issues.append(f"{candidate.name}: {exc}")
                continue
            pair.session.candidates = [*pair.session.candidates, candidate]
            added.append((pair.id, candidate.id))
        if added:
            for pair in staged.pairs:
                if pair.session.fixed is None:
                    pair.session.fixed = reference.model_copy(deep=True)
            # Validate the full staged import before changing the live study or its results.
            staged = ValidationStudy.model_validate(staged.model_dump())
            previous = self.study.pairs[self._pair_index]
            self._generation += 1
            self._cancel.set()
            for pair_id, candidate_id in added:
                self._drop_result(pair_id)
                self._selected_candidates[pair_id] = candidate_id
            self.study = staged
            target_id = previous.id if any(pair_id == previous.id for pair_id, _ in added) else added[0][0]
            self._pair_index = next(i for i, pair in enumerate(staged.pairs) if pair.id == target_id)
            self.session = staged.pairs[self._pair_index].session
            self._dirty = True
            self._adding_pair, self._pending_pair = False, None
            self.edit_landmarks.setChecked(False)
            if previous.id != self._pair_id:
                self._offer_landmark_copy(previous)
            self._refresh_inputs()
            self._load_views()
            self._display_analysis()
        self._set_status(
            "\n".join(issues)
            if issues
            else "Add independent landmarks or corresponding masks, then use Analyze pair or Analyze all.",
            f"Imported {len(added)} registration path(s); {len(issues)} skipped or failed",
        )

    def choose_register(self) -> None:
        """Choose a saved landmark registration."""
        path, _ = QFileDialog.getOpenFileName(self, "Add fiducial registration", "", "Register (*.i2r.json *.i2r.toml)")
        if path:
            self._try(partial(self.open_path, path))

    def choose_elastix(self) -> None:
        """Choose a completed Elastix project directory."""
        path = QFileDialog.getExistingDirectory(self, "Add completed image2image-reg project")
        if path:
            self._try(partial(self.open_path, path))

    def choose_session(self) -> None:
        """Replace the current study after preserving unsaved annotations."""
        if not self._may_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open validation study", "", "Validation (*.i2validate.json)")
        if path:
            self._try(partial(self.open_path, path))

    def choose_save(self) -> bool:
        """Save editable evidence and transform references, without embedding image data."""
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save validation study",
            str(self._session_path or "validation-study.i2validate.json"),
            "Validation (*.i2validate.json)",
        )
        if not path:
            return False
        saved = self._try(partial(save_validation_study, path, self.study))
        if saved:
            self._session_path, self._dirty = saved, False
            self._set_status(f"Saved {saved}")
        return bool(saved)

    def _may_discard(self) -> bool:
        if not self._dirty:
            return True
        choice = QMessageBox.question(
            self,
            "Unsaved validation annotations",
            "Save this study before closing it?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
        )
        return choice == QMessageBox.StandardButton.Discard or (
            choice == QMessageBox.StandardButton.Save and self.choose_save()
        )

    def new_session(self) -> None:
        """Start a fresh comparison without silently discarding annotations."""
        if self._may_discard():
            self.invalidate()
            self._clear_study_results()
            self.study, self._pair_index = ValidationStudy(), 0
            self.session = self.study.pairs[0].session
            self._display_states.clear()
            self._camera_states.clear()
            self._selected_candidates.clear()
            self._session_path = None
            self._dirty = False
            self.undo_stack.clear()
            self._refresh_inputs()
            self._load_views()

    def choose_image(self, index: int) -> None:
        """Choose the fixed reference or load additional moving images into separate pairs."""
        if index == 1:
            self.choose_moving_images()
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose fixed image")
        if not path:
            return
        image = self._try(partial(inspect_validation_image, path))
        if not image or not self._confirm_calibration(image):
            return
        existing = self.session.fixed
        changed = existing is not None and (
            Path(existing.path).resolve() != Path(image.path).resolve()
            or existing.geometry != image.geometry
            or existing.reader_kws != image.reader_kws
        )
        if changed and (self.session.candidates or len(self.study.pairs) > 1):
            self._set_status("The image must match its registration and reference. Start a new pair or study.")
            return
        if changed and (self.session.landmarks or self.session.fixed_mask):
            if not hp.confirm(
                self, "Replacing the fixed image removes landmarks and the fixed mask placed on the previous image."
            ):
                return
            self.undo_stack.clear()
            self.session.landmarks, self.session.mask_pairs, self.session.fixed_mask = [], [], None
        self.invalidate()
        self.session.fixed = image
        for pair in self.study.pairs:
            if pair.session.fixed is None:
                pair.session.fixed = image.model_copy(deep=True)
        self._refresh_inputs()
        self._load_views()

    def remove_candidate(self) -> None:
        """Remove the selected registration without touching its source files."""
        item = self.candidate_list.currentItem()
        if item is not None:
            _pair_id, candidate_id = item.data(Qt.ItemDataRole.UserRole)
            if not hp.confirm(self, f"Remove {item.text()} from the study? Its files are not changed."):
                return
            self.invalidate()
            self.session.candidates = [c for c in self.session.candidates if c.id != candidate_id]
            self._selected_candidates.pop(self._pair_id, None)
            self._refresh_inputs()
            self._ensure_transform()
            self._refresh_landmarks(sync_layers=False)

    def _channel_changed(self, index: int, channel: int) -> None:
        if self._updating or channel < 0:
            return
        image = self.session.fixed if index == 0 else self.session.moving
        if image:
            image.display_channel = self.channels[index].itemData(channel)
            self._dirty = True
            self._load_views()
            self._refresh_inputs()
            if self.analysis is not None and self.result_mode.currentIndex() == 1:
                camera = self.result_view.viewer.camera
                center, zoom = tuple(camera.center), camera.zoom
                self.show_result()
                camera.center, camera.zoom = center, zoom

    def _resolution_changed(self, value: int) -> None:
        if not self._updating:
            self.session.downsample = value
            self.invalidate()
            self._refresh_inputs()

    def _automatic_changed(self, _value: Any = None) -> None:
        """Apply automatic-evidence settings to every image pair, so study results stay comparable."""
        if self._updating:
            return
        settings = {
            "auto_residual_map": self.auto_residual_map.isChecked(),
            "auto_tile_um": self.auto_tile.value() or None,
        }
        for pair in self.study.pairs:
            for key, value in settings.items():
                setattr(pair.session, key, value)
            if pair.id != self._pair_id:
                self._drop_result(pair.id)
        self.invalidate()

    def _units_changed(self, value: str) -> None:
        if not self._updating:
            self.session.landmark_units = value
            self._dirty = True

    def _load_views(self) -> None:
        cameras = []
        previous_pair = next((key[0] for key in self._view_keys if key is not None), None)
        for index, view in enumerate(self.views):
            cameras.append((tuple(view.viewer.camera.center), view.viewer.camera.zoom))
            if self._view_keys[index] is not None and "Image" in view.viewer.layers:
                layer = view.viewer.layers["Image"]
                self._display_states[self._view_keys[index]] = {
                    "contrast_limits": tuple(layer.contrast_limits),
                    "gamma": layer.gamma,
                    "colormap": layer.colormap,
                    "opacity": layer.opacity,
                }
        if previous_pair is not None:
            self._camera_states[previous_pair] = cameras
        self._updating = True
        try:
            for view in self.views:
                view.viewer.layers.clear()
            self.point_layers.clear()
            for index, image in enumerate((self.session.fixed, self.session.moving)):
                view = self.views[index]
                key = None
                if image and Path(image.path).exists():
                    key = (
                        self._pair_id,
                        index,
                        image.path,
                        image.display_channel if image.display_channel is not None else image.channel,
                        image.fingerprint,
                        image.geometry.spacing,
                        image.geometry.origin,
                        image.geometry.shape,
                        json.dumps(image.reader_kws, sort_keys=True),
                    )
                    previous = self._view_keys[index]
                    if self._readers[index] is not None and (
                        previous is None
                        or previous[2] != image.path
                        or previous[4] != image.fingerprint
                        or previous[-2:] != key[-2:]
                    ):
                        self._readers[index].close()
                        self._readers[index] = None
                    reader = self._readers[index]
                    if reader is None:
                        reader = open_validation_reader(image)
                        self._readers[index] = reader
                    image.is_rgb = bool(reader.is_rgb)
                    if image.is_rgb:
                        image.channel_names = ["R", "G", "B"]
                    rgb = key[3] == "rgb" and image.is_rgb
                    arrays = validation_image_pyramid(reader, image.channel if rgb else int(key[3]), rgb=rgb)
                    limits = validation_contrast_limits(arrays)
                    settings = self._display_states.get(key)
                    if (
                        settings is None
                        and previous is not None
                        and previous[:5] == key[:5]
                        and previous[-2:] == key[-2:]
                    ):
                        settings = self._display_states.get(previous)
                    settings = settings or {"contrast_limits": limits}
                    layer = view.viewer.add_image(
                        arrays if len(arrays) > 1 else arrays[0],
                        multiscale=len(arrays) > 1,
                        rgb=rgb,
                        name="Image",
                        scale=image.geometry.spacing[::-1],
                        translate=image.geometry.origin[::-1],
                        units=("um", "um"),
                        **settings,
                    )
                    self._refresh_channels(index, image)
                elif self._readers[index] is not None:
                    self._readers[index].close()
                    self._readers[index] = None
                self._view_keys[index] = key
                layer = view.viewer.add_points(
                    np.empty((0, 2)),
                    name="Validation landmarks",
                    size=6 * max(image.geometry.spacing) if image else 6,
                    canvas_size_limits=(6, 40),
                    units=("um", "um"),
                )
                layer.mode = "select"
                layer.events.data.connect(partial(self._points_moved, index))
                layer.selected_data.events.active.connect(partial(self._point_selected, index))
                self.point_layers.append(layer)
                view.viewer.reset_view()
        finally:
            self._updating = False
        self._clear_pending()
        self._refresh_landmarks()
        self._load_mask_overlays()
        self._landmark_edit_mode()
        self._ensure_transform()
        if self._pair_id in self._camera_states:
            self._syncing = True
            try:
                for view, (center, zoom) in zip(self.views, self._camera_states[self._pair_id], strict=True):
                    view.viewer.camera.center, view.viewer.camera.zoom = center, zoom
            finally:
                self._syncing = False
            self._link_views_changed()
        else:
            QTimer.singleShot(0, self._reset_native_views)

    def _load_mask_overlays(self) -> None:
        for index, mask in enumerate((self.session.fixed_mask, self.session.moving_mask)):
            viewer = self.views[index].viewer
            if "Region labels" in viewer.layers:
                viewer.layers.remove("Region labels")
            if self._mask_readers[index] is not None:
                self._mask_readers[index].close()
                self._mask_readers[index] = None
            if mask is not None and Path(mask.path).exists():
                try:
                    reader = open_validation_reader(mask)
                    self._mask_readers[index] = reader
                    levels = validation_image_pyramid(reader, mask.channel, labels=True)
                except (OSError, ValueError) as exc:
                    self._set_status(f"Cannot display mask: {exc}")
                    continue
                viewer.add_labels(
                    levels if len(levels) > 1 else levels[0],
                    multiscale=len(levels) > 1,
                    name="Region labels",
                    opacity=0.35,
                    scale=mask.geometry.spacing[::-1],
                    translate=mask.geometry.origin[::-1],
                    units=("um", "um"),
                    visible=self.tabs.currentIndex() == 2,
                )
            self._select_native_layer(index)

    def _reset_native_views(self) -> None:
        if self._closing:
            return
        self._syncing = True
        try:
            for view in self.views:
                view.viewer.reset_view()
            zoom = min(view.viewer.camera.zoom for view in self.views)
            images = (self.session.fixed, self.session.moving)
            loaded = [
                index for index, image in enumerate(images) if image is not None and self._readers[index] is not None
            ]
            if loaded:
                zoom = min(self.views[index].viewer.camera.zoom for index in loaded)
            center = self.views[loaded[0] if loaded else 0].viewer.camera.center
            for view in self.views:
                view.viewer.camera.zoom = zoom
                if self.link_views.isChecked():
                    view.viewer.camera.center = center
        finally:
            self._syncing = False
        if self._active_candidate() is not None:
            self._link_views_changed()

    def _set_landmarks(self, landmarks: list[ValidationLandmark], text: str, sync_layers: bool = True) -> None:
        """Replace the selected pair's landmarks through the undo stack."""
        self.undo_stack.push(LandmarkCommand(self, self._pair_id, self.session.landmarks, landmarks, text, sync_layers))

    def _apply_landmarks(self, pair_id: str, landmarks: list[ValidationLandmark], sync_layers: bool = True) -> None:
        pair = next((pair for pair in self.study.pairs if pair.id == pair_id), None)
        if pair is None:
            return
        pair.session.landmarks = [point.model_copy() for point in landmarks]
        if pair_id == self._pair_id:
            self.invalidate()
            self._refresh_landmarks(sync_layers)
        else:
            self._dirty = True
            self._drop_result(pair_id)

    def _live_errors(self) -> dict[str, dict[str, Any]]:
        """Measure landmark residuals with the cached registration, without dense maps."""
        transform = self._current_transform()
        if transform is None or not self.session.landmarks or self.session.fixed is None or not self.session.moving:
            return {}
        try:
            exclusions = landmark_exclusions(self.session, [transform])
            rows, _ = landmark_errors(
                transform, self.session.landmarks, exclusions, landmark_warnings(self.session, [transform])
            )
        except (ValueError, RuntimeError):
            return {}
        return {row["id"]: row for row in rows}

    def _refresh_landmarks(self, sync_layers: bool = True) -> None:
        errors = self._live_errors()
        self._updating = True
        try:
            self.landmark_table.setRowCount(len(self.session.landmarks))
            for row, point in enumerate(self.session.landmarks):
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
                use.setCheckState(Qt.CheckState.Checked if point.enabled else Qt.CheckState.Unchecked)
                self.landmark_table.setItem(row, 0, use)
                self.landmark_table.setItem(row, 1, QTableWidgetItem(point.id))
                for col, value in enumerate((*point.fixed, *point.moving), 2):
                    item = QTableWidgetItem(f"{value:.4f}")
                    item.setData(Qt.ItemDataRole.UserRole, float(value))
                    item.setToolTip(repr(float(value)))
                    self.landmark_table.setItem(row, col, item)
                self.landmark_table.setItem(row, 6, QTableWidgetItem(point.note))
                live = errors.get(point.id, {})
                source = point.source + (f" ({point.score:.2f})" if point.score is not None else "")
                error = (
                    f"{live['error_um']:.4g}"
                    if live.get("error_um") is not None
                    else live.get("excluded", "") or ("" if errors else "No registration loaded")
                )
                for col, value in ((7, source), (8, error)):
                    item = QTableWidgetItem(value)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    item.setToolTip(live.get("warning", ""))
                    self.landmark_table.setItem(row, col, item)
            for index, layer in enumerate(self.point_layers if sync_layers else []):
                layer.data = np.asarray(
                    [p.fixed[::-1] if index == 0 else p.moving[::-1] for p in self.session.landmarks]
                ).reshape(-1, 2)
            for index, layer in enumerate(self.point_layers):
                layer.face_color = [
                    ("cyan" if p.source == "automatic" else "lime") if p.enabled else "gray"
                    for p in self.session.landmarks
                ] or "lime"
                labels = [p.id for p in self.session.landmarks]
                if index == 1 and errors:
                    labels = [
                        f"{p.id}: {errors[p.id]['error_um']:.3g} um"
                        if errors.get(p.id, {}).get("error_um") is not None
                        else p.id
                        for p in self.session.landmarks
                    ]
                layer.text = {"string": labels, "color": "white", "size": 10}
        finally:
            self._updating = False

    def _landmark_edited(self, item: QTableWidgetItem) -> None:
        if self._updating:
            return
        row, col = item.row(), item.column()
        if not 0 <= row < len(self.session.landmarks) or col > 6:
            return
        point = self.session.landmarks[row]
        try:
            if col == 0:
                update: dict[str, Any] = {"enabled": item.checkState() == Qt.CheckState.Checked}
            elif col == 1:
                update = {"id": item.text().strip()}
            elif col == 6:
                update = {"note": item.text()}
            else:
                value = float(item.text())
                target = "fixed" if col < 4 else "moving"
                coordinates = list(getattr(point, target))
                coordinates[(col - 2) % 2] = value
                update = {target: tuple(coordinates)}
            edited = ValidationLandmark.model_validate({**point.model_dump(), **update})
            landmarks = [edited if i == row else p for i, p in enumerate(self.session.landmarks)]
            ValidationSession(landmarks=landmarks)  # check identifiers before recording the edit
        except ValueError as exc:
            self._set_status(f"Invalid landmark edit: {exc}")
            self._refresh_landmarks()
            return
        self._set_landmarks(landmarks, "Edit landmark")

    def _points_moved(self, index: int, _event: Any = None) -> None:
        if self._updating:
            return
        action = getattr(_event, "action", None)
        if action in ("changing", "adding", "removing"):
            return
        data = self.point_layers[index].data
        if action == "removed":
            removed = {int(i) for i in getattr(_event, "data_indices", ())}
            if len(data) + len(removed) == len(self.session.landmarks):
                landmarks = [p for i, p in enumerate(self.session.landmarks) if i not in removed]
                self._set_landmarks(landmarks, "Delete landmark")
                return
        if len(data) != len(self.session.landmarks):
            if len(data) > len(self.session.landmarks):
                self._set_status("Use Add landmark to add points to both images.")
            self._refresh_landmarks()
            return
        pairs = [p.model_copy() for p in self.session.landmarks]
        for point, yx in zip(pairs, data, strict=True):
            setattr(point, "fixed" if index == 0 else "moving", tuple(float(v) for v in yx[::-1]))
        self._set_landmarks(pairs, "Move landmark", sync_layers=False)

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
                coordinates = (point.fixed, point.moving)
                for view, layer, xy in zip(self.views, self.point_layers, coordinates, strict=True):
                    view.viewer.camera.center = (0, xy[1], xy[0])
                    layer.selected_data = {row}
            finally:
                self._updating = False

    def _active_candidate(self) -> ValidationCandidate | None:
        candidates = self.session.candidates
        if not candidates:
            return None
        selected = self._selected_candidates.get(self._pair_id)
        return next((c for c in candidates if c.id == selected), candidates[0])

    def _transform_key(self) -> tuple | None:
        if self.session.fixed is None or self.session.moving is None:
            return None
        candidate = self._active_candidate() or IDENTITY
        return (
            self._pair_id,
            candidate.id,
            candidate.fingerprint,
            self.session.fixed.geometry.model_dump_json(),
            self.session.moving.geometry.model_dump_json(),
        )

    def _current_transform(self) -> ValidationTransform | None:
        """Return the selected registration for navigation and live errors, if loaded."""
        key = self._transform_key()
        if key is None:
            return None
        if key[1] == IDENTITY.id:
            return self._transforms.setdefault(key, ValidationTransform(IDENTITY, matrix=np.eye(3)))
        return self._transforms.get(key)

    def _ensure_transform(self) -> None:
        """Load the selected registration once in the background; views stay physically linked meanwhile."""
        key = self._transform_key()
        candidate = self._active_candidate()
        if key is None or candidate is None or key in self._transforms or key in self._transform_errors:
            self._update_link_label()
            return
        if any(getattr(worker, "_validation_key", None) == key for worker in self._transform_workers):
            return
        session = self.session.model_copy(deep=True)
        worker = create_worker(
            load_validation_transform, candidate.model_copy(), session, _start_thread=False, _ignore_errors=True
        )
        worker._validation_key = key
        worker.returned.connect(partial(self._transform_loaded, key))
        worker.errored.connect(partial(self._transform_failed, key))
        worker.finished.connect(partial(self._worker_finished, self._transform_workers, worker))
        self._transform_workers.append(worker)
        self._update_link_label()
        worker.start()

    def _transform_loaded(self, key: tuple, transform: ValidationTransform) -> None:
        if self._closing:
            return
        self._transforms[key] = transform
        if key == self._transform_key():
            self._update_link_label()
            self._refresh_landmarks(sync_layers=False)
            self._link_views_changed()

    def _transform_failed(self, key: tuple, error: Exception) -> None:
        if self._closing:
            return
        self._transform_errors[key] = str(error)
        if key == self._transform_key():
            self._update_link_label()
            self._refresh_landmarks(sync_layers=False)

    def _worker_finished(self, workers: list[Any], worker: Any) -> None:
        if worker in workers:
            workers.remove(worker)
        if self._closing and not self._workers and not self._transform_workers and not self._assist_workers:
            self.close()

    def _update_link_label(self) -> None:
        candidate = self._active_candidate()
        key = self._transform_key()
        if candidate is None:
            text = "Link views in physical coordinates"
        elif key in self._transforms:
            text = f"Link views through {candidate.name}"
        elif key in self._transform_errors:
            text = "Link views in physical coordinates (registration unavailable)"
            self.link_views.setToolTip(self._transform_errors[key])
        else:
            text = f"Link views through {candidate.name} (loading...)"
        self.link_views.setText(text)

    def _linked_camera(self, index: int) -> tuple[tuple[float, float], float] | None:
        """Map one view's center and zoom to the other view through the selected registration."""
        source = self.views[index].viewer.camera
        xy = np.array([source.center[-1], source.center[-2]], dtype=float)
        transform = self._current_transform() if self._active_candidate() is not None else None
        if transform is None:
            return (float(xy[0]), float(xy[1])), source.zoom
        try:
            if index == 0:
                fixed_xy, target = xy, transform.map_points(xy[None])[0]
            else:
                initial = np.array(self.views[0].viewer.camera.center[-1:-3:-1], dtype=float)
                fixed_xy = target = transform.inverse_points(xy[None], initial[None])[0]
            step = 4 * max(self.session.fixed.geometry.spacing)
            scale = float(np.sqrt(abs(np.linalg.det(transform.jacobian_at(fixed_xy[None], step)[0]))))
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            return None
        if not np.isfinite(target).all() or not np.isfinite(scale) or scale <= 0:
            return None
        scale = 1.0 if abs(scale - 1) < 1e-6 else scale
        zoom = source.zoom / scale if index == 0 else source.zoom * scale
        return (float(target[0]), float(target[1])), zoom

    def _sync_zoom(self, index: int, _event: Any = None) -> None:
        if self._syncing or self._updating or len(self.views) != 2 or not self.link_views.isChecked():
            return
        linked = self._linked_camera(index)
        if linked is None:
            return
        (x, y), zoom = linked
        self._syncing = True
        try:
            camera = self.views[1 - index].viewer.camera
            camera.zoom = zoom
            camera.center = (*camera.center[:-2], y, x)
        finally:
            self._syncing = False

    def _link_views_changed(self, _state: int = 0) -> None:
        if self.link_views.isChecked():
            self._sync_zoom(0 if self.session.fixed is not None else 1)

    def _select_native_layer(self, index: int, _event: Any = None) -> None:
        if self._updating or self._selecting_layer or index >= len(self.point_layers):
            return
        viewer = self.views[index].viewer
        editing = self._adding_pair or self.edit_landmarks.isChecked()
        layer = self.point_layers[index] if editing else viewer.layers["Image"] if "Image" in viewer.layers else None  # noqa: SIM401 - LayerList is not a mapping.
        if viewer.layers.selection.active is not layer:
            self._selecting_layer = True
            try:
                viewer.layers.selection.active = layer
            finally:
                self._selecting_layer = False

    def _landmark_edit_mode(self, _state: int = 0) -> None:
        if not self.edit_landmarks.isChecked():
            self._adding_pair = False
            self._clear_pending()
        for layer in self.point_layers:
            layer.mode = "pan_zoom" if self._adding_pair else "select"
        if hasattr(self, "_cancel_shortcut"):
            self._cancel_shortcut.setEnabled(self._adding_pair)
        for index in range(len(self.views)):
            self._select_native_layer(index)

    def _show_pending(self, index: int, xy: tuple[float, float] | None, name: str = "Pending landmark") -> None:
        viewer = self.views[index].viewer
        if name in viewer.layers:
            viewer.layers.remove(name)
        if xy is None:
            return
        image = self.session.fixed if index == 0 else self.session.moving
        self._updating = True
        try:
            viewer.add_points(
                np.array([xy[::-1]]),
                name=name,
                size=6 * max(image.geometry.spacing) if image else 6,
                canvas_size_limits=(8, 40),
                face_color="transparent",
                border_color="yellow" if name == "Pending landmark" else "cyan",
                border_width=0.3,
                units=("um", "um"),
            )
        finally:
            self._updating = False
        self._select_native_layer(index)

    def _clear_pending(self) -> None:
        self._pending_pair, self._suggestion = None, None
        self._assist_generation += 1
        for shortcut in getattr(self, "_accept_shortcuts", []):
            shortcut.setEnabled(False)
        for index in range(len(self.views)):
            for name in ("Pending landmark", "Suggested landmark"):
                if name in self.views[index].viewer.layers:
                    self.views[index].viewer.layers.remove(name)

    def _cursor_changed(self, index: int | None, _event: Any = None) -> None:
        view = self.result_view if index is None else self.views[index]
        label = self.result_coordinates if index is None else self.coordinate_labels[index]
        y, x = view.viewer.cursor.position[-2:]
        label.setText(f"X: {x:.6g} um, Y: {y:.6g} um")

    def begin_pair(self) -> None:
        """Start adding landmarks; no transformed location is inserted as evidence without confirmation."""
        if not self.session.fixed or not self.session.moving:
            self._set_status("Load both images before annotating.")
            return
        self._adding_pair = True
        self._clear_pending()
        self.tabs.setCurrentIndex(1)
        self.edit_landmarks.setChecked(True)
        self._landmark_edit_mode()
        self._set_status(
            "Click the independent landmark in the fixed image, then its counterpart in moving. Press Esc to stop."
        )

    def cancel_pair(self) -> None:
        """Stop adding landmarks and discard an incomplete pair."""
        if self._adding_pair:
            self._adding_pair = False
            self.edit_landmarks.setChecked(False)
            self._landmark_edit_mode()
            self._set_status("Stopped adding landmarks.")

    def _pair_click(self, index: int, _viewer: Any, event: Any) -> None:
        if not self._adding_pair or event.type != "mouse_press" or event.button != 1:
            return
        xy = tuple(float(v) for v in event.position[-2:][::-1])
        image = self.session.fixed if index == 0 else self.session.moving
        if image is None or not image.geometry.contains(np.array(xy)):
            self._set_status("Click inside the image to place a landmark.")
            return
        if index == 0:
            self._clear_pending()
            self._pending_pair = xy
            self._show_pending(0, xy)
            if self.assist.isChecked():
                self._suggest(xy)
            else:
                self._set_status("Now click the corresponding landmark in the moving image.")
        elif self._pending_pair is not None:
            self._add_landmark(ValidationLandmark(fixed=self._pending_pair, moving=xy))
        else:
            self._set_status("Click the fixed image first.")

    def _add_landmark(self, landmark: ValidationLandmark) -> None:
        self._clear_pending()
        self._set_landmarks([*self.session.landmarks, landmark], "Add landmark")
        self.landmark_table.selectRow(len(self.session.landmarks) - 1)
        self._landmark_edit_mode()
        self._set_status(
            f"Landmark {landmark.id} added ({landmark.source}). Click the next fixed landmark or press Esc."
        )

    def _assist_sizes(self) -> tuple[float, float]:
        spacing = max(self.session.fixed.geometry.spacing)
        return (
            self.assist_template.value() or 48 * spacing,
            self.assist_search.value() or 24 * spacing,
        )

    def _suggest(self, fixed_xy: tuple[float, float]) -> None:
        """Search the moving image for the fixed content near the registration's prediction."""
        transform = self._current_transform()
        if transform is None:
            self._set_status("The registration is still loading; click the moving image to place the point.")
            return
        template_um, search_um = self._assist_sizes()
        fixed, moving = self.session.fixed.model_copy(), self.session.moving.model_copy()
        generation = self._assist_generation

        def run() -> TemplateMatch:
            readers = [open_validation_reader(fixed), open_validation_reader(moving)]
            try:
                return match_through_transform(
                    readers[0].get_channel(fixed.channel, split_rgb=True),
                    fixed.geometry,
                    readers[1].get_channel(moving.channel, split_rgb=True),
                    moving.geometry,
                    transform,
                    fixed_xy,
                    template_um,
                    search_um,
                )
            finally:
                for reader in readers:
                    reader.close()

        worker = create_worker(run, _start_thread=False, _ignore_errors=True)
        worker.returned.connect(partial(self._suggestion_ready, generation))
        worker.errored.connect(lambda exc: self._set_status(f"Suggestion failed: {exc}"))
        worker.finished.connect(partial(self._worker_finished, self._assist_workers, worker))
        self._assist_workers.append(worker)
        self._set_status("Searching the moving image for the clicked content...")
        worker.start()

    def _suggestion_ready(self, generation: int, match: TemplateMatch) -> None:
        if generation != self._assist_generation or self._closing or self._pending_pair is None:
            return
        if not match.ok or match.reason or match.score is None or match.score < 0.5:
            reason = match.reason or (f"correlation {match.score:.2f} is below 0.5" if match.score is not None else "")
            self._set_status(f"No reliable suggestion ({reason}). Click the moving image to place the point.")
            return
        self._suggestion = match
        for shortcut in self._accept_shortcuts:
            shortcut.setEnabled(True)
        self._show_pending(1, match.moving_xy, "Suggested landmark")
        self._updating = True
        try:
            self.views[1].viewer.camera.center = (0, match.moving_xy[1], match.moving_xy[0])
        finally:
            self._updating = False
        self._set_status(
            f"Suggested moving point, correlation {match.score:.2f}. Press Enter to accept, "
            "or click the moving image to place it yourself."
        )

    def accept_suggestion(self) -> None:
        """Store the template-matching suggestion as an assisted landmark."""
        if self._suggestion is None or self._pending_pair is None:
            return
        match = self._suggestion
        self._add_landmark(
            ValidationLandmark(fixed=self._pending_pair, moving=match.moving_xy, source="assisted", score=match.score)
        )

    def detect_landmarks(self) -> None:
        """Propose automatic landmarks through the selected registration; they are added for review."""
        if not self.session.fixed or not self.session.moving:
            self._set_status("Load both images before detecting landmarks.")
            return
        if self._assist_workers:
            return
        transform = self._current_transform()
        if transform is None:
            self._set_status("The registration is still loading. Try again in a moment.")
            return
        from image2image.utils.validation_auto import detect_landmarks

        template_um, search_um = self._assist_sizes()
        session, pair_id = self.session.model_copy(deep=True), self._pair_id
        self._detect_cancel = Event()
        worker = create_worker(
            detect_landmarks,
            session,
            transform,
            max_points=self.detect_count.value(),
            template_um=template_um,
            search_um=search_um,
            cancel=self._detect_cancel,
            progress=lambda text: self.progress_changed.emit(self._generation, text),
            _start_thread=False,
            _ignore_errors=True,
        )
        worker.returned.connect(partial(self._landmarks_detected, pair_id, session.landmarks))
        worker.errored.connect(
            lambda exc: self._set_status(
                "Detection cancelled." if isinstance(exc, AnalysisCancelled) else f"Detection failed: {exc}"
            )
        )
        worker.finished.connect(partial(self._worker_finished, self._assist_workers, worker))
        worker.finished.connect(lambda: self.detect_btn.setEnabled(True))
        self._assist_workers.append(worker)
        self.detect_btn.setEnabled(False)
        self._set_status("Detecting landmarks...")
        worker.start()

    def _landmarks_detected(
        self, pair_id: str, before: list[ValidationLandmark], result: tuple[list[ValidationLandmark], dict]
    ) -> None:
        if self._closing:
            return
        landmarks, summary = result
        if pair_id != self._pair_id or [p.id for p in self.session.landmarks] != [p.id for p in before]:
            self._set_status("Landmarks changed during detection; detect again to add automatic landmarks.")
            return
        if not landmarks:
            self._set_status(summary.get("reason") or "No reliable matches were found.", "No landmarks detected")
            return
        self._set_landmarks([*self.session.landmarks, *landmarks], "Detect landmarks")
        outliers = sum(not point.enabled for point in landmarks)
        self._set_status(
            "Review them in the table and viewers: Del removes selected rows and Ctrl+Z removes them all. "
            "Automatic landmarks are reported separately from manual and assisted ones.",
            f"Added {len(landmarks)} automatic landmarks ({outliers} outliers unticked)",
        )

    def remove_automatic(self) -> None:
        """Delete every automatic landmark of the selected image pair."""
        kept = [point for point in self.session.landmarks if point.source != "automatic"]
        if len(kept) != len(self.session.landmarks):
            self._set_landmarks(kept, "Remove automatic landmarks")

    def delete_pair(self) -> None:
        """Remove the selected landmarks from both native views."""
        rows = {index.row() for index in self.landmark_table.selectionModel().selectedRows()}
        if not rows and self.landmark_table.currentRow() >= 0:
            rows = {self.landmark_table.currentRow()}
        if rows:
            landmarks = [p for i, p in enumerate(self.session.landmarks) if i not in rows]
            self._set_landmarks(landmarks, "Delete landmark" if len(rows) == 1 else "Delete landmarks")

    def choose_landmarks(self) -> None:
        """Import paired CSV annotations, retaining existing pairs with distinct IDs."""
        path, _ = QFileDialog.getOpenFileName(self, "Import independent landmarks", "", "Landmarks (*.csv)")
        if path:

            def load() -> None:
                points = import_validation_landmarks(path, self.session, self.session.landmark_units)
                landmarks, renames = merge_landmarks(self.session.landmarks, points)
                self._set_landmarks(landmarks, "Import landmarks")
                message = f"Imported {len(points)} landmark(s)."
                if renames:
                    message += " Renamed duplicate IDs: " + ", ".join(f"{a} to {b}" for a, b in renames.items()) + "."
                self._set_status(message)

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
        if self._confirm_calibration(mask, force=True):
            self._try(partial(self.set_mask, index, mask))

    def _confirm_calibration(self, image: ValidationImage, force: bool = False) -> bool:
        if image.calibration_confirmed and not force:
            return True
        dlg = QDialog(self)
        dlg.setWindowTitle("Confirm physical calibration")
        form = hp.make_form_layout(parent=dlg)
        form.addRow(
            hp.make_label(dlg, f"{Path(image.path).name}\nConfirm the native physical geometry in um.", wrap=True)
        )
        controls = []
        for name, value in zip(
            ("Spacing X (um)", "Spacing Y (um)", "Origin X (um)", "Origin Y (um)"),
            (*image.geometry.spacing, *image.geometry.origin),
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
            return False
        values = [c.value() for c in controls]
        image.geometry = ImageGeometry(shape=image.geometry.shape, spacing=tuple(values[:2]), origin=tuple(values[2:]))
        image.calibration_source, image.calibration_confirmed = "user", True
        return True

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
        others = [pair for pair in self.study.pairs if pair.id != self._pair_id]
        if index == 0 and others and hp.confirm(self, "Use this fixed mask for every image pair?"):
            for pair in others:
                pair.session.fixed_mask = mask.model_copy(deep=True)
                self._drop_result(pair.id)
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
        self._load_mask_overlays()
        self._set_status(
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
            self._set_status(f"Invalid label pair: {exc}")
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
        if (self.session.fixed_mask or self.session.moving_mask) and not hp.confirm(
            self, "Clear both masks and their label pairs for this image pair? Mask files are not changed."
        ):
            return
        self.session.fixed_mask = self.session.moving_mask = None
        self.session.mask_pairs = []
        self.invalidate()
        self._refresh_inputs()
        self._load_mask_overlays()

    def start_analysis(self) -> None:
        """Analyze a frozen input snapshot outside the Qt thread."""
        self._start_analysis(all_pairs=False)

    def start_analysis_all(self) -> None:
        """Analyze every moving image sequentially with independent pair outcomes."""
        self._start_analysis(all_pairs=True)

    def _start_analysis(self, all_pairs: bool) -> None:
        if self._workers:
            return
        pairs = self.study.pairs if all_pairs else [self.study.pairs[self._pair_index]]
        force = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
        if all_pairs and not force:
            pairs = [pair for pair in pairs if pair.id not in self._results]
            if not pairs:
                self._set_status("Every image pair has current results. Hold Shift and click Analyze all to repeat.")
                return
        for pair in pairs:
            for image in (pair.session.fixed, pair.session.moving):
                if image is not None and not self._confirm_calibration(image):
                    return
        self._generation += 1
        for pair in pairs:
            self._drop_result(pair.id)
        self._clear_result_widgets("Analyzing available evidence...")
        generation = self._generation
        self._cancel = Event()
        directory = TemporaryDirectory(prefix="i2validate-")
        snapshot = (
            ValidationStudy(pairs=[pair.model_copy(deep=True) for pair in pairs])
            if all_pairs
            else self.session.model_copy(deep=True)
        )
        worker = create_worker(
            analyze_validation_study if all_pairs else analyze_validation_session,
            snapshot,
            directory.name,
            self._cancel,
            lambda text: self.progress_changed.emit(generation, text),
            _start_thread=False,
            _ignore_errors=True,
        )
        if all_pairs:
            worker.returned.connect(partial(self.accept_study_analysis, generation, directory))
        else:
            worker.returned.connect(partial(self.accept_analysis, generation, directory, pair_id=self._pair_id))
        worker.errored.connect(partial(self._analysis_error, generation, pair_id=None if all_pairs else self._pair_id))
        worker.finished.connect(partial(self._analysis_finished, worker, directory))
        self._workers.append(worker)
        self.run_btn.setEnabled(False)
        self.run_all_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        for button in (*self.geometry_buttons, *self.remove_image_buttons):
            button.setEnabled(False)
        self._set_status(
            "Loading registration definitions..." if self.session.candidates else "Analyzing aligned images..."
        )
        worker.start()

    def accept_analysis(
        self, generation: int, directory: TemporaryDirectory, result: ValidationAnalysis, pair_id: str | None = None
    ) -> None:
        """Publish only results belonging to the current input generation."""
        if generation != self._generation or self._cancel.is_set() or self._closing:
            return
        pair_id = pair_id or self._pair_id
        self._results[pair_id] = result
        self._directories[pair_id] = directory
        self.export_btn.setEnabled(True)
        if pair_id == self._pair_id:
            self._display_analysis()
            self.tabs.setCurrentIndex(3)
        self._set_status("Analysis complete. Results describe the supplied evidence and sampled spatial resolution.")

    def accept_study_analysis(self, generation: int, directory: TemporaryDirectory, result: StudyAnalysis) -> None:
        """Publish a completed study only if its input generation is still current."""
        if generation != self._generation or self._cancel.is_set() or self._closing:
            return
        for pair in result.pairs:
            if pair.analysis is not None:
                self._results[pair.pair_id] = pair.analysis
                self._directories[pair.pair_id] = directory
            else:
                self._failures[pair.pair_id] = pair.error
        self._display_analysis()
        self.tabs.setCurrentIndex(3)
        failures = sum(bool(pair.error) for pair in result.pairs)
        self._set_status(
            f"Study analysis complete: {len(result.pairs) - failures} pair(s) completed, {failures} failed."
        )

    def _display_analysis(self) -> None:
        self._clear_result_widgets(
            self._failures.get(self._pair_id, "Analyze this pair to see available measurements.")
        )
        self.analysis = self._results.get(self._pair_id)
        self._result_directory = self._directories.get(self._pair_id)
        if self.analysis is None:
            return
        rows = comparison_rows(self.analysis)
        self.summary_table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for col, value in enumerate(
                (
                    values["candidate"],
                    f"{values['validation_count']}/{values['validation_eligible']}",
                    values["validation_mean_um"],
                    values["validation_median_um"],
                    values["validation_rms_um"],
                    values["validation_p95_um"],
                    values["validation_maximum_um"],
                    values["automatic_median_um"],
                    values["residual_map_median_um"],
                )
            ):
                self.summary_table.setItem(row, col, QTableWidgetItem(self._format(value)))
        with hp.qt_signals_blocked(self.result_candidate):
            self.result_candidate.clear()
            for candidate in self.analysis.candidates:
                self.result_candidate.addItem(candidate.name, candidate.id)
            selected = self.result_candidate.findData(self._selected_candidates.get(self._pair_id))
            self.result_candidate.setCurrentIndex(max(selected, 0))
        self.show_result()

    def _analysis_error(self, generation: int, error: Exception, pair_id: str | None = None) -> None:
        if generation == self._generation and not self._closing:
            self._set_status(
                "Analysis cancelled." if isinstance(error, AnalysisCancelled) else f"Analysis failed: {error}"
            )
            if pair_id and not isinstance(error, AnalysisCancelled):
                self._failures[pair_id] = str(error)
                if pair_id == self._pair_id:
                    self._display_analysis()

    def _analysis_finished(self, worker: Any, directory: TemporaryDirectory) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        if directory not in self._directories.values():
            directory.cleanup()
        self.run_btn.setEnabled(True)
        self.run_all_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        for index, image in enumerate((self.session.fixed, self.session.moving)):
            self.geometry_buttons[index].setEnabled(image is not None)
            self.remove_image_buttons[index].setEnabled(image is not None)
        if self._cancel.is_set() and not self._closing:
            self._set_status("Analysis cancelled. Inputs are ready for a new run.")
        if self._closing and not self._transform_workers and not self._assist_workers:
            self.close()

    def cancel_analysis(self) -> None:
        """Request cancellation at the next bounded processing step."""
        self._cancel.set()
        self._generation += 1
        self._set_status("Cancelling analysis...")

    def _show_progress(self, generation: int, text: str) -> None:
        if generation == self._generation and not self._closing:
            self._set_status(text)

    @staticmethod
    def _format(value: Any) -> str:
        return "Unavailable" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)

    def _tab_changed(self, index: int) -> None:
        self.view_stack.setCurrentIndex(1 if index == 3 else 0)
        if index != 1:
            self.edit_landmarks.setChecked(False)
        for view in self.views:
            if "Region labels" in view.viewer.layers:
                view.viewer.layers["Region labels"].visible = index == 2

    def _error_labels_changed(self, checked: bool) -> None:
        viewer = self.result_view.viewer
        if "Validation error (um)" in viewer.layers:
            viewer.layers["Validation error (um)"].text.visible = checked

    def _show_error_summary(self, result: CandidateAnalysis) -> None:
        evaluated = int(result.validation.get("count", 0))
        eligible = int(result.validation.get("eligible", 0))
        independent = self._rows(result, automatic=False)
        total = len(independent)
        counts = (
            f"{evaluated}/{eligible} eligible landmarks evaluated; {total} supplied, {total - eligible} excluded, "
            f"{eligible - evaluated} not evaluated."
        )
        if evaluated:
            statistics = "; ".join(
                f"{label}: {self._format(result.validation.get(key))} um"
                for key, label in (
                    ("mean", "Mean"),
                    ("median", "Median"),
                    ("rms", "RMS"),
                    ("p95", "P95"),
                    ("maximum", "Max"),
                )
            )
            self._set_message(self.validation_summary, counts + "\n" + statistics, "Measured positional errors")
        else:
            reasons = list(dict.fromkeys(point["excluded"] for point in independent if point["excluded"]))
            reason = (
                "Add independent landmark pairs in Landmarks."
                if not independent
                else " ".join(reasons) + " See Landmark residuals for each point."
            )
            self._set_message(
                self.validation_summary, counts + "\n" + reason, "No independent validation errors to display"
            )
        self.error_labels.setEnabled(bool(evaluated))
        self.validation_legend.show()

    def _add_native_moving(self, viewer: Any) -> None:
        """Show the native moving image as context for moving-frame point measurements."""
        image = self.session.moving
        if image is None or self._readers[1] is None:
            return
        display = image.display_channel if image.display_channel is not None else image.channel
        rgb = display == "rgb" and image.is_rgb
        levels = validation_image_pyramid(self._readers[1], image.channel if rgb else int(display), rgb=rgb)
        viewer.add_image(
            levels if len(levels) > 1 else levels[0],
            multiscale=len(levels) > 1,
            rgb=rgb,
            name="Native moving",
            scale=image.geometry.spacing[::-1],
            translate=image.geometry.origin[::-1],
            units=("um", "um"),
            contrast_limits=validation_contrast_limits(levels),
        )

    def _add_point_errors(self, viewer: Any, points: list[dict[str, Any]], name: str, maximum: float) -> float:
        """Draw moving-frame residual points and arrows; return the shared color limit."""
        grid = self.analysis.grid
        data = np.array([[p["moving_y"], p["moving_x"]] for p in points])
        errors = np.array([p["error_um"] for p in points])
        error_limit = maximum if maximum > 0 else 1.0
        layer = viewer.add_points(
            data,
            name=name,
            features={"error_um": errors, "landmark_id": [point["id"] for point in points]},
            face_color="error_um",
            face_colormap="magma",
            face_contrast_limits=(0, error_limit),
            size=6 * max(grid.spacing),
            canvas_size_limits=(6, 40),
            units=("um", "um"),
            text={
                "string": "{landmark_id}: {error_um:.4g} um",
                "visible": self.error_labels.isChecked(),
                "size": max(11, THEMES.get_font_size()),
                "color": "white",
                "anchor": "upper_left",
                "translation": (4 * max(grid.spacing), 4 * max(grid.spacing)),
            },
        )
        layer.face_colorbar.visible = True
        layer.face_colorbar.position = "top_right"
        layer.face_colorbar.font_size = max(11, THEMES.get_font_size())
        vectors = np.stack((data, np.array([[p["dy_um"], p["dx_um"]] for p in points])), axis=1)
        viewer.add_vectors(
            vectors,
            name="Residual G(fixed)-moving",
            edge_color="cyan",
            units=("um", "um"),
            vector_style="arrow",
            length=1,
            blending="translucent_no_depth",
        )
        return error_limit

    @staticmethod
    def _rows(result: CandidateAnalysis, *, automatic: bool) -> list[dict[str, Any]]:
        """Residual rows for automatic landmarks, or for manual and assisted (independent) landmarks."""
        return [row for row in result.landmarks if (row.get("source") == "automatic") == automatic]

    def _shared_maximum(self, *, automatic: bool) -> float:
        """Largest finite error for this pair, so candidates share one color scale."""
        return max(
            [
                p["error_um"]
                for c in self.analysis.candidates
                for p in self._rows(c, automatic=automatic)
                if p["error_um"] is not None
            ]
            or [1]
        )

    def show_result(self, _index: int = 0) -> None:
        """Display calibrated diagnostics for the selected candidate without interpolating landmark errors."""
        if self.analysis is None or self.result_candidate.currentIndex() < 0:
            return
        result = self.analysis.candidates[self.result_candidate.currentIndex()]
        self._selected_candidates[self._pair_id] = result.id
        with hp.qt_signals_blocked(self.candidate_list):
            for row in range(self.candidate_list.count()):
                item = self.candidate_list.item(row)
                if item is not None and item.data(Qt.ItemDataRole.UserRole) == (self._pair_id, result.id):
                    self.candidate_list.setCurrentItem(item)
                    break
        with hp.qt_signals_blocked(self.summary_table):
            self.summary_table.selectRow(self.result_candidate.currentIndex())
        self._ensure_transform()
        if result.map_path is None:
            return
        viewer = self.result_view.viewer
        viewer.layers.clear()
        self.validation_legend.hide()
        grid = self.analysis.grid
        kwargs = {"scale": grid.spacing[::-1], "translate": grid.origin[::-1], "units": ("um", "um")}
        mode = list(RESULT_MODES)[self.result_mode.currentIndex()]
        with np.load(result.map_path) as arrays:
            legend = ""
            if mode == "overlap":
                fixed, moving = normalize_image(arrays["fixed_image"]), normalize_image(arrays["registered_image"])
                viewer.add_image(fixed, colormap="magenta", blending="additive", name="Fixed", **kwargs)
                viewer.add_image(moving, colormap="green", blending="additive", name="Registered moving", **kwargs)
                legend = "Magenta: fixed; green: registered moving; gray: agreement."
            elif mode == "validation_errors":
                self._show_error_summary(result)
                self._add_native_moving(viewer)
                points = [p for p in self._rows(result, automatic=False) if p["error_um"] is not None]
                if points:
                    maximum = self._shared_maximum(automatic=False)
                    error_limit = self._add_point_errors(viewer, points, "Validation error (um)", maximum)
                    legend = (
                        f"Point color: positional error from 0 to {error_limit:.4g} um; "
                        "the color range is shared by registration candidates for this pair. "
                        "Cyan arrows: observed moving point to predicted position."
                    )
                    if maximum == 0:
                        legend += " All measured errors are zero; the 0 to 1 um scale is for display."
                else:
                    legend = "No independent validation landmarks available."
            elif mode == "mask_disagreement":
                if "fixed_mask" in arrays and self.analysis.session.mask_pairs:
                    disagreement = np.zeros((*grid.shape, 3), dtype=np.float32)
                    for pair in self.analysis.session.mask_pairs:
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
                    legend = result.mask_reason or "No corresponding mask labels available."
            elif mode == "residual_shift":
                legend = self._show_residual_map(viewer, arrays, result, kwargs)
            elif mode == "automatic_landmarks":
                self._add_native_moving(viewer)
                points = [p for p in self._rows(result, automatic=True) if p["error_um"] is not None]
                if points:
                    error_limit = self._add_point_errors(
                        viewer, points, "Automatic error (um)", self._shared_maximum(automatic=True)
                    )
                    legend = (
                        f"{len(points)} automatic landmarks; point color 0 to {error_limit:.4g} um. Moving frame. "
                        f"{result.automatic.get('outliers', 0)} unticked outliers are not shown. "
                        "Image-similarity evidence, reported separately from manual and assisted landmarks."
                    )
                elif not result.automatic:
                    legend = "No automatic landmarks. Use Detect landmarks in the Landmarks tab."
                else:
                    legend = result.automatic.get("reason") or "No automatic landmarks were measured."
            else:
                key = {
                    "area_change": "log2_area_ratio",
                    "signed_jacobian": "jacobian",
                    "nonpositive_jacobian": "nonpositive",
                }[mode]
                data = arrays[key]
                finite = data[np.isfinite(data)]
                limit = max(float(np.max(np.abs(finite))), 0.1) if finite.size else 1
                limits = (
                    (-limit, limit)
                    if mode == "area_change"
                    else (float(finite.min()), float(finite.max()) + 1e-9)
                    if finite.size
                    else (0, 1)
                )
                viewer.add_image(
                    np.nan_to_num(data).astype(np.float32),
                    name=key,
                    colormap="coolwarm" if mode == "area_change" else "viridis",
                    contrast_limits=limits,
                    **kwargs,
                )
                invalid = ~np.isfinite(data) if mode != "nonpositive_jacobian" else ~arrays["valid"]
                overlay = np.zeros((*grid.shape, 4), dtype=np.float32)
                overlay[invalid] = (0.5, 0.5, 0.5, 1)
                if mode == "area_change":
                    overlay[arrays["nonpositive"]] = (0.5, 0, 0.5, 1)
                viewer.add_image(overlay, rgb=True, name="Unavailable / nonpositive", **kwargs)
                legend = (
                    f"Display range {limits[0]:.4g} to {limits[1]:.4g}. Gray: unavailable. "
                    "log2 area ratio: positive expansion, negative compression, moving to fixed. "
                    "Signed Jacobian: fixed to moving. Nonpositive values can include intended reflections. "
                    "All maps include global affine scaling."
                )
        viewer.reset_view()
        if not self._rows(result, automatic=False):
            evidence_note = "Independent accuracy unavailable: add held-out landmark pairs in Landmarks."
        elif not result.validation.get("eligible"):
            evidence_note = "Independent accuracy unavailable: all landmarks are excluded; see Landmark residuals."
        elif not result.validation.get("count"):
            evidence_note = "Independent accuracy unavailable: the transform could not evaluate eligible landmarks."
        else:
            evidence_note = (
                f"Independent landmarks: {result.validation['count']}/{result.validation['eligible']} evaluated."
            )
        if result.automatic:
            evidence_note += (
                f" Automatic landmarks (reported separately): "
                f"{result.automatic.get('count', 0)}/{result.automatic.get('eligible', 0)} evaluated."
            )
        reference = result.deformation.get("reference", "")
        self._set_message(
            self.result_detail,
            f"{evidence_note}\n{legend}\n{reference}\nDiagnostic grid spacing XY: {grid.spacing} um.",
        )
        self._fill_table(
            self.fitting_table,
            [
                ("Fitting points", result.fitting.get("count", 0)),
                *[
                    (f"{label} (um, moving frame)", result.fitting.get(key))
                    for key, label in (
                        ("mean", "Mean"),
                        ("median", "Median"),
                        ("rms", "RMS"),
                        ("p95", "P95"),
                        ("maximum", "Maximum"),
                    )
                ],
                (
                    "Evidence",
                    result.fitting.get("reason") or "Original registration fitting landmarks; not independent accuracy",
                ),
            ],
        )
        self._fill_table(
            self.overlap_table,
            [
                (
                    row["name"],
                    row["dice"],
                    row["iou"],
                    row["fixed_area_um2"],
                    row["moving_area_um2"],
                    f"{row['joint_coverage']:.1%}",
                    row["reason"],
                )
                for row in result.masks
            ]
            or [
                (
                    "Unavailable",
                    "",
                    "",
                    "",
                    "",
                    "",
                    result.mask_reason
                    or "Load both segmentation masks and pair their labels in Region overlap (optional).",
                )
            ],
        )
        self._fill_table(
            self.deformation_table,
            [
                ("Source coverage", f"{result.deformation['coverage']:.1%}"),
                ("Derivative coverage", f"{result.deformation['derivative_coverage']:.1%}"),
                (
                    "Valid derivative samples",
                    f"{result.deformation['valid_count']}/{result.deformation['total_count']}",
                ),
                ("Nonpositive Jacobian fraction", result.deformation["nonpositive_fraction"]),
                *[(f"Jacobian {key}", result.deformation[key]) for key in ("minimum", "median", "p95", "maximum")],
                ("Map direction", "fixed to moving; includes global affine scaling"),
                ("Area ratio direction", "moving to fixed; sampled deformation is not independent accuracy"),
                ("Transform reference", reference),
            ],
        )
        self._fill_automatic_table(result)
        self.error_table.setRowCount(len(result.landmarks))
        for row, point in enumerate(result.landmarks):
            for col, key in enumerate(("id", "source", "dx_um", "dy_um", "error_um", "excluded", "warning")):
                self.error_table.setItem(row, col, QTableWidgetItem(self._format(point.get(key, ""))))

    def _show_residual_map(self, viewer: Any, arrays: Any, result: CandidateAnalysis, kwargs: dict) -> str:
        if "shift_um" not in arrays:
            if not self.analysis.session.auto_residual_map:
                return "Enable Residual shift map under Automatic evidence and analyze again."
            return result.residual_map.get("reason") or "Residual shift map unavailable."
        viewer.add_image(normalize_image(arrays["fixed_image"]), colormap="gray", name="Fixed", **kwargs)
        valid = arrays["shift_valid"]
        magnitude = np.nan_to_num(arrays["shift_um"]).astype(np.float32)
        tile_kwargs = {
            "scale": tuple(arrays["shift_spacing_xy_um"][::-1]),
            "translate": tuple(arrays["shift_origin_xy_um"][::-1]),
            "units": ("um", "um"),
        }
        limit = float(magnitude[valid].max()) if valid.any() else 1.0
        viewer.add_image(
            magnitude,
            name="Residual shift (um)",
            colormap="magma",
            contrast_limits=(0, max(limit, 1e-6)),
            opacity=0.6,
            **tile_kwargs,
        )
        overlay = np.zeros((*valid.shape, 4), dtype=np.float32)
        overlay[~valid] = (0.5, 0.5, 0.5, 0.6)
        viewer.add_image(overlay, rgb=True, name="Rejected tiles", **tile_kwargs)
        if valid.any():
            rows, cols = np.nonzero(valid)
            origin, spacing = arrays["shift_origin_xy_um"], arrays["shift_spacing_xy_um"]
            starts = np.column_stack((origin[1] + rows * spacing[1], origin[0] + cols * spacing[0]))
            directions = np.column_stack((arrays["shift_y_um"][valid], arrays["shift_x_um"][valid]))
            viewer.add_vectors(
                np.stack((starts, directions), axis=1),
                name="Residual shift",
                edge_color="cyan",
                vector_style="arrow",
                length=1,
                units=("um", "um"),
            )
        summary = result.residual_map
        legend = (
            f"Residual shift magnitude 0 to {limit:.4g} um, fixed frame; arrows show predicted minus observed "
            f"position at physical length. {summary.get('valid_tiles', 0)}/{summary.get('tiles', 0)} tiles reliable; "
            f"gray tiles were rejected. Resolution about {self._format(summary.get('resolution_um'))} um."
        )
        if summary.get("reason"):
            legend += f" {summary['reason']}"
        return legend

    def _fill_automatic_table(self, result: CandidateAnalysis) -> None:
        shift, auto = result.residual_map, result.automatic
        rows = [
            (
                "Status",
                self._automatic_status(shift),
                (auto.get("reason") or "Measured") if auto else "Not detected",
            ),
            ("Frame", shift.get("frame", ""), auto.get("frame", "")),
            (
                "Measurements",
                f"{shift.get('valid_tiles', '')}/{shift.get('tiles', '')} tiles" if shift else "",
                f"{auto.get('count', 0)}/{auto.get('eligible', 0)} eligible points "
                f"({auto.get('outliers', 0)} unticked outliers)"
                if auto
                else "",
            ),
        ]
        for key, label in (("mean", "Mean"), ("median", "Median"), ("rms", "RMS"), ("p95", "P95"), ("maximum", "Max")):
            rows.append((f"{label} (um)", shift.get(key), auto.get(key)))
        self._fill_table(self.automatic_table, rows)

    def _automatic_status(self, summary: dict[str, Any]) -> str:
        if not self.analysis.session.auto_residual_map:
            return "Not requested"
        return summary.get("reason") or "Measured"

    def _fill_table(self, table: QTableWidget, rows: list[tuple[Any, ...]]) -> None:
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(self._format(value)))

    def choose_export(self) -> None:
        """Write HTML, PDF and measurement exports using the selected path privacy option."""
        if not self._results and not self._failures:
            return
        parent = QFileDialog.getExistingDirectory(self, "Choose parent directory for report")
        if not parent:
            return
        name, ok = QInputDialog.getText(self, "Report directory", "New directory name", text="validation-report")
        if not ok or not name or Path(name).name != name:
            return
        outcomes = [
            PairAnalysis(pair.id, pair.name, self._results.get(pair.id), self._failures.get(pair.id, ""))
            for pair in self.study.pairs
        ]
        analysis = StudyAnalysis(self.study.model_copy(deep=True), outcomes)
        path = self._try(
            partial(
                export_validation_study_report,
                analysis,
                Path(parent) / name,
                include_paths=self.report_include_paths.isChecked(),
            )
        )
        if path:
            self._set_status(f"Exported HTML and PDF reports to {path.parent}")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Cancel running work before disposing readers, views, and temporary maps."""
        if not self._closing and not self._may_discard():
            event.ignore()
            return
        self._closing = True
        if self._workers or self._transform_workers or self._assist_workers:
            self._detect_cancel.set()
            if self._workers:
                self.cancel_analysis()
            event.ignore()
            return
        self._updating = True
        for view in [*self.views, self.result_view]:
            view.viewer.layers.clear()
            view.close()
        for reader in (*self._readers, *self._mask_readers):
            if reader is not None:
                reader.close()
        self._readers = [None, None]
        self._mask_readers = [None, None]
        for directory in set(self._directories.values()):
            directory.cleanup()
        self._directories.clear()
        event.accept()
