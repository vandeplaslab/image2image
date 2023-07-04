"""Widget for loading data."""
import typing as ty
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from koyo.typing import PathLike
from loguru import logger
from qtextra.utils.utilities import connect
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QWidget
from superqt.utils import thread_worker

from image2image._dialogs import OverlayTableDialog, SelectChannelsTableDialog, SelectTransformTableDialog
from image2image.config import CONFIG
from image2image.enums import ALLOWED_FORMATS, VIEW_TYPE_TRANSLATIONS
from image2image.models import DataModel, TransformModel
from image2image.utilities import log_exception, style_form_layout

if ty.TYPE_CHECKING:
    from image2image.readers.coordinate_reader import CoordinateReader


class LoadMixin(QWidget):
    """Load data mixin."""

    evt_loading = Signal()
    evt_loaded = Signal(object, object)
    evt_closed = Signal(object)

    IS_FIXED: bool
    INFO_TEXT = "Select data..."
    FILE_TITLE = "Select data..."
    FILE_FORMATS = ALLOWED_FORMATS

    def __init__(self, parent, view):
        """Init."""
        super().__init__(parent=parent)
        self._setup_ui()
        self.view = view
        self.model = DataModel(is_fixed=self.IS_FIXED)

    @property
    def resolution(self) -> float:
        """Get resolution."""
        return 1.0

    @resolution.setter
    def resolution(self, value: str):
        """Set resolution."""

    def _on_close_dataset(self, force: bool = False) -> bool:
        """Close dataset."""
        from image2image._dialogs import CloseDatasetDialog

        if self.model.n_paths:
            paths = None
            if not force:  # only ask user if not forced
                dlg = CloseDatasetDialog(self, self.model)
                if dlg.exec_():
                    paths = dlg.paths
            else:
                paths = self.model.paths
            if paths:
                self.model.remove_paths(paths)
                self.evt_closed.emit(self.model)
            if not self.model.n_paths:
                self.resolution = "1.0"
            return True
        return False

    def on_set_path(self, paths: ty.Union[PathLike, ty.List[PathLike]]):
        """Set the path and immediately load it."""
        if isinstance(paths, (str, Path)):
            paths = [paths]
        self._on_load_dataset(paths)

    def on_select_dataset(self, _evt=None):
        """Load data."""
        path = hp.get_filename(
            self,
            title=self.FILE_TITLE,
            base_dir=CONFIG.fixed_dir if self.IS_FIXED else CONFIG.moving_dir,
            file_filter=self.FILE_FORMATS,
        )
        # path = self.FILENAME
        if path:
            if self.IS_FIXED:
                CONFIG.fixed_dir = str(Path(path).parent)
            else:
                CONFIG.moving_dir = str(Path(path).parent)
            self._on_load_dataset([path])

    def _on_load_dataset(self, paths: ty.List[PathLike]):
        """Load data."""
        self.evt_loading.emit()
        model = self.model
        model.add_paths(paths)
        func = thread_worker(
            model.load,
            start_thread=True,
            connect={"returned": self._on_loaded_dataset, "errored": self._on_failed_dataset},
        )
        func()
        logger.info(f"Started loading dataset - '{model.paths}'")

    def _on_loaded_dataset(self, model: "DataModel"):
        """Finished loading data."""
        # setup resolution
        self.resolution = str(model.resolution)
        # select what should be loaded
        dlg = SelectChannelsTableDialog(self, model)
        channel_list = []
        if dlg.exec_():
            channel_list = dlg.channels
        logger.info(f"Selected channels: {channel_list}")
        if not channel_list:
            model.remove_paths(model.just_added)
            model, channel_list = None, None
            logger.warning("No channels selected - dataset not loaded")

        # load data into an image
        self.evt_loaded.emit(model, channel_list)

    def _on_failed_dataset(self, exception: Exception):
        """Failed to load dataset."""
        logger.error("Error occurred while loading dataset.")
        log_exception(exception)
        self.evt_loaded.emit(None, None)

    def _on_extract_channels(self):
        """Extract channels from the list."""
        from image2image._dialogs import ExtractChannelsDialog

        if not self.model.get_extractable_paths():
            logger.warning("No paths to extract data from.")
            hp.warn(
                self,
                "No paths to extract data from. Only <b>.imzML</b>, <b>.tdf</b> and <b>.tsf</b> files support data"
                " extraction.",
            )
            return

        dlg = ExtractChannelsDialog(self, self.model)
        path, mzs, ppm = None, None, None
        if dlg.exec_():
            path = dlg.path_to_extract
            mzs = dlg.mzs
            ppm = dlg.ppm

        if path and mzs and ppm:
            reader: "CoordinateReader" = self.model.get_reader(path)  # noqa
            if reader:
                func = thread_worker(
                    partial(reader.extract, mzs=mzs, ppm=ppm),
                    start_thread=True,
                    connect={"returned": self._on_update_dataset, "errored": self._on_failed_update_dataset},
                )
                func()

    def _on_update_dataset(self, result: ty.Tuple[Path, ty.List[str]]):
        """Finished loading data."""
        path, channel_list = result
        # load data into an image
        self.evt_loaded.emit(self.model, channel_list)

    @staticmethod
    def _on_failed_update_dataset(exception: Exception):
        """Failed to load dataset."""
        logger.error("Error occurred while extracting images.", exception)
        log_exception(exception)


class LoadWidget(LoadMixin):
    """Widget for loading data."""

    evt_toggle_channel = Signal(str, bool)
    evt_toggle_all_channels = Signal(bool)

    IS_FIXED: bool = True

    def __init__(self, parent, view):
        """Init."""
        super().__init__(parent, view)
        self.table_dlg = OverlayTableDialog(self, self.model, self.view)

    @property
    def resolution(self) -> float:
        """Get resolution."""
        return self.resolution_edit.value()

    @resolution.setter
    def resolution(self, value: str):
        self.resolution_edit.setValue(float(value))

    def _setup_ui(self):
        """Setup UI."""
        self.resolution_edit = hp.make_double_spin_box(
            self, minimum=0, maximum=100, n_decimals=2, func=self._on_set_resolution
        )

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(hp.make_label(self, self.INFO_TEXT, bold=True, wrap=True, alignment=Qt.AlignCenter))
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Add image...", func=self.on_select_dataset),
                hp.make_qta_btn(self, "add", func=self._on_extract_channels, normal=True),
                hp.make_btn(self, "Remove image...", func=self._on_close_dataset),
                stretch_id=(0, 2),
            )
        )
        layout.addRow(hp.make_btn(self, "Select channels...", func=self._on_select_channels))
        layout.addRow(hp.make_label(self, "Pixel size (μm)"), self.resolution_edit)
        self.setLayout(layout)
        return layout

    def _on_set_resolution(self, _evt=None):
        """Specify resolution."""
        self.model.resolution = self.resolution_edit.value()

    def _on_select_channels(self):
        """Select channels from the list."""
        self.table_dlg.show()


class MovingWidget(LoadWidget):
    """Widget for loading moving data."""

    # class attrs
    IS_FIXED = False
    INFO_TEXT = "Select 'moving' data..."
    FILE_TITLE = "Select 'moving' data..."

    # events
    evt_show_transformed = Signal(str)
    evt_view_type = Signal(object)

    def __init__(self, parent, view):
        super().__init__(parent, view)

        # extra events
        connect(self.evt_loaded, self._on_update_choice)
        connect(self.evt_closed, self._on_clear_choice)

    def _on_update_choice(self, _model, _channel_list):
        """Update list of available options."""
        hp.combobox_setter(self.transformed_choice, clear=True, items=["None", *self.model.channel_names()])
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _on_clear_choice(self, _model):
        """Clear list of available options."""
        hp.combobox_setter(self.transformed_choice, clear=True, items=["None"])
        self._on_toggle_transformed(self.transformed_choice.currentText())

    def _setup_ui(self):
        """Setup UI."""
        layout = super()._setup_ui()

        self.view_type_choice = hp.make_combobox(
            self,
            data=VIEW_TYPE_TRANSLATIONS,
            value=str(CONFIG.view_type),
            func=self._on_update_view_type,
        )
        layout.addRow(hp.make_label(self, "View type"), self.view_type_choice)

        self.transformed_choice = hp.make_combobox(
            self,
            tooltip="Select which image should be displayed on the fixed modality.",
            func=self._on_toggle_transformed,
        )
        layout.addRow(
            hp.make_label(self, "Overlay"),
            self.transformed_choice,
        )
        return layout

    def _on_update_view_type(self, value: str):
        """Update view type."""
        CONFIG.view_type = value
        self.evt_view_type.emit(value)

    def _on_toggle_transformed(self, value: str):
        """Toggle visibility of transformed."""
        CONFIG.show_transformed = value != "None"
        self.evt_show_transformed.emit(value)


class FixedWidget(LoadWidget):
    """Widget for loading fixed data."""

    # class attrs
    IS_FIXED = True
    INFO_TEXT = "Select 'fixed' data..."
    FILE_TITLE = "Select 'fixed' data..."


class LoadWithTransformWidget(LoadMixin):
    """Widget for loading data."""

    IS_FIXED = True

    evt_transform_changed = Signal(Path)

    def __init__(self, parent, view):
        """Init."""
        super().__init__(parent, view)
        self.transform_model = TransformModel()
        self.transform_model.add_transform("Identity matrix", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.table_dlg = SelectTransformTableDialog(self, self.model, self.transform_model, self.view)

    def _setup_ui(self):
        """Setup UI."""
        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(hp.make_label(self, self.INFO_TEXT, bold=True, wrap=True, alignment=Qt.AlignCenter))
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "Add image(s)...", func=self.on_select_dataset),
                hp.make_qta_btn(self, "add", func=self._on_extract_channels, normal=True),
                hp.make_btn(self, "Remove...", func=self._on_close_dataset),
                stretch_id=(0, 2),
            )
        )
        layout.addRow(hp.make_btn(self, "Select transformation...", func=self._on_select_transform))
        return layout

    def _on_select_transform(self):
        """Select transformation data."""
        self.table_dlg.show()
