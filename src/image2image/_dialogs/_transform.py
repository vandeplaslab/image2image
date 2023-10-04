import typing as ty
from pathlib import Path

from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout

from image2image.utilities import style_form_layout

if ty.TYPE_CHECKING:
    from image2image._select import LoadWithTransformWidget
    from image2image.models import DataModel, TransformModel


class SelectTransformDialog(QtFramelessTool):
    """Dialog to enable creation of overlays."""

    evt_transform = Signal(Path)

    HIDE_WHEN_CLOSE = True

    TABLE_CONFIG = (
        TableConfig()
        .add("", "check", "bool", 25, no_sort=True)
        .add("dataset", "dataset", "str", 250)
        .add("dataset_path", "dataset_path", "str", 0, no_sort=True, hidden=True)
        .add("transform", "transform", "str", 250)
    )

    def __init__(self, parent: "LoadWithTransformWidget", model: "DataModel", transform_model: "TransformModel", view):
        self.model = model
        self.transform_model = transform_model
        self.view = view
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    def connect_events(self, state: bool = True):
        """Connect events."""
        # TODO: connect event that updates checkbox state when user changes visibility in layer list
        # change of model events
        connect(self.parent().dataset_dlg.evt_loaded, self.on_update_data_list, state=state)
        connect(self.parent().dataset_dlg.evt_closed, self.on_update_data_list, state=state)

    def show(self) -> None:
        """Force update of the data/transform list."""
        self.on_update_transform_list()
        self.on_update_data_list()
        super().show()

    def on_apply_transform(self):
        """On transform choice."""
        indices = reversed(self.table.get_all_checked())
        if indices:
            transform_name = self.transform_choice.currentText()
            matrix = self.transform_model.get_matrix(transform_name)
            for index in indices:
                self.table.set_value(self.TABLE_CONFIG.transform, index, transform_name)
                reader_path = self.table.get_value(self.TABLE_CONFIG.dataset_path, index)
                # get reader appropriate for the path
                reader = self.model.get_reader(reader_path)
                if reader:
                    # if reader.transform_name != transform_name:
                    # transform information need to be updated
                    reader.transform_name = transform_name
                    reader.transform = matrix
                    self.table.update_value(index, self.TABLE_CONFIG.transform, transform_name)
                    self.evt_transform.emit(reader_path)
                    logger.trace(f"Updated transformation matrix for '{reader_path}'")
                else:
                    logger.warning(f"Could not update transformation matrix for '{reader_path}'")

    def on_update_transform_list(self):
        """Update list of transforms."""
        transforms = [t.name for t in self.transform_model.transforms] if self.transform_model.transforms else []
        hp.combobox_setter(self.transform_choice, items=transforms, set_item=self.transform_choice.currentText())

    def on_update_data_list(self):
        """Update the list of datasets."""
        data = []
        wrapper = self.model.get_wrapper()
        if wrapper:
            for path, reader in wrapper.path_reader_iter():
                data.append([False, path.name, path, reader.transform_name])
        self.table.reset_data()
        self.table.add_data(data)

    def on_load_transform(self):
        """Load transformation matrix."""
        from image2image.config import CONFIG
        from image2image.enums import ALLOWED_EXPORT_REGISTER_FORMATS
        from image2image.models.transformation import load_transform_from_file
        from image2image.utilities import compute_transform

        path = hp.get_filename(
            self,
            "Load transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_EXPORT_REGISTER_FORMATS,
        )
        if path:
            # load transformation
            path = Path(path)
            CONFIG.output_dir = str(path.parent)

            # get info on which settings should be imported

            # load data from config file
            try:
                (
                    transformation_type,
                    _fixed_paths,
                    _fixed_paths_missing,
                    fixed_points,
                    _moving_paths,
                    _moving_paths_missing,
                    moving_points,
                ) = load_transform_from_file(path)
            except ValueError as e:
                hp.warn(self, f"Failed to load transformation from {path}\n{e}", "Failed to load transformation")
                return
            affine = compute_transform(moving_points, fixed_points, transformation_type)
            self.transform_model.add_transform(path, affine.params)
            self.on_update_transform_list()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Transformation Selection")

        self.transform_choice = hp.make_combobox(self, ["Identity matrix"])  # , func=self.on_transform_choice)
        self.add_btn = hp.make_qta_btn(self, "add", func=self.on_load_transform, normal=True)

        self.table = QtCheckableTableView(
            self, config=self.TABLE_CONFIG, enable_all_check=True, sortable=True, double_click_to_check=True
        )
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )

        self.info = hp.make_label(self, "", enable_url=True)

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(
            hp.make_label(self, "Transformation name"),
            hp.make_h_layout(
                self.transform_choice,
                self.add_btn,
                stretch_id=0,
            ),
        )
        layout.addRow(hp.make_btn(self, "Apply to selected", func=self.on_apply_transform))
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Check/uncheck images to select where the current transformation matrix should be applied."
                "<br><b>Tip.</b> You can quickly check/uncheck row by double-clicking on a row.",
                alignment=Qt.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        return layout
