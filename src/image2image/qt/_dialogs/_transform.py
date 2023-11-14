"""Transform widget."""
import typing as ty
from copy import deepcopy
from pathlib import Path

from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QFormLayout

from image2image.models.data import DataModel
from image2image.models.transform import TransformData, TransformModel

if ty.TYPE_CHECKING:
    from qtextra._napari.image.wrapper import NapariImageView

    from image2image.qt._select import LoadWithTransformWidget


logger = logger.bind(src="TransformDialog")


class SelectTransformDialog(QtFramelessTool):
    """Dialog to enable creation of overlays."""

    evt_transform = Signal(str)

    HIDE_WHEN_CLOSE = True

    TABLE_CONFIG = (
        TableConfig()  # type: ignore[no-untyped-call]
        .add("", "check", "bool", 25, no_sort=True)
        .add("dataset", "dataset", "str", 250)
        .add("key", "key", "str", 0, no_sort=True, hidden=True)
        .add("transform", "transform", "str", 250)
    )

    def __init__(
        self,
        parent: "LoadWithTransformWidget",
        model: "DataModel",
        transform_model: "TransformModel",
        view: "NapariImageView",
    ):
        self.model = model
        self.transform_model = transform_model
        self.view = view
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        parent: "LoadWithTransformWidget" = self.parent()  # type: ignore[assignment]
        connect(parent.dataset_dlg.evt_loaded, self.on_update_data_list, state=state)
        connect(parent.dataset_dlg.evt_closed, self.on_update_data_list, state=state)

    def show(self) -> None:
        """Force update of the data/transform list."""
        self.on_update_transform_list()
        self.on_update_data_list()
        self.on_show_transform()
        super().show()

    def on_show_transform(self) -> None:
        """Show metadata about the transform."""
        transform_name = self.transform_choice.currentText()
        transform_data = self.transform_model.get_matrix(transform_name)
        if not transform_data:
            info = "No metadata available"
        else:
            info = ""
            transform = transform_data.compute()
            if hasattr(transform, "scale"):
                scale = transform.scale
                scale = (scale, scale) if isinstance(scale, float) else scale
                info += f"\nScale: {scale[0]:.3f}, {scale[1]:.3f}"
            if hasattr(transform, "translation"):
                translation = transform.translation
                translation = (translation, translation) if isinstance(translation, float) else translation
                info += f"\nTranslation: {translation[0]:.3f}, {translation[1]:.3f}"
            if hasattr(transform, "rotation"):
                rotation = transform.rotation
                info += f"\nRotation: {rotation:.3f}"
        self.transform_metadata.setText(info)

    def on_apply_transform(self) -> None:
        """On transform choice."""
        indices = reversed(self.table.get_all_checked())
        if indices:
            transform_name = self.transform_choice.currentText()
            transform_data = self.transform_model.get_matrix(transform_name)
            if not transform_data:
                return
            for index in indices:
                self.table.set_value(self.TABLE_CONFIG.transform, index, transform_name)
                key = self.table.get_value(self.TABLE_CONFIG.key, index)
                # get reader appropriate for the path
                reader = self.model.get_reader_for_key(key)
                if reader:
                    # transform information need to be updated
                    reader.transform_name = transform_name
                    reader.transform_data = deepcopy(transform_data)
                    self.table.update_value(index, self.TABLE_CONFIG.transform, transform_name)
                    self.evt_transform.emit(key)
                    logger.trace(f"Updated transformation matrix for '{key}'")
                else:
                    logger.warning(f"Could not update transformation matrix for '{key}'")

    def on_update_transform_list(self) -> None:
        """Update list of transforms."""
        transforms = self.transform_model.transform_names
        hp.combobox_setter(self.transform_choice, items=transforms, set_item=self.transform_choice.currentText())

    def on_update_data_list(self) -> None:
        """Update the list of datasets."""
        data = []
        wrapper = self.model.get_wrapper()
        if wrapper:
            for reader in wrapper.reader_iter():
                data.append([False, reader.name, reader.key, reader.transform_name])
        self.table.reset_data()
        self.table.add_data(data)

    def on_load_transform(self):
        """Load transformation matrix."""
        from image2image.config import CONFIG
        from image2image.enums import ALLOWED_EXPORT_REGISTER_FORMATS

        path = hp.get_filename(
            self,
            "Load transformation",
            base_dir=CONFIG.output_dir,
            file_filter=ALLOWED_EXPORT_REGISTER_FORMATS,
        )
        if path:
            # load transformation
            path_ = Path(path)
            CONFIG.output_dir = str(path_.parent)

            # get info on which settings should be imported

            # load data from config file
            try:
                transform_data = TransformData.from_i2r(path_)
            except ValueError as e:
                hp.warn(self, f"Failed to load transformation from {path_}\n{e}", "Failed to load transformation")
                return
            self.transform_model.add_transform(path_, transform_data)
            self.on_update_transform_list()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Transformation Selection")

        self.transform_choice = hp.make_combobox(self, ["Identity matrix"], func=self.on_show_transform)
        self.add_btn = hp.make_qta_btn(self, "add", func=self.on_load_transform, normal=True)

        self.transform_metadata = hp.make_label(self, "\n\n\n", wrap=True)

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
        hp.style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(
            hp.make_label(self, "Transformation name"),
            hp.make_h_layout(
                self.transform_choice,
                self.add_btn,
                stretch_id=0,
            ),
        )
        layout.addRow(self.transform_metadata)
        layout.addRow(hp.make_btn(self, "Apply to selected", func=self.on_apply_transform))
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Check/uncheck images to select where the current transformation matrix should be applied."
                "<br><b>Tip.</b> You can quickly check/uncheck row by double-clicking on a row.",
                alignment=Qt.AlignHCenter,  # type: ignore[attr-defined]
                object_name="tip_label",
                enable_url=True,
            )
        )
        return layout
