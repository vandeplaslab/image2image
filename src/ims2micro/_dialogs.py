"""Various dialogs."""
import typing as ty
from pathlib import Path

import numpy as np
import qtextra.helpers as hp
from koyo.typing import PathLike
from loguru import logger
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog, QtFramelessPopup, QtFramelessTool
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QFormLayout

from ims2micro.utilities import style_form_layout

if ty.TYPE_CHECKING:
    from ims2micro._select import LoadWidget
    from ims2micro.models import DataModel

OverlayConfig = (
    TableConfig()
    .add("", "check", "bool", 25, no_sort=True)
    .add("channel name", "channel_name", "str", 125, no_sort=True)
    .add("dataset", "dataset", "str", 250, no_sort=True)
)
SelectConfig = (
    TableConfig()
    .add("", "check", "bool", 25, no_sort=True)
    .add("channel name", "channel_name", "str", 200)
    .add("channel name (full)", "channel_name_full", "str", 0, hidden=True)
)
LocateConfig = (
    TableConfig()
    .add("", "check", "bool", 0, no_sort=True, hidden=True)
    .add("old path", "old_path", "str", 250)
    .add("new path", "new_path", "str", 250)
    .add("comment", "valid", "str", 100)
)
FiducialConfig = (
    TableConfig()
    .add("", "check", "bool", 0, no_sort=True, hidden=True)
    .add("index", "index", "int", 50)
    .add("y-m(px)", "y_px_micro", "float", 50)
    .add("x-m(px)", "x_px_micro", "float", 50)
    .add("y-i(px)", "y_px_ims", "float", 50)
    .add("x-i(px)", "x_px_ims", "float", 50)
)
ExtractConfig = TableConfig().add("", "check", "bool", 0, no_sort=True, hidden=True).add("m/z", "mz", "float", 100)


class LocateFilesDialog(QtDialog):
    """Dialog to locate files."""

    def __init__(self, parent, micro_paths: ty.List[PathLike], ims_paths: ty.List[PathLike]):
        paths = ims_paths + micro_paths
        self.paths: ty.List[ty.Dict[str, ty.Optional[PathLike]]] = [
            {"old_path": Path(path), "new_path": None} for path in paths
        ]
        super().__init__(parent)
        self.setWindowTitle("Locate files...")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.on_load()

    def connect_events(self, state: bool = True):
        """Connect events."""
        connect(self.table.doubleClicked, self.on_double_click, state=state)

    def keyPressEvent(self, evt):
        """Key press event."""
        if evt.key() == Qt.Key_Escape:
            evt.ignore()
        else:
            super().keyPressEvent(evt)

    def fix_missing_paths(self, paths_missing: ty.List[PathLike], paths: ty.List[PathLike]):
        """Locate missing paths."""
        if paths is None:
            paths = []
        for path in paths_missing:
            for path_pair in self.paths:
                if path_pair["old_path"] == Path(path) and path_pair["new_path"] is not None:
                    paths.append(path_pair["new_path"])
        return paths

    def on_double_click(self, index):
        """Zoom in."""
        row = index.row()
        path = self.paths[row]["old_path"]
        new_path = self.paths[row]["new_path"]
        if new_path and new_path.exists():
            path = new_path
        suffix = path.suffix.lower()
        base_dir = ""
        if path.parent.exists():
            base_dir = str(path.parent)
        # looking for a file
        if suffix in [".tiff", ".jpg", ".jpeg", ".png", ".h5", ".imzml", ".tdf", ".tsf", ".npy"]:
            new_path = hp.get_filename(
                self,
                title="Locate file...",
                base_dir=base_dir,
                base_filename=path.name,
                file_filter=f"All files (*.*);; File type (*{suffix})",
            )
            if not new_path:
                logger.warning("No file selected.")
                return
            self.paths[row]["new_path"] = Path(new_path)
            self.on_load()
            logger.info(f"Located file - {new_path}")

    def on_load(self, _evt=None):
        """On load."""
        data = []
        for path_pair in self.paths:
            old_path = path_pair["old_path"]
            new_path = path_pair["new_path"]
            comment = "File found at old location" if old_path.exists() else "File not found"
            if new_path and new_path.exists():
                comment = "File found at new location"
                if old_path.name != new_path.name:
                    comment += " but has different name."
            data.append([True, str(old_path), str(new_path) if new_path else "", comment])
        self.table.reset_data()
        self.table.add_data(data)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.table = QtCheckableTableView(self, config=LocateConfig, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(LocateConfig.header, LocateConfig.no_sort_columns, LocateConfig.hidden_columns)
        self.table.setTextElideMode(Qt.TextElideMode.ElideLeft)

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(
            hp.make_label(
                self,
                "At least one file read from the configuration file is no longer at the specified path."
                " Please locate it on your hard drive or it won't be imported.",
                alignment=Qt.AlignHCenter,
            )
        )
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Double-click on a row to zoom in on the point.",
                alignment=Qt.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout


class FiducialTableDialog(QtFramelessTool):
    """Dialog to display fiducial marker information."""

    HIDE_WHEN_CLOSE = True

    # event emitted when the popup closes
    evt_close = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.points_data = None
        self.on_load()

    def connect_events(self, state: bool = True):
        """Connect events."""
        # change of model events
        connect(self.parent().fixed_points_layer.events.data, self.on_load, state=state)
        connect(self.parent().moving_points_layer.events.data, self.on_load, state=state)
        connect(self.parent().evt_predicted, self.on_load, state=state)
        # table events
        connect(self.table.doubleClicked, self.on_double_click, state=state)

    def keyPressEvent(self, evt):
        """Key press event."""
        if evt.key() == Qt.Key_Escape:
            evt.ignore()
        elif evt.key() == Qt.Key_Backspace or evt.key() == Qt.Key_Delete:
            self.on_delete_row()
            evt.accept()
        else:
            super().keyPressEvent(evt)

    def on_delete_row(self):
        """Delete row."""
        sel_model = self.table.selectionModel()
        if sel_model.hasSelection():
            indices = [index.row() for index in sel_model.selectedRows()]
            indices = sorted(indices, reverse=True)
            for index in indices:
                fixed_points = self.parent().fixed_points_layer.data
                moving_points = self.parent().moving_points_layer.data
                if index < len(fixed_points):
                    fixed_points = np.delete(fixed_points, index, axis=0)
                    self.parent().fixed_points_layer.data = fixed_points
                if index < len(moving_points):
                    moving_points = np.delete(moving_points, index, axis=0)
                    self.parent().moving_points_layer.data = moving_points
                logger.debug(f"Deleted {index} from fiducial table")

    def on_double_click(self, index):
        """Zoom in."""
        row = index.row()
        y_micro, x_micro, y_ims, x_ims = self.points_data[row]
        # zoom-in on fixed data
        if not np.isnan(x_micro):
            view_fixed = self.parent().view_fixed
            view_fixed.viewer.camera.center = (0.0, y_micro, x_micro)
            view_fixed.viewer.camera.zoom = 5
            logger.debug(
                f"Applied focus center=({y_micro:.1f}, {x_micro:.1f}) zoom={view_fixed.viewer.camera.zoom:.3f} on micro"
                f" data"
            )
        # zoom-in on moving data
        if not np.isnan(x_ims):
            view_moving = self.parent().view_moving
            view_moving.viewer.camera.center = (0.0, y_ims, x_ims)
            view_moving.viewer.camera.zoom = 50
            logger.debug(
                f"Applied focus center=({y_ims:.1f}, {x_ims:.1f}) zoom={view_moving.viewer.camera.zoom:.3f} on IMS data"
            )

    def on_load(self, _evt=None):
        """On load."""

        def _str_fmt(value):
            if np.isnan(value):
                return ""
            return f"{value:.3f}"

        fixed_points_layer = self.parent().fixed_points_layer
        moving_points_layer = self.parent().moving_points_layer
        n = max([len(fixed_points_layer.data), len(moving_points_layer.data)])
        array = np.full((n, 4), fill_value=np.nan)
        array[0 : len(fixed_points_layer.data), 0:2] = fixed_points_layer.data
        array[0 : len(moving_points_layer.data), 2:] = moving_points_layer.data

        data = []
        for index, row in enumerate(array, start=1):
            data.append([True, str(index), *map(_str_fmt, row)])
        self.table.reset_data()
        self.table.add_data(data)
        self.points_data = array

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("Fiducial markers")

        self.table = QtCheckableTableView(self, config=FiducialConfig, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(FiducialConfig.header, FiducialConfig.no_sort_columns, FiducialConfig.hidden_columns)

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Double-click on a row to zoom in on the point."
                "<b>Tip.</b> Press  <b>Delete</b> or <b>Backspace</b> to delete a point.",
                alignment=Qt.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        return layout


class OverlayTableDialog(QtFramelessTool):
    """Dialog to enable creation of overlays."""

    HIDE_WHEN_CLOSE = True

    # event emitted when the popup closes
    evt_close = Signal()

    def __init__(self, parent: "LoadWidget", model: "DataModel", view):
        self.model = model
        self.view = view
        super().__init__(parent)
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)

    def connect_events(self, state: bool = True):
        """Connect events."""
        # TODO: connect event that updates checkbox state when user changes visibility in layer list
        # change of model events
        connect(self.parent().evt_loaded, self.on_load, state=state)
        connect(self.parent().evt_closed, self.on_load, state=state)
        # table events
        connect(self.table.evt_checked, self.on_toggle_channel, state=state)

    def on_toggle_channel(self, index: int, state: bool):
        """Toggle channel."""
        if index == -1:
            self.parent().evt_toggle_all_channels.emit(state)
        else:
            channel_name = self.table.get_value(OverlayConfig.channel_name, index)
            dataset = self.table.get_value(OverlayConfig.dataset, index)
            self.parent().evt_toggle_channel.emit(f"{channel_name} | {dataset}", state)

    def on_load(self, model: "DataModel"):
        """On load."""
        if not model:
            return

        self.model = model
        data = []
        reader = self.model.get_wrapper()
        if reader:
            for name in reader.channel_names():
                channel_name, dataset = name.split(" | ")
                data.append([True, channel_name, dataset])
        existing_data = self.table.get_data()
        if existing_data:
            for exist_row in existing_data:
                for new_row in data:
                    if (
                        exist_row[OverlayConfig.channel_name] == new_row[OverlayConfig.channel_name]
                        and exist_row[OverlayConfig.dataset] == new_row[OverlayConfig.dataset]
                    ):
                        new_row[OverlayConfig.check] = exist_row[OverlayConfig.check]
        self.table.reset_data()
        self.table.add_data(data)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        which = "Fixed" if self.model.is_fixed else "Moving"
        self._title_label.setText(f"'{which}' Channel Selection")

        self.table = QtCheckableTableView(self, config=OverlayConfig, enable_all_check=True, sortable=True)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(OverlayConfig.header, OverlayConfig.no_sort_columns, OverlayConfig.hidden_columns)

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Check/uncheck a row to toggle visibility of the channel.",
                alignment=Qt.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        return layout


class CloseDatasetDialog(QtDialog):
    """Dialog where user can select which path(s) should be removed."""

    def __init__(self, parent, model: "DataModel"):
        self.model = model

        super().__init__(parent)
        self.paths = self.get_paths()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.all_check = hp.make_checkbox(
            self,
            "Check all",
            clicked=self.on_check_all,
            value=True,
        )
        # iterate over all available paths
        self.checkboxes = []
        for path in self.model.paths:
            # make checkbox for each path
            checkbox = hp.make_checkbox(self, str(path), value=True, clicked=self.on_apply)
            self.checkboxes.append(checkbox)

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(
            hp.make_label(
                self,
                "Please select which images should be <b>removed</b> from the project."
                " Fiducial markers will be unaffected.",
                alignment=Qt.AlignHCenter,
                enable_url=True,
                wrap=True,
            )
        )
        layout.addRow(self.all_check)
        layout.addRow(hp.make_h_line())
        for checkbox in self.checkboxes:
            layout.addRow(checkbox)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def on_check_all(self, state: bool):
        """Check all."""
        for checkbox in self.checkboxes:
            checkbox.setChecked(state)

    def on_apply(self):
        """Apply."""
        self.paths = self.get_paths()
        all_checked = len(self.config) == len(self.checkboxes)
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)

    def get_paths(self) -> ty.List[Path]:
        """Return state."""
        paths = []
        for checkbox in self.checkboxes:
            if checkbox.isChecked():
                paths.append(Path(checkbox.text()))
        return paths


class ImportSelectDialog(QtDialog):
    """Dialog that lets you select what should be imported."""

    def __init__(self, parent):
        super().__init__(parent)
        self.config = self.get_config()

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.all_check = hp.make_checkbox(self, "Check all", clicked=self.on_check_all, value=True)
        self.micro_check = hp.make_checkbox(self, "Fixed images (if exist)", value=True, func=self.on_apply)
        self.ims_check = hp.make_checkbox(self, "Moving images (if exist)", value=True, func=self.on_apply)
        self.fixed_check = hp.make_checkbox(self, "Fixed fiducials", value=True, func=self.on_apply)
        self.moving_check = hp.make_checkbox(self, "Moving fiducials", value=True, func=self.on_apply)

        layout = hp.make_form_layout()
        style_form_layout(layout)
        layout.addRow(
            hp.make_label(self, "Please select what should be imported.", alignment=Qt.AlignHCenter, bold=True)
        )
        layout.addRow(self.all_check)
        layout.addRow(hp.make_h_line())
        layout.addRow(self.micro_check)
        layout.addRow(self.ims_check)
        layout.addRow(self.fixed_check)
        layout.addRow(self.moving_check)
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def on_check_all(self, state: bool):
        """Check all."""
        self.micro_check.setChecked(state)
        self.ims_check.setChecked(state)
        self.fixed_check.setChecked(state)
        self.moving_check.setChecked(state)

    def on_apply(self):
        """Apply."""
        self.config = self.get_config()
        all_checked = all(self.config.values())
        self.all_check.setCheckState(Qt.Checked if all_checked else Qt.Unchecked)

    def get_config(self) -> ty.Dict[str, bool]:
        """Return state."""
        return {
            "micro": self.micro_check.isChecked(),
            "ims": self.ims_check.isChecked(),
            "fixed": self.fixed_check.isChecked(),
            "moving": self.moving_check.isChecked(),
        }


class SelectChannelsTableDialog(QtDialog):
    """Dialog to enable creation of overlays."""

    def __init__(self, parent: "LoadWidget", model: "DataModel"):
        super().__init__(parent, title="Select Channels to Load")
        self.model = model

        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        self.on_load()
        self.channels = self.get_channels()

    def connect_events(self, state: bool = True):
        """Connect events."""
        connect(self.table.evt_checked, self.on_select_channel, state=state)

    def on_select_channel(self, _index: int, _state: bool):
        """Toggle channel."""
        self.channels = self.get_channels()

    def get_channels(self):
        """Select all channels."""
        channels = []
        for index in self.table.get_all_checked():
            channels.append(self.table.get_value(SelectConfig.channel_name_full, index))
        return channels

    def on_load(self):
        """On load."""
        data = []
        wrapper = self.model.get_wrapper()
        if wrapper:
            for name in wrapper.channel_names_for_names(self.model.just_added):
                channel_name, _ = name.split(" | ")
                data.append([True, channel_name, name])
        else:
            logger.warning(f"Wrapper was not specified - {wrapper}")
        self.table.add_data(data)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.table = QtCheckableTableView(
            self, config=SelectConfig, enable_all_check=True, sortable=True, double_click_to_check=True
        )
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(SelectConfig.header, SelectConfig.no_sort_columns, SelectConfig.hidden_columns)

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> You can quickly check/uncheck row by double-clicking on a row.<br>"
                "<b>Tip.</b> Check/uncheck a row to select which channels should be immediately loaded.<br>"
                "<b>Tip.</b> You can quickly check/uncheck all rows by clicking on the first column header.",
                alignment=Qt.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout


class ExtractChannelsDialog(QtDialog):
    """Dialog to extract ion images."""

    def __init__(self, parent: "LoadWidget", model: "DataModel"):
        self.model = model
        super().__init__(parent, title="Extract Ion Images")
        self.setFocus()
        self.path_to_extract = None
        self.mzs = None
        self.ppm = None
        self.on_select_path()

    def get_paths(self) -> ty.List[str]:
        """Get paths."""
        paths = self.model.get_extractable_paths()
        return [str(path) for path in paths]

    def on_add(self):
        """Add peak."""
        value = self.mz_edit.value()
        values = self.table.get_col_data(ExtractConfig.mz)
        if value is not None and value not in values:
            self.table.add_data([[True, value]])
        self.mzs = self.table.get_col_data(ExtractConfig.mz)

    def on_select_path(self, _value: ty.Optional[str] = None):
        """Select path."""
        self.path_to_extract = self.path_choice.currentText()
        self.ppm = self.ppm_edit.value()

    def on_delete_row(self):
        """Delete row."""
        sel_model = self.table.selectionModel()
        if sel_model.hasSelection():
            indices = [index.row() for index in sel_model.selectedRows()]
            indices = sorted(indices, reverse=True)
            for index in indices:
                self.table.remove_row(index)
                logger.trace(f"Deleted '{index}' from m/z table")

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.path_choice = hp.make_combobox(
            self, self.get_paths(), "Dataset from which to extract data.", func=self.on_select_path
        )
        self.mz_edit = hp.make_double_spin_box(self, minimum=0, maximum=2500, step=0.1, n_decimals=3)
        self.ppm_edit = hp.make_double_spin_box(
            self, minimum=0.5, maximum=25, value=10, step=1, n_decimals=1, suffix=" ppm"
        )

        self.table = QtCheckableTableView(self, config=ExtractConfig, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(ExtractConfig.header, ExtractConfig.no_sort_columns, ExtractConfig.hidden_columns)

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(self.path_choice)
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "m/z"),
                self.mz_edit,
                hp.make_qta_btn(self, "add", tooltip="Add peak", func=self.on_add, normal=True),
                stretch_id=1,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_label(self, "ppm"),
                self.ppm_edit,
                stretch_id=1,
            )
        )
        layout.addRow(self.table)
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Press <b>Delete</b> or <b>Backspace</b> to delete a peak.",
                alignment=Qt.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout

    def keyPressEvent(self, evt):
        """Key press event."""
        key = evt.key()
        if key == Qt.Key_Escape:
            evt.ignore()
        elif key == Qt.Key_Backspace or key == Qt.Key_Delete:
            self.on_delete_row()
            evt.accept()
        elif key == Qt.Key_Plus or key == Qt.Key_A:
            self.on_add()
            evt.accept()
        else:
            super().keyPressEvent(evt)


def open_about(parent):
    """Open a dialog with information about the app."""
    dlg = DialogAbout(parent)
    dlg.show()


class DialogAbout(QtFramelessPopup):
    """About dialog."""

    def __init__(self, parent):
        super().__init__(parent)

    # noinspection PyAttributeOutsideInit
    def make_panel(self):
        """Make panel."""
        from qtextra.widgets.qt_svg import QtColoredSVGIcon

        from ims2micro import __version__
        from ims2micro.assets import ICON_SVG

        links = {
            "project": "https://github.com/vandeplaslab/ims2micro",
            "github": "https://github.com/lukasz-migas",
            "website": "https://lukasz-migas.com/",
        }

        text = f"""
        <p><h2><strong>ims2micro</strong></h2></p>
        <p><strong>Version:</strong> {__version__}</p>
        <p><strong>Author:</strong> Lukasz G. Migas</p>
        <p><strong>Email:</strong> {hp.parse_link_to_link_tag("mailto:l.g.migas@tudelft.nl",
                                                              "l.g.migas@tudelft.nl")}</p>
        <p><strong>GitHub:</strong>&nbsp;{hp.parse_link_to_link_tag(links["project"])}</p>
        <p><strong>Project's GitHub:</strong>&nbsp;{hp.parse_link_to_link_tag(links["github"])}</p>
        <p><strong>Author's website:</strong>&nbsp;{hp.parse_link_to_link_tag(links["website"])}</p>
        <br>
        <p>Developed in the Van de Plas lab</p>
        """

        pix = QtColoredSVGIcon(ICON_SVG)
        self._image = hp.make_label(self, "")
        self._image.setPixmap(pix.pixmap(300, 300))

        # about label
        self.about_label = hp.make_label(self)
        self.about_label.setText(text)
        self.about_label.setAlignment(Qt.AlignCenter)

        # set layout
        vertical_layout = hp.make_v_layout()
        vertical_layout.addWidget(self._image, alignment=Qt.AlignHCenter)
        vertical_layout.addWidget(self.about_label)
        return vertical_layout
