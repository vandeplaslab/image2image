import typing as ty
from pathlib import Path

from koyo.typing import PathLike
from loguru import logger
from qtextra import helpers as hp
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog
from qtextra.widgets.qt_table_view import QtCheckableTableView
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFormLayout

from image2image.utilities import style_form_layout


class LocateFilesDialog(QtDialog):
    """Dialog to locate files."""

    TABLE_CONFIG = (
        TableConfig()
        .add("", "check", "bool", 0, no_sort=True, hidden=True)
        .add("old path", "old_path", "str", 250)
        .add("new path", "new_path", "str", 250)
        .add("comment", "valid", "str", 100)
    )

    def __init__(
        self,
        parent,
        fixed_paths: ty.Sequence[PathLike],
        moving_paths: ty.Optional[ty.Sequence[ty.List[PathLike]]] = None,
    ):
        paths = list(fixed_paths)
        if moving_paths:
            paths.extend(list(moving_paths))
        self.paths: ty.List[ty.Dict[str, ty.Optional[PathLike]]] = [
            {"old_path": Path(path), "new_path": None} for path in paths
        ]
        super().__init__(parent)
        self.setWindowTitle("Locate files...")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.on_update_data_list()

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
            self.on_update_data_list()
            logger.info(f"Located file - {new_path}")

    def on_update_data_list(self, _evt=None):
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
        self.table = QtCheckableTableView(self, config=self.TABLE_CONFIG, enable_all_check=False, sortable=False)
        self.table.setCornerButtonEnabled(False)
        hp.set_font(self.table)
        self.table.setup_model(
            self.TABLE_CONFIG.header, self.TABLE_CONFIG.no_sort_columns, self.TABLE_CONFIG.hidden_columns
        )
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
