"""Locate files."""

import typing as ty
from pathlib import Path

from koyo.typing import PathLike
from loguru import logger
from qtextra import helpers as hp
from qtextra.config.theme import THEMES
from qtextra.utils.table_config import TableConfig
from qtextra.utils.utilities import connect
from qtextra.widgets.qt_dialog import QtDialog
from qtpy.QtCore import QModelIndex, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QFormLayout, QTableWidgetItem, QWidget

from image2image.config import SingleAppConfig

FILE_SUFFIXES = [
    # microscopy
    ".czi",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
    ".png",
    # mass spectrometry
    ".h5",
    ".imzml",
    ".tdf",
    ".tsf",
    ".baf",
    ".ser",
    # other
    ".npy",
]


class LocateFilesDialog(QtDialog):
    """Dialog to locate files."""

    TABLE_CONFIG = (
        TableConfig()
        .add("original path", "old_path", "str", 250, sizing="stretch")
        .add("replacement path", "new_path", "str", 250, sizing="stretch")
    )

    def __init__(self, parent: QWidget, config: SingleAppConfig, fixed_paths: ty.Sequence[PathLike]):
        self.CONFIG = config
        paths = list(fixed_paths)
        self.paths: list[dict[str, ty.Optional[PathLike]]] = [
            {"old_path": Path(path), "new_path": None} for path in paths
        ]
        super().__init__(parent)
        self.setWindowTitle("Locate files...")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        self.on_update_data_list()

    def connect_events(self, state: bool = True) -> None:
        """Connect events."""
        connect(self.table.doubleClicked, self.on_double_click, state=state)

    def keyPressEvent(self, evt):
        """Key press event."""
        if evt.key() == Qt.Key.Key_Escape:
            evt.ignore()
        else:
            super().keyPressEvent(evt)

    def fix_missing_paths(self, paths_missing: ty.Sequence[PathLike], paths: ty.Sequence[PathLike]) -> list[Path]:
        """Locate missing paths."""
        if paths is None:
            paths = []
        paths = [Path(p) for p in paths]
        for path in paths_missing:
            for path_pair in self.paths:
                if path_pair["old_path"] == Path(path) and path_pair["new_path"] is not None:
                    paths.append(Path(path_pair["new_path"]))
        return paths

    def on_double_click(self, index: QModelIndex) -> None:
        """Double-click event."""
        row = index.row()
        old_path = Path(self.paths[row]["old_path"])  # type: ignore[arg-type]
        logger.trace(f"Locating file: '{old_path}'")
        new_path = self.paths[row]["new_path"]
        if new_path and Path(new_path).exists():
            old_path = Path(new_path)
        base_dir = ""
        if old_path.parent.exists():
            base_dir = str(old_path.parent)
        # looking for a file
        directory = hp.get_directory(self, title="Locate directory...", base_dir=base_dir)
        if directory:
            directory = Path(directory)
            self.CONFIG.update(last_dir=directory)
            new_path = directory / old_path.name
            if not new_path.exists():
                logger.warning(f"File not found: '{new_path!r}'")
                return
            self.paths[row]["new_path"] = new_path
            logger.info(f"Located file - '{new_path!r}'")
        self.on_update_data_list()

    def on_find_files(self) -> None:
        """Find files."""
        directory = hp.get_directory(self, title="Locate directory...", base_dir=self.CONFIG.last_dir)
        if not directory:
            return
        self.CONFIG.update(last_dir=directory)
        for paths in self.paths:
            old_path = Path(paths["old_path"])  # type: ignore[arg-type]
            if old_path.exists():
                continue
            name = old_path.name
            suffix = old_path.suffix.lower()
            is_file = suffix in FILE_SUFFIXES
            if is_file:
                new_path = Path(directory) / name
                if new_path.exists():
                    paths["new_path"] = new_path
                    logger.info(f"Located file - '{paths['new_path']!r}'")
            else:
                paths["new_path"] = Path(directory)
        self.on_update_data_list()

    def on_update_data_list(self, _evt: ty.Any = None) -> None:
        """On load."""

        def _exists(path: Path) -> bool:
            try:
                return path.exists()
            except PermissionError:
                return False

        show_valid = self.show_valid_check.isChecked()
        show_missing = self.show_missing_check.isChecked()

        hp.clear_table(self.table)
        for paths in self.paths:
            old_path = Path(paths["old_path"])  # type: ignore[arg-type]
            old_exists = _exists(old_path)
            new_path = paths["new_path"]
            new_exists = False
            if new_path:
                new_path = Path(new_path)
                new_exists = _exists(new_path)
            exists = old_exists or new_exists
            if not show_valid and exists:
                continue
            if not show_missing and not exists:
                continue

            text_color = THEMES.get_hex_color("error") if not exists else THEMES.get_hex_color("text")

            # get model information
            index = self.table.rowCount()
            self.table.insertRow(index)

            # add old-path
            old_path_item = QTableWidgetItem(str(old_path))
            old_path_item.setToolTip(str(old_path))
            old_path_item.setFlags(old_path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            old_path_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            old_path_item.setForeground(QColor(text_color))
            self.table.setItem(index, self.TABLE_CONFIG.old_path, old_path_item)

            # add new-path
            new_path_item = QTableWidgetItem(str(new_path) if new_path else "")
            new_path_item.setToolTip(str(new_path) if new_path else "")
            new_path_item.setFlags(new_path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            new_path_item.setTextAlignment(Qt.AlignmentFlag.AlignLeft)
            new_path_item.setForeground(QColor(text_color))
            self.table.setItem(index, self.TABLE_CONFIG.new_path, new_path_item)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        self.table = hp.make_table(self, self.TABLE_CONFIG, elide=Qt.TextElideMode.ElideLeft)

        self.show_valid_check = hp.make_checkbox(self, "Show valid", value=True, func=self.on_update_data_list)
        self.show_missing_check = hp.make_checkbox(self, "Show missing", value=True, func=self.on_update_data_list)

        layout = hp.make_form_layout(parent=self)
        layout.addRow(
            hp.make_label(
                self,
                "At least one file read from the configuration file is <b>no longer at the specified path</b>."
                "<br>Please locate it on your hard drive by <b>specifying directory where it might be located</b> or "
                " it won't be imported.",
                alignment=Qt.AlignmentFlag.AlignHCenter,  # type: ignore[attr-defined]
            )
        )
        layout.addRow(self.table)
        layout.addRow(hp.make_h_line(self))
        layout.addRow(
            hp.make_label(
                self,
                "<b>Tip.</b> Double-click on a row to open search dialog for a specific file.",
                alignment=Qt.AlignmentFlag.AlignHCenter,
                object_name="tip_label",
                enable_url=True,
            )
        )
        layout.addRow(hp.make_h_line(self))
        layout.addRow(
            hp.make_h_layout(
                self.show_valid_check,
                self.show_missing_check,
                hp.make_btn(
                    self,
                    "Search",
                    func=self.on_find_files,
                    tooltip="Click here to specify a 'parent' directory where the files ae located. This will be used"
                    " to locate the missing files by searching for them in the specified directory.",
                ),
                spacing=2,
                stretch_id=(1,),
            )
        )

        layout.addRow(
            hp.make_h_layout(
                hp.make_btn(self, "OK", func=self.accept),
                hp.make_btn(self, "Cancel", func=self.reject),
            )
        )
        return layout


if __name__ == "__main__":  # pragma: no cover
    from qtextra.utils.dev import apply_style, qapplication

    from image2image.config import SingleAppConfig

    app = qapplication()
    dlg = LocateFilesDialog(
        None,
        SingleAppConfig(),  # type: ignore[call-arg]
        [
            "/Users/lgmigas/Documents/_projects_/2024_hickey_kruse2/B011-reg007/codex2ims.valis/Images/he.ome.tiff",
            "/Users/lgmigas/Documents/_projects_/2024_hickey_kruse2/B011-reg007/codex2ims.valis/he.ome.tiff",
        ],
    )
    apply_style(dlg)
    dlg.exec_()
