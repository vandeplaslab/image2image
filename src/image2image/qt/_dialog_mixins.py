"""Mixin classes."""

from __future__ import annotations

import typing as ty
from pathlib import Path

import qtextra.helpers as hp
from koyo.typing import PathLike
from loguru import logger
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtpy.QtWidgets import QDialog, QMenuBar

from image2image.qt.dialog_base import Window

if ty.TYPE_CHECKING:
    from qtextra._napari.image.wrapper import NapariImageView

    from image2image.models.data import DataModel
    from image2image.qt._dialogs._select import LoadWidget


class SingleViewerMixin(Window):
    """Mixin class for single viewer."""

    _output_dir = None
    view: NapariImageView
    _image_widget: LoadWidget

    WINDOW_CONSOLE_ARGS: tuple[str, ...] = ()

    def on_set_output_dir(self) -> None:
        """Set output directory."""
        self.output_dir = hp.get_directory(self, "Select output directory", self.CONFIG.output_dir)

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if self.CONFIG.output_dir is None:
                return Path.cwd()
            return Path(self.CONFIG.output_dir)
        return Path(self._output_dir)

    @output_dir.setter
    def output_dir(self, directory: PathLike) -> None:
        if directory:
            self._output_dir = directory
            self.CONFIG.update(output_dir=directory)
            formatted_output_dir = f".{self.output_dir.parent}/{self.output_dir.name}"
            self.output_dir_label.setText(hp.hyper(self.output_dir, value=formatted_output_dir))
            logger.debug(f"Output directory set to {self._output_dir}")

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def _make_menu(self) -> None:
        """Make menu items."""
        # File menu
        menu_file = hp.make_menu(self, "File")
        hp.make_menu_item(
            self,
            "Add image (.tiff, .czi, + others)...",
            "Ctrl+I",
            menu=menu_file,
            func=self._image_widget.on_select_dataset,
        )
        menu_file.addSeparator()
        hp.make_menu_item(
            self,
            "Clear data",
            menu=menu_file,
            func=self._image_widget.on_close_dataset,
        )
        menu_file.addSeparator()
        hp.make_menu_item(self, "Quit", menu=menu_file, func=self.close)

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(self._make_tools_menu(scalebar=True).menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        super()._make_statusbar()
        self._make_scalebar_statusbar()
        self._make_export_statusbar()

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if (
            not force
            or not self.CONFIG.confirm_close
            or QtConfirmCloseDialog(
                self,
                "confirm_close",
                self.on_save_to_project,
                self.CONFIG,
            ).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if (
            evt.spontaneous()
            and self.CONFIG.confirm_close
            and QtConfirmCloseDialog(
                self,
                "confirm_close",
                self.on_save_to_project,
                self.CONFIG,
            ).exec_()  # type: ignore[attr-defined]
            != QDialog.DialogCode.Accepted
        ):
            evt.ignore()
            return
        if self._console:
            self._console.close()
        self.CONFIG.save()
        evt.accept()

    def _get_console_variables(self) -> dict:
        """Get variables for the console."""

        def _get_nester_arg(args: tuple[str, ...]) -> ty.Any:
            """Get nested argument."""
            obj = self
            for a in args:
                obj = getattr(obj, a)
            return obj

        variables = super()._get_console_variables()
        for arg in self.WINDOW_CONSOLE_ARGS:
            if isinstance(arg, tuple):
                variables[arg[-1]] = _get_nester_arg(arg)
            else:
                variables[arg] = getattr(self, arg)
        return variables
