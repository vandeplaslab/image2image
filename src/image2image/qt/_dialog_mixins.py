"""Mixin classes."""

from __future__ import annotations

import typing as ty
from functools import partial
from pathlib import Path

import qtextra.helpers as hp
from koyo.typing import PathLike
from loguru import logger
from qtextra.widgets.qt_close_window import QtConfirmCloseDialog
from qtpy.QtWidgets import QDialog, QMenuBar, QStatusBar

from image2image.config import CONFIG
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

    WINDOW_CONFIG_ATTR: str = ""
    WINDOW_CONSOLE_ARGS: tuple[str, ...] = ()

    def on_set_output_dir(self) -> None:
        """Set output directory."""
        self.output_dir = hp.get_directory(self, "Select output directory", CONFIG.output_dir)

    @property
    def output_dir(self) -> Path:
        """Output directory."""
        if self._output_dir is None:
            if CONFIG.output_dir is None:
                return Path.cwd()
            return Path(CONFIG.output_dir)
        return Path(self._output_dir)

    @output_dir.setter
    def output_dir(self, directory: PathLike) -> None:
        if directory:
            self._output_dir = directory
            CONFIG.output_dir = directory
            formatted_output_dir = f".{self.output_dir.parent}/{self.output_dir.name}"
            self.output_dir_label.setText(hp.hyper(self.output_dir, value=formatted_output_dir))
            logger.debug(f"Output directory set to {self._output_dir}")

    @property
    def data_model(self) -> DataModel:
        """Return transform model."""
        return self._image_widget.model

    def on_activate_scalebar(self) -> None:
        """Activate scalebar."""
        self.view.viewer.scale_bar.visible = not self.view.viewer.scale_bar.visible

    def on_show_scalebar(self) -> None:
        """Show scale bar controls for the viewer."""
        from image2image.qt._dialogs._scalebar import QtScaleBarControls

        dlg = QtScaleBarControls(self.view.viewer, self.view.widget)
        dlg.set_px_size(self.data_model.min_resolution)
        dlg.show_above_widget(self.scalebar_btn)

    def on_show_save_figure(self) -> None:
        """Show scale bar controls for the viewer."""
        from qtextra._napari.common.widgets.screenshot_dialog import QtScreenshotDialog

        dlg = QtScreenshotDialog(self.view, self)
        dlg.show_above_widget(self.clipboard_btn)

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

        # Tools menu
        menu_tools = hp.make_menu(self, "Tools")
        hp.make_menu_item(
            self, "Show scale bar controls...", "Ctrl+S", menu=menu_tools, icon="ruler", func=self.on_show_scalebar
        )
        menu_tools.addSeparator()
        hp.make_menu_item(self, "Show Logger...", "Ctrl+L", menu=menu_tools, func=self.on_show_logger)
        hp.make_menu_item(
            self, "Show IPython console...", "Ctrl+T", menu=menu_tools, icon="ipython", func=self.on_show_console
        )

        # set actions
        self.menubar = QMenuBar(self)
        self.menubar.addAction(menu_file.menuAction())
        self.menubar.addAction(menu_tools.menuAction())
        self.menubar.addAction(self._make_apps_menu().menuAction())
        self.menubar.addAction(self._make_config_menu().menuAction())
        self.menubar.addAction(self._make_help_menu().menuAction())
        self.setMenuBar(self.menubar)

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        from qtextra.widgets.qt_image_button import QtThemeButton

        from image2image.qt._dialogs._sentry import send_feedback

        self.statusbar = QStatusBar()
        self.statusbar.setSizeGripEnabled(False)

        self.screenshot_btn = hp.make_qta_btn(
            self,
            "save",
            tooltip="Save snapshot of the canvas to file. Right-click to show dialog with more options.",
            func=self.view.widget.on_save_figure,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.screenshot_btn)

        self.clipboard_btn = hp.make_qta_btn(
            self,
            "screenshot",
            tooltip="Take a snapshot of the canvas and copy it into your clipboard. Right-click to show dialog with"
            " more options.",
            func=self.view.widget.clipboard,
            func_menu=self.on_show_save_figure,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.clipboard_btn)
        self.scalebar_btn = hp.make_qta_btn(
            self,
            "ruler",
            tooltip="Show scalebar.",
            func=self.on_activate_scalebar,
            func_menu=self.on_show_scalebar,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.scalebar_btn)

        self.feedback_btn = hp.make_qta_btn(
            self,
            "feedback",
            tooltip="Send feedback to the developers.",
            func=partial(send_feedback, parent=self),
            small=True,
        )
        self.statusbar.addPermanentWidget(self.feedback_btn)

        self.theme_btn = QtThemeButton(self)
        self.theme_btn.auto_connect()
        with hp.qt_signals_blocked(self.theme_btn):
            self.theme_btn.dark = CONFIG.theme == "dark"
        self.theme_btn.clicked.connect(self.on_toggle_theme)
        self.theme_btn.set_small()
        self.statusbar.addPermanentWidget(self.theme_btn)

        self.tutorial_btn = hp.make_qta_btn(
            self, "help", tooltip="Give me a quick tutorial!", func=self.on_show_tutorial, small=True
        )
        self.statusbar.addPermanentWidget(self.tutorial_btn)
        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "ipython",
                tooltip="Open IPython console",
                small=True,
                func=self.on_show_console,
            )
        )
        self.update_status_btn = hp.make_btn(
            self,
            "Update available - click here to download!",
            tooltip="Show information about available updates.",
            func=self.on_show_update_info,
        )
        self.update_status_btn.setObjectName("update_btn")
        self.update_status_btn.hide()
        self.statusbar.addPermanentWidget(self.update_status_btn)
        self.setStatusBar(self.statusbar)

    def close(self, force=False):
        """Override to handle closing app or just the window."""
        if not self.WINDOW_CONFIG_ATTR:
            return super().close()
        if (
            not force
            or not getattr(CONFIG, self.WINDOW_CONFIG_ATTR)
            or QtConfirmCloseDialog(self, self.WINDOW_CONFIG_ATTR, self.on_save_to_project, CONFIG).exec_()  # type: ignore[attr-defined]
            == QDialog.DialogCode.Accepted
        ):
            return super().close()
        return None

    def closeEvent(self, evt):
        """Close."""
        if self.WINDOW_CONFIG_ATTR:
            if (
                evt.spontaneous()
                and getattr(CONFIG, self.WINDOW_CONFIG_ATTR)
                and QtConfirmCloseDialog(self, self.WINDOW_CONFIG_ATTR, self.on_save_to_project, CONFIG).exec_()  # type: ignore[attr-defined]
                != QDialog.DialogCode.Accepted
            ):
                evt.ignore()
                return
        if self._console:
            self._console.close()
        CONFIG.save()
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
