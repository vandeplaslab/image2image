"""Dialog window base class."""
from __future__ import annotations

import typing as ty
from functools import partial

import qtextra.helpers as hp
from image2image_reader.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Layer, Shapes
from qtextra._napari.mixins import ImageViewMixin
from qtextra.config import THEMES
from qtextra.mixins import IndicatorMixin
from qtextra.widgets.qt_image_button import QtThemeButton
from qtpy.QtCore import Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QMainWindow, QMenu, QProgressBar, QStatusBar, QWidget
from superqt.utils import create_worker, ensure_main_thread

from image2image.config import CONFIG
from image2image.models.data import DataModel
from image2image.qt._dialogs._update import check_version
from image2image.utils.utilities import get_colormap, log_exception_or_error

if ty.TYPE_CHECKING:
    from qtextra._napari.image.wrapper import NapariImageView


class Window(QMainWindow, IndicatorMixin, ImageViewMixin):
    """Base class window for all apps."""

    _console = None

    allow_drop: bool = True
    evt_dropped = Signal("QEvent")

    def __init__(self, parent: QWidget | None, title: str, delay_events: bool = False):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)  # type: ignore[attr-defined]
        self.setWindowTitle(title)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setMinimumSize(1200, 800)

        self._setup_ui()
        if not delay_events:
            self.setup_events()
        else:
            hp.call_later(self, self.setup_events, 3000)

        # check for updates every now and in then every 4 hours
        hp.call_later(self, self.on_check_new_version, 5 * 1000)
        self.version_timer = hp.make_periodic_timer(self, self.on_check_new_version, 4 * 3600 * 1000)

        # synchronize themes
        THEMES.evt_theme_changed.connect(self.on_changed_theme)

        # most apps will benefit from this
        READER_CONFIG.init_pyramid = True
        READER_CONFIG.auto_pyramid = True
        READER_CONFIG.split_czi = True

    def on_toggle_theme(self) -> None:
        """Toggle theme."""
        THEMES.theme = "dark" if self.theme_btn.dark else "light"
        CONFIG.theme = THEMES.theme

    def on_changed_theme(self) -> None:
        """Update theme of the app."""
        CONFIG.theme = THEMES.theme
        THEMES.set_theme_stylesheet(self)
        # update console
        if self._console:
            self._console._console._update_theme()

    def on_check_new_version(self) -> None:
        """Check for the new version."""
        create_worker(
            check_version,
            _connect={
                "returned": self._on_set_new_version,
                "errored": lambda: hp.toast(
                    self, "Failed", "Failed checking for new version", icon="error", position="top_left"
                ),
            },
        )

    @ensure_main_thread()
    def _on_set_new_version(self, res: tuple[bool, str]) -> None:
        """Set the result of version check."""
        is_new_available, reason = res
        if not is_new_available:
            logger.debug("Using the latest version of the app.")
            return
        hp.long_toast(self, "New version available!", reason, 15_000, icon="info", position="top_left")
        logger.debug("Checked for latest version.")
        self.update_status_btn.show()

    def _setup_ui(self) -> None:
        """Create panel."""
        raise NotImplementedError("Must implement method")

    def setup_events(self, state: bool = True) -> None:
        """Additional setup."""
        raise NotImplementedError("Must implement method")

    def _toggle_channel(
        self, model: DataModel, view_wrapper: NapariImageView, name: str, state: bool, view_kind: str
    ) -> None:
        if name not in view_wrapper.layers:
            logger.warning(f"Layer '{name}' not found in the view.")
            self.image_layer, self.shape_layer = self._plot_image_layers(model, view_wrapper, [name], view_kind, True)
            return
        view_wrapper.layers[name].visible = state

    def _toggle_all_channels(
        self,
        model: DataModel,
        view_wrapper: NapariImageView,
        state: bool,
        view_kind: str,
        channel_names: list[str] | None = None,
    ) -> None:
        wrapper = model.wrapper
        if not wrapper:
            for layer in view_wrapper.layers:
                if isinstance(layer, (Image, Shapes)):
                    if channel_names and layer.name not in channel_names:
                        continue
                    layer.visible = state
        else:
            for name in wrapper.channel_names():
                if channel_names and name not in channel_names:
                    continue
                if name not in view_wrapper.layers:
                    logger.warning(f"Layer '{name}' not found in the view.")
                    self.image_layer, self.shape_layer = self._plot_image_layers(
                        model, view_wrapper, [name], view_kind, True
                    )
                    continue
                view_wrapper.layers[name].visible = state

    @staticmethod
    def _plot_image_layers(
        model: DataModel,
        view_wrapper: NapariImageView,
        channel_list: list[str] | None = None,
        view_kind: str = "view",
        scale: bool = False,
    ) -> tuple[list[Image] | None, list[Shapes] | None]:
        wrapper = model.wrapper
        if not wrapper:
            logger.error("Failed to get wrapper.")
            return None, None
        if channel_list is None:
            channel_list = wrapper.channel_names()
        image_layer, shape_layer = [], []
        for index, (name, array, reader) in enumerate(wrapper.channel_image_reader_iter()):
            if name not in channel_list:
                continue
            logger.trace(f"Adding '{name}' to view...")
            with MeasureTimer() as timer:
                if name in view_wrapper.layers:
                    if reader.reader_type == "shapes":
                        shape_layer.append(view_wrapper.layers[name])
                    else:
                        image_layer.append(view_wrapper.layers[name])
                    continue

                # get current transform and scale
                # current_affine = reader.transform
                current_affine = wrapper.get_affine(reader, reader.resolution) if scale else reader.transform
                # current_affine = (
                #     wrapper.update_affine(reader.transform, reader.resolution) if scale else reader.transform
                # )
                current_scale = reader.scale if scale else (1, 1)
                if reader.reader_type == "shapes" and hasattr(reader, "to_shapes_kwargs"):
                    shape_layer.append(
                        view_wrapper.viewer.add_shapes(**reader.to_shapes_kwargs(name=name, affine=current_affine))
                    )
                else:
                    image_layer.append(
                        view_wrapper.viewer.add_image(
                            array,
                            name=name,
                            blending="additive",
                            colormap=get_colormap(index, view_wrapper.layers),
                            visible=name in channel_list,
                            affine=current_affine,
                            scale=current_scale,
                        )
                    )
                logger.trace(f"Added '{name}' to {view_kind} in {timer()}.")
        return image_layer, shape_layer

    @staticmethod
    def _close_model(model: DataModel, view_wrapper: NapariImageView, view_kind: str = "view") -> None:
        """Close model."""
        try:
            channel_names = model.channel_names()
            layer_names = [layer.name for layer in view_wrapper.layers if isinstance(layer, (Image, Shapes))]
            for name in layer_names:
                if name not in channel_names:
                    del view_wrapper.layers[name]
                    logger.trace(f"Removed '{name}' from {view_kind}.")
        except Exception as e:
            log_exception_or_error(e)

    @staticmethod
    def _move_layer(view: NapariImageView, layer: Layer, new_index: int = -1, select: bool = True) -> None:
        """Move a layer and select it."""
        view.layers.move(view.layers.index(layer), new_index)
        if select:
            view.layers.selection.select_only(layer)

    def on_show_console(self) -> None:
        """View console."""
        if self._console is None:
            from qtextra.dialogs.qt_console import QtConsoleDialog

            self._console = QtConsoleDialog(self)
        self._console.push_variables(self._get_console_variables())
        self._console.show()

    def _get_console_variables(self) -> dict:
        """Get variables for the console."""
        return {"window": self, "config": CONFIG}

    def _make_icon(self) -> None:
        """Make icon."""
        from image2image.assets import ICON_ICO

        icon = hp.get_icon_from_img(ICON_ICO)
        if icon:
            self.setWindowIcon(icon)

    def _make_config_menu(self) -> QMenu:
        from koyo.path import open_directory_alt

        from image2image.utils._appdirs import USER_CONFIG_DIR, USER_LOG_DIR

        menu_config = hp.make_menu(self, "Config")
        hp.make_menu_item(
            self, "Open 'Config' directory", menu=menu_config, func=lambda: open_directory_alt(USER_CONFIG_DIR)
        )
        hp.make_menu_item(self, "Open 'Log' directory", menu=menu_config, func=lambda: open_directory_alt(USER_LOG_DIR))
        return menu_config

    def _make_help_menu(self) -> QMenu:
        from image2image.qt._dialogs import open_about
        from image2image.qt._sentry import ask_opt_in, send_feedback
        from image2image.utils.utilities import open_bug_report, open_docs, open_github, open_request

        menu_help = hp.make_menu(self, "Help")
        hp.make_menu_item(self, "Documentation (online)", menu=menu_help, icon="web", func=open_docs, shortcut="F1")
        hp.make_menu_item(
            self,
            "GitHub (online)",
            menu=menu_help,
            status_tip="Open project's GitHub page.",
            icon="github",
            func=open_github,
            disabled=True,
        )
        hp.make_menu_item(
            self,
            "Request Feature (online)",
            menu=menu_help,
            status_tip="Open project's GitHub feature request page.",
            icon="request",
            func=open_request,
            disabled=True,
        )
        hp.make_menu_item(
            self,
            "Report Bug (online)",
            menu=menu_help,
            status_tip="Open project's GitHub bug report page.",
            icon="bug",
            func=open_bug_report,
            disabled=True,
        )
        menu_help.addSeparator()
        hp.make_menu_item(
            self,
            "Send feedback or request...",
            menu=menu_help,
            func=partial(send_feedback, parent=self),
            icon="feedback",
        )
        hp.make_menu_item(self, "Telemetry...", menu=menu_help, func=partial(ask_opt_in, parent=self), icon="telemetry")
        hp.make_menu_item(self, "About...", menu=menu_help, func=partial(open_about, parent=self), icon="info")
        return menu_help

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        from image2image.qt._sentry import send_feedback

        self.statusbar = QStatusBar()  # noqa
        self.statusbar.setSizeGripEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximumWidth(200)
        self.statusbar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()

        self.feedback_btn = hp.make_qta_btn(
            self,
            "feedback",
            tooltip="Refresh task list ahead of schedule.",
            func=partial(send_feedback, parent=self),
            small=True,
        )
        self.statusbar.addPermanentWidget(self.feedback_btn)

        self.theme_btn = QtThemeButton(self)
        self.theme_btn.auto_connect()
        with hp.qt_signals_blocked(self.theme_btn):
            self.theme_btn.dark = CONFIG.theme == "dark"
        self.theme_btn.clicked.connect(self.on_toggle_theme)  # noqa
        self.theme_btn.set_small()
        self.statusbar.addPermanentWidget(self.theme_btn)

        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "ipython",
                tooltip="Open IPython console",
                small=True,
                func=self.on_show_console,
            )
        )

        self.tutorial_btn = hp.make_qta_btn(
            self, "help", tooltip="Give me a quick tutorial!", func=self.on_show_tutorial, small=True
        )
        self.statusbar.addPermanentWidget(self.tutorial_btn)

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

    def on_show_update_info(self) -> None:
        """Show information about available updates."""
        from koyo.release import format_version, get_latest_git
        from qtextra.widgets.changelog import ChangelogDialog

        data = get_latest_git(package="image2image-docs")
        text = format_version(data)
        dlg = ChangelogDialog(self, text)
        dlg.exec_()  # type: ignore

    def on_show_tutorial(self) -> None:
        """Quick tutorial."""
        hp.toast(self, "Tutorial", "Coming soon...", icon="info", position="top_left")

    def dragEnterEvent(self, event):
        """Override Qt method.

        Provide style updates on event.
        """
        if self.allow_drop:
            hp.toast(
                self,
                "Drag & drop",
                "Drop the files in the app and we will try to open them..",
                icon="info",
                position="top_left",
            )
            hp.update_property(self.centralWidget(), "drag", True)
            if event.mimeData().hasUrls():
                event.accept()
            else:
                event.ignore()
        else:
            hp.toast(
                self,
                "Drag & drop not allowed",
                "Drag & drop is not allowed in this app.",
                icon="error",
                position="top_left",
            )

    def dragLeaveEvent(self, event):
        """Override Qt method."""
        if self.allow_drop:
            hp.update_property(self.centralWidget(), "drag", False)

    def dropEvent(self, event):
        """Override Qt method."""
        if self.allow_drop:
            hp.update_property(self.centralWidget(), "drag", False)
            self.evt_dropped.emit(event)
