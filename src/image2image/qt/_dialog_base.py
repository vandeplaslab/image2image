"""Dialog window base class."""

from __future__ import annotations

import typing as ty
from functools import partial

import qtextra.helpers as hp
from image2image_io.config import CONFIG as READER_CONFIG
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image, Layer, Points, Shapes
from qtextra.config import THEMES
from qtextra.dialogs.qt_logger import QtLoggerDialog
from qtextra.mixins import IndicatorMixin
from qtextra.widgets.qt_button_icon import QtThemeButton
from qtextraplot._napari.mixins import ImageViewMixin
from qtpy.QtCore import QProcess, Qt, Signal  # type: ignore[attr-defined]
from qtpy.QtWidgets import QMainWindow, QMenu, QProgressBar, QStatusBar, QWidget
from superqt.utils import create_worker, ensure_main_thread

from image2image.config import STATE, SingleAppConfig, get_app_config
from image2image.models.data import DataModel
from image2image.qt._dialogs._update import check_version
from image2image.utils._appdirs import USER_LOG_DIR
from image2image.utils.utilities import (
    get_colormap,
    get_contrast_limits,
    get_next_color,
    log_exception_or_error,
)

if ty.TYPE_CHECKING:
    from image2image_io.readers import BaseReader
    from qtextraplot._napari.image.wrapper import NapariImageView


class Window(QMainWindow, IndicatorMixin, ImageViewMixin):
    """Base class window for all apps."""

    APP_NAME: str = ""
    CONFIG: SingleAppConfig

    _console = None
    view: NapariImageView | None = None
    data_model: DataModel

    allow_drop: bool = True
    evt_dropped = Signal("QEvent")
    evt_initialized = Signal("QEvent")

    def __init__(self, parent: QWidget | None, title: str, delay_events: bool = False, run_check_version: bool = True):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setMinimumSize(1200, 800)

        self._setup_ui()
        # check for updates every now and in then every 4 hours
        if run_check_version:
            hp.call_later(self, self.on_check_new_version, 5 * 1000)
            self.version_timer = hp.make_periodic_timer(self, self.on_check_new_version, 4 * 3600 * 1000)

        if not delay_events:
            self.setup_events()
        else:
            hp.call_later(self, self.setup_events, 3000)

        # synchronize themes
        THEMES.evt_theme_changed.connect(self.on_changed_theme)

        # add logger
        self.logger_dlg = QtLoggerDialog(self, USER_LOG_DIR)
        self.temporary_layers: dict[str, Layer] = {}
        self._setup_config()

    @staticmethod
    def _setup_config() -> None:
        raise NotImplementedError("Must implement method")

    def on_toggle_theme(self) -> None:
        """Toggle theme."""
        THEMES.theme = "dark" if self.theme_btn.dark else "light"
        get_app_config().theme = THEMES.theme

    def on_changed_theme(self) -> None:
        """Update theme of the app."""
        get_app_config().theme = THEMES.theme
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
            hp.toast(
                self, "No new version", "You are using the latest version of the app.", icon="info", position="top_left"
            )
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
            self.image_layer, self.shape_layer, self.points_layer = self._plot_reader_layers(
                model, view_wrapper, [name], view_kind, True
            )
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
                if state and name not in view_wrapper.layers:
                    logger.warning(f"Layer '{name}' not found in the view.")
                    self.image_layer, self.shape_layer, self.points_layer = self._plot_reader_layers(
                        model, view_wrapper, [name], view_kind, True
                    )
                    continue
                if name in view_wrapper.layers:
                    view_wrapper.layers[name].visible = state

    @staticmethod
    def _plot_reader_layers(
        model: DataModel,
        view_wrapper: NapariImageView,
        channel_list: list[str] | None = None,
        view_kind: str = "view",
        scale: bool = False,
    ) -> tuple[list[Image] | None, list[Shapes] | None, list[Points] | None]:
        wrapper = model.wrapper
        if not wrapper:
            logger.error("Failed to get wrapper.")
            return None, None, None
        if channel_list is None:
            channel_list = wrapper.channel_names()
        image_layer, shape_layer, points_layer = [], [], []

        need_reset = len(view_wrapper.layers) == 0
        zoom = view_wrapper.viewer.camera.zoom
        center = view_wrapper.viewer.camera.center
        image_index = 0
        for _index, (name, array, reader) in enumerate(wrapper.channel_image_for_channel_names_iter(channel_list)):
            if name not in channel_list:
                continue
            logger.trace(f"Adding '{name}' to view...")
            with MeasureTimer() as timer:
                if name in view_wrapper.layers:
                    if reader.reader_type == "shapes":
                        shape_layer.append(view_wrapper.layers[name])
                    elif reader.reader_type == "points":
                        points_layer.append(view_wrapper.layers[name])
                    else:
                        image_layer.append(view_wrapper.layers[name])
                    continue

                # get current transform and scale
                # current_affine = reader.transform
                current_affine = wrapper.get_affine(reader, reader.resolution) if scale else reader.transform

                current_scale = reader.scale if scale else (1, 1)
                try:
                    if reader.reader_type == "shapes" and hasattr(reader, "to_shapes_kwargs"):
                        display_type = reader.display_type or READER_CONFIG.shape_display
                        if display_type == "points" and hasattr(reader, "to_points_kwargs"):
                            face_color = get_next_color(0, view_wrapper.layers, "points")
                            kws = reader.to_points_kwargs(name=name, affine=current_affine, face_color=face_color)
                            logger.trace(f"Adding '{name}' to {view_kind} with {len(kws['data']):,} points...")
                            points_layer.append(view_wrapper.viewer.add_points(**kws))
                        else:
                            edge_color = get_next_color(0, view_wrapper.layers, "shapes")
                            kws = reader.to_shapes_kwargs(name=name, affine=current_affine, edge_color=edge_color)
                            logger.trace(f"Adding '{name}' to {view_kind} with {len(kws['data']):,} shapes...")
                            shape_layer.append(view_wrapper.viewer.add_shapes(**kws))
                    elif reader.reader_type == "points" and hasattr(reader, "to_points_kwargs"):
                        face_color = get_next_color(0, view_wrapper.layers, "points")
                        kws = reader.to_points_kwargs(name=name, affine=current_affine, face_color=face_color)
                        logger.trace(f"Adding '{name}' to {view_kind} with {len(kws['data']):,} points...")
                        points_layer.append(view_wrapper.viewer.add_points(**kws))
                    else:
                        if array is None:
                            raise ValueError(f"Failed to get array for '{name}'.")
                        contrast_limits, contrast_limits_range = get_contrast_limits(array)
                        if any(v in name.lower() for v in ("brightfield", "bright")):
                            colormap = "gray"
                        else:
                            colormap = get_colormap(image_index, view_wrapper.layers)
                        image_index += 1
                        image_layer.append(
                            view_wrapper.viewer.add_image(
                                array,
                                name=name,
                                blending="additive",
                                colormap=colormap,
                                visible=name in channel_list,
                                affine=current_affine,
                                scale=current_scale,
                                contrast_limits=contrast_limits,
                            )
                        )
                        if contrast_limits_range:
                            image_layer[-1].contrast_limits_range = contrast_limits_range
                    logger.trace(f"Added '{name}' to {view_kind} in {timer()}.")
                except (TypeError, ValueError, AssertionError):
                    logger.exception(f"Failed to add '{name}' to {view_kind}.")
                    continue
        if need_reset:
            view_wrapper.viewer.reset_view()
        else:
            if zoom:
                view_wrapper.viewer.camera.zoom = zoom
            if center:
                view_wrapper.viewer.camera.center = center
        return image_layer, shape_layer, points_layer

    @staticmethod
    def _get_reader_for_key(model: DataModel, key: str) -> tuple[BaseReader | None, str | None]:
        wrapper = model.wrapper
        if not wrapper:
            logger.error("Failed to get wrapper.")
            return None, None
        reader = wrapper.data[key]
        return reader, f"temporary-{reader.reader_type}"

    def _plot_temporary_layer(
        self,
        model: DataModel,
        view_wrapper: NapariImageView,
        key: str,
        channel_index: int,
        scale: bool = False,
    ) -> None:
        wrapper = model.wrapper
        if not wrapper:
            logger.error("Failed to get wrapper.")
            return
        with MeasureTimer() as timer:
            reader = wrapper.data[key]

            # get current transform and scale
            current_affine = wrapper.get_affine(reader, reader.resolution) if scale else reader.transform
            current_scale = reader.scale if scale else (1, 1)
            full_channel_name = f"{reader.channel_names[channel_index]} | {key}"

            # get temporary layer
            name = f"temporary-{reader.reader_type}"
            layer, updated = None, False
            if name in view_wrapper.layers:
                layer = view_wrapper.layers[name]

            if reader.reader_type == "shapes" and hasattr(reader, "to_shapes_kwargs"):
                view_wrapper.remove_layer(name)
                if READER_CONFIG.shape_display == "points" and hasattr(reader, "to_points_kwargs"):
                    face_color = get_next_color(0, view_wrapper.layers, "points")
                    layer_data = reader.to_points_kwargs(face_color=face_color, name=name, affine=current_affine)
                    collected_in = timer(since_last=True)
                    layer = view_wrapper.viewer.add_points(**layer_data)
                    plotted_in = timer(since_last=True)
                else:
                    edge_color = get_next_color(0, view_wrapper.layers, "shapes")
                    layer_data = reader.to_shapes_kwargs(name=name, affine=current_affine, edge_color=edge_color)
                    collected_in = timer(since_last=True)
                    layer = view_wrapper.viewer.add_shapes(**layer_data)
                    plotted_in = timer(since_last=True)
            elif reader.reader_type == "points" and hasattr(reader, "to_points_kwargs"):
                view_wrapper.remove_layer(name)
                channel_name = reader.channel_names[channel_index]
                layer_data = reader.to_points_kwargs(face_color=channel_name, name=name, affine=current_affine)
                collected_in = timer(since_last=True)
                layer = view_wrapper.viewer.add_points(**layer_data)
                plotted_in = timer(since_last=True)
            else:
                array = reader.get_channel_pyramid(channel_index)
                collected_in = timer(since_last=True)
                contrast_limits, contrast_limits_range = get_contrast_limits(array)
                if any(v in name.lower() for v in ("brightfield", "bright")):
                    colormap = "gray"
                elif layer:
                    colormap = layer.colormap
                else:
                    colormap = get_colormap(channel_index, view_wrapper.layers)
                if layer and len(array) == 1:
                    layer_name = layer.metadata.get("name", " | ")
                    if layer_name.split(" | ")[1] == full_channel_name.split(" | ")[1]:
                        layer.data = array[0]
                        layer.contrast_limits = contrast_limits
                        updated = True
                if not updated:
                    view_wrapper.remove_layer(name)
                    layer = view_wrapper.viewer.add_image(
                        array,
                        name=name,
                        blending="additive",
                        colormap=colormap,
                        affine=current_affine,
                        scale=current_scale,
                        contrast_limits=contrast_limits,
                        metadata={"name": full_channel_name},
                    )
                plotted_in = timer(since_last=True)
                if contrast_limits_range:
                    layer.contrast_limits_range = contrast_limits_range
            self.temporary_layers[key] = layer
        logger.trace(
            f"Plotted temporary layer for '{key}' in {timer()} (collected={collected_in}; plotted={plotted_in})."
        )

    @staticmethod
    def _closing_model(
        model: DataModel, channel_names: list[str], view_wrapper: NapariImageView, view_kind: str = "view"
    ) -> None:
        """Close model."""
        try:
            for name in channel_names:
                if view_wrapper.remove_layer(name, silent=True):
                    logger.trace(f"Removed '{name}' from {view_kind}.")
        except Exception:  # noqa: BLE001
            log_exception_or_error(exc)

    @staticmethod
    def _close_model(
        model: DataModel, view_wrapper: NapariImageView, view_kind: str = "view", exclude_names: list[str] | None = None
    ) -> None:
        """Close model."""
        if not exclude_names:
            exclude_names = []
        try:
            channel_names = model.channel_names()
            layer_names = [layer.name for layer in view_wrapper.layers if isinstance(layer, (Image, Shapes, Points))]
            for name in layer_names:
                if (
                    name not in exclude_names
                    and name not in channel_names
                    and view_wrapper.remove_layer(name, silent=True)
                ):
                    logger.trace(f"Removed '{name}' from {view_kind}.")
        except Exception as exc:  # noqa: BLE001
            log_exception_or_error(exc)

    @staticmethod
    def _move_layer(view: NapariImageView, layer: Layer, new_index: int = -1, select: bool = True) -> None:
        """Move a layer and select it."""
        view.layers.move(view.layers.index(layer), new_index)
        if select:
            view.layers.selection.select_only(layer)
        else:
            view.layers.selection.toggle(layer)

    def on_show_logger(self) -> None:
        """View console."""
        self.logger_dlg.show()

    def on_show_console(self) -> None:
        """View console."""
        if self._console is None:
            from qtextra.dialogs.qt_console import QtConsoleDialog

            self._console = QtConsoleDialog(self)
        self._console.push_variables(self._get_console_variables())
        self._console.show()

    def on_save_to_project(self) -> None:
        """Save data to config file."""

    def on_load_from_project(self) -> None:
        """Load previous data."""

    def _get_console_variables(self) -> dict:
        """Get variables for the console."""
        import numpy as np

        return {
            "window": self,
            "APP_CONFIG": get_app_config(),
            "CONFIG": self.CONFIG,
            "READER_CONFIG": READER_CONFIG,
            "np": np,
        }

    def _make_icon(self) -> None:
        """Make icon."""

    #     from image2image.assets import ICON_ICO
    #
    #     icon = hp.get_icon_from_img(ICON_ICO)
    #     if icon:
    #         self.setWindowIcon(icon)

    def _make_tools_menu(self, scalebar: bool = False, shortcut: bool = False) -> QMenu:
        """Make tools menu."""
        menu_tools = hp.make_menu(self, "Tools")
        if scalebar:
            hp.make_menu_item(
                self,
                "Show scale bar controls...",
                "Ctrl+S",
                menu=menu_tools,
                icon="ruler",
                func=self.on_show_scalebar,
            )
        if shortcut:
            hp.make_menu_item(
                self, "Show shortcuts...", "Ctrl+Y", menu=menu_tools, func=self.on_show_shortcuts, icon="shortcut"
            )
        hp.make_menu_item(self, "Show Log window...", "Ctrl+L", menu=menu_tools, func=self.on_show_logger, icon="log")
        hp.make_menu_item(
            self, "Show IPython console...", "Ctrl+T", menu=menu_tools, func=self.on_show_console, icon="ipython"
        )
        return menu_tools

    def _make_config_menu(self) -> QMenu:
        from koyo.path import open_directory_alt

        from image2image.utils._appdirs import USER_CONFIG_DIR, USER_LOG_DIR

        menu_config = hp.make_menu(self, "Config")
        hp.make_menu_item(self, "Save config", menu=menu_config, icon="save", func=self.on_save_config)
        hp.make_menu_item(
            self, "Open Config directory", menu=menu_config, func=lambda: open_directory_alt(USER_CONFIG_DIR)
        )
        hp.make_menu_item(self, "Open Log directory", menu=menu_config, func=lambda: open_directory_alt(USER_LOG_DIR))
        return menu_config

    def _make_apps_menu(self) -> QMenu:
        # from koyo.system import IS_MAC_ARM, IS_PYINSTALLER

        menu_apps = hp.make_menu(self, "Apps")
        menu_apps.addSection("Viewers")
        hp.make_menu_item(self, "Open Viewer App", menu=menu_apps, func=self.on_open_viewer, icon="viewer")
        menu_apps.addSection("Register")
        hp.make_menu_item(self, "Open Register App", menu=menu_apps, func=self.on_open_register, icon="register")
        hp.make_menu_item(self, "Open Elastix App", menu=menu_apps, func=self.on_open_elastix, icon="elastix")
        hp.make_menu_item(self, "Open Valis App", menu=menu_apps, func=self.on_open_valis, icon="valis")
        menu_apps.addSection("Utilities")
        hp.make_menu_item(self, "Open Crop App", menu=menu_apps, func=self.on_open_crop, icon="crop")
        hp.make_menu_item(
            self,
            "Open Convert App",
            menu=menu_apps,
            func=self.on_open_convert,
            # disabled=STATE.allow_convert,
            icon="convert",
        )
        hp.make_menu_item(self, "Open Merge App", menu=menu_apps, func=self.on_open_merge, icon="merge")
        hp.make_menu_item(self, "Open Fusion App", menu=menu_apps, func=self.on_open_fusion, icon="fusion")
        menu_apps.addSeparator()
        hp.make_menu_item(self, "Open Launcher App", menu=menu_apps, func=self.on_open_launcher, icon="launch")
        return menu_apps

    def _make_help_menu(self) -> QMenu:
        from image2image.qt._dialogs import open_about, open_sysinfo
        from image2image.qt._dialogs._sentry import ask_opt_in, send_feedback
        from image2image.utils.utilities import open_bug_report, open_docs, open_github, open_request

        menu_help = hp.make_menu(self, "Help")
        hp.make_menu_item(
            self,
            "Documentation (online)",
            menu=menu_help,
            icon="web",
            func=lambda: open_docs(app=self.APP_NAME),
            shortcut="F1",
        )
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
        hp.make_menu_item(
            self, "System info...", menu=menu_help, func=partial(open_sysinfo, parent=self), icon="settings"
        )
        hp.make_menu_item(self, "About...", menu=menu_help, func=partial(open_about, parent=self), icon="info")
        menu_help.addSeparator()
        hp.make_menu_item(
            self,
            "Check for updates...",
            "",
            icon="reload",
            menu=menu_help,
            func=self.on_check_new_version,
        )
        return menu_help

    def _make_theme_statusbar(self) -> None:
        self.theme_btn = QtThemeButton(self)
        self.theme_btn.auto_connect()
        with hp.qt_signals_blocked(self.theme_btn):
            self.theme_btn.dark = get_app_config().theme == "dark"
        self.theme_btn.clicked.connect(self.on_toggle_theme)  # noqa
        self.theme_btn.set_small()
        self.statusbar.addPermanentWidget(self.theme_btn)

    def _make_update_statusbar(self) -> None:
        self.update_status_btn = hp.make_btn(
            self,
            "Update available - click here to download!",
            tooltip="Show information about available updates.",
            func=self.on_show_update_info,
        )
        self.update_status_btn.setObjectName("update_btn")
        self.update_status_btn.hide()
        self.statusbar.addPermanentWidget(self.update_status_btn)

    def _make_tutorial_statusbar(self) -> None:
        self.tutorial_btn = hp.make_qta_btn(
            self, "help", tooltip="Give me a quick tutorial!", func=self.on_show_tutorial, small=True
        )
        self.statusbar.addPermanentWidget(self.tutorial_btn)

    def _make_ipython_statusbar(self) -> None:
        self.statusbar.addPermanentWidget(
            hp.make_qta_btn(
                self,
                "ipython",
                tooltip="Open IPython console",
                small=True,
                func=self.on_show_console,
            )
        )

    def _make_shortcut_statusbar(self) -> None:
        self.shortcuts_btn = hp.make_qta_btn(
            self, "shortcut", tooltip="Show me shortcuts", func=self.on_show_shortcuts, small=True
        )
        self.statusbar.addPermanentWidget(self.shortcuts_btn)
        self.shortcuts_btn.hide()

    def _make_feedback_statusbar(self) -> None:
        from image2image.qt._dialogs._sentry import send_feedback

        self.feedback_btn = hp.make_qta_btn(
            self,
            "feedback",
            tooltip="Send feedback to the developers.",
            func=partial(send_feedback, parent=self),
            small=True,
        )
        self.statusbar.addPermanentWidget(self.feedback_btn)

    def _make_logger_statusbar(self) -> None:
        self.logger_btn = hp.make_qta_btn(
            self,
            "log",
            tooltip="Open log window.",
            func=self.on_show_logger,
            small=True,
        )
        self.statusbar.addPermanentWidget(self.logger_btn)

    def _make_scalebar_statusbar(self, front: bool = True) -> None:
        self.scalebar_btn = hp.make_qta_btn(
            self,
            "ruler",
            tooltip="Show scalebar.",
            func=self.on_activate_scalebar,
            func_menu=self.on_show_scalebar,
            small=True,
        )
        if front:
            self.statusbar.insertPermanentWidget(0, self.scalebar_btn)
        else:
            self.statusbar.addPermanentWidget(self.scalebar_btn)

    def _make_export_statusbar(self, front: bool = True) -> None:
        if self.view is None:
            raise ValueError("View is not initialized.")

        self.clipboard_btn = hp.make_qta_btn(
            self,
            "screenshot",
            tooltip="Take a snapshot of the canvas and copy it into your clipboard. Right-click to show dialog with"
            " more options.",
            func=lambda _: self.view.widget.clipboard(),
            func_menu=self.on_show_save_figure,
            small=True,
        )
        if front:
            self.statusbar.insertPermanentWidget(0, self.clipboard_btn)
        else:
            self.statusbar.addPermanentWidget(self.clipboard_btn)

        self.screenshot_btn = hp.make_qta_btn(
            self,
            "save",
            tooltip="Save snapshot of the canvas to file. Right-click to show dialog with more options.",
            func=lambda _: self.view.widget.on_save_figure(),
            func_menu=self.on_show_save_figure,
            small=True,
        )
        if front:
            self.statusbar.insertPermanentWidget(0, self.screenshot_btn)
        else:
            self.statusbar.addPermanentWidget(self.screenshot_btn)

    def _make_statusbar(self) -> None:
        """Make statusbar."""
        self.statusbar = QStatusBar()  # noqa
        self.statusbar.setSizeGripEnabled(False)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximumWidth(200)
        self.statusbar.addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()

        self._make_feedback_statusbar()
        self._make_theme_statusbar()
        self._make_shortcut_statusbar()
        self._make_tutorial_statusbar()
        self._make_logger_statusbar()
        self._make_ipython_statusbar()
        self._make_update_statusbar()
        self.setStatusBar(self.statusbar)

    def on_show_shortcuts(self) -> None:
        """Show shortcuts."""

    def on_activate_scalebar(self) -> None:
        """Activate scalebar."""
        if self.view is None:
            raise ValueError("View is not initialized.")
        self.view.viewer.scale_bar.visible = not self.view.viewer.scale_bar.visible

    def on_show_scalebar(self) -> None:
        """Show scale bar controls for the viewer."""
        from image2image.qt._dialogs._scalebar import QtScaleBarControls

        if self.view is None:
            raise ValueError("View is not initialized.")

        dlg = QtScaleBarControls(self.view.viewer, self.view.widget)
        dlg.set_px_size(self.data_model.min_resolution)
        dlg.show_above_widget(self.scalebar_btn)

    def on_show_save_figure(self) -> None:
        """Show scale bar controls for the viewer."""
        from qtextraplot._napari.common.widgets.screenshot_dialog import QtScreenshotDialog

        if self.view is None:
            raise ValueError("View is not initialized.")

        dlg = QtScreenshotDialog(self.view.widget, self)
        dlg.show_above_widget(self.screenshot_btn)

    def on_show_update_info(self) -> None:
        """Show information about available updates."""
        from koyo.release import format_version, get_latest_git
        from qtextra.dialogs.qt_changelog import ChangelogDialog

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
            hp.update_property(self.centralWidget(), "drag", True)
            hp.call_later(self, lambda: hp.update_property(self.centralWidget(), "drag", False), 2000)
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
                position="top_right",
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

    def _status_changed(self, event):
        """Update status bar."""
        if isinstance(event.value, str):
            self.statusbar.showMessage(event.value)
        elif isinstance(event.value, dict):
            d = event.value
            self.statusbar.showMessage(d["coordinates"])
        else:
            self.statusbar.showMessage("")

    def on_save_config(self) -> None:
        """Save configuration file."""
        self.CONFIG.save()

    @staticmethod
    def on_open_launcher(*args: str) -> None:
        """Open launcher application."""
        create_new_window("", *args)

    @staticmethod
    def on_open_convert(*args: str) -> None:
        """Open convert application."""
        if not STATE.allow_convert:
            hp.warn_pretty(
                None,
                "Not available on Apple Silicon - there is a bug that I can't find nor fix - sorry!",
                "App not available on this platform.",
            )
            return
        create_new_window("convert", *args)

    @staticmethod
    def on_open_fusion(*args: str) -> None:
        """Open fusion application."""
        create_new_window("fusion", *args)

    @staticmethod
    def on_open_merge(*args: str) -> None:
        """Open merge application."""
        create_new_window("merge", *args)

    @staticmethod
    def on_open_register(*args: str) -> None:
        """Open register application."""
        create_new_window("register", *args)

    @staticmethod
    def on_open_viewer(*args: str) -> None:
        """Open viewer application."""
        create_new_window("viewer", *args)

    @staticmethod
    def on_open_crop(*args: str) -> None:
        """Open crop application."""
        create_new_window("crop", *args)

    @staticmethod
    def on_open_elastix(*args: str) -> None:
        """Open elastix application."""
        create_new_window("elastix", *args)

    @staticmethod
    def on_open_valis(*args: str) -> None:
        """Open valis application."""
        if not STATE.allow_valis:
            logger.error("Valis and/or pyvips is not installed on this machine.")
        #     hp.warn_pretty(
        #     None,
        #     "Valis and/or pyvips is not installed on this machine. This app is not available, sorry!",
        #     "App not available on this platform.",
        # )
        # return

        create_new_window("valis", *args)


def create_new_window(plugin: str, *extra_args: ty.Any) -> None:
    """Create new window."""
    import os
    import sys

    from koyo.system import IS_WIN

    from image2image.utils.utilities import pad_str

    program = os.environ.get("IMAGE2IMAGE_PROGRAM", None)

    process = QProcess()
    args = sys.argv
    if not program and args:
        program = args.pop(0)
    process.setProgram(program)
    arguments = []
    if "--dev" in args:
        arguments.append("--dev")
    if plugin:
        arguments.append("--tool")
        # arguments.append(pad_str(plugin))
        arguments.append(plugin)
    if extra_args:
        for arg in extra_args:
            if isinstance(arg, str):
                arguments.append(pad_str(arg))
    logger.trace(f"Executing {program} {' '.join(arguments)}...")
    if IS_WIN and hasattr(process, "setNativeArguments"):
        process.setNativeArguments(" ".join(arguments))
    else:
        process.setArguments(arguments)
    process.startDetached()
