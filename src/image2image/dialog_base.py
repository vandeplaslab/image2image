"""Dialog window base class."""
import typing as ty
from functools import partial

import qtextra.helpers as hp
from koyo.timer import MeasureTimer
from loguru import logger
from napari.layers import Image
from qtextra._napari.mixins import ImageViewMixin
from qtextra.mixins import IndicatorMixin
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QMainWindow

# need to load to ensure all assets are loaded properly
import image2image.assets  # noqa: F401
from image2image._sentry import install_error_monitor
from image2image.config import CONFIG
from image2image.models import DataModel
from image2image.utilities import get_colormap, log_exception

if ty.TYPE_CHECKING:
    from qtextra._napari.image.viewer import NapariImageView


class Window(QMainWindow, IndicatorMixin, ImageViewMixin):
    """Base class window for all apps.."""

    _console = None

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)  # noqa
        self.setWindowTitle(title)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setMouseTracking(True)
        self.setMinimumSize(1200, 800)

        # load configuration
        CONFIG.load()

        self._setup_ui()
        self.setup_events()

        # delay asking for telemetry opt-in by 10s
        hp.call_later(self, install_error_monitor, 5_000)

    def _setup_ui(self):
        """Create panel."""
        raise NotImplementedError("Must implement method")

    def setup_events(self, state: bool = True):
        """Additional setup."""
        raise NotImplementedError("Must implement method")

    @staticmethod
    def _plot_image_layers(
        model: "DataModel",
        view_wrapper: "NapariImageView",
        channel_list: ty.Optional[ty.List[str]] = None,
        view_kind: str = "view",
        scale: bool = False,
    ):
        wrapper = model.get_wrapper()
        if channel_list is None:
            channel_list = wrapper.channel_names()
        image_layer = []
        for index, (name, array, reader) in enumerate(wrapper.channel_image_reader_iter()):
            logger.trace(f"Adding '{name}' to view...")
            with MeasureTimer() as timer:
                if name in view_wrapper.layers:
                    image_layer.append(view_wrapper.layers[name])
                    continue
                # get current transform and scale
                current_affine = (
                    wrapper.update_affine(reader.transform, reader.resolution) if scale else reader.transform
                )
                current_scale = reader.scale if scale else (1, 1)
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
        return image_layer

    @staticmethod
    def _close_model(model: "DataModel", view_wrapper: "NapariImageView", view_kind: str = "view"):
        """Close model."""
        try:
            channel_names = model.channel_names()
            layer_names = [layer.name for layer in view_wrapper.layers if isinstance(layer, Image)]
            for name in layer_names:
                if name not in channel_names:
                    del view_wrapper.layers[name]
                    logger.trace(f"Removed '{name}' from {view_kind}.")
        except Exception as e:
            log_exception(e)

    @staticmethod
    def _move_layer(view, layer, new_index: int = -1, select: bool = True):
        """Move a layer and select it."""
        view.layers.move(view.layers.index(layer), new_index)
        if select:
            view.layers.selection.select_only(layer)

    def on_show_console(self):
        """View console."""
        if self._console is None:
            from qtextra.dialogs.qt_console import QtConsoleDialog

            self._console = QtConsoleDialog(self)
            self._console.push_variables(self._get_console_variables())
        self._console.show()

    def _get_console_variables(self) -> ty.Dict:
        """Get variables for the console."""
        raise NotImplementedError("Must implement method")

    def _make_icon(self):
        """Make icon."""
        from image2image.assets import ICON_ICO

        self.setWindowIcon(hp.get_icon_from_img(ICON_ICO))

    def _make_help_menu(self):
        from image2image._dialogs import open_about
        from image2image._sentry import ask_opt_in, send_feedback
        from image2image.utilities import open_bug_report, open_docs, open_github, open_request

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
