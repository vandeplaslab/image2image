import re
import sys
import typing as ty
from ipykernel.inprocess.ipkernel import InProcessInteractiveShell
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython import get_ipython
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from qtconsole.inprocess import QtInProcessKernelManager
from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QFormLayout
from qtextra.widgets.qt_dialog import QtFramelessTool
import qtextra.helpers as hp
from ims2micro.utilities import style_form_layout
from ims2micro.config import CONFIG


def str_to_rgb(arg):
    """Convert an rgb string 'rgb(x,y,z)' to a list of ints [x,y,z]."""
    return list(map(int, re.match(r"rgb\((\d+),\s*(\d+),\s*(\d+)\)", arg).groups()))


"""
set default asyncio policy to be compatible with tornado

Tornado 6 (at least) is not compatible with the default
asyncio implementation on Windows

Pick the older SelectorEventLoopPolicy on Windows
if the known-incompatible default policy is in use.

FIXME: if/when tornado supports the defaults in asyncio,
remove and bump tornado requirement for py38
borrowed from ipykernel:  https://github.com/ipython/ipykernel/pull/456
"""
if sys.platform.startswith("win"):
    import asyncio

    try:
        from asyncio import (
            WindowsProactorEventLoopPolicy,
            WindowsSelectorEventLoopPolicy,
        )
    except ImportError:
        pass
        # not affected
    else:
        if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
            # WindowsProactorEventLoopPolicy is not compatible with tornado 6
            # fallback to the pre-3.8 default of Selector
            asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())


class QtConsole(RichJupyterWidget):
    """Qt view for the console, an integrated iPython terminal in napari.

    Parameters
    ----------
    user_variables : dict
        Dictionary of user variables to declare in console name space.

    Attributes
    ----------
    kernel_client : qtconsole.inprocess.QtInProcessKernelClient,
                    qtconsole.client.QtKernelClient, or None
        Client for the kernel if it exists, None otherwise.
    shell : ipykernel.inprocess.ipkernel.InProcessInteractiveShell,
            ipykernel.zmqshell.ZMQInteractiveShell, or None.
        Shell for the kernel if it exists, None otherwise.
    """

    def __init__(self, variables: ty.Dict[str, ty.Any] = None):
        super().__init__()
        # Connect theme update
        if variables is None:
            variables = {}

        # this makes calling `setFocus()` on a QtConsole give keyboard focus to
        # the underlying `QTextEdit` widget
        self.setFocusProxy(self._control)

        # get current running instance or create new instance
        shell = get_ipython()

        if shell is None:
            # If there is no currently running instance create an in-process
            # kernel.
            kernel_manager = QtInProcessKernelManager()
            kernel_manager.start_kernel(show_banner=False)
            kernel_manager.kernel.gui = "qt"

            kernel_client = kernel_manager.client()
            kernel_client.start_channels()

            self.kernel_manager = kernel_manager
            self.kernel_client = kernel_client
            self.shell = kernel_manager.kernel.shell
            self.push = self.shell.push
        elif type(shell) == InProcessInteractiveShell:
            # If there is an existing running InProcessInteractiveShell
            # it is likely because multiple viewers have been launched from
            # the same process. In that case create a new kernel.
            # Connect existing kernel
            kernel_manager = QtInProcessKernelManager(kernel=shell.kernel)
            kernel_client = kernel_manager.client()

            self.kernel_manager = kernel_manager
            self.kernel_client = kernel_client
            self.shell = kernel_manager.kernel.shell
            self.push = self.shell.push
        elif isinstance(shell, (TerminalInteractiveShell, ZMQInteractiveShell)):
            # if launching from an ipython or jupyter then adding a console is
            # not supported. Instead users should use the existing interactive
            # terminal for the same functionality.
            self.kernel_client = None
            self.kernel_manager = None
            self.shell = None
            self.push = lambda var: None

        else:
            raise ValueError("ipython shell not recognized; " f"got {type(shell)}")
        # Add any user variables
        self.push(variables)

        self.enable_calltips = False

        # Set stylings
        self._update_theme()

        # TODO: Try to get console from jupyter to run without a shift click
        # self.execute_on_complete_input = True

    def _update_theme(self, event=None):
        """Update the napari GUI theme."""
        from qtextra.config.theme import THEMES

        # qtconsole unfortunately won't inherit the parent stylesheet
        # so it needs to be directly set
        # template and apply the primary stylesheet
        # (should probably be done by napari)
        # After napari 0.4.11, themes are evented models rather than
        # dicts.
        self.style_sheet = THEMES.get_theme_stylesheet()

        # After napari 0.4.6 the following syntax will be allowed
        # self.style_sheet = get_stylesheet(self.viewer.theme)

        # Set syntax styling and highlighting using theme
        self.syntax_style = THEMES.syntax_style
        bracket_color = QColor(*str_to_rgb(THEMES.get_rgb_color("highlight")))
        self._bracket_matcher.format.setBackground(bracket_color)

    def closeEvent(self, event):
        """Clean up the integrated console in napari."""
        if self.kernel_client is not None:
            self.kernel_client.stop_channels()
        if self.kernel_manager is not None and self.kernel_manager.has_kernel:
            self.kernel_manager.shutdown_kernel()

        # RichJupyterWidget doesn't clean these up
        self._completion_widget.deleteLater()
        self._call_tip_widget.deleteLater()
        self.deleteLater()
        event.accept()


class QtConsoleDialog(QtFramelessTool):
    """Console popup."""

    HIDE_WHEN_CLOSE = True

    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

    # noinspection PyAttributeOutsideInit
    def make_panel(self) -> QFormLayout:
        """Make panel."""
        _, header_layout = self._make_hide_handle()
        self._title_label.setText("IPython Console")

        self._console = QtConsole(
            variables={
                "viewer_fixed": self.parent().view_fixed.viewer,
                "model_fixed": self.parent()._fixed_widget.model,
                "viewer_moving": self.parent().view_moving.viewer,
                "model_moving": self.parent()._moving_widget.model,
            }
        )

        layout = hp.make_form_layout(self)
        style_form_layout(layout)
        layout.addRow(header_layout)
        layout.addRow(self._console)
        return layout
