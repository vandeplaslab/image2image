"""Event loop."""
import os
import sys
from typing import Optional
from warnings import warn

from napari._qt.dialogs.qt_notification import NapariQtNotification
from napari._qt.qt_event_loop import _ipython_has_eventloop, _pycharm_has_eventloop
from napari._qt.qthreading import wait_for_workers_to_quit
from napari._qt.utils import _maybe_allow_interrupt
from napari.plugins import plugin_manager
from napari.resources._icons import _theme_path
from napari.utils.notifications import notification_manager, show_console_notification
from napari.utils.theme import _themes
from qtpy.QtCore import QDir, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication
from superqt import QMessageHandler

from ims2micro import __version__
from ims2micro.assets import ICON_PNG

APP_ID = f"ims2micro.app.{__version__}"


def set_app_id(app_id):
    """Get app ID."""
    if os.name == "nt" and app_id and not getattr(sys, "frozen", False):
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)


_defaults = {
    "app_name": "ims2micro",
    "app_version": __version__,
    "icon": ICON_PNG,
    "org_name": "ims2micro",
    "org_domain": "",
    "app_id": APP_ID,
}


# store reference to QApplication to prevent garbage collection
_app_ref = None


def get_app(
    *,
    app_name: Optional[str] = None,
    app_version: Optional[str] = None,
    icon: Optional[str] = None,
    org_name: Optional[str] = None,
    org_domain: Optional[str] = None,
    app_id: Optional[str] = None,
    ipy_interactive: Optional[bool] = None,
) -> QApplication:
    """Get or create the Qt QApplication.

    There is only one global QApplication instance, which can be retrieved by
    calling get_app again, (or by using QApplication.instance())

    Parameters
    ----------
    app_name : str, optional
        Set app name (if creating for the first time), by default 'ims2micro'
    app_version : str, optional
        Set app version (if creating for the first time), by default __version__
    icon : str, optional
        Set app icon (if creating for the first time), by default
        NAPARI_ICON_PATH
    org_name : str, optional
        Set organization name (if creating for the first time), by default
        'napari'
    org_domain : str, optional
        Set organization domain (if creating for the first time), by default
        'napari.org'
    app_id : str, optional
        Set organization domain (if creating for the first time).  Will be
        passed to set_app_id (which may also be called independently), by
        default NAPARI_APP_ID
    ipy_interactive : bool, optional
        Use the IPython Qt event loop ('%gui qt' magic) if running in an
        interactive IPython terminal.

    Returns
    -------
    QApplication
        [description]

    Notes
    -----
    Substitutes QApplicationWithTracing when the NAPARI_PERFMON env variable
    is set.

    If the QApplication already exists, we call convert_app_for_tracing() which
    deletes the QApplication and creates a new one. However here with get_app
    we need to create the correct QApplication up front, or we will crash
    because we'd be deleting the QApplication after we created QWidgets with
    it, such as we do for the splash screen.
    """
    # napari defaults are all-or nothing.  If any of the keywords are used
    # then they are all used.
    set_values = {k for k, v in locals().items() if v}
    kwargs = locals() if set_values else _defaults
    global _app_ref

    with QMessageHandler():
        app = QApplication.instance()
        if app:
            set_values.discard("ipy_interactive")
            if set_values:
                warn(
                    f"QApplication already existed, these arguments to to 'get_app' were ignored: {set_values}",
                    stacklevel=1,
                )
        else:
            # automatically determine monitor DPI.
            # Note: this MUST be set before the QApplication is instantiated
            os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
            QApplication.setAttribute(Qt.AA_UseStyleSheetPropagationInWidgetStyles, True)
            QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)

            # if this is the first time the Qt app is being instantiated, we set
            # the name and metadata
            app = QApplication(sys.argv)
            app.setApplicationName(kwargs.get("app_name"))
            app.setApplicationVersion(kwargs.get("app_version"))
            app.setOrganizationName(kwargs.get("org_name"))
            app.setOrganizationDomain(kwargs.get("org_domain"))
            set_app_id(kwargs.get("app_id"))

        if not _ipython_has_eventloop():
            notification_manager.notification_ready.connect(NapariQtNotification.show_notification)
            notification_manager.notification_ready.connect(show_console_notification)

        if app.windowIcon().isNull():
            app.setWindowIcon(QIcon(kwargs.get("icon")))

        if not _app_ref:  # running get_app for the first time
            # see docstring of `wait_for_workers_to_quit` for caveats on killing
            # workers at shutdown.
            app.aboutToQuit.connect(wait_for_workers_to_quit)

            # Setup search paths for currently installed themes.
            for name in _themes:
                QDir.addSearchPath(f"theme_{name}", str(_theme_path(name)))

            try:
                # this will register all of our resources (icons) with Qt, so that they
                # can be used in qss files and elsewhere.
                plugin_manager.discover_icons()
                plugin_manager.discover_qss()
            except AttributeError:
                pass

        _app_ref = app  # prevent garbage collection

        return app


def quit_app():
    """Close all windows and quit the QApplication if ims2micro started it."""
    QApplication.closeAllWindows()
    # if we started the application then the app will be named 'ims2micro'.
    if QApplication.applicationName() == "ims2micro" and not _ipython_has_eventloop():
        QApplication.quit()

    # otherwise, something else created the QApp before us (such as
    # %gui qt IPython magic).  If we quit the app in this case, then
    # *later* attempts to instantiate a ims2micro viewer won't work until
    # the event loop is restarted with app.exec_().  So rather than
    # quit just close all the windows (and clear our app icon).
    else:
        QApplication.setWindowIcon(QIcon())


def run(*, force=False, max_loop_level=1, _func_name="run"):
    """Start the Qt Event Loop.

    Parameters
    ----------
    force : bool, optional
        Force the application event_loop to start, even if there are no top
        level widgets to show.
    max_loop_level : int, optional
        The maximum allowable "loop level" for the execution thread.  Every
        time `QApplication.exec_()` is called, Qt enters the event loop,
        increments app.thread().loopLevel(), and waits until exit() is called.
        This function will prevent calling `exec_()` if the application already
        has at least ``max_loop_level`` event loops running.  By default, 1.
    _func_name : str, optional
        name of calling function, by default 'run'.  This is only here to
        provide functions like `gui_qt` a way to inject their name into the
        warning message.

    Raises
    ------
    RuntimeError
        (To avoid confusion) if no widgets would be shown upon starting the
        event loop.
    """
    if _ipython_has_eventloop():
        # If %gui qt is active, we don't need to block again.
        return

    app = QApplication.instance()
    if _pycharm_has_eventloop(app):
        # explicit check for PyCharm pydev console
        return

    if not app:
        raise RuntimeError(
            "No Qt app has been created. One can be created by calling `get_app()` or `qtpy.QtWidgets.QApplication([])`"
        )
    if not app.topLevelWidgets() and not force:
        warn(
            f"Refusing to run a QApplication with no topLevelWidgets. To run the app anyway, use `{_func_name}"
            "(force=True)`",
            stacklevel=1,
        )
        return

    if app.thread().loopLevel() >= max_loop_level:
        loops = app.thread().loopLevel()
        warn(
            f"A QApplication is already running with 1 event loop. To enter *another* event loop, use `{_func_name}"
            f"(max_loop_level={max_loop_level})` A QApplication is already running with {loops} event loops. To enter"
            f" *another* event loop, use `{_func_name}(max_loop_level={max_loop_level})`",
            stacklevel=1,
        )
        return
    with _maybe_allow_interrupt(app):
        app.exec_()
