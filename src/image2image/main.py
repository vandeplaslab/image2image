"""Main window."""

from __future__ import annotations

import os
import sys
import typing as ty

from loguru import logger

LOG_FMT = "[<level>{level: <8}</level>][{time:YYYY-MM-DD HH:mm:ss:SSS}][{extra[src]}] {message}"
COLOR_LOG_FMT = (
    "<green>[<level>{level: <8}</level>]</green>"
    "<cyan>[{time:YYYY-MM-DD HH:mm:ss:SSS}]</cyan>"
    "<red>[{process}]</red>"
    "<blue>[{extra[src]}]</blue>"
    " {message}"
)

AvailableTools = ty.Literal[
    "launcher", "register", "viewer", "crop", "fusion", "convert", "merge", "wsiprep", "elastix", "valis"
]


def setup_logger(
    level: int = 10,
    no_color: bool = False,
    modules: tuple[str, ...] = ("image2image", "image2image_io", "image2image_reg", "koyo"),
) -> None:
    """Setup logger."""
    from koyo.logging import set_loguru_log

    import image2image.assets  # noqa: F401

    # setup console logger
    set_loguru_log(
        level=level,
        no_color=no_color,
        diagnose=True,
        catch=True,
        logger=logger,
        remove=False,
        fmt=LOG_FMT if no_color else COLOR_LOG_FMT,
    )
    logger.configure(extra={"src": "CLI"})
    [logger.enable(module) for module in modules]  # type: ignore
    logger.trace(f"Enabled logger at level={level}")


def run(
    level: int = 10,
    no_color: bool = False,
    dev: bool = False,
    tool: str | AvailableTools = "launcher",
    log: bool = True,
    **kwargs: ty.Any,
) -> None:
    """Execute command."""
    import warnings

    from image2image_io.config import CONFIG as READER_CONFIG
    from koyo.faulthandler import install_segfault_handler, maybe_submit_segfault
    from koyo.timer import MeasureTimer
    from qtextra.config import THEMES
    from qtextra.utils.context import _maybe_allow_interrupt

    import image2image.assets  # noqa: F401
    from image2image.config import get_app_config
    from image2image.qt.event_loop import get_app
    from image2image.utils._appdirs import USER_LOG_DIR

    # setup file logger
    with MeasureTimer() as timer:
        dev_modules = ["qtextra", "qtextraplot", "koyo", "image2image", "image2image_io", "image2image_reg"]
        if log:
            from koyo.logging import set_loguru_log

            log_path = USER_LOG_DIR / f"log_tool={tool}.txt"
            set_loguru_log(
                log_path,
                level=level,
                no_color=True,
                diagnose=True,
                catch=True,
                logger=logger,
                fmt=LOG_FMT,
            )

            # setup console logger
            set_loguru_log(
                level=level,
                no_color=no_color,
                diagnose=True,
                catch=True,
                logger=logger,
                remove=False,
                fmt=LOG_FMT if no_color else COLOR_LOG_FMT,
            )
            logger.configure(extra={"src": "CLI"})
            for module in dev_modules:
                logger.enable(module)
            logger.info(f"Enabled logger - logging to '{log_path}' at level={level}")

        run_check_version = not dev
        if dev:
            from koyo.hooks import install_debugger_hook

            install_debugger_hook()
            logger.debug(f"Installed debugger hook: {sys.excepthook.__name__}")

        # make app
        app = get_app()

        # load config
        try:
            READER_CONFIG.load()
            # setup theme
            for theme in THEMES.themes:
                THEMES[theme].font_size = "10pt"
            THEMES.theme = get_app_config().theme
        except Exception as e:
            logger.exception(f"Failed to load config: {e}")

        # install error monitor
        if not dev:
            from image2image.qt._dialogs._sentry import install_error_monitor

            install_error_monitor()

        # install segfault handler
        segfault_filename = f"segfault_{tool}.log"
        maybe_submit_segfault(USER_LOG_DIR, segfault_filename)
        install_segfault_handler(USER_LOG_DIR, segfault_filename)

        args = sys.argv
        if args and ("image2image" in args or "image2image" in args[0]):
            os.environ["IMAGE2IMAGE_PROGRAM"] = args[0]
            logger.trace(f"Updated environment variable. IMAGE2IMAGE_PROGRAM={os.environ['IMAGE2IMAGE_PROGRAM']}")

        if tool == "launcher":
            from image2image.qt.launcher import Launcher

            dlg = Launcher(None)  # type: ignore[no-untyped-call]
        elif tool == "register":
            from image2image.qt.dialog_register import ImageRegistrationWindow

            dlg = ImageRegistrationWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(1800, 1200)
        elif tool == "viewer":
            from image2image.qt.dialog_viewer import ImageViewerWindow

            dlg = ImageViewerWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(1200, 800)
        elif tool == "crop":
            from image2image.qt.dialog_crop import ImageCropWindow

            dlg = ImageCropWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(1200, 800)
        elif tool == "fusion":
            from image2image.qt.dialog_fusion import ImageFusionWindow

            dlg = ImageFusionWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(600, 400)
        elif tool == "convert":
            from image2image.qt.dialog_convert import ImageConvertWindow

            dlg = ImageConvertWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(600, 400)
        elif tool == "merge":
            from image2image.qt.dialog_merge import ImageMergeWindow

            dlg = ImageMergeWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(600, 400)
        elif tool == "elastix3d":
            from image2image.qt.dialog_elastix3d import ImageElastix3dWindow

            dlg = ImageElastix3dWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(1200, 800)
        elif tool == "elastix":
            from image2image.qt.dialog_elastix import ImageElastixWindow

            dlg = ImageElastixWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(1200, 800)
        elif tool == "valis":
            from image2image.qt.dialog_valis import ImageValisWindow

            dlg = ImageValisWindow(None, run_check_version=run_check_version, **kwargs)  # type: ignore[assignment]
            dlg.setMinimumSize(1200, 800)
        elif tool == "napari":
            from napari import Viewer

            viewer = Viewer()
            dlg = viewer.window._qt_window
        else:
            raise ValueError("Launcher is not implemented yet.")

        THEMES.set_theme_stylesheet(dlg)
        THEMES.evt_theme_changed.connect(lambda: THEMES.set_theme_stylesheet(dlg))

        # disable some annoying warnings from napari
        warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered in multiply")
        warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered in square")
        warnings.filterwarnings("ignore", message="RuntimeWarning: overflow encountered in cast")

        if dev:
            import logging

            from qtextra.dialogs.qt_dev import QDevPopup
            from qtextra.helpers import make_qta_btn

            # from qtextra.utils.dev import qdev

            logger.enable("qtextra")
            logging.getLogger("qtreload").setLevel(level)

            dev_dlg = QDevPopup(dlg, modules=dev_modules)
            dev_dlg.qdev.evt_stylesheet.connect(lambda: THEMES.set_theme_stylesheet(dlg))
            if hasattr(dlg, "statusbar"):
                dlg.dev_btn = make_qta_btn(  # type: ignore[attr-defined]
                    dlg,
                    "dev",
                    tooltip="Open development tools.",
                    func=dev_dlg.show,
                    small=True,
                )
                dlg.statusbar.addPermanentWidget(dlg.dev_btn)  # type: ignore[attr-defined]
        os.environ["IMAGE2IMAGE_DEV_MODE"] = "1" if dev else "0"

        # show dialog
        dlg.show()
        with _maybe_allow_interrupt(app):
            logger.trace(f"Launched {tool} in {timer()}")
            sys.exit(app.exec_())
