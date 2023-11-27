"""Main window."""
import os
import sys
import typing as ty

from loguru import logger


def run(
    level: int = 10,
    no_color: bool = False,
    dev: bool = False,
    tool: ty.Literal["launcher", "register", "viewer", "crop", "fusion", "convert"] = "launcher",
) -> None:
    """Execute command."""
    import warnings

    from image2image_reader.config import CONFIG as READER_CONFIG
    from koyo.logging import set_loguru_log
    from qtextra.config import THEMES
    from qtextra.utils.context import _maybe_allow_interrupt

    import image2image.assets  # noqa: F401
    from image2image.config import CONFIG
    from image2image.qt._sentry import install_error_monitor
    from image2image.qt.event_loop import get_app
    from image2image.utils._appdirs import USER_LOG_DIR
    from image2image.utils.utilities import install_segfault_handler, maybe_submit_segfault

    log_path = USER_LOG_DIR / f"log_tool={tool}.txt"
    set_loguru_log(log_path, level=level, no_color=True, diagnose=True, catch=True, logger=logger)
    set_loguru_log(level=level, no_color=no_color, diagnose=True, catch=True, logger=logger, remove=False)
    logger.enable("image2image")
    logger.enable("image2image_reader")
    logger.info(f"Enabled logger - logging to '{log_path}' at level={level}")

    if dev:
        from koyo.hooks import install_debugger_hook

        install_debugger_hook()

    # load config
    READER_CONFIG.load()
    CONFIG.load()
    # setup theme
    for theme in THEMES.themes:
        THEMES[theme].font_size = "9pt"
    THEMES.theme = CONFIG.theme

    # make app
    app = get_app()

    # install error monitor
    install_error_monitor()
    maybe_submit_segfault()
    install_segfault_handler()

    if tool == "launcher":
        from image2image.qt.launcher import Launcher

        dlg = Launcher(None)  # type: ignore[no-untyped-call]
        dlg.setMinimumSize(400, 450)
    elif tool == "register":
        from image2image.qt.dialog_register import ImageRegistrationWindow

        dlg = ImageRegistrationWindow(None)  # type: ignore[assignment]
        dlg.setMinimumSize(1200, 800)
    elif tool == "viewer":
        from image2image.qt.dialog_viewer import ImageViewerWindow

        dlg = ImageViewerWindow(None)  # type: ignore[assignment]
        dlg.setMinimumSize(1200, 800)
    elif tool == "crop":
        from image2image.qt.dialog_crop import ImageCropWindow

        dlg = ImageCropWindow(None)  # type: ignore[assignment]
        dlg.setMinimumSize(1200, 800)
    elif tool == "fusion":
        from image2image.qt.dialog_export import ImageExportWindow

        dlg = ImageExportWindow(None)  # type: ignore[assignment]
        dlg.setMinimumSize(600, 200)
    elif tool == "convert":
        from image2image.qt.dialog_convert import ImageConvertWindow

        dlg = ImageConvertWindow(None)  # type: ignore[assignment]
        dlg.setMinimumSize(600, 200)
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

        from qtextra.utils.dev import qdev

        logger.enable("qtextra")
        logging.getLogger("qtreload").setLevel(logging.DEBUG)

        dev = qdev(dlg, modules=["qtextra", "image2image", "image2image_reader", "koyo"])
        dev.evt_theme.connect(lambda: THEMES.set_theme_stylesheet(dlg))
        if hasattr(dlg, "centralWidget"):
            dlg.centralWidget().layout().addWidget(dev)
        else:
            dlg.layout().addWidget(dev)

        # install_debugger_hook()
        os.environ["IMAGE2IMAGE_DEV_MODE"] = "1"
    else:
        os.environ["IMAGE2IMAGE_DEV_MODE"] = "0"

    dlg.show()
    with _maybe_allow_interrupt(app):
        sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run(dev=True, tool="viewer")
