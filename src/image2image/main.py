"""Main window."""
import os
import sys
import typing as ty

from loguru import logger


def run(
    level: int = 10,
    no_color: bool = False,
    dev: bool = False,
    tool: ty.Literal["launcher", "register", "viewer", "crop", "export"] = "launcher",
):
    """Execute command."""
    from koyo.logging import set_loguru_log
    from qtextra.config import THEMES

    from image2image.config import CONFIG
    from image2image.qt.event_loop import get_app
    from image2image.utils._appdirs import USER_LOG_DIR

    log_path = USER_LOG_DIR / f"log_tool={tool}.txt"
    set_loguru_log(log_path, level=level, no_color=True, diagnose=True, catch=True, logger=logger)
    set_loguru_log(level=level, no_color=no_color, diagnose=True, catch=True, logger=logger, remove=False)
    logger.enable("image2image")
    logger.info(f"Enabled logger - logging to '{log_path}' at level={level}")

    # load config
    CONFIG.load()
    # setup theme
    for theme in THEMES.themes:
        THEMES[theme].font_size = "9pt"
    THEMES.theme = CONFIG.theme

    # make app
    app = get_app()
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
    elif tool == "export":
        from image2image.qt.dialog_export import ImageExportWindow

        dlg = ImageExportWindow(None)  # type: ignore[assignment]
        dlg.setMinimumSize(600, 500)
    else:
        raise ValueError("Launcher is not implemented yet.")

    THEMES[THEMES.theme].font_size = "9pt"
    THEMES.set_theme_stylesheet(dlg)
    THEMES.evt_theme_changed.connect(lambda: THEMES.set_theme_stylesheet(dlg))

    if dev:
        import faulthandler
        import logging

        from qtextra.utils.dev import qdev

        segfault_path = USER_LOG_DIR / "segfault.log"
        segfault_file = open(segfault_path, "w+")
        faulthandler.enable(segfault_file, all_threads=True)
        logger.trace(f"Enabled fault handler - logging to '{segfault_path}'")
        logger.enable("qtextra")
        logging.getLogger("qtreload").setLevel(logging.DEBUG)

        dev = qdev(dlg, modules=["qtextra", "image2image"])
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
    # if tool in ["launcher", "export", "crop"]:
    #     dlg.show()
    # else:
    #     dlg.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run(dev=True, tool="viewer")
