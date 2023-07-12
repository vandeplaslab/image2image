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

    from image2image._appdirs import USER_LOG_DIR
    from image2image.event_loop import get_app

    log_path = USER_LOG_DIR / f"log_tool={tool}.txt"
    set_loguru_log(log_path, level=level, no_color=True, diagnose=True, catch=True, logger=logger)
    set_loguru_log(level=level, no_color=no_color, diagnose=True, catch=True, logger=logger, remove=False)
    logger.enable("image2image")
    logger.info(f"Enabled logger - logging to '{log_path}' at level={level}")

    # make app
    app = get_app()
    if tool == "register":
        from image2image.dialog_register import ImageRegistrationWindow

        dlg = ImageRegistrationWindow(None)
        dlg.setMinimumSize(1200, 500)
    elif tool == "viewer":
        from image2image.dialog_viewer import ImageViewerWindow

        dlg = ImageViewerWindow(None)
        dlg.setMinimumSize(1200, 500)
    elif tool == "launcher":
        from image2image.launcher import Launcher

        dlg = Launcher(None)
        dlg.setMinimumSize(300, 500)
    elif tool == "crop":
        from image2image.dialog_crop import ImageCropWindow

        dlg = ImageCropWindow(None)
        dlg.setMinimumSize(1200, 500)
    elif tool == "export":
        from image2image.dialog_export import ImageExportWindow

        dlg = ImageExportWindow(None)
        dlg.setMinimumSize(600, 500)
    else:
        raise ValueError("Launcher is not implemented yet.")

    THEMES[THEMES.theme].font_size = "9pt"
    THEMES.set_theme_stylesheet(dlg)
    THEMES.evt_theme_changed.connect(lambda: THEMES.set_theme_stylesheet(dlg))

    if dev:
        import faulthandler

        from qtextra.utils.dev import qdev

        segfault_path = USER_LOG_DIR / "segfault.log"
        segfault_file = open(segfault_path, "w+")
        faulthandler.enable(segfault_file, all_threads=True)
        logger.trace(f"Enabled fault handler - logging to '{segfault_path}'")
        logger.enable("qtextra")
        logger.enable("qtreload")

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
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run(dev=True, tool="viewer")
