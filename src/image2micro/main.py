"""Main window."""
import sys

from loguru import logger


def run(level: int = 10, no_color: bool = False, dev: bool = False):
    """Execute command."""
    from koyo.logging import set_loguru_log
    from qtextra.config import THEMES

    from image2image._appdirs import USER_LOG_DIR
    from image2image.dialog_register import ImageRegistrationWindow
    from image2image.event_loop import get_app

    log_path = USER_LOG_DIR / "log.txt"
    set_loguru_log(log_path, level=level, no_color=True, diagnose=True, catch=True, logger=logger)
    set_loguru_log(level=level, no_color=no_color, diagnose=True, catch=True, logger=logger, remove=False)
    logger.enable("image2image")
    logger.info(f"Enabled logger - logging to '{log_path}' at level={level}")

    # make app
    app = get_app()
    dlg = ImageRegistrationWindow(None)
    dlg.setMinimumSize(1200, 500)
    THEMES[THEMES.theme].font_size = "9pt"
    THEMES.set_theme_stylesheet(dlg)

    if dev:
        import faulthandler

        from qtextra.utils.dev import qdev

        segfault_path = USER_LOG_DIR / "segfault.log"
        segfault_file = open(segfault_path, "w")
        faulthandler.enable(segfault_file, all_threads=True)
        logger.trace(f"Enabled fault handler - logging to '{segfault_path}'")
        logger.enable("qtextra")
        logger.enable("qtreload")

        dev = qdev(dlg, modules=["qtextra", "image2image"])
        dev.evt_theme.connect(lambda: THEMES.set_theme_stylesheet(dlg))
        dlg.centralWidget().layout().addWidget(dev)

        # install_debugger_hook()
        # os.environ["IMS2MICRO_DEV_MODE"] = "1"
    # else:
    #     os.environ["IMS2MICRO_DEV_MODE"] = "0"

    dlg.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run(dev=True)
