"""Main window."""
import sys

from loguru import logger


def run(level: int = 10, no_color: bool = False, dev: bool = False):
    """Execute command."""
    from koyo.logging import set_loguru_log
    from qtextra.config import THEMES

    from ims2micro.appdirs import USER_LOG_DIR
    from ims2micro.dialog_register import ImageRegistrationWindow
    from ims2micro.event_loop import get_app

    log_path = USER_LOG_DIR / "log.txt"
    set_loguru_log(log_path, level=level, no_color=True, diagnose=True, catch=True, logger=logger)
    set_loguru_log(level=level, no_color=False, diagnose=True, catch=True, logger=logger, remove=False)
    logger.enable("ims2micro")
    logger.info(f"Logging to '{log_path}' at level={level}")

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

        dev = qdev(dlg, modules=["qtextra", "ims2micro"])
        dev.evt_theme.connect(lambda: THEMES.set_theme_stylesheet(dlg))
        dlg.centralWidget().layout().addWidget(dev)

    dlg.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run()
