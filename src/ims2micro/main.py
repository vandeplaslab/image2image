"""Main window."""
import sys

from loguru import logger


def run(level: int = 10, no_color: bool = False, dev: bool = False):
    """Execute command."""
    from koyo.logging import set_loguru_log
    from qtextra.config import THEMES

    from ims2micro.appdirs import USER_LOG_DIR
    from ims2micro.dialog import ImageRegistrationWindow
    from ims2micro.event_loop import get_app

    set_loguru_log(USER_LOG_DIR / "log.txt", level=level, no_color=no_color, diagnose=True, catch=True)
    logger.enable("ims2micro")

    # make app
    app = get_app()
    dlg = ImageRegistrationWindow(None)
    dlg.setMinimumSize(1200, 500)
    THEMES[THEMES.theme].font_size = "9pt"
    THEMES.set_theme_stylesheet(dlg)

    if dev:
        import faulthandler

        from qtextra.utils.dev import qdev

        segfault_filename = USER_LOG_DIR / "segfault.log"
        segfault_file = open(segfault_filename, "w")
        faulthandler.enable(segfault_file, all_threads=True)
        logger.trace(f"Enabled fault handler to '{segfault_filename}'.")

        # enable extra loggers
        logger.enable("qtextra")
        logger.enable("qtreload")
        dev = qdev(dlg, modules=["qtextra", "ims2micro"])
        dev.evt_theme.connect(lambda: THEMES.set_theme_stylesheet(dlg))
        dlg.centralWidget().layout().addWidget(dev)

    dlg.show()
    sys.exit(app.exec_())


if __name__ == "__main__":  # pragma: no cover
    run()
