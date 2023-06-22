"""Main window."""


if __name__ == "__main__":  # pragma: no cover
    import faulthandler
    import sys

    from loguru import logger
    from qtextra.config import THEMES
    from qtextra.utils.dev import qdev

    from ims2micro.dialog import ImageRegistrationDialog
    from ims2micro.event_loop import get_app

    faulthandler.enable()
    logger.enable("ims2micro")
    logger.enable("qtextra")
    logger.enable("qtreload")

    app = get_app()
    dlg = ImageRegistrationDialog(None)
    dlg.setMinimumSize(1200, 500)
    THEMES[THEMES.theme].font_size = "8pt"
    THEMES.set_theme_stylesheet(dlg)

    dev = qdev(dlg, modules=["qtextra", "ims2micro"])
    dev.evt_theme.connect(lambda: THEMES.set_theme_stylesheet(dlg))

    dlg.show()
    sys.exit(app.exec_())
