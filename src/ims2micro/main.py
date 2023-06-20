"""Main window."""


if __name__ == "__main__":  # pragma: no cover
    import sys
    import faulthandler

    from qtextra.utils.dev import qdev
    from ims2micro.dialog import ImageRegistrationDialog
    from qtextra.config import THEMES
    from ims2micro.event_loop import get_app

    faulthandler.enable()

    app = get_app()
    dlg = ImageRegistrationDialog(None)
    dlg.setMinimumSize(1200, 500)
    THEMES.set_theme_stylesheet(dlg)

    dev = qdev(dlg, modules=["qtextra", "ims2micro"])
    dlg.show()
    sys.exit(app.exec_())
