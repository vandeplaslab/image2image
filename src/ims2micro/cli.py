"""CLI."""
import sys

import click
from loguru import logger

from ims2micro import __version__


def dev_options(func):
    """Setup dev options."""
    # if os.environ.get("IONGLOW_DEV_MODE", "0") == "1":
    func = click.option(
        "--dev",
        help="Flat to indicate that CLI should run in development mode and catch all errors.",
        default=False,
        is_flag=True,
        show_default=True,
    )(func)
    return func


@click.version_option(__version__, prog_name="ims2micro")
@dev_options
@click.option(
    "--no_color",
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("--quiet", "-q", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=5, help="Maximum output")
@click.option(
    "--verbose",
    "-v",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120, "ignore_unknown_options": True},
    invoke_without_command=True,
)
@click.pass_context
def cli(ctx, verbosity: int, no_color: bool, dev: bool = False, extras=None):
    """Execute imimspy CLI."""
    from koyo.logging import set_loguru_log
    from qtextra.config import THEMES

    from ims2micro.appdirs import USER_LOG_DIR
    from ims2micro.dialog import ImageRegistrationDialog
    from ims2micro.event_loop import get_app

    min(1, verbosity) * 10
    level = min(1, verbosity) * 10
    set_loguru_log(USER_LOG_DIR / "log.txt", level=level, no_color=no_color, diagnose=True, catch=True)
    logger.enable("ims2micro")

    # make app
    app = get_app()
    dlg = ImageRegistrationDialog(None)
    dlg.setMinimumSize(1200, 500)

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
