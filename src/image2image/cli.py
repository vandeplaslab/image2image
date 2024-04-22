"""CLI."""

import typing as ty

import click
from koyo.system import IS_MAC, IS_MAC_ARM, IS_PYINSTALLER

from image2image import __version__

if IS_MAC_ARM and IS_PYINSTALLER:
    AVAILABLE_TOOLS = ty.Literal["launcher", "register", "viewer", "crop", "fusion"]  # type: ignore
else:
    AVAILABLE_TOOLS = ty.Literal["launcher", "register", "viewer", "crop", "fusion", "convert"]  # type: ignore


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


@click.version_option(__version__, prog_name="image2image")
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
@click.option(
    "-t",
    "--tool",
    type=click.Choice(["launcher", "register", "viewer", "fusion", "crop", "convert", "wsiprep"]),
    default="launcher",
    show_default=True,
)
@click.command(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120, "ignore_unknown_options": True},
)
@click.pass_context
def cli(
    ctx: click.Context,
    tool: AVAILABLE_TOOLS,
    verbosity: float,
    no_color: bool,
    dev: bool = False,
    extras: ty.Any = None,
):
    """Launch image2image app.

    \b
    Available tools:
    launcher - opens dialog where you can launch any of the tools.
    register - opens dialog where you can co-register images.
    viewer - opens dialog where you can view images.
    export - opens dialog where you can export images.
    sync - opens dialog where you can sync images (not yet implemented)
    crop - opens dialog where you can crop images (not yet implemented)
    """
    import os

    from image2image.main import run

    if IS_MAC:
        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    if dev:
        if IS_PYINSTALLER:
            click.echo("Developer mode is disabled in bundled app.")
            dev = False
        else:
            verbosity = 0.5
    level = min(0.5, verbosity) * 10
    run(level=int(level), no_color=no_color, dev=dev, tool=tool)
