"""CLI."""

from __future__ import annotations

import sys
import typing as ty
from multiprocessing import freeze_support, set_start_method

import click
import koyo.compat  # noqa
from click_groups import GroupedGroup
from koyo.click import cli_parse_paths_sort, dev_options
from koyo.system import IS_MAC, IS_PYINSTALLER
from koyo.utilities import is_installed

from image2image import __version__

AVAILABLE_TOOLS = [
    "launcher",
    "register",
    "viewer",
    "crop",
    "elastix",
    "valis",
    "elastix3d",
    "fusion",
    "convert",
    "merge",
    "napari",
]
# if IS_MAC_ARM and IS_PYINSTALLER:
#     AVAILABLE_TOOLS.pop(AVAILABLE_TOOLS.index("convert"))


def _cli_setup(verbosity: float, no_color: bool, info: bool = False, dev: bool = False) -> float:
    from image2image.main import setup_logger

    if info:
        from image2image.utils.system import get_system_info

        print(get_system_info())
        return sys.exit(0)

    if IS_MAC:
        import os

        os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

    if dev:
        if IS_PYINSTALLER:
            click.echo("Developer mode is disabled in bundled app.")
            dev = False
        else:
            verbosity = 0.5
    else:
        verbosity = 5 - int(verbosity)  # default is WARNING
    verbosity = max(0, verbosity)
    level = max(0.5, verbosity) * 10

    setup_logger(level=int(level), no_color=no_color)
    return level


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    },
    invoke_without_command=True,
    cls=GroupedGroup,
)
@click.option(
    "-t",
    "--tool",
    type=click.Choice(AVAILABLE_TOOLS),
    default="launcher",
    show_default=True,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the Elastix/Valis project directory. It usually ends in .i2reg extension"
    " (for 'elastix' or 'valis' tool).",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
)
@click.option(
    "-f",
    "--file",
    help="Path to microscopy images (for 'viewer' tool).",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-F",
    "--file_dir",
    help="Path to directory with microscopy images (for 'viewer' tool).",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
)
@click.version_option(__version__, prog_name="image2image")
@dev_options
@click.option(
    "--no_color",
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("-q", "--quiet", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=45, help="Maximum output")
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
@click.option("--info", is_flag=True, help="Print program information and exit.")
@click.pass_context
def cli(
    ctx: click.Context,
    tool: str,
    verbosity: float,
    no_color: bool,
    info: bool = False,
    dev: bool = False,
    project_dir: str | None = None,
    file: str | list[str] | None = None,
    file_dir: str | None = None,
    extras: ty.Any = None,
) -> None:
    """Launch image2image app.

    \b
    Available tools:
    launcher - opens a dialog where you can launch any of the tools.
    viewer - opens a dialog where you can view images, shapes and points data
    register - opens a dialog where you can co-register images using affine transformation
    elastix - opens a dialog where you can co-register whole slide images using i2reg-elastix
    valis - opens a dialog where you can co-register whole-slide images using i2reg-valis
    convert - opens a dialog where you can convert images to OME-TIFF
    merge - opens a dialog where you can merge multiple image channels and images together
    crop - opens a dialog where you can crop (and mask) images
    fusion - opens a dialog where you can prepare data for image fusion
    """
    level = _cli_setup(verbosity, no_color, info, dev)
    if ctx.invoked_subcommand is None:
        from image2image.main import run

        run(
            level=int(level),
            no_color=no_color,
            dev=dev,
            tool=tool,
            image_path=file,
            image_dir=file_dir,
            project_dir=project_dir,
        )
    return


merge = None
if is_installed("image2image_io"):
    from image2image_io.cli import convert, merge, thumbnail, transform  # type: ignore[assignment]

    cli.add_command(convert, help_group="Utility")  # type: ignore[attr-defined]
    cli.add_command(thumbnail, help_group="Utility")  # type: ignore[attr-defined]
    cli.add_command(transform, help_group="Utility")  # type: ignore[attr-defined]


if is_installed("image2image_reg"):
    from image2image_reg.cli import elastix, merge, valis  # type: ignore[assignment]

    # registration
    cli.add_command(elastix, help_group="Registration")  # type: ignore[attr-defined]
    if valis:
        cli.add_command(valis, help_group="Registration")  # type: ignore[attr-defined]

if merge:
    cli.add_command(merge, help_group="Utility")  # type: ignore[attr-defined]


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    },
    invoke_without_command=True,
    cls=GroupedGroup,
)
@click.option(
    "-f",
    "--file",
    help="Path to microscopy images (for 'viewer' tool).",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-F",
    "--file_dir",
    help="Path to directory with microscopy images (for 'viewer' tool).",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
)
@click.option(
    "--no_color",
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("-q", "--quiet", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=45, help="Maximum output")
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
def cli_viewer(
    verbosity: float, no_color: bool, file: str | list[str] | None = None, file_dir: str | None = None
) -> None:
    """Launch image2image Viewer app."""
    from image2image.main import run

    level = _cli_setup(verbosity, no_color)

    run(level=int(level), no_color=no_color, tool="viewer", image_path=file, image_dir=file_dir)


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    },
    invoke_without_command=True,
    cls=GroupedGroup,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the Elastix/Valis project directory. It usually ends in .i2reg extension"
    " (for 'elastix' or 'valis' tool).",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
)
@click.option(
    "--no_color",
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("-q", "--quiet", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=45, help="Maximum output")
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
def cli_elastix(
    verbosity: float,
    no_color: bool,
    project_dir: str | None = None,
) -> None:
    """Launch image2image Viewer app."""
    from image2image.main import run

    level = _cli_setup(verbosity, no_color)

    run(level=int(level), no_color=no_color, tool="elastix", project_dir=project_dir)


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    },
    invoke_without_command=True,
    cls=GroupedGroup,
)
@click.option(
    "--no_color",
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("-q", "--quiet", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=45, help="Maximum output")
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
def cli_crop(verbosity: float, no_color: bool) -> None:
    """Launch image2image Viewer app."""
    from image2image.main import run

    level = _cli_setup(verbosity, no_color)

    run(level=int(level), no_color=no_color, tool="crop")


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    },
    invoke_without_command=True,
    cls=GroupedGroup,
)
@click.option(
    "--no_color",
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("-q", "--quiet", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=45, help="Maximum output")
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
def cli_convert(verbosity: float, no_color: bool) -> None:
    """Launch image2image Viewer app."""
    from image2image.main import run

    level = _cli_setup(verbosity, no_color)

    run(level=int(level), no_color=no_color, tool="convert")


@click.group(
    context_settings={
        "help_option_names": ["-h", "--help"],
        "max_content_width": 120,
        "ignore_unknown_options": True,
    },
    invoke_without_command=True,
    cls=GroupedGroup,
)
@click.option(
    "--no_color",
    help="Flag to enable colored logs.",
    default=False,
    is_flag=True,
    show_default=True,
)
@click.option("-q", "--quiet", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=45, help="Maximum output")
@click.option(
    "-v",
    "--verbose",
    "verbosity",
    default=1,
    count=True,
    help="Verbose output. This is additive flag so `-vvv` will print `INFO` messages and -vvvv will print `DEBUG`"
    " information.",
)
def cli_register(verbosity: float, no_color: bool) -> None:
    """Launch image2image Viewer app."""
    from image2image.main import run

    level = _cli_setup(verbosity, no_color)

    run(level=int(level), no_color=no_color, tool="register")


def main() -> None:
    """Execute the "i2i" command line program."""
    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    cli.main(windows_expand_args=False)  # type: ignore[attr-defined]


def main_viewer() -> None:
    """Execute the "i2i" command line program."""
    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    cli_viewer.main(windows_expand_args=False)  # type: ignore[attr-defined]


def main_elastix() -> None:
    """Execute the "i2i" command line program."""
    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    cli_elastix.main(windows_expand_args=False)  # type: ignore[attr-defined]


def main_register() -> None:
    """Execute the "i2i" command line program."""
    freeze_support()
    if sys.platform == "darwin":
        set_start_method("spawn", True)
    cli_register.main(windows_expand_args=False)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
