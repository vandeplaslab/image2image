"""CLI."""

from __future__ import annotations

import typing as ty

import click
from click_groups import GroupedGroup
from koyo.click import cli_parse_paths_sort
from koyo.system import IS_MAC, IS_MAC_ARM, IS_PYINSTALLER
from koyo.utilities import is_installed

from image2image import __version__

AVAILABLE_TOOLS = [
    "launcher",
    "register",
    "viewer",
    "crop",
    "wsiprep",
    "elastix",
    "valis",
    "fusion",
    "convert",
    "merge",
]
if IS_MAC_ARM and IS_PYINSTALLER:
    AVAILABLE_TOOLS.pop(AVAILABLE_TOOLS.index("convert"))


def dev_options(func: ty.Callable) -> ty.Callable:
    """Setup dev options."""
    if not IS_PYINSTALLER:
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
@click.option("-q", "--quiet", "verbosity", flag_value=0, help="Minimal output")
@click.option("--debug", "verbosity", flag_value=5, help="Maximum output")
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
@click.option(
    "-d",
    "--image_dir",
    help="Path to directory with microscopy images (for 'viewer' tool).",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
)
@click.option(
    "-i",
    "--image_path",
    help="Path to microscopy images (for 'viewer' tool).",
    type=click.UNPROCESSED,
    show_default=True,
    multiple=True,
    callback=cli_parse_paths_sort,
)
@click.option(
    "-p",
    "--project_dir",
    help="Path to the Elastix/Valis project directory. It usually ends in .i2reg extension (for 'elastix' or 'valis' tool).",
    type=click.Path(exists=True, resolve_path=True, file_okay=False, dir_okay=True),
    show_default=True,
)
@click.option(
    "-t",
    "--tool",
    type=click.Choice(AVAILABLE_TOOLS),
    default="launcher",
    show_default=True,
)
@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "max_content_width": 120, "ignore_unknown_options": True},
    invoke_without_command=True,
    cls=GroupedGroup,
)
@click.pass_context
# @click.group(
#     context_settings={
#         "help_option_names": ["-h", "--help"],
#         "max_content_width": 120,
#         "ignore_unknown_options": True,
#         "allow_extra_args": False,
#     },
#     invoke_without_command=True,
# )
def cli(
    ctx: click.Context,
    tool: str,
    verbosity: float,
    no_color: bool,
    info: bool = False,
    dev: bool = False,
    project_dir: str | None = None,
    image_path: str | list[str] | None = None,
    image_dir: str | None = None,
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
    import os
    import sys

    if info:
        from image2image.utils.system import get_system_info

        print(get_system_info())
        return sys.exit()

    if ctx.invoked_subcommand is None:
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
        run(
            level=int(level),
            no_color=no_color,
            dev=dev,
            tool=tool,
            image_path=image_path,
            image_dir=image_dir,
            project_dir=project_dir,
        )
        return None
    return None


if is_installed("image2image_reg"):
    from image2image_reg.cli.elastix import elastix
    from image2image_reg.cli.valis import valis

    cli.add_command(elastix, help_group="Registration")  # type: ignore
    if valis:
        cli.add_command(valis, help_group="Registration")  # type: ignore

    # from image2image_reg.cli._common import (
    #     as_uint8_,
    #     fmt_,
    #     n_parallel_,
    #     original_size_,
    #     overwrite_,
    #     parallel_mode_,
    #     project_path_multi_,
    #     remove_merged_,
    #     write_merged_,
    #     write_not_registered_,
    #     write_registered_,
    # )
    # from image2image_reg.cli.elastix import register_runner
    # from image2image_reg.enums import WriterMode
    #
    # @overwrite_
    # @parallel_mode_
    # @n_parallel_
    # @as_uint8_
    # @original_size_
    # @remove_merged_
    # @write_merged_
    # @write_not_registered_
    # @write_registered_
    # @fmt_
    # @click.option(
    #     "-w/-W",
    #     "--write/--no_write",
    #     help="Write images to disk.",
    #     is_flag=True,
    #     default=True,
    #     show_default=True,
    # )
    # @click.option(
    #     "--histogram_match/--no_histogram_match",
    #     help="Match image histograms before co-registering - this might improve co-registration.",
    #     is_flag=True,
    #     default=False,
    #     show_default=True,
    # )
    # @project_path_multi_
    # @cli.command("i2reg")
    # def register_cmd(
    #     project_dir: ty.Sequence[str],
    #     histogram_match: bool,
    #     write: bool,
    #     fmt: WriterMode,
    #     write_registered: bool,
    #     write_not_registered: bool,
    #     write_merged: bool,
    #     remove_merged: bool,
    #     original_size: bool,
    #     as_uint8: bool | None,
    #     n_parallel: int,
    #     parallel_mode: str,
    #     overwrite: bool,
    # ) -> None:
    #     """Register images."""
    #     register_runner(
    #         project_dir,
    #         histogram_match=histogram_match,
    #         write_images=write,
    #         fmt=fmt,
    #         write_registered=write_registered,
    #         write_merged=write_merged,
    #         remove_merged=remove_merged,
    #         write_not_registered=write_not_registered,
    #         original_size=original_size,
    #         as_uint8=as_uint8,
    #         n_parallel=n_parallel,
    #         parallel_mode=parallel_mode,
    #         overwrite=overwrite,
    #     )
