"""Headless entry points used by the Serial Sections subprocess queue."""

from pathlib import Path

import click


@click.group()
def serial() -> None:
    """Register and reconstruct serial-section projects."""


@serial.command("register")
@click.option("--project", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
def register_command(project: Path) -> None:
    """Register all included sections and save reusable transformations."""
    from image2image.utils.serial import register_serial

    try:
        register_serial(project, progress=click.echo)
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


@serial.command("reconstruct")
@click.option("--project", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--output", required=True, type=click.Path(path_type=Path))
def reconstruct_command(project: Path, output: Path) -> None:
    """Write a new OME-Zarr volume from completed registration results."""
    from image2image.utils.serial import reconstruct_serial

    try:
        reconstruct_serial(project, output, progress=click.echo)
    except (ValueError, OSError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
