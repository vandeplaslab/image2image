"""Test CLI."""

import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner
from koyo.system import is_installed

from image2image.cli import cli

has_i2reg = is_installed("image2image_reg")
has_i2io = is_installed("image2image_io")


def test_cli_entrypoint() -> None:
    """Test CLI entrypoint."""
    exit_status = os.system(f"{sys.executable} -m image2image --help")
    assert exit_status == 0, "i2i command failed."


@pytest.mark.skipif(not has_i2reg, reason="image2image-reg is not installed.")
def test_cli_reg():
    """Test CLI entrypoint."""
    exit_status = os.system(f"{sys.executable} -m image2image elastix --help")
    assert exit_status == 0, "i2i elastix command failed."


@pytest.mark.skipif(not has_i2io, reason="image2image-io is not installed.")
def test_cli_io():
    """Test CLI entrypoint."""
    exit_status = os.system(f"{sys.executable} -m image2image convert --help")
    assert exit_status == 0, "i2i convert command failed."

    exit_status = os.system(f"{sys.executable} -m image2image thumbnail --help")
    assert exit_status == 0, "i2i thumbnail command failed."

    exit_status = os.system(f"{sys.executable} -m image2image transform --help")
    assert exit_status == 0, "i2i transform command failed."


@pytest.mark.parametrize(
    ("tool", "suffix"),
    [
        ("elastix", ".config.json"),
        ("register", ".i2r.json"),
    ],
)
def test_cli_accepts_project_file(
    tool: str, suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test project files are forwarded to the selected GUI tool."""
    project_path = tmp_path / f"project{suffix}"
    project_path.write_text("{}")
    calls: list[dict[str, object]] = []

    monkeypatch.setattr("image2image.cli._cli_setup", lambda *_args, **_kwargs: (10, False))
    monkeypatch.setattr("image2image.main.run", lambda **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(cli, ["-t", tool, "-p", str(project_path)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["tool"] == tool
    assert calls[0]["project_dir"] == str(project_path.resolve())
