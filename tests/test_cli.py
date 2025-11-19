"""Test CLI."""

import os

import pytest
from koyo.utilities import is_installed

has_i2reg = is_installed("image2image_reg")
has_i2io = is_installed("image2image_io")


def test_cli_entrypoint():
    """Test CLI entrypoint."""
    exit_status = os.system("i2i --help")
    assert exit_status == 0, "i2i command failed."


@pytest.mark.skipif(not has_i2reg, reason="image2image-reg is not installed.")
def test_cli_reg():
    """Test CLI entrypoint."""
    exit_status = os.system("i2i elastix --help")
    assert exit_status == 0, "i2i elastix command failed."


@pytest.mark.skipif(not has_i2io, reason="image2image-io is not installed.")
def test_cli_io():
    """Test CLI entrypoint."""
    exit_status = os.system("i2i convert --help")
    assert exit_status == 0, "i2i convert command failed."

    exit_status = os.system("i2i thumbnail --help")
    assert exit_status == 0, "i2i thumbnail command failed."

    exit_status = os.system("i2i transform --help")
    assert exit_status == 0, "i2i transform command failed."
