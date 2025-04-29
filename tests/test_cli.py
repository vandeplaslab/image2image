"""Test CLI."""

import os

import pytest


@pytest.mark.xfail(reason="need to fix")
def test_cli_entrypoint():
    """Test CLI entrypoint."""
    exit_status = os.system("i2i --help")
    assert exit_status == 0
