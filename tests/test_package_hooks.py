"""Check native backend inclusion without importing platform-specific vendor SDKs."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

HOOKS = Path(__file__).resolve().parents[1] / "package" / "_hooks"


@pytest.fixture
def hook_tools(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Provide controlled collector outputs for executing the packaging hooks."""
    hooks = SimpleNamespace(
        can_import_module=Mock(return_value=True),
        collect_data_files=Mock(return_value=[]),
        collect_dynamic_libs=Mock(return_value=[]),
        collect_submodules=Mock(return_value=["pytsf", "pytsf.tdf"]),
        copy_metadata=Mock(side_effect=lambda name: [(f"/{name}.dist-info", f"{name}.dist-info")]),
    )
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    return hooks


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_imzy_collects_platform_binaries_and_preserves_resources(
    monkeypatch: pytest.MonkeyPatch, hook_tools: SimpleNamespace, platform: str
) -> None:
    monkeypatch.setitem(
        sys.modules, "koyo.system", SimpleNamespace(IS_MAC=platform == "darwin", IS_WIN=platform == "win32")
    )
    # Backslashes and nested vendor directories occur in Windows collector output.
    bruker = r"imzy\_readers\bruker\sdk"
    waters = "imzy/_readers/waters/sdk"
    libraries = [
        ("/bruker/timsdata.dll", bruker),
        ("/bruker/libtimsdata.so", bruker),
        ("/bruker/libtimsdata.so.1", bruker),
        ("/bruker/libtimsdata.dylib", bruker),
        ("/waters/MassLynxRaw.dll", waters),
    ]
    resources = [("/bruker/LICENSE.txt", bruker), ("/imzy/schema.json", "imzy")]
    hook_tools.collect_data_files.return_value = resources
    hook_tools.collect_dynamic_libs.side_effect = lambda name, **_: libraries if name == "imzy" else []

    hook = runpy.run_path(str(HOOKS / "hook-imzy.py"))

    expected = {
        "darwin": [],
        "win32": [libraries[0], libraries[4]],
        "linux": [libraries[1], libraries[2]],
    }[platform]
    assert hook["binaries"] == expected
    assert hook["datas"][: len(resources)] == resources
    assert hook["filter_imzy_data"]([*resources, *libraries]) == [*resources, *expected]
    hook_tools.collect_data_files.assert_called_once_with(
        "imzy", excludes=["**/*.dll", "**/*.dylib", "**/*.so", "**/*.so.*"]
    )
    assert "imzy._imzml_rust" in hook["hiddenimports"]
    assert "pytsf.tdf" in hook["hiddenimports"]
    assert ("/pytsf.dist-info", "pytsf.dist-info") in hook["datas"]


def test_mac_hook_preserves_native_extension_and_pytsf_library(
    monkeypatch: pytest.MonkeyPatch, hook_tools: SimpleNamespace
) -> None:
    monkeypatch.setitem(sys.modules, "koyo.system", SimpleNamespace(IS_MAC=True, IS_WIN=False))
    native = ("/imzy/_imzml_rust.abi3.so", "imzy")
    pytsf_library = ("/pytsf/libtimsdata_tsf.dylib", "pytsf")
    hook_tools.collect_dynamic_libs.side_effect = lambda name, **_: [pytsf_library] if name == "pytsf" else []

    hook = runpy.run_path(str(HOOKS / "hook-imzy.py"))

    assert hook["filter_imzy_data"]([native]) == [native]
    assert pytsf_library in hook["binaries"]


def test_imzy_hook_rejects_missing_rust_backend(hook_tools: SimpleNamespace) -> None:
    hook_tools.can_import_module.return_value = False

    with pytest.raises(RuntimeError, match="--update-deps"):
        runpy.run_path(str(HOOKS / "hook-imzy.py"))

    hook_tools.can_import_module.assert_called_once_with("imzy._imzml_rust")


def test_imzy_hook_does_not_collect_pyspa(hook_tools: SimpleNamespace) -> None:
    hook = runpy.run_path(str(HOOKS / "hook-imzy.py"))

    assert not any(name.startswith("pyspa") for name in hook["hiddenimports"])
    assert not any("pyspa" in source for source, _ in hook["datas"])
    assert not any(call.args[0] == "pyspa" for call in hook_tools.copy_metadata.call_args_list)
