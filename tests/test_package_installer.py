"""Check native backend installation without modifying the packaging environment."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from package.install_dependencies import NATIVE_BUILD_ENV, _run, install_dependencies, main

ROOT = Path(__file__).resolve().parents[1]


def test_update_rebuilds_all_backends_from_local_sources() -> None:
    repo_dir = Path("/github/image2image")
    with patch("package.install_dependencies._run") as run:
        install_dependencies(repo_dir, update_deps=True, dry_run=True)

    command = run.call_args_list[0].args[0]
    assert command[:6] == ["uv", "pip", "install", "--python", sys.executable, "--no-cache"]
    assert command[6:12] == [
        "--reinstall-package",
        "imzy",
        "--reinstall-package",
        "pytsf",
        "--reinstall-package",
        "ims-utils",
    ]
    assert command[12:15] == ["/github/imzy", "/github/timsMac", "/github/ims-utils"]
    assert command[15] == "--override"
    assert not any("imspy" in argument or "pyspa" in argument for argument in command)
    assert run.call_args_list[0].kwargs["env"] == NATIVE_BUILD_ENV
    assert "import imzy._imzml_rust, pytsf" in run.call_args_list[1].args[0][-1]


def test_build_without_updates_validates_installed_backends() -> None:
    with patch("package.install_dependencies._run") as run:
        install_dependencies(ROOT)

    run.assert_called_once()
    assert run.call_args.args[0][:2] == (sys.executable, "-c")
    assert "import imzy._imzml_rust, pytsf" in run.call_args.args[0][-1]
    assert "pyspa" not in run.call_args.args[0][-1]


def test_resolver_override_selects_local_ims_utils_source() -> None:
    overrides: list[str] = []

    def run(command: list[str], **_: object) -> None:
        if command[0] == "uv":
            overrides.append(Path(command[-1]).read_text())

    with patch("package.install_dependencies._run", side_effect=run):
        install_dependencies(Path("/github/image2image"), update_deps=True, dry_run=True)

    assert overrides == ["ims-utils @ file:///github/ims-utils\n"]


def test_failed_native_import_aborts_packaging() -> None:
    with (
        patch("package.install_dependencies._run", side_effect=subprocess.CalledProcessError(1, "python")),
        pytest.raises(subprocess.CalledProcessError),
    ):
        install_dependencies(ROOT)


def test_missing_checkout_is_rejected_before_installing(tmp_path: Path) -> None:
    repo_dir = tmp_path / "image2image"
    repo_dir.mkdir()
    (tmp_path / "imzy").mkdir()

    with (
        patch("package.install_dependencies._run") as run,
        pytest.raises(FileNotFoundError, match="timsMac"),
    ):
        install_dependencies(repo_dir, update_deps=True)

    run.assert_not_called()


def test_build_flags_override_environment_only_in_subprocess(tmp_path: Path) -> None:
    with (
        patch.dict(os.environ, {"IMZY_BUILD_RUST": "0", "PATH": "/toolchain"}),
        patch("package.install_dependencies.subprocess.run") as run,
    ):
        _run(("uv", "pip", "install", "."), cwd=tmp_path, dry_run=False, env=NATIVE_BUILD_ENV)

        assert run.call_args.kwargs["env"]["IMZY_BUILD_RUST"] == "1"
        assert "PYSPA_REQUIRE_RUST" not in NATIVE_BUILD_ENV
        assert run.call_args.kwargs["env"]["PATH"] == "/toolchain"
        assert run.call_args.kwargs["check"]
        assert os.environ["IMZY_BUILD_RUST"] == "0"


def test_dry_run_prints_commands_without_installing(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("package.install_dependencies.subprocess.run") as run:
        assert main(["--update-deps", "--dry-run"]) == 0

    run.assert_not_called()
    output = capsys.readouterr().out
    assert "timsMac" in output
    assert "IMZY_BUILD_RUST" in output
    assert "pyspa" not in output.lower()
    assert "import imzy._imzml_rust, pytsf" in output


def test_build_scripts_validate_native_backends_after_dependency_updates() -> None:
    macos = (ROOT / "package/macos-pyinstaller/build.sh").read_text()
    windows = (ROOT / "package/windows-pyinstaller/build.ps1").read_text()

    assert "$update_deps && dependency_args+=(--update-deps)" in macos
    assert macos.index("# after pip installs") < macos.index('python "$app_dir/package/install_dependencies.py"')
    assert macos.index('python "$app_dir/package/install_dependencies.py"') < macos.index("pyinstaller --noconfirm")
    assert '"${dependency_args[@]}" || exit 1' in macos
    assert '$dependency_args += "--update-deps"' in windows
    assert windows.index("foreach ($package in $after_install)") < windows.index('"package\\install_dependencies.py"')
    assert windows.index('"package\\install_dependencies.py"') < windows.index("if ($no_build)")
    assert 'throw "Native backend installation or validation failed' in windows


def test_universal_spec_triggers_imzy_hook() -> None:
    spec = (ROOT / "package/image2image.spec").read_text()
    hiddenimports = spec.split("hiddenimports=[", maxsplit=1)[1].split("]", maxsplit=1)[0]

    assert '"imzy"' in hiddenimports
    assert '"pyspa"' not in hiddenimports


@pytest.mark.parametrize(
    "spec_path", ["image2image.spec", "macos-pyinstaller/image2image.spec", "windows-pyinstaller/image2image.spec"]
)
def test_all_specs_explicitly_exclude_pyspa(spec_path: str) -> None:
    tree = ast.parse((ROOT / "package" / spec_path).read_text())
    exclusions = [
        ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.keyword) and node.arg == "excludes"
    ]

    assert len(exclusions) == 1
    assert "pyspa" in exclusions[0]
    assert not (ROOT / "package/_hooks/hook-pyspa.py").exists()
