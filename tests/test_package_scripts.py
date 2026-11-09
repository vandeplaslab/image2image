"""Check artifact helpers and shell orchestration without building the application."""

from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_helper(relative_path: str) -> ModuleType:
    """Load a packaging helper without invoking its CLI."""
    spec = importlib.util.spec_from_file_location("packaging_helper", ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_zip_preserves_distribution_contents_and_version(tmp_path: Path) -> None:
    helper = load_helper("package/windows-pyinstaller/zip.py")
    app_dir = tmp_path / "dist" / "image2image"
    (app_dir / "_internal").mkdir(parents=True)
    (app_dir / "image2image.exe").write_bytes(b"executable")
    (app_dir / "_internal" / "resource.txt").write_text("resource")
    output = helper.create_archive(tmp_path, "0.3.0")
    assert output.name == "image2image-v0.3.0-win_amd64.zip"
    with zipfile.ZipFile(output) as archive:
        assert archive.read("image2image.exe") == b"executable"
        assert archive.read("_internal/resource.txt") == b"resource"
    assert (app_dir / "image2image.exe").exists()


def test_zip_rejects_missing_executable(tmp_path: Path) -> None:
    helper = load_helper("package/windows-pyinstaller/zip.py")
    with pytest.raises(FileNotFoundError, match="Build the application first"):
        helper.create_archive(tmp_path, "0.3.0")
    assert not (tmp_path / "dist").exists()


def test_legacy_dmg_rename(tmp_path: Path) -> None:
    helper = load_helper("package/macos-pyinstaller/rename.py")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "image2image.dmg").write_bytes(b"dmg")
    output = helper.rename_package(tmp_path, "0.3.0")
    assert output.name == "image2image-v0.3.0-macosx_arm64.dmg"
    assert output.read_bytes() == b"dmg"


@pytest.fixture
def shell_build(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    if not shutil.which("zsh"):
        pytest.skip("zsh is required for macOS script tests")
    repo = tmp_path / "repo with spaces"
    script_dir = repo / "package" / "macos-pyinstaller"
    script_dir.mkdir(parents=True)
    shutil.copy(ROOT / "package/macos-pyinstaller/build.sh", script_dir)
    (script_dir / "package.sh").write_text('echo "package invoked" >> "$BUILD_LOG"\n')
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()
    log = tmp_path / "calls.log"
    python_stub = stub_dir / "python"
    python_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['BUILD_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1].endswith('install_dependencies.py'):\n"
        "    sys.exit(int(os.environ.get('FAIL_INSTALL', '0')))\n"
        "if '-m' in sys.argv and 'PyInstaller' in sys.argv:\n"
        "    sys.exit(int(os.environ.get('FAIL_BUILD', '0')))\n"
    )
    python_stub.chmod(0o755)
    (stub_dir / "uname").write_text('#!/bin/sh\nif [ "$1" = "-s" ]; then echo Darwin; else echo arm64; fi\n')
    (stub_dir / "uv").write_text(
        '#!/bin/sh\n[ "${FAIL_UV:-0}" = 0 ] || exit 1\n'
        'for last; do :; done\nmkdir -p "$last/bin"\ncp "$STUB_PYTHON" "$last/bin/python"\n'
    )
    for name in ("uname", "uv"):
        (stub_dir / name).chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}",
        "BUILD_LOG": str(log),
        "STUB_PYTHON": str(python_stub),
    }
    return script_dir, env, log


def test_shell_bootstrap_and_reuse_from_another_directory(shell_build: tuple[Path, dict[str, str], Path]) -> None:
    import json

    script_dir, env, log = shell_build
    script = str(script_dir / "build.sh")
    subprocess.run(["zsh", script, "-as"], cwd=script_dir.parents[2], env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == [str(script_dir.parent / "install_dependencies.py"), "--update-app", "--bootstrap"]
    log.unlink()
    subprocess.run(["zsh", script], cwd=script_dir.parents[2], env=env, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == [str(script_dir.parent / "install_dependencies.py")]
    assert any(call[:2] == ["-m", "PyInstaller"] for call in calls)


@pytest.mark.parametrize("failure", ["FAIL_UV", "FAIL_INSTALL", "FAIL_BUILD"])
def test_shell_failure_stops_downstream_steps(shell_build: tuple[Path, dict[str, str], Path], failure: str) -> None:
    script_dir, env, log = shell_build
    result = subprocess.run(["zsh", str(script_dir / "build.sh"), "-z"], env={**env, failure: "1"}, capture_output=True)
    assert result.returncode != 0
    if log.exists():
        assert "package invoked" not in log.read_text()
    if failure == "FAIL_UV":
        assert not log.exists()
    elif failure == "FAIL_INSTALL":
        assert "PyInstaller" not in log.read_text()


@pytest.mark.parametrize(
    "value, expected", [("0", False), ("false", False), ("off", False), ("all", "all"), ("1", "all")]
)
def test_specs_normalize_debug_mode(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool | str) -> None:
    monkeypatch.setenv("PYINSTALLER_DEBUG", value)
    for platform in ("macos", "windows"):
        tree = ast.parse((ROOT / f"package/{platform}-pyinstaller/image2image.spec").read_text())
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id in {"debug_mode", "DEBUG_MODE"} for target in node.targets
            )
        ]
        scope: dict[str, object] = {"os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "debug-mode", "exec"), scope)
        assert scope["DEBUG_MODE"] == expected


@pytest.mark.parametrize(
    "flags, forwarded",
    [
        (["-u"], ["--update"]),
        (["-r", "-w", "-j"], ["--update-just-reader", "--update-just-register", "--update-just-app"]),
        (["-d", "-q", "-i"], ["--update-deps", "--update-qt", "--update-pip"]),
        (["-p", "-n"], ["--update-pip"]),
    ],
)
def test_shell_forwards_update_flags(
    shell_build: tuple[Path, dict[str, str], Path], flags: list[str], forwarded: list[str]
) -> None:
    import json

    script_dir, env, log = shell_build
    subprocess.run(["zsh", str(script_dir / "build.sh"), *flags, "-s"], env=env, check=True)
    call = json.loads(log.read_text().splitlines()[0])
    assert call[1:] == [*forwarded, "--bootstrap"]


def test_shell_help_does_not_create_environment(shell_build: tuple[Path, dict[str, str], Path]) -> None:
    script_dir, env, log = shell_build
    subprocess.run(["zsh", str(script_dir / "build.sh"), "-h"], env=env, check=True, capture_output=True)
    assert not log.exists()
    assert not (script_dir / "venv_package").exists()


@pytest.mark.parametrize("signed, failure", [(False, ""), (True, ""), (True, "codesign"), (False, "hdiutil")])
def test_dmg_staging_preserves_app_and_cleans_up(tmp_path: Path, signed: bool, failure: str) -> None:
    if not shutil.which("zsh"):
        pytest.skip("zsh is required for macOS script tests")
    script_dir = tmp_path / "packaging with spaces"
    script_dir.mkdir()
    shutil.copy(ROOT / "package/macos-pyinstaller/package.sh", script_dir)
    (script_dir / "entitlements.plist").write_text("entitlements")
    app = script_dir / "dist" / "image2image.app"
    app.mkdir(parents=True)
    (app / "executable").write_text("original")
    venv = script_dir / "venv_package" / "bin"
    venv.mkdir(parents=True)
    (venv / "python").write_text("#!/bin/sh\necho 0.3.0\n")
    (venv / "python").chmod(0o755)
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "ditto").write_text('#!/bin/sh\ncp -R "$1" "$2"\n')
    log = tmp_path / "commands.log"
    for name in ("codesign", "hdiutil"):
        (stubs / name).write_text(
            f'#!/bin/sh\necho "{name} $*" >> "$DMG_LOG"\n'
            f'[ "$FAIL_COMMAND" != "{name}" ] || exit 1\n'
            + ('for last; do :; done\ntouch "$last"\n' if name == "hdiutil" else "")
        )
    for path in stubs.iterdir():
        path.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}",
        "MACOS_SIGN_IDENTITY": "Test Identity" if signed else "",
        "FAIL_COMMAND": failure,
        "DMG_LOG": str(log),
    }
    result = subprocess.run(["zsh", str(script_dir / "package.sh")], cwd=tmp_path, env=env, capture_output=True)
    assert result.returncode == (1 if failure else 0), result.stderr.decode()
    assert (app / "executable").read_text() == "original"
    assert not list((script_dir / "dist").glob(".dmg-*"))
    assert (script_dir / "dist" / "image2image-v0.3.0-macosx_arm64.dmg").exists() is (not failure)
    if signed and not failure:
        assert "codesign --verify --deep --strict" in log.read_text()
    if not signed:
        assert "codesign" not in log.read_text()
