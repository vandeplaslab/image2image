"""Exercise dependency selection and installation without changing any environment."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from package.install_dependencies import (
    APP_PACKAGES,
    LOCAL_REPOSITORIES,
    NATIVE_BUILD_ENV,
    QT_PACKAGES,
    InstallRequest,
    _run,
    create_parser,
    install_dependencies,
    main,
    needs_bootstrap,
    require_packaging_environment,
    selected_packages,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def local_repos(tmp_path: Path) -> Path:
    for directory in LOCAL_REPOSITORIES.values():
        path = tmp_path / directory
        path.mkdir()
        (path / "pyproject.toml").write_text("[project]\n")
    return tmp_path / "image2image"


@pytest.mark.parametrize("install_request", [InstallRequest(bootstrap=True), InstallRequest(update=True)])
def test_full_setup_selects_all_local_sources(install_request: InstallRequest) -> None:
    assert selected_packages(install_request) == tuple(LOCAL_REPOSITORIES)


@pytest.mark.parametrize(
    ("install_request", "expected"),
    [
        (InstallRequest(), ()),
        (InstallRequest(update_app=True), APP_PACKAGES),
        (InstallRequest(update_just_app=True), ("image2image",)),
        (InstallRequest(update_just_reader=True), ("image2image-io",)),
        (InstallRequest(update_just_register=True), ("image2image-reg",)),
        (InstallRequest(update_qt=True), QT_PACKAGES),
        (InstallRequest(update_pip=True), ()),
    ],
)
def test_selective_groups(install_request: InstallRequest, expected: tuple[str, ...]) -> None:
    assert selected_packages(install_request) == expected


def test_local_dependency_update_includes_native_and_qt_packages() -> None:
    packages = selected_packages(InstallRequest(update_deps=True, update_qt=True))
    assert {"imzy", "pytsf", "ims-utils", "yoki5", "koyo", *QT_PACKAGES} <= set(packages)
    assert not set(APP_PACKAGES) & set(packages)
    assert len(packages) == len(set(packages))


@pytest.mark.parametrize("alias", ["--update-app", "--update_app", "-update_app"])
def test_parser_preserves_aliases(alias: str) -> None:
    assert create_parser().parse_args([alias]).update_app


def test_native_update_preserves_sources_and_targets_interpreter(local_repos: Path) -> None:
    with patch("package.install_dependencies._run") as run:
        install_dependencies(local_repos, update_deps=True, dry_run=True)
    command = run.call_args_list[0].args[0]
    assert command[:6] == ["uv", "pip", "install", "--python", sys.executable, "--no-sources"]
    assert "--no-cache" in command
    assert "--upgrade" not in command
    assert "--force-reinstall" not in command
    assert "--editable" not in command
    assert str(local_repos.parent / "timsMac") in command
    assert str(local_repos) + "[pyqt6]" in command
    assert "--override" in command
    assert not any("imspy" in argument or "pyspa" in argument for argument in command)
    assert run.call_args_list[0].kwargs["env"] == NATIVE_BUILD_ENV
    assert "import imzy._imzml_rust, pytsf" in run.call_args_list[1].args[0][-1]


def test_build_without_updates_only_validates() -> None:
    with patch("package.install_dependencies._run") as run:
        install_dependencies(ROOT)
    run.assert_called_once()
    assert run.call_args.args[0][:2] == (sys.executable, "-c")
    assert "qtpy.API_NAME != 'PyQt6'" in run.call_args.args[0][-1]
    assert run.call_args.kwargs["env"] == {"QT_API": "pyqt6"}


def test_registry_updates_do_not_force_reinstall_local_packages(local_repos: Path) -> None:
    with patch("package.install_dependencies._run") as run:
        install_dependencies(local_repos, InstallRequest(update_pip=True), dry_run=True)
    command = run.call_args_list[0].args[0]
    assert "--upgrade" in command
    assert "--reinstall-package" not in command
    assert "--no-cache" not in command


def test_app_only_updates_do_not_force_reinstall_native_packages(local_repos: Path) -> None:
    with patch("package.install_dependencies._run") as run:
        install_dependencies(local_repos, InstallRequest(update_just_app=True), dry_run=True)
    command = run.call_args_list[0].args[0]
    reinstalls = [command[i + 1] for i, value in enumerate(command) if value == "--reinstall-package"]
    assert reinstalls == ["image2image"]


def test_resolver_override_selects_every_local_source(local_repos: Path) -> None:
    overrides: list[str] = []

    def run(command: list[str], **_: object) -> None:
        if command[:3] == ["uv", "pip", "install"]:
            overrides.append(Path(command[-1]).read_text())

    with patch("package.install_dependencies._run", side_effect=run):
        install_dependencies(local_repos, InstallRequest(bootstrap=True))
    assert f"ims-utils @ {(local_repos.parent / 'ims-utils').as_uri()}\n" in overrides[0]
    assert f"pytsf @ {(local_repos.parent / 'timsMac').as_uri()}\n" in overrides[0]
    assert len(overrides[0].splitlines()) == len(LOCAL_REPOSITORIES)


@pytest.mark.parametrize("failure_at", [0, 1])
def test_failed_command_aborts_packaging(local_repos: Path, failure_at: int) -> None:
    effects = [None] * failure_at + [subprocess.CalledProcessError(1, "command")]
    with (
        patch("package.install_dependencies._run", side_effect=effects) as run,
        pytest.raises(subprocess.CalledProcessError),
    ):
        install_dependencies(local_repos, InstallRequest(update=True))
    assert run.call_count == failure_at + 1


def test_missing_checkout_is_rejected_before_installing(local_repos: Path) -> None:
    (local_repos.parent / "timsMac" / "pyproject.toml").unlink()
    with (
        patch("package.install_dependencies._run") as run,
        pytest.raises(FileNotFoundError, match="timsMac"),
    ):
        install_dependencies(local_repos, update_deps=True)
    run.assert_not_called()


def test_build_flags_override_environment_only_in_subprocess(tmp_path: Path) -> None:
    with (
        patch.dict(os.environ, {"IMZY_BUILD_RUST": "0", "PATH": "/toolchain"}),
        patch("package.install_dependencies.subprocess.run") as run,
    ):
        _run(("uv", "pip", "install", "."), cwd=tmp_path, dry_run=False, env=NATIVE_BUILD_ENV)
        assert run.call_args.kwargs["env"]["IMZY_BUILD_RUST"] == "1"
        assert run.call_args.kwargs["env"]["PATH"] == "/toolchain"
        assert run.call_args.kwargs["check"]
        assert os.environ["IMZY_BUILD_RUST"] == "0"


def test_dry_run_has_no_files_processes_or_imports(local_repos: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("package.install_dependencies.subprocess.run") as run,
        patch("package.install_dependencies.tempfile.TemporaryDirectory") as temporary,
        patch("package.install_dependencies.require_packaging_environment") as require,
    ):
        install_dependencies(local_repos, InstallRequest(update=True), dry_run=True)
        assert main(["--dry-run"]) == 0
    run.assert_not_called()
    temporary.assert_not_called()
    require.assert_not_called()
    output = capsys.readouterr().out
    assert "timsMac" in output
    assert "IMZY_BUILD_RUST" in output
    assert "pyspa" not in output.lower()


@pytest.mark.parametrize("incomplete", [False, True])
def test_main_bootstraps_only_incomplete_environments(incomplete: bool) -> None:
    with (
        patch("package.install_dependencies.require_packaging_environment"),
        patch("package.install_dependencies.needs_bootstrap", return_value=incomplete),
        patch("package.install_dependencies.install_dependencies") as install,
    ):
        assert main([]) == 0
    assert install.call_args.args[1].bootstrap is incomplete


def test_bootstrap_detection_uses_metadata_only() -> None:
    with patch("package.install_dependencies.version", return_value="1.0"):
        assert not needs_bootstrap()
    from importlib.metadata import PackageNotFoundError

    with patch("package.install_dependencies.version", side_effect=PackageNotFoundError("pytsf")):
        assert needs_bootstrap()


@pytest.mark.parametrize("problem", ["system", "old_python", "32_bit", "rosetta", "windows_arm"])
def test_incompatible_environment_is_rejected(monkeypatch: pytest.MonkeyPatch, problem: str) -> None:
    monkeypatch.setattr(sys, "prefix", "/venv_package")
    monkeypatch.setattr(sys, "base_prefix", "/base")
    monkeypatch.setattr(sys, "version_info", (3, 12))
    monkeypatch.setattr(sys, "platform", "linux")
    if problem == "system":
        monkeypatch.setattr(sys, "prefix", "/base")
    elif problem == "old_python":
        monkeypatch.setattr(sys, "version_info", (3, 11))
    elif problem == "32_bit":
        monkeypatch.setattr("package.install_dependencies.struct.calcsize", lambda _: 4)
    else:
        monkeypatch.setattr(sys, "platform", "darwin" if problem == "rosetta" else "win32")
        monkeypatch.setattr(
            "package.install_dependencies.platform.machine", lambda: "x86_64" if problem == "rosetta" else "ARM64"
        )
    with pytest.raises(RuntimeError):
        require_packaging_environment()


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


def test_main_reports_incompatible_environment_without_installing() -> None:
    with (
        patch(
            "package.install_dependencies.require_packaging_environment", side_effect=RuntimeError("Use Python 3.12")
        ),
        patch("package.install_dependencies.install_dependencies") as install,
        pytest.raises(SystemExit, match=r"Use Python 3\.12"),
    ):
        main([])
    install.assert_not_called()
