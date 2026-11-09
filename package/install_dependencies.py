"""Install and validate the shared PyInstaller packaging environment."""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import struct
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

LOCAL_REPOSITORIES = {
    "image2image": "image2image",
    "image2image-io": "image2image-io",
    "image2image-reg": "image2image-reg",
    "koyo": "koyo",
    "qtextra": "qtextra",
    "qtextraplot": "qtextraplot",
    "imzy": "imzy",
    "ims-utils": "ims-utils",
    "yoki5": "yoki5",
    "pytsf": "timsMac",
}
APP_PACKAGES = ("image2image", "image2image-io", "image2image-reg")
QT_PACKAGES = ("qtextra", "qtextraplot")
LOCAL_DEPENDENCIES = tuple(name for name in LOCAL_REPOSITORIES if name not in APP_PACKAGES)
# The shared hooks collect pkg_resources, which setuptools removed in version 82.
PACKAGING_REQUIREMENTS = ("pyinstaller>=6", "PyQt6>=6.9.1", "setuptools<82")
LOCAL_EXTRAS = {"image2image": "pyqt6", "qtextra": "console,sentry", "qtextraplot": "1d,2d,mpl"}
NATIVE_BUILD_ENV = {"IMZY_BUILD_RUST": "1"}


@dataclass(frozen=True)
class InstallRequest:
    """Select the package groups to refresh; no flags means validate only."""

    bootstrap: bool = False
    update: bool = False
    update_app: bool = False
    update_deps: bool = False
    update_pip: bool = False
    update_qt: bool = False
    update_just_app: bool = False
    update_just_reader: bool = False
    update_just_register: bool = False


def selected_packages(request: InstallRequest) -> tuple[str, ...]:
    """Expand update groups into a deduplicated list of local distributions."""
    if request.bootstrap or request.update:
        return tuple(LOCAL_REPOSITORIES)
    packages: list[str] = []
    if request.update_app:
        packages.extend(APP_PACKAGES)
    for selected, name in (
        (request.update_just_app, "image2image"),
        (request.update_just_reader, "image2image-io"),
        (request.update_just_register, "image2image-reg"),
    ):
        if selected:
            packages.append(name)
    if request.update_deps:
        packages.extend(LOCAL_DEPENDENCIES)
    if request.update_qt:
        packages.extend(QT_PACKAGES)
    return tuple(dict.fromkeys(packages))


def _run(command: Sequence[str], *, cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> None:
    """Print and optionally execute one command in the packaging environment."""
    rendered = subprocess.list2cmdline(command) if sys.platform.startswith("win") else shlex.join(command)
    if env:
        print(f"Build environment: {env}")
    print(f"[{cwd}] {rendered}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True, env={**os.environ, **env} if env else None)  # noqa: S603


def require_packaging_environment() -> None:
    """Reject unsupported interpreters before changing any installed packages."""
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this installer with the venv_package Python interpreter, not the system Python.")
    if sys.version_info < (3, 12):
        raise RuntimeError(
            "Packaging requires Python 3.12 or newer. Rename the old venv_package directory and rerun build "
            "to create a Python 3.12 environment."
        )
    if struct.calcsize("P") != 8:
        raise RuntimeError("Packaging requires a 64-bit Python interpreter.")
    if sys.platform == "darwin" and platform.machine() != "arm64":
        raise RuntimeError("macOS packaging requires native arm64 Python. Run: arch -arm64 /bin/zsh")
    if sys.platform == "win32" and platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("Windows packaging requires an x64 Python interpreter.")


def needs_bootstrap() -> bool:
    """Detect an empty or interrupted packaging installation without importing Qt."""
    try:
        for name in (*LOCAL_REPOSITORIES, "pyinstaller", "PyQt6", "setuptools"):
            version(name)
    except PackageNotFoundError:
        return True
    return False


def install_dependencies(
    repo_dir: Path,
    request: InstallRequest | None = None,
    *,
    update_deps: bool = False,
    dry_run: bool = False,
) -> None:
    """Resolve local sources together, refresh selected groups, and validate native imports.

    The legacy ``update_deps`` keyword remains available for direct callers. All
    installations target this interpreter explicitly and use non-editable sources.
    """
    request = request or InstallRequest(update_deps=update_deps)
    packages = selected_packages(request)
    upgrade_registry = request.bootstrap or request.update or request.update_pip
    if packages or upgrade_registry:
        package_dirs = {name: repo_dir.parent / directory for name, directory in LOCAL_REPOSITORIES.items()}
        for package_dir in package_dirs.values():
            if not (package_dir / "pyproject.toml").is_file():
                raise FileNotFoundError(f"Local package pyproject.toml does not exist: {package_dir}")

        command = ["uv", "pip", "install", "--python", sys.executable, "--no-sources"]
        if upgrade_registry:
            command.append("--upgrade")
        if "imzy" in packages or "pytsf" in packages:
            command.append("--no-cache")
        for package in packages:
            command.extend(("--reinstall-package", package))
        command.extend(PACKAGING_REQUIREMENTS)
        for name, directory in package_dirs.items():
            extras = LOCAL_EXTRAS.get(name, "")
            command.append(f"{directory}[{extras}]" if extras else str(directory))

        # Override sibling version minima (notably ims-utils) and prevent the
        # resolver from substituting registry wheels for any of our local builds.
        overrides = "".join(f"{name} @ {directory.as_uri()}\n" for name, directory in package_dirs.items())
        if dry_run:
            print(f"Local source overrides:\n{overrides}", end="")
            _run([*command, "--override", "<local-source-overrides>"], cwd=repo_dir, dry_run=True, env=NATIVE_BUILD_ENV)
        else:
            with tempfile.TemporaryDirectory(prefix="image2image-packaging-") as directory:
                override_path = Path(directory) / "overrides.txt"
                override_path.write_text(overrides, encoding="utf-8")
                # One resolver transaction preserves all sibling sources and their native builds.
                _run([*command, "--override", str(override_path)], cwd=repo_dir, dry_run=False, env=NATIVE_BUILD_ENV)

    _run(
        (
            sys.executable,
            "-c",
            "import imzy._imzml_rust, pytsf; "
            "import PyInstaller, qtpy; "
            "\nif qtpy.API_NAME != 'PyQt6':\n    raise RuntimeError('Packaging requires PyQt6')\n"
            "print('imzy native:', imzy._imzml_rust.__file__); print('pytsf:', pytsf.__file__)",
        ),
        cwd=repo_dir,
        dry_run=dry_run,
        env={"QT_API": "pyqt6"},
    )
    # uv pip check cannot account for the intentional local ims-utils version override.
    # Dependency requirements are resolved together during installations instead.


def create_parser() -> argparse.ArgumentParser:
    """Create the shared installer CLI, including legacy update aliases."""
    parser = argparse.ArgumentParser(description=__doc__)
    flags = {
        "bootstrap": "Install all packages into an empty or incomplete environment.",
        "update": "Refresh all local and registry dependencies.",
        "update_app": "Refresh image2image, image2image-io, and image2image-reg.",
        "update_deps": "Refresh local ecosystem dependencies, including native backends and Qt helpers.",
        "update_pip": "Upgrade registry dependencies within project requirements.",
        "update_qt": "Refresh qtextra and qtextraplot.",
        "update_just_app": "Refresh only the image2image project.",
        "update_just_reader": "Refresh only the image2image-io project.",
        "update_just_register": "Refresh only the image2image-reg project.",
    }
    for name, help_text in flags.items():
        aliases = tuple(dict.fromkeys((f"--{name.replace('_', '-')}", f"--{name}", f"-{name}")))
        parser.add_argument(*aliases, dest=name, action="store_true", help=help_text)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without changing the environment.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Bootstrap, update, or validate the current packaging virtual environment."""
    args = create_parser().parse_args(argv)
    try:
        if not args.dry_run:
            require_packaging_environment()
            args.bootstrap = args.bootstrap or needs_bootstrap()
        request = InstallRequest(**{field: getattr(args, field) for field in InstallRequest.__dataclass_fields__})
        install_dependencies(Path(__file__).resolve().parent.parent, request, dry_run=args.dry_run)
    except subprocess.CalledProcessError as error:
        raise SystemExit(f"Packaging command failed with exit code {error.returncode}. See the output above.") from None
    except (OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from None
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
