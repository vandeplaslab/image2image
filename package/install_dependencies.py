"""Build and validate local mass-spectrometry backends for packaged applications."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

LOCAL_REPOSITORIES = {"imzy": "imzy", "pytsf": "timsMac", "ims-utils": "ims-utils"}
NATIVE_BUILD_ENV = {"IMZY_BUILD_RUST": "1"}


def _run(command: Sequence[str], *, cwd: Path, dry_run: bool, env: dict[str, str] | None = None) -> None:
    """Print and optionally execute one command in the packaging environment."""
    rendered = subprocess.list2cmdline(command) if sys.platform.startswith("win") else shlex.join(command)
    if env:
        print(f"Build environment: {env}")
    print(f"[{cwd}] {rendered}")
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True, env={**os.environ, **env} if env else None)  # noqa: S603


def install_dependencies(repo_dir: Path, *, update_deps: bool = False, dry_run: bool = False) -> None:
    """Rebuild requested local backends, then require their imports to succeed."""
    if update_deps:
        package_dirs = [repo_dir.parent / directory for directory in LOCAL_REPOSITORIES.values()]
        if not dry_run:
            for package_dir in package_dirs:
                if not package_dir.is_dir():
                    raise FileNotFoundError(f"Local package directory does not exist: {package_dir}")

        command = ["uv", "pip", "install", "--python", sys.executable, "--no-cache"]
        for package in LOCAL_REPOSITORIES:
            command.extend(("--reinstall-package", package))
        command.extend(str(directory) for directory in package_dirs)
        # imzy's ims-utils minimum is ahead of the sibling checkout's development version.
        override = f"ims-utils @ {(repo_dir.parent / 'ims-utils').as_uri()}\n"
        with tempfile.TemporaryDirectory(prefix="image2image-packaging-") as directory:
            override_path = Path(directory) / "overrides.txt"
            override_path.write_text(override, encoding="utf-8")
            command.extend(("--override", str(override_path)))
            print(f"Local dependency override: {override.strip()}")
            # Resolve the local projects together so none can be replaced by a registry wheel.
            _run(command, cwd=repo_dir, dry_run=dry_run, env=NATIVE_BUILD_ENV)

    _run(
        (
            sys.executable,
            "-c",
            (
                "import imzy._imzml_rust, pytsf; "
                "print('imzy native:', imzy._imzml_rust.__file__); "
                "print('pytsf:', pytsf.__file__)"
            ),
        ),
        cwd=repo_dir,
        dry_run=dry_run,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Install or validate native backends in the current packaging environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-deps", action="store_true", help="Rebuild imzy and pytsf from sibling checkouts.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without changing the environment.")
    args = parser.parse_args(argv)
    install_dependencies(Path(__file__).resolve().parent.parent, update_deps=args.update_deps, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
