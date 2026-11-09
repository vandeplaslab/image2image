"""Create a versioned archive of the Windows PyInstaller distribution."""

from __future__ import annotations

from pathlib import Path
from shutil import make_archive


def create_archive(package_dir: Path, version: str) -> Path:
    """Archive the contents of a built application, rejecting missing executables."""
    source_dir = package_dir / "dist" / "image2image"
    if not (source_dir / "image2image.exe").is_file():
        raise FileNotFoundError(f"Build the application first: {source_dir / 'image2image.exe'}")
    output = package_dir / "dist" / f"image2image-v{version}-win_amd64"
    return Path(make_archive(str(output), "zip", source_dir))


def main() -> None:
    """Read the packaging environment's version and create its ZIP archive."""
    from image2image import __version__

    print(f"Created ZIP: {create_archive(Path(__file__).resolve().parent, __version__)}")


if __name__ == "__main__":
    main()
