"""Rename a legacy unversioned DMG using the installed application version."""

from pathlib import Path


def rename_package(package_dir: Path, version: str) -> Path:
    """Rename the legacy DMG and return its versioned path."""
    source = package_dir / "dist" / "image2image.dmg"
    destination = source.with_name(f"image2image-v{version}-macosx_arm64.dmg")
    source.rename(destination)
    return destination


if __name__ == "__main__":
    from image2image import __version__

    print(rename_package(Path(__file__).resolve().parent, __version__))
