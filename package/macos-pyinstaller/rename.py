"""Rename package to represent the macOS platform + version."""

import inspect
from pathlib import Path

from image2image import __version__

parent = Path(inspect.getfile(lambda: None)).parent.resolve()
source_dir = parent / "dist" / "image2image.dmg"
output_dir = parent / "dist" / f"image2image-v{__version__}-macosx_arm64.dmg"
source_dir.rename(output_dir)
